"""Training-free CLoRA-style antennas on frozen EAT, with real general audio (ESC-50) as the reference.

Features: per layer, each clip's tokens averaged over time within each of the 8 frequency rows (clip level).
mu = this machine's normal mean per row. Both covariances are taken around mu, pooled over rows:
  S_norm = E[(x - mu)(x - mu)^T] over the machine's normal clips
  S_gen  = E[(g - mu)(g - mu)^T] over ESC-50 clips (2000 recordings, each looped to 10 s)
Antennas (rank r):
  gen-loud   top solutions of S_gen a = lambda S_norm a   -> directions real-world sounds use and this machine's
             normal sound does not. Score = energy along them (a^T S_norm a = 1).
  signature  top solutions of S_norm a = lambda S_gen a   -> this machine's own directions, where general audio is
             quiet. Score = Mahalanobis inside them (normal variance per direction = lambda).
Per clip: max over rows (pre-committed headline) and mean over rows; layers z-normalised on 100 held-out normal clips
and summed.  Usage: uv run python -u clora_gen.py --name v1 --machines ToyCar,fan
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch

from clora_ad import LAYERS, fbank, load_eat, tokens
from clora_ft import EscMixer
from dcase_ae.efficientad import WavStore, build_cache
from dcase_ae.evaluation import compute_metrics

KINDS = ("gen-loud", "signature")
AGGS = ("max", "mean")


DASHENG_LAYERS = [7, 13, 20, 27, 33, 40]  # same relative depths as EAT's 2,4,...,12 of 12


def get_encoder(name, dev):
    """-> (layers, fn(waves) -> {layer: (N, rows, D)} time-averaged tokens per frequency row).
    EAT: 8 frequency rows of 16 mel bins. Dasheng: tokens span all 64 mel bins, so 1 row."""
    if name == "eat":
        model, grab = load_eat(dev)

        @torch.no_grad()
        def fn(waves, bs=8):
            out = {L: [] for L in LAYERS}
            for i in range(0, len(waves), bs):
                fbs = [fbank(torch.as_tensor(w, device=dev).float()) for w in waves[i : i + bs]]
                tc = tokens(model, grab, fbs)
                for L in LAYERS:
                    out[L].append(tc[L].mean(1).cpu())
            return {L: torch.cat(v) for L, v in out.items()}
        return LAYERS, fn

    from dasheng import dasheng_12B
    model = dasheng_12B(path="pretrained/dasheng/dasheng_12b.pt").to(dev).eval()
    grab = {}
    for L in DASHENG_LAYERS:
        model.blocks[L - 1].register_forward_hook(lambda m, a, o, L=L: grab.__setitem__(L, o))

    @torch.no_grad()
    def fn(waves, bs=8):
        out = {L: [] for L in DASHENG_LAYERS}
        for i in range(0, len(waves), bs):
            w = torch.stack([torch.as_tensor(x, device=dev).float() for x in waves[i : i + bs]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(w)
            for L in DASHENG_LAYERS:
                out[L].append(grab[L].float().mean(1, keepdim=True).cpu())  # (B, 1, D)
        return {L: torch.cat(v) for L, v in out.items()}
    return DASHENG_LAYERS, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--machines", default="ToyCar,fan")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder", default="eat", choices=["eat", "dasheng12b"])
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = Path("outputs/clora_gen") / args.name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"=== run '{args.name}': {vars(args)}")
    LAYERS, row_features = get_encoder(args.encoder, dev)

    gcache = Path(f"cache/esc50_{args.encoder}_rows.pt")
    if gcache.exists():
        G = torch.load(gcache)
    else:
        t0 = time.time()
        esc = EscMixer(dev).wav.float()  # (2000, 80000) at 16 kHz
        G = row_features(esc.repeat(1, 2))  # loop each 5 s recording to 10 s
        torch.save(G, gcache)
        log(f"general audio: EAT row features of {len(esc)} ESC-50 clips in {time.time() - t0:.0f}s")

    results = {}
    for m in args.machines.split(","):
        for split in ("train", "test"):
            build_cache("data", "cache", m, split)
        tr, te = WavStore("cache", m, "train"), WavStore("cache", m, "test")
        idx = list(range(len(tr)))
        random.Random(args.seed).shuffle(idx)
        val_idx, fit_idx = idx[:100], idx[100:]
        t0 = time.time()
        feats = {}
        for name, store, ids in (("fit", tr, fit_idx), ("val", tr, val_idx), ("test", te, list(range(len(te))))):
            parts = [row_features(store.get(ids[i : i + 64])[:, 0]) for i in range(0, len(ids), 64)]
            feats[name] = {L: torch.cat([p[L] for p in parts]) for L in LAYERS}
        log(f"--- {m}: features in {time.time() - t0:.0f}s")

        scores = {}
        for L in LAYERS:
            X = feats["fit"][L].double()
            mu = X.mean(0)  # (rows, D)
            DIM = X.shape[-1]
            Xc = (X - mu).reshape(-1, DIM)
            Gc = (G[L].double() - mu).reshape(-1, DIM)
            Sn = (Xc.T @ Xc / len(Xc)).numpy()
            Sg = (Gc.T @ Gc / len(Gc)).numpy()
            reg = lambda S: S + 1e-3 * np.trace(S) / DIM * np.eye(DIM)
            Sn, Sg = reg(Sn), reg(Sg)
            lam_g, Vg = scipy.linalg.eigh(Sg, Sn, subset_by_index=[DIM - args.rank, DIM - 1])  # Vg^T Sn Vg = I
            lam_s, Vs = scipy.linalg.eigh(Sn, Sg, subset_by_index=[DIM - args.rank, DIM - 1])  # Vs^T Sg Vs = I, Vs^T Sn Vs = lam_s
            log(f"  layer {L:2d}: general/normal variance in gen-loud directions {lam_g.mean():.0f}x (min {lam_g.min():.0f}x) | "
                f"normal/general in signature directions {lam_s.mean():.2f}x (max {lam_s.max():.2f}x)")
            Vg, Vs = torch.tensor(Vg), torch.tensor(Vs)
            ls = torch.tensor(lam_s)
            for split in ("val", "test"):
                Z = feats[split][L].double() - mu  # (N, FR, DIM)
                e = {"gen-loud": (Z @ Vg).pow(2).sum(-1), "signature": ((Z @ Vs).pow(2) / ls).sum(-1)}  # (N, FR)
                for k in KINDS:
                    scores[(split, L, k, "max")] = e[k].amax(1).numpy()
                    scores[(split, L, k, "mean")] = e[k].mean(1).numpy()

        lab, dom = np.array(te.meta["label"]), np.array(te.meta["domain"])
        z = lambda L, k, a: (scores[("test", L, k, a)] - scores[("val", L, k, a)].mean()) / (scores[("val", L, k, a)].std() + 1e-9)
        res = {}
        log(f"--- {m} results (AUC; all layers = AUC/official):")
        for k in KINDS:
            for a in AGGS:
                per = {L: compute_metrics(scores[("test", L, k, a)], lab, dom) for L in LAYERS}
                allr = compute_metrics(sum(z(L, k, a) for L in LAYERS), lab, dom)
                res[f"{k} {a}"] = {"layers": {L: per[L] for L in LAYERS}, "all": allr}
                log(f"  {k:9s} {a:4s} | " + "  ".join(f"L{L} {per[L]['auc_all']:.3f}" for L in LAYERS)
                    + f" | all layers {allr['auc_all']:.3f}/{allr['official']:.3f}")
        results[m] = res
    (out / "summary.json").write_text(json.dumps(results, indent=1, default=str))
    log("=== headline (all layers, max over rows):")
    for m, r in results.items():
        log(f"  {m:11s} " + "  ".join(f"{k} {r[k + ' max']['all']['auc_all']:.3f}/{r[k + ' max']['all']['official']:.3f}" for k in KINDS))
    log("=== done")


if __name__ == "__main__":
    main()
