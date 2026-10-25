"""CLoRA-style 'antenna' anomaly detection on frozen EAT activations (training-free, per machine).

For each EAT layer, with patch tokens x (768-d) at time t and frequency row f (64 x 8 patches per clip):
  S_norm  covariance of normal tokens after removing each frequency row's mean       ("normal variation")
  S_delta covariance of paired differences f(x~) - f(x), x~ = the same clip with a subtle synthetic spectral bump
Bases (rank r):
  contrastive  top solutions of S_delta a = lambda (S_norm + eps I) a  -> loud for anomaly-like changes, quiet for
               normal variation (Fisher / Rayleigh quotient); scaled so a^T S_norm a = 1
  null         bottom eigenvectors of S_norm                            -> directions normal clips never use
  full         Mahalanobis over all directions (reference: no direction selection)
Score per patch = energy along the basis; per clip: mean over time per frequency row, then max over rows ('steady')
or mean over everything ('mean'); each layer z-normalised on held-out normal clips; layers summed.

Usage:  uv run python clora_ad.py [--machines ToyCar] [--rank 32]
Log:    outputs/clora_ad/<name>/train.log
"""
import argparse
import json
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torchaudio
from safetensors.torch import load_file

from dcase_ae.efficientad import WavStore, build_cache
from dcase_ae.evaluation import compute_metrics

sys.path.insert(0, str(Path(__file__).parent))
from third_party.eat.eat_model import EAT  # noqa: E402

LAYERS = [2, 4, 6, 8, 10, 12]
FR, TR, DIM = 8, 64, 768  # frequency rows, time steps, token dim
LN_PER_DB = np.log(10) / 10
AGGS = ("steady", "mean", "clip-level max", "clip-level mean")


def load_eat(dev):
    cfg = types.SimpleNamespace(
        model_variant="pretrain", img_size=[1024, 128], patch_size=16, in_chans=1, embed_dim=768, stride=16,
        drop_rate=0.0, fixed_positions=True, norm_eps=1e-6, norm_affine=True, start_drop_path_rate=0.0,
        end_drop_path_rate=0.0, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, attn_drop_rate=0.0,
        activation_dropout=0.0, post_mlp_drop=0.0, layer_norm_first=False, mel_bins=128, max_length=768, num_classes=527)
    m = EAT(cfg)
    sd = {k.removeprefix("model."): v for k, v in load_file("pretrained/EAT-base_epoch30_pretrain/model.safetensors").items()}
    m.load_state_dict(sd, strict=False)
    m = m.to(dev).eval()
    grab = {}
    for L in LAYERS:
        m.blocks[L - 1].register_forward_hook(lambda mod, a, out, L=L: grab.__setitem__(L, out[0]))
    return m, grab


def fbank(w):
    """near-mic waveform (L,) on GPU -> (1024, 128) natural-log mel energies."""
    w = w - w.mean()
    fb = torchaudio.compliance.kaldi.fbank(w[None], htk_compat=True, sample_frequency=16000, use_energy=False,
                                           window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
    return torch.nn.functional.pad(fb, (0, 0, 0, max(0, 1024 - len(fb))))[:1024]


def bump(fb, rng):
    """Subtle synthetic spectral change: Gaussian bump over 1-4 mel bins, +-1..4 dB, steady (70%) or 0.3-2 s."""
    f = torch.arange(128, device=fb.device).float()
    prof = torch.exp(-0.5 * ((f - rng.uniform(0, 127)) / rng.uniform(1, 4)) ** 2)
    h = rng.choice([-1, 1]) * rng.uniform(1, 4) * LN_PER_DB
    out = fb.clone()
    if rng.random() < 0.7:
        out += h * prof[None]
    else:
        d = int(rng.uniform(30, 200))
        t0 = rng.randrange(0, 1024 - d)
        out[t0 : t0 + d] += h * prof[None]
    return out


@torch.no_grad()
def tokens(model, grab, fbs):
    """list of (1024, 128) fbanks -> {layer: (B, TR, FR, DIM)}"""
    x = torch.stack([(f + 4.268) / (4.569 * 2) for f in fbs])[:, None]
    model.extract_features(x)
    return {L: grab[L][:, 1:].float().reshape(len(fbs), TR, FR, DIM) for L in LAYERS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--machines", default="ToyCar")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = Path("outputs/clora_ad") / args.name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"=== run '{args.name}': {vars(args)}")
    model, grab = load_eat(dev)
    results = {}
    for m in args.machines.split(","):
        for split in ("train", "test"):
            build_cache("data", "cache", m, split)
        tr, te = WavStore("cache", m, "train"), WavStore("cache", m, "test")
        rng = random.Random(args.seed)
        idx = list(range(len(tr)))
        rng.shuffle(idx)
        val_idx, fit_idx = idx[:100], idx[100:]

        # ---- pass 1: per-row moments of normal tokens + covariance of paired differences --------------------
        t0 = time.time()
        acc = {L: {"n": torch.zeros(FR, device=dev, dtype=torch.float64),
                   "s": torch.zeros(FR, DIM, device=dev, dtype=torch.float64),
                   "ss": torch.zeros(FR, DIM, DIM, device=dev, dtype=torch.float64),
                   "dd": torch.zeros(DIM, DIM, device=dev, dtype=torch.float64), "nd": 0} for L in LAYERS}
        for i in range(0, len(fit_idx), 8):
            w = tr.get(fit_idx[i : i + 8])[:, 0].to(dev)  # near mic
            clean = [fbank(x) for x in w]
            pert = [bump(f, rng) for f in clean]
            tc, tp = tokens(model, grab, clean), tokens(model, grab, pert)
            for L in LAYERS:
                a = acc[L]
                x = tc[L].double().permute(2, 0, 1, 3).reshape(FR, -1, DIM)  # (FR, B*TR, DIM)
                a["n"] += x.shape[1]
                a["s"] += x.sum(1)
                a["ss"] += torch.einsum("fni,fnj->fij", x, x)
                d = (tp[L] - tc[L]).double().reshape(-1, DIM)
                a["dd"] += d.T @ d
                a["nd"] += len(d)
                xc = tc[L].double().mean(1).permute(1, 0, 2)  # (FR, B, DIM) time-averaged per row
                a["cn"] = a.get("cn", 0) + xc.shape[1]
                a["cs"] = a.get("cs", 0) + xc.sum(1)
                a["css"] = a.get("css", 0) + torch.einsum("fni,fnj->fij", xc, xc)
                dc = (tp[L] - tc[L]).double().mean(1).reshape(-1, DIM)
                a["cdd"] = a.get("cdd", 0) + dc.T @ dc
                a["cnd"] = a.get("cnd", 0) + len(dc)
        log(f"--- {m}: fitted moments on {len(fit_idx)} normal clips (+ perturbed copies) in {time.time() - t0:.0f}s")

        def make_bases(n, s, ss, dd, nd):
            mu = s / n  # (FR, DIM)
            S = (ss - torch.einsum("fi,fj->fij", mu, mu) * n).sum(0) / (n * FR)
            Sd = dd / nd
            S, Sd = S.cpu().numpy(), Sd.cpu().numpy()
            Sr = S + 1e-3 * np.trace(S) / DIM * np.eye(DIM)
            lam, Vc = scipy.linalg.eigh(Sd, Sr, subset_by_index=[DIM - args.rank, DIM - 1])  # V^T Sr V = I
            ev, U = np.linalg.eigh(Sr)
            to = lambda M: torch.tensor(M, device=dev, dtype=torch.float32)
            return {"mu": mu.float(), "contrastive": to(Vc.T), "null": to(U[:, : args.rank].T), "full": to((U / np.sqrt(ev)).T)}, lam

        bases, cbases = {}, {}
        for L in LAYERS:
            a = acc[L]
            n = a["n"][0]  # same count for every row
            bases[L], lam = make_bases(n, a["s"], a["ss"], a["dd"], a["nd"])
            cbases[L], clam = make_bases(a["cn"], a["cs"], a["css"], a["cdd"], a["cnd"])
            log(f"  layer {L:2d}: anomaly-change / normal-variation ratio in the best directions: patch level "
                f"{lam.mean():.2f}x (max {lam.max():.2f}x) | clip level (time-averaged) {clam.mean():.2f}x (max {clam.max():.2f}x)")

        # ---- pass 2: scores ----------------------------------------------------------------------------------
        @torch.no_grad()
        def scores(store, ids):
            res = {(L, b, agg): [] for L in LAYERS for b in ("contrastive", "null", "full") for agg in AGGS}
            for i in range(0, len(ids), 8):
                w = store.get(ids[i : i + 8])[:, 0].to(dev)
                tc = tokens(model, grab, [fbank(x) for x in w])
                for L in LAYERS:
                    x = tc[L] - bases[L]["mu"][None, None]  # (B, TR, FR, DIM)
                    xc = tc[L].mean(1) - cbases[L]["mu"][None]  # (B, FR, DIM) time-averaged
                    for b in ("contrastive", "null", "full"):
                        e = (x @ bases[L][b].T).pow(2).sum(-1)  # (B, TR, FR) energy per patch
                        res[(L, b, "steady")].append(e.mean(1).amax(1).cpu())
                        res[(L, b, "mean")].append(e.mean(dim=(1, 2)).cpu())
                        ec = (xc @ cbases[L][b].T).pow(2).sum(-1)  # (B, FR) energy per row of the clip average
                        res[(L, b, "clip-level max")].append(ec.amax(1).cpu())
                        res[(L, b, "clip-level mean")].append(ec.mean(1).cpu())
            return {k: torch.cat(v).numpy() for k, v in res.items()}

        sv, st = scores(tr, val_idx), scores(te, list(range(len(te))))
        lab, dom = np.array(te.meta["label"]), np.array(te.meta["domain"])
        z = lambda k: (st[k] - sv[k].mean()) / (sv[k].std() + 1e-9)
        res = {}
        for b in ("contrastive", "null", "full"):
            for agg in AGGS:
                for L in LAYERS:
                    res[f"{b} {agg} L{L}"] = compute_metrics(st[(L, b, agg)], lab, dom)
                res[f"{b} {agg} all layers"] = compute_metrics(sum(z((L, b, agg)) for L in LAYERS), lab, dom)
                res[f"{b} {agg} early layers 2-6"] = compute_metrics(sum(z((L, b, agg)) for L in (2, 4, 6)), lab, dom)
        log(f"--- {m} results (AUC / official):")
        for b in ("contrastive", "null", "full"):
            for agg in AGGS:
                per = "  ".join(f"L{L} {res[f'{b} {agg} L{L}']['auc_all']:.3f}" for L in LAYERS)
                a, e = res[f"{b} {agg} all layers"], res[f"{b} {agg} early layers 2-6"]
                log(f"  {b:11s} {agg:15s} | {per} | all layers {a['auc_all']:.3f}/{a['official']:.3f}  early {e['auc_all']:.3f}/{e['official']:.3f}")
        results[m] = res
    (out / "summary.json").write_text(json.dumps(results, indent=1))
    key = "null clip-level max all layers"  # pre-committed configuration (chosen on ToyCar)
    log(f"=== fixed config '{key}':")
    vals = []
    for m, res in results.items():
        r = res[key]
        vals += [r["auc_source"], r["auc_target"], r["pauc"]]
        log(f"  {m:11s} AUC {r['auc_all']:.3f}  src {r['auc_source']:.3f}  tgt {r['auc_target']:.3f}  pAUC {r['pauc']:.3f}  official {r['official']:.3f}")
    from scipy.stats import hmean
    log(f"  OVERALL     mean AUC {np.mean([res[key]['auc_all'] for res in results.values()]):.3f}  Omega {hmean(vals):.3f}")
    log("=== done")


if __name__ == "__main__":
    main()
