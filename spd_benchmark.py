"""Spectral-perturbation detection (SPD): per machine, learn to detect and locate tiny synthetic spectral bumps.

Input: generic 4-channel stereo features (dcase_ae/efficientad.py), 256 bands, 4 s crops while training.
Perturbation (on normalised log-power, applied to near / far / both mics): Gaussian bump across 1-8 bands,
height 0.5-4 dB (sign random), over the whole crop (steady) or a 0.3-2 s part (transient). 50% of crops perturbed.
Network: spectrogram PDN (narrow in frequency, long in time) -> time-average -> per-band logits + clip logit.
Loss: BCE(clip logit, perturbed?) + BCE(per-band logits, bump location).
Scores on full 10 s test clips: clip logit, max per-band logit, and kNN on the time-averaged embedding; each
z-normalised with held-out normal training clips; 'final' = sum of the first two. Same settings for every machine.

Usage:  uv run python spd_benchmark.py --name spd_v1 [--machines ToyCar] [--steps 3000]
Live log: outputs/spd/<name>/train.log
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dcase_ae.data.prepare import list_machines
from dcase_ae.efficientad import CLIP, PDN, SR, WavStore, build_cache, features, logpower_stats
from dcase_ae.evaluation import compute_metrics

HOP = 256


class SPDNet(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.pdn = PDN(dim)
        self.band = nn.Sequential(nn.Conv1d(dim, 64, 5, padding=2), nn.GELU(), nn.Conv1d(64, 1, 1))
        self.clip = nn.Sequential(nn.Linear(2 * dim, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        e = F.gelu(self.pdn(x)).mean(3)  # (B, D, F) time-averaged embedding
        band = self.band(e)[:, 0]  # (B, F)
        clip = self.clip(torch.cat([e.mean(2), e.amax(2)], 1))[:, 0]
        return clip, band, e


def perturb(x, stats, rng, dev):
    """x (B, 4, F, T) normalised features -> (perturbed x, clip labels (B,), band targets (B, F))."""
    B, _, Fn, T = x.shape
    x = x.clone()
    lab = torch.zeros(B, device=dev)
    tgt = torch.zeros(B, Fn, device=dev)
    f = torch.arange(Fn, device=dev).float()
    for i in range(B):
        if rng.random() < 0.5:
            continue
        lab[i] = 1
        c, w = rng.uniform(0, Fn - 1), rng.uniform(0.5, 4.0)  # centre band, width (sigma, bands)
        h_db = rng.choice([-1, 1]) * rng.uniform(0.5, 4.0)
        prof = torch.exp(-0.5 * ((f - c) / w) ** 2)
        chans = rng.choice([[0], [1], [0, 1]])  # near, far or both mics
        if rng.random() < 0.7:  # steady
            t0, t1 = 0, T
        else:  # transient
            d = int(rng.uniform(0.3, 2.0) * SR / HOP)
            t0 = rng.randrange(0, max(1, T - d))
            t1 = t0 + d
        for ch in chans:
            x[i, ch, :, t0:t1] += (h_db / stats[1][ch]) * prof[:, None]
        tgt[i] = (prof > 0.5).float()
    return x, lab, tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="spd_v1")
    ap.add_argument("--machines", default="ToyCar")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = Path("outputs/spd") / args.name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"=== run '{args.name}': {vars(args)}")
    results = {}
    for m in args.machines.split(","):
        for split in ("train", "test"):
            build_cache("data", "cache", m, split)
        store, test = WavStore("cache", m, "train"), WavStore("cache", m, "test")
        stats = logpower_stats(store, dev)
        rng = random.Random(args.seed)
        torch.manual_seed(args.seed)
        idx = list(range(len(store)))
        rng.shuffle(idx)
        val_idx, tr_idx = idx[:100], idx[100:]
        dom = store.meta["domain"]
        lab_t, dom_t = np.array(test.meta["label"]), np.array(test.meta["domain"])
        net = SPDNet().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
        length = int(args.crop * SR)

        @torch.no_grad()
        def outputs(st, ids):
            net.eval()
            c, b, e = [], [], []
            for i in range(0, len(ids), 8):
                x = features(st.get(ids[i : i + 8]).to(dev), stats)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ci, bi, ei = net(x)
                c.append(ci.float())
                b.append(bi.float().amax(1))
                e.append(F.avg_pool1d(ei.float(), 8))  # (B, D, 32) for kNN
            net.train()
            return torch.cat(c), torch.cat(b), torch.cat(e).flatten(1)

        @torch.no_grad()
        def evaluate(step):
            vc, vb, ve = outputs(store, val_idx)
            tc, tb, te_ = outputs(test, list(range(len(test))))
            ac, ab, ae = outputs(store, tr_idx)
            z = lambda a, ref: ((a - ref.mean()) / (ref.std() + 1e-9)).cpu().numpy()
            mu, sd = ae.mean(0), ae.std(0) + 1e-6
            knn = lambda q: torch.cdist((q - mu) / sd, (ae - mu) / sd).topk(1, largest=False).values[:, 0]
            scores = {"clip logit": z(tc, vc), "max band logit": z(tb, vb), "kNN embedding": z(knn(te_), knn(ve))}
            scores["final"] = scores["clip logit"] + scores["max band logit"]
            res = {k: compute_metrics(v, lab_t, dom_t) for k, v in scores.items()}
            f = res["final"]
            log(f"  [{m} eval step {step}] final: AUC {f['auc_all']:.3f} (src {f['auc_source']:.3f} / tgt {f['auc_target']:.3f})"
                f" pAUC {f['pauc']:.3f} official {f['official']:.3f} | " + "  ".join(
                    f"{k} {v['auc_all']:.3f}" for k, v in res.items() if k != "final"))
            return res

        log(f"--- {m}: {len(tr_idx)} training clips, {len(val_idx)} held out")
        t0 = time.time()
        for step in range(1, args.steps + 1):
            picks = rng.sample(tr_idx, args.batch)
            starts = [rng.randrange(0, CLIP - length + 1) for _ in picks]
            w = torch.stack([store.get([i], s, length)[0] for i, s in zip(picks, starts)]).to(dev)
            x, lab, tgt = perturb(features(w, stats), stats, rng, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, b, _ = net(x)
            loss_c = F.binary_cross_entropy_with_logits(c.float(), lab)
            loss_b = F.binary_cross_entropy_with_logits(b.float(), tgt)
            loss = loss_c + loss_b
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 50 == 0:
                acc = ((c.float() > 0) == (lab > 0)).float().mean().item()
                log(f"  [{m}] step {step:5d}/{args.steps}  loss clip {loss_c.item():.3f}  band {loss_b.item():.3f}"
                    f"  clip acc {acc:.2f}  ({(time.time() - t0) / step:.2f}s/step)")
            if step % args.eval_every == 0 or step == args.steps:
                res = evaluate(step)
        results[m] = res
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
    log("=== done")


if __name__ == "__main__":
    main()
