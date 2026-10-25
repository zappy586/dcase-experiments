"""Frequency-conditioned normalizing flow on the leave-one-machine-out teacher's features (see dcase_ae/flow.py).

Variants:
  rows      one vector per clip per frequency row = time-average of the teacher features
  positions every (frequency row, time step) vector; at test time the NLL is averaged over time per row
Score per clip: NLL per row, then max / top-10% / mean over rows. Same settings for every machine.

Usage:  uv run python flow_benchmark.py --name flow_v1 [--machines ToyCar] [--steps 3000]
Live log: outputs/flow/<name>/train.log
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import hmean

from dcase_ae.data.prepare import list_machines
from dcase_ae.efficientad import D, PDN, WavStore, build_cache, features, logpower_stats, train_teacher
from dcase_ae.evaluation import compute_metrics
from dcase_ae.flow import CondFlow

T_SAMPLES = 16  # time steps kept per training clip for the 'positions' variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="flow_v1")
    ap.add_argument("--machines", default="ToyCar")
    ap.add_argument("--variants", default="rows,positions")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--teacher_steps", type=int, default=1500, help="only used if no saved teacher exists")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = Path("outputs/flow") / args.name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"=== run '{args.name}': {vars(args)}")
    all_machines = list_machines("dataset")
    for m in all_machines:
        for split in ("train", "test"):
            build_cache("data", "cache", m, split)
    train = {m: WavStore("cache", m, "train") for m in all_machines}
    stats = {m: logpower_stats(train[m], dev) for m in all_machines}

    results = {}
    for m in args.machines.split(","):
        saved = Path(f"outputs/efficientad/v1/model_{m}.pt")
        if saved.exists():
            teacher = PDN(D).to(dev).eval()
            teacher.load_state_dict(torch.load(saved, map_location=dev)["teacher"])
            for p in teacher.parameters():
                p.requires_grad_(False)
            log(f"--- {m}: loaded leave-one-out teacher from {saved}")
        else:
            others = [o for o in all_machines if o != m]
            log(f"--- {m}: training leave-one-out teacher on {others}")
            teacher = train_teacher([train[o] for o in others], [stats[o] for o in others], dev, args.teacher_steps, log, args.seed)

        test = WavStore("cache", m, "test")
        lab, dom = np.array(test.meta["label"]), np.array(test.meta["domain"])
        rng = np.random.default_rng(args.seed)

        @torch.no_grad()
        def teacher_maps(store, ids):
            """Yields (ids chunk, (B, D, F, T') teacher features)."""
            for i in range(0, len(ids), 8):
                chunk = ids[i : i + 8]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    yield chunk, teacher(features(store.get(chunk).to(dev), stats[m])).float()

        # training vectors from normal clips: row means and sampled positions
        ids = list(range(len(train[m])))
        rows, pos = [], []
        for _, f in teacher_maps(train[m], ids):
            rows.append(f.mean(3).permute(0, 2, 1))  # (B, F, D)
            sel = torch.from_numpy(rng.choice(f.shape[3], T_SAMPLES, replace=False)).to(dev)
            pos.append(f[..., sel].permute(0, 2, 3, 1))  # (B, F, T_SAMPLES, D)
        rows, pos = torch.cat(rows), torch.cat(pos)
        n_freq = rows.shape[1]
        log(f"  {m}: {len(ids)} normal clips, {n_freq} frequency rows, feature dim {D}")

        for variant in args.variants.split(","):
            data = rows if variant == "rows" else pos
            vecs = data.reshape(-1, D)
            f_of = torch.arange(n_freq, device=dev).repeat_interleave(1 if variant == "rows" else T_SAMPLES).repeat(len(ids))
            mu, sd = vecs.mean(0), vecs.std(0) + 1e-6
            torch.manual_seed(args.seed)
            flow = CondFlow(D, n_freq, blocks=args.blocks).to(dev)
            opt = torch.optim.Adam(flow.parameters(), lr=2e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

            @torch.no_grad()
            def evaluate(step):
                flow.eval()
                per_row = []
                for chunk, f in teacher_maps(test, list(range(len(test)))):
                    B, _, Fn, T = f.shape
                    v = f.mean(3).permute(0, 2, 1) if variant == "rows" else f.permute(0, 2, 3, 1)
                    flat = ((v.reshape(-1, D) - mu) / sd)
                    fi = torch.arange(Fn, device=dev).repeat_interleave(1 if variant == "rows" else T).repeat(B)
                    nll = torch.cat([flow.nll(flat[j : j + 65536], fi[j : j + 65536]) for j in range(0, len(flat), 65536)])
                    per_row.append(nll.reshape(B, Fn, -1).mean(2).cpu())
                per_row = torch.cat(per_row).numpy()
                k = max(1, per_row.shape[1] // 10)
                aggs = {"max": per_row.max(1), "top-10%": np.sort(per_row, 1)[:, -k:].mean(1), "mean": per_row.mean(1)}
                res = {a: compute_metrics(s, lab, dom) for a, s in aggs.items()}
                log(f"  [{m} | {variant} | eval step {step}] " + " | ".join(
                    f"{a}: AUC {r['auc_all']:.3f} (src {r['auc_source']:.3f}/tgt {r['auc_target']:.3f}) pAUC {r['pauc']:.3f} off {r['official']:.3f}"
                    for a, r in res.items()))
                flow.train()
                return res

            t0 = time.time()
            for step in range(1, args.steps + 1):
                j = torch.randint(len(vecs), (args.batch,), device=dev)
                loss = flow.nll((vecs[j] - mu) / sd, f_of[j]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
                opt.step()
                sched.step()
                if step % 100 == 0:
                    log(f"  [{m} | {variant}] step {step:5d}/{args.steps}  train NLL/dim {loss.item():.4f}  ({(time.time() - t0) / step * 1000:.0f} ms/step)")
                if step % args.eval_every == 0 or step == args.steps:
                    res = evaluate(step)
            results[f"{m} | {variant}"] = res

    lines = []
    for key, res in results.items():
        for a, r in res.items():
            lines.append(f"{key:24s} {a:8s} AUC {r['auc_all']:.3f}  src {r['auc_source']:.3f}  tgt {r['auc_target']:.3f}  pAUC {r['pauc']:.3f}  official {r['official']:.3f}")
    log("=== summary (last step)\n" + "\n".join(lines))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=1))


if __name__ == "__main__":
    main()
