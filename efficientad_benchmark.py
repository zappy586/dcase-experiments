"""EfficientAD for machine sound on every machine type, identical settings (see dcase_ae/efficientad.py).

Per machine: teacher pretrained on the OTHER machines (leave-one-machine-out), frozen; student + autoencoder trained
on this machine's normal clips (other machines = penalty data); evaluated every --eval_every steps; the reported
result is the last step.

Usage:  uv run python efficientad_benchmark.py --name v1 [--machines fan,ToyCar] [--steps 2000]
Live log: outputs/efficientad/<name>/train.log
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import hmean

from dcase_ae.data.prepare import ensure_extracted, list_machines
from dcase_ae.efficientad import (CLIP, SR, Detector, WavStore, build_cache, clip_scores, features, logpower_stats,
                                  quantiles, random_crops, train_teacher)
from dcase_ae.evaluation import compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--machines", default=None)
    ap.add_argument("--teacher_steps", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=float, default=4.0, help="training crop length in seconds")
    ap.add_argument("--target_share", type=float, default=0.2, help="share of target-domain clips in training batches")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda")
    out = Path("outputs/efficientad") / args.name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    all_machines = list_machines("dataset")
    machines = args.machines.split(",") if args.machines else all_machines
    log(f"=== run '{args.name}': {vars(args)}")
    for m in all_machines:
        ensure_extracted("dataset", "data", m)
        for split in ("train", "test"):
            build_cache("data", "cache", m, split)
    train = {m: WavStore("cache", m, "train") for m in all_machines}
    stats = {m: logpower_stats(train[m], dev) for m in all_machines}
    log("waveform caches and log-power statistics ready")

    results = {}
    for m in machines:
        t_start = time.time()
        others = [o for o in all_machines if o != m]
        log(f"--- {m}: teacher on {others}")
        teacher = train_teacher([train[o] for o in others], [stats[o] for o in others], dev, args.teacher_steps, log, args.seed)

        rng = random.Random(args.seed)
        torch.manual_seed(args.seed)
        store, test = train[m], WavStore("cache", m, "test")
        idx = list(range(len(store)))
        rng.shuffle(idx)
        val_idx, tr_idx = idx[:100], idx[100:]
        dom = store.meta["domain"]
        n_t = sum(dom[i] for i in tr_idx)
        weights = [(args.target_share / n_t if dom[i] else (1 - args.target_share) / (len(tr_idx) - n_t)) for i in tr_idx]
        sub = WavStoreView(store, tr_idx)

        det = Detector(teacher, dev)
        with torch.no_grad():
            x = torch.cat([features(store.get(tr_idx[i : i + 8]).to(dev), stats[m]) for i in range(0, 64, 8)])
            det.set_teacher_norm(x)
        params = list(det.s.parameters()) + list(det.ae.parameters())
        opt = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)
        length = int(args.crop * SR)

        def full_maps(st, ids):
            st_maps, ae_maps = [], []
            for i in range(0, len(ids), 8):
                x = features(st.get(ids[i : i + 8]).to(dev), stats[m])
                a, b = det.maps(x)
                st_maps.append(a)
                ae_maps.append(b)
            return torch.cat(st_maps), torch.cat(ae_maps)

        def evaluate(step):
            det.s.eval(), det.ae.eval()
            v_st, v_ae = full_maps(store, val_idx)
            norm = {"st": quantiles(v_st), "ae": quantiles(v_ae)}
            vs, vt, vparts = clip_scores(v_st, v_ae, norm)
            t_st, t_ae = full_maps(test, list(range(len(test))))
            ts, tt, tparts = clip_scores(t_st, t_ae, norm)
            z = lambda a, ref: ((a - ref.mean()) / (ref.std() + 1e-9)).cpu().numpy()
            scores = {"final": z(ts, vs) + z(tt, vt), "steady": z(ts, vs), "transient": z(tt, vt),
                      **{k: z(tparts[k], vparts[k]) for k in tparts}}
            lab, d = np.array(test.meta["label"]), np.array(test.meta["domain"])
            res = {k: compute_metrics(v, lab, d) for k, v in scores.items()}
            f = res["final"]
            log(f"  [{m} eval step {step}] final: AUC {f['auc_all']:.3f} (src {f['auc_source']:.3f} / tgt {f['auc_target']:.3f})"
                f"  pAUC {f['pauc']:.3f}  official {f['official']:.3f} | AUC by part: "
                + "  ".join(f"{k} {v['auc_all']:.3f}" for k, v in res.items() if k != "final"))
            det.s.train(), det.ae.train()
            return res

        evaluate(0)
        t0 = time.time()
        for step in range(1, args.steps + 1):
            x = features(random_crops(sub, args.batch, length, rng, weights).to(dev), stats[m])
            pen_m = [rng.choice(others) for _ in range(args.batch)]
            w_pen = torch.stack([random_crops(train[o], 1, length, rng)[0] for o in pen_m]).to(dev)
            x_pen = features(w_pen, (torch.stack([stats[o][0] for o in pen_m]), torch.stack([stats[o][1] for o in pen_m])))
            l_hard, l_pen, l_ae, l_stae = det.losses(x, x_pen)
            loss = l_hard + l_pen + l_ae + l_stae
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                log(f"  [{m}] step {step:5d}/{args.steps}  loss {loss.item():.4f}  hard {l_hard.item():.4f}  pen {l_pen.item():.4f}"
                    f"  ae {l_ae.item():.4f}  st-ae {l_stae.item():.4f}  ({(time.time() - t0) / step:.2f}s/step)")
            if step % args.eval_every == 0 or step == args.steps:
                res = evaluate(step)
        results[m] = res
        torch.save({"student": det.s.state_dict(), "ae": det.ae.state_dict(), "teacher": teacher.state_dict()},
                   out / f"model_{m}.pt")
        log(f"--- {m} done in {(time.time() - t_start) / 60:.1f} min")

    lines = [f"{'machine':<12}" + "".join(f"{c:>12}" for c in ("auc_source", "auc_target", "auc_all", "pauc", "official"))]
    for m, r in results.items():
        f = r["final"]
        lines.append(f"{m:<12}" + "".join(f"{f[c]:>12.3f}" for c in ("auc_source", "auc_target", "auc_all", "pauc", "official")))
    omega = hmean([r["final"][c] for r in results.values() for c in ("auc_source", "auc_target", "pauc")])
    lines.append(f"DCASE-style overall score: {omega:.3f}   (mean AUC {np.mean([r['final']['auc_all'] for r in results.values()]):.3f})")
    log("=== summary (last step)\n" + "\n".join(lines))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "results": results, "overall": omega}, indent=1))


class WavStoreView:
    """A subset of a WavStore's clips (same interface as WavStore for random_crops)."""

    def __init__(self, store, idx):
        self.store, self.idx = store, idx

    def __len__(self):
        return len(self.idx)

    def get(self, idx, start=0, length=CLIP):
        return self.store.get([self.idx[i] for i in idx], start, length)


if __name__ == "__main__":
    main()
