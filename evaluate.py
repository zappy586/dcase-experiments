"""Evaluate a trained checkpoint on the test split.

Usage: uv run python evaluate.py outputs/<name>/<run>/best.pt [--heatmaps 8] [a.b=value ...]
"""
import argparse
import csv
from pathlib import Path

import torch

from dcase_ae.config import apply_overrides, to_attr
from dcase_ae.data import build_datasets
from dcase_ae.evaluation import evaluate, format_metrics
from dcase_ae.models import build_model
from dcase_ae.visualize import save_step_plots, select_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--heatmaps", type=int, default=8, help="number of samples to plot (0 = none)")
    ap.add_argument("overrides", nargs="*", help="dotted config overrides, e.g. data.machine=ToyCar")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = to_attr(apply_overrides(ck["cfg"], args.overrides))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _, ds = build_datasets(cfg)
    C, n_mels, n_frames = ds.specs.shape[1:]
    model = build_model(cfg, C, (n_mels, n_frames)).to(device)
    model.load_state_dict(ck["model"])

    res = evaluate(model, ds, device, cfg.eval.batch_size, cfg.train.amp, cfg.eval.pauc_max_fpr)
    print(f"checkpoint step {ck['step']}")
    print(format_metrics(res["metrics"], cfg.eval.score))
    for k, m in res["metrics"].items():
        print(f"  {k:>10}: " + "  ".join(f"{a}={b:.3f}" for a, b in m.items()))

    out = Path(args.ckpt).parent / "eval"
    out.mkdir(exist_ok=True)
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "domain", *res["scores"].keys()])
        for i, name in enumerate(ds.names):
            w.writerow([name, int(ds.labels[i]), int(ds.domains[i]), *(f"{s[i]:.6f}" for s in res["scores"].values())])
    if args.heatmaps:
        idx = select_indices(ds.labels.numpy(), ds.domains.numpy(), args.heatmaps, cfg.seed)
        d = save_step_plots(out, ck["step"], idx, ds, res["recon"], res["err"], res["scores"][cfg.eval.score], cfg.audio, cfg.vis.err_vmax)
        print(f"Heatmaps in {d}")
    print(f"Scores in {out / 'scores.csv'}")


if __name__ == "__main__":
    main()
