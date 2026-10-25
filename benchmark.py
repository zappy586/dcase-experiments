"""Train and evaluate ONE config on every machine type with identical settings, then summarise.

Usage: uv run python benchmark.py --config configs/baseline.yaml --name my_exp [--machines fan,ToyCar] [a.b=value ...]

Only the machine type changes between runs. The reported numbers are the final-step metrics of
the primary score (eval.score), so nothing is selected using the labelled test clips.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.stats import hmean

from dcase_ae.config import apply_overrides, load_config, to_attr, to_plain
from dcase_ae.data.prepare import list_machines
from dcase_ae.trainer import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--name", required=True, help="experiment name (outputs/<name>/<machine>/...)")
    ap.add_argument("--machines", default=None, help="comma-separated, e.g. fan,ToyCar (default: every machine in the zips)")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. model.encoder.dropout=0.1")
    args = ap.parse_args()

    base = load_config(args.config, args.overrides)
    machines = args.machines.split(",") if args.machines else list_machines(base.data.zip_dir)
    primary = base.eval.score
    results = {}
    for m in machines:
        cfg = to_attr(copy.deepcopy(to_plain(base)))
        cfg.data.machine = m
        cfg.name = f"{args.name}/{m}"
        results[m] = train(cfg)[primary]

    cols = ("auc_source", "auc_target", "auc_all", "pauc", "official")
    lines = [f"{'machine':<12}" + "".join(f"{c:>12}" for c in cols)]
    for m, r in results.items():
        lines.append(f"{m:<12}" + "".join(f"{r[c]:>12.3f}" for c in cols))
    lines.append(f"{'MEAN':<12}" + "".join(f"{np.mean([r[c] for r in results.values()]):>12.3f}" for c in cols))
    omega = hmean([r[c] for r in results.values() for c in ("auc_source", "auc_target", "pauc")])
    lines.append(f"DCASE-style overall score (harmonic mean of every AUC_source, AUC_target, pAUC): {omega:.3f}")
    table = "\n".join(lines)
    print(f"\n=== {args.name}  (score: {primary}, last step) ===\n{table}")

    out = Path(base.output.dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"overrides": args.overrides, "results": results, "overall": omega}, indent=1))
    (out / "summary.txt").write_text(table + "\n")


if __name__ == "__main__":
    main()
