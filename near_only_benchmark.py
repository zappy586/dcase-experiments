"""Near-only spectrum + three-scorer ensemble on every machine type (see dcase_ae/nearonly.py).

Usage: uv run python near_only_benchmark.py [--machines fan,ToyCar] [--seed 0] [--name near_only]

Writes outputs/<name>/summary.txt, summary.json and scores_<machine>.csv. Same settings for every machine;
the DCASE-style overall score is the harmonic mean of every machine's AUC_source, AUC_target and pAUC.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import hmean

from dcase_ae.data.prepare import ensure_extracted, list_machines
from dcase_ae.evaluation import compute_metrics
from dcase_ae.nearonly import anomaly_scores, load_spectra

COLS = ("auc_source", "auc_target", "auc_all", "pauc", "official")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", default=None, help="comma-separated (default: every machine in dataset/)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="near_only")
    ap.add_argument("--zip_dir", default="dataset")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--cache_dir", default="cache")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    machines = args.machines.split(",") if args.machines else list_machines(args.zip_dir)
    out = Path("outputs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for m in machines:
        ensure_extracted(args.zip_dir, args.data_root, m)
        train = load_spectra(args.data_root, args.cache_dir, m, "train", device)
        test = load_spectra(args.data_root, args.cache_dir, m, "test", device)
        scores = anomaly_scores(train, test, device, args.seed)
        results[m] = {k: compute_metrics(v, test["label"], test["domain"]) for k, v in scores.items()}
        with open(out / f"scores_{m}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "label", "domain", *scores])
            for i, name in enumerate(test["names"]):
                w.writerow([name, int(test["label"][i]), int(test["domain"][i]), *(f"{s[i]:.6f}" for s in scores.values())])
        r = results[m]["ensemble"]
        print(f"{m:11s} AUC {r['auc_all']:.3f} (src {r['auc_source']:.3f} / tgt {r['auc_target']:.3f})  pAUC {r['pauc']:.3f}")

    lines = []
    for scorer in ("ensemble", "knn", "gauss", "synth"):
        lines.append(f"\n[{scorer}]\n{'machine':<12}" + "".join(f"{c:>12}" for c in COLS))
        for m in machines:
            lines.append(f"{m:<12}" + "".join(f"{results[m][scorer][c]:>12.3f}" for c in COLS))
        lines.append(f"{'MEAN':<12}" + "".join(f"{np.mean([results[m][scorer][c] for m in machines]):>12.3f}" for c in COLS))
        omega = hmean([results[m][scorer][c] for m in machines for c in ("auc_source", "auc_target", "pauc")])
        lines.append(f"DCASE-style overall score: {omega:.3f}")
    table = "\n".join(lines)
    print(table)
    (out / "summary.txt").write_text(table.lstrip() + "\n")
    (out / "summary.json").write_text(json.dumps({"seed": args.seed, "results": results}, indent=1))


if __name__ == "__main__":
    main()
