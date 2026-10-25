"""For a few test clips, find their nearest normal training clips with the near-only spectrum (dcase_ae/nearonly.py).

Usage: uv run python tools/nearest_normals.py [--machines fan,valveEmu] [--per_label 4] [--k 5]
Writes outputs/span_nn/neighbours.json: {machine: [{"test": file, "label": ..., "neighbours": [train files]}]}
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dcase_ae.nearonly import load_spectra, near_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", default="fan,valveEmu")
    ap.add_argument("--per_label", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(0)
    out = {}
    for m in args.machines.split(","):
        train = load_spectra("data", "cache", m, "train", device)
        test = load_spectra("data", "cache", m, "test", device)
        A, B = near_only(train), near_only(test)
        mu, sd = A.mean(0), A.std(0) + 1e-6
        A, B = (A - mu) / sd, (B - mu) / sd
        names = list(test["names"])
        chosen = []
        for lab in ("normal", "anomaly"):
            pool = [i for i, n in enumerate(names) if f"source_test_{lab}" in n]
            chosen += [(i, lab) for i in rng.sample(pool, args.per_label)]
        out[m] = []
        for i, lab in chosen:
            d = np.linalg.norm(A - B[i], axis=1)
            nn = np.argsort(d)[: args.k]
            out[m].append({"test": names[i], "label": lab, "neighbours": [str(train["names"][j]) for j in nn],
                           "distances": [round(float(d[j]), 2) for j in nn]})
    path = Path("outputs/span_nn")
    path.mkdir(parents=True, exist_ok=True)
    (path / "neighbours.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {path / 'neighbours.json'}")


if __name__ == "__main__":
    main()
