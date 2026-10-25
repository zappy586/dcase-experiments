"""Which EnCodec codebook entries do ToyCar's near-minus-far clips use, and do normal and anomalous clips differ?

Audio: near-minus-far residual (as tools/encodec_knn.py), 16 -> 24 kHz, ONE gain for the whole machine (median level of
the training clips), so level differences between clips are kept. EnCodec 24 kHz at 6 kbps = 8 residual codebooks
(q1 coarsest ... q8 finest), 1024 entries each, 75 codes per second per codebook.
Reported per codebook: entries used; how concentrated usage is; the most used entries in training normals vs test
normals vs test anomalies; entries most over-used by anomalies; and a label-free check: score each test clip by how
unusual its codes are under the training-clip code frequencies (mean -log p), AUC per codebook.
Usage: uv run python tools/encodec_codes.py --machine ToyCar
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from encodec_knn import residual  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402

K = 1024


@torch.no_grad()
def codes(model, machine, split, gain, dev):
    wav = np.load(ROOT / "cache" / f"wav_{machine}_{split}.npy", mmap_mode="r")
    out = []
    for i in range(0, len(wav), 16):
        r = torchaudio.functional.resample(residual(torch.from_numpy(wav[i : i + 16].astype(np.float32) / 32768).to(dev)), 16000, 24000)
        frames = model.encode((r * gain)[:, None])
        out.append(frames[0][0].cpu())  # (B, n_q, T)
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--bandwidth", type=float, default=6.0)
    args = ap.parse_args()
    from encodec import EncodecModel
    dev = torch.device("cuda")
    model = EncodecModel.encodec_model_24khz().to(dev).eval()
    model.set_target_bandwidth(args.bandwidth)
    wtr = np.load(ROOT / "cache" / f"wav_{args.machine}_train.npy", mmap_mode="r")
    with torch.no_grad():
        r = residual(torch.from_numpy(wtr[:200].astype(np.float32) / 32768).to(dev))
        gain = float(0.1 / r.pow(2).mean(-1).sqrt().median())
    Ctr, Cte = codes(model, args.machine, "train", gain, dev), codes(model, args.machine, "test", gain, dev)
    meta = json.loads((ROOT / "cache" / f"wav_{args.machine}_test.json").read_text())
    lab, dom = np.array(meta["label"]), np.array(meta["domain"])
    nq = Ctr.shape[1]
    print(f"{args.machine}: near-minus-far, one gain {20 * np.log10(gain):+.1f} dB; {nq} codebooks x {K} entries; "
          f"train {Ctr.shape}, test {Cte.shape}\n")
    hist = lambda C, q: np.bincount(C[:, q].ravel(), minlength=K) / C[:, q].size
    for q in range(nq):
        htr, hn, ha = hist(Ctr, q), hist(Cte[lab == 0], q), hist(Cte[lab == 1], q)
        used = (htr > 0).sum()
        top10 = np.sort(htr)[::-1][:10].sum()
        ent = -(htr[htr > 0] * np.log2(htr[htr > 0])).sum()
        p = (np.bincount(Ctr[:, q].ravel(), minlength=K) + 0.5) / (Ctr[:, q].size + 0.5 * K)
        score = (-np.log(p[Cte[:, q]])).mean(1)
        r = compute_metrics(score, lab, dom)
        tv = 0.5 * np.abs(hn - ha).sum()  # total-variation distance between normal and anomaly code frequencies
        tv_nn = 0.5 * np.abs(hist(Cte[(lab == 0) & (dom == 0)], q) - hist(Cte[(lab == 0) & (dom == 1)], q)).sum()
        print(f"q{q + 1}: entries used {used:4d}/{K} | top-10 entries cover {top10:5.1%} | entropy {ent:4.1f} bits | "
              f"normal-vs-anomaly difference {tv:5.1%} (source-vs-target normals {tv_nn:5.1%}) | "
              f"rarity score AUC src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} official {r['official']:.3f}")
        if q in (0, 1):
            order = np.argsort(htr)[::-1][:8]
            print("     most used (train normal / test normal / test anomaly): "
                  + "  ".join(f"#{c}: {htr[c]:.1%}/{hn[c]:.1%}/{ha[c]:.1%}" for c in order))
            ratio = (ha + 1e-4) / (hn + 1e-4)
            over = [c for c in np.argsort(ratio)[::-1] if ha[c] > 0.002][:6]
            print("     most over-used by anomalies: " + "  ".join(f"#{c}: {hn[c]:.2%} -> {ha[c]:.2%}" for c in over))
    np.savez(ROOT / "outputs" / "probe_cache" / f"encodec_codes_{args.machine}.npz", train=Ctr, test=Cte, lab=lab, dom=dom)


if __name__ == "__main__":
    main()
