"""Diagnostic (no training): does the near-only machine sound carry fault information in its AMPLITUDE MODULATION that the
time-averaged power spectrum (the autoencoder's current target) throws away?

For every test clip of a machine:
  residual waveform r = near - H(f) * far  (same least-squares H as ae_noiseaware.Spec, STFT 1024/512, then iSTFT)
  A. power features (current AE target): time-averaged residual power in 256 bands, dB                    -> 256 numbers
  B. modulation features: residual STFT 256/16 (1 kHz envelope rate) -> 16 carrier bands of 500 Hz, envelope e_b(t);
     Welch spectrum of e_b - mean (2 s segments, 0.5 Hz resolution) divided by mean(e_b)^2 (modulation depth, so the
     overall level is removed) -> 16 log-spaced modulation bands 0.5-500 Hz, dB                            -> 256 numbers
Per feature: AUC of normal vs anomalous WITHIN each domain (source, target), averaged over the two domains.
Test labels are used only to measure information content here; nothing is selected or tuned on them.
Reported per machine and feature set: mean over the top 10% of features of max(AUC, 1-AUC), and how many features
reach 0.70. Both sets have 256 features, so chance maxima are comparable.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
MACHINES = ["ToyCar", "ToyCarEmu", "bearingEmu", "fan", "gearboxEmu", "sliderEmu", "valveEmu"]
torch.set_num_threads(8)


def residual(w):
    """w (B, 2, L) -> near-only residual waveform (B, L)"""
    win = torch.hann_window(1024)
    S = torch.stft(w.reshape(-1, w.shape[-1]), 1024, 512, window=win, return_complex=True).reshape(len(w), 2, 513, -1)
    N, F = S[:, 0], S[:, 1]
    H = (N * F.conj()).mean(-1, keepdim=True) / (F.abs().pow(2).mean(-1, keepdim=True) + 1e-12)
    return torch.istft(N - H * F, 1024, 512, window=win, length=w.shape[-1])


def power_feats(r):
    S = torch.stft(r, 1024, 512, window=torch.hann_window(1024), return_complex=True)[:, 1:]
    P = S.abs().pow(2).reshape(len(r), 256, 2, -1).sum(2).mean(-1)
    return 10 * torch.log10(P + 1e-12)


MOD_EDGES = np.geomspace(0.5, 500, 17)


def modulation_feats(r):
    S = torch.stft(r, 256, 16, window=torch.hann_window(256), return_complex=True)[:, 1:]  # (B, 128, T) at 1 kHz
    e = S.abs().pow(2).reshape(len(r), 16, 8, -1).sum(2).sqrt()  # (B, 16, T) carrier-band envelopes
    m = e.mean(-1, keepdim=True)
    x = e - m
    seg, hop = 2000, 1000  # 2 s Welch segments
    frames = x.unfold(-1, seg, hop) * torch.hann_window(seg)
    M = torch.fft.rfft(frames, dim=-1).abs().pow(2).mean(-2)  # (B, 16, seg//2+1)
    freqs = np.fft.rfftfreq(seg, 1 / 1000)
    out = []
    for lo, hi in zip(MOD_EDGES[:-1], MOD_EDGES[1:]):
        sel = torch.from_numpy((freqs >= lo) & (freqs < hi))
        out.append(M[..., sel].mean(-1))
    D = torch.stack(out, -1) / m.pow(2)  # modulation depth
    return 10 * torch.log10(D + 1e-12).reshape(len(r), -1)


def per_feature_auc(X, lab, dom):
    a = []
    for d in (0, 1):
        s = dom == d
        a.append([roc_auc_score(lab[s], X[s, j]) for j in range(X.shape[1])])
    a = np.mean(a, 0)
    return np.maximum(a, 1 - a)


def main():
    machines = sys.argv[1:] or MACHINES
    res = {}
    for mname in machines:
        wav = np.load(ROOT / "cache" / f"wav_{mname}_test.npy", mmap_mode="r")
        meta = json.loads((ROOT / "cache" / f"wav_{mname}_test.json").read_text())
        lab, dom = np.array(meta["label"]), np.array(meta["domain"])
        P, M = [], []
        with torch.no_grad():
            for i in range(0, len(wav), 20):
                w = torch.from_numpy(wav[i : i + 20].astype(np.float32) / 32768)
                r = residual(w)
                P.append(power_feats(r))
                M.append(modulation_feats(r))
        P, M = torch.cat(P).numpy(), torch.cat(M).numpy()
        row = {}
        for k, X in (("power", P), ("modulation", M)):
            a = np.sort(per_feature_auc(X, lab, dom))[::-1]
            row[k] = {"top10pct": float(a[:26].mean()), "best": float(a[0]), "n_ge_0.70": int((a >= 0.70).sum())}
        res[mname] = row
        print(f"{mname:11s} power: top10% {row['power']['top10pct']:.3f} best {row['power']['best']:.3f} "
              f"n>=.70 {row['power']['n_ge_0.70']:3d} | modulation: top10% {row['modulation']['top10pct']:.3f} "
              f"best {row['modulation']['best']:.3f} n>=.70 {row['modulation']['n_ge_0.70']:3d}", flush=True)
        np.savez(ROOT / "outputs" / f"modprobe_{mname}.npz", P=P, M=M, lab=lab, dom=dom)
    (ROOT / "outputs" / "modulation_probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
