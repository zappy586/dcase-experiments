"""Diagnostic (no training), follow-up to tools/modulation_probe.py: does averaging the near-only residual over TIME
throw away fault information? Per test clip, residual frame power in 256 bands (dB, STFT 1024/512), pooled 3 ways:
  mean (the AE's current target, as power-mean in dB), std over time (dB), 95th - 50th percentile (dB, burstiness),
plus coherence between the mics |X|^2/(Pn Pf) per band (dB). 256 features each. Same per-feature AUC as the
modulation probe (within domain, averaged; labels used only to measure information)."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modulation_probe import residual, per_feature_auc, MACHINES, ROOT
torch.set_num_threads(8)

def feats(w):
    win = torch.hann_window(1024)
    r = residual(w)
    S = torch.stft(r, 1024, 512, window=win, return_complex=True)[:, 1:]
    P = S.abs().pow(2).reshape(len(r), 256, 2, -1).sum(2)
    L = 10 * torch.log10(P + 1e-12)
    q = torch.quantile(L, torch.tensor([0.5, 0.95]), dim=-1)
    W = torch.stft(w.reshape(-1, w.shape[-1]), 1024, 512, window=win, return_complex=True)[:, 1:].reshape(len(w), 2, 256, 2, -1)
    N, F = W[:, 0], W[:, 1]
    X = (N * F.conj()).mean(-1).abs().pow(2).sum(-1)
    coh = X / (N.abs().pow(2).mean(-1).sum(-1) * F.abs().pow(2).mean(-1).sum(-1) + 1e-20)
    return {"mean": 10 * torch.log10(P.mean(-1) + 1e-12), "std": L.std(-1), "burst": q[1] - q[0],
            "coherence": 10 * torch.log10(coh + 1e-12)}

res = {}
for m in (sys.argv[1:] or MACHINES):
    wav = np.load(ROOT / "cache" / f"wav_{m}_test.npy", mmap_mode="r")
    meta = json.loads((ROOT / "cache" / f"wav_{m}_test.json").read_text())
    lab, dom = np.array(meta["label"]), np.array(meta["domain"])
    acc = {}
    with torch.no_grad():
        for i in range(0, len(wav), 20):
            for k, v in feats(torch.from_numpy(wav[i:i+20].astype(np.float32) / 32768)).items():
                acc.setdefault(k, []).append(v)
    row = {}
    for k, v in acc.items():
        a = np.sort(per_feature_auc(torch.cat(v).numpy(), lab, dom))[::-1]
        row[k] = (float(a[:26].mean()), float(a[0]), int((a >= .7).sum()))
    res[m] = row
    print(f"{m:11s} " + " | ".join(f"{k}: top10% {t:.3f} best {b:.3f} n>=.70 {n:3d}" for k, (t, b, n) in row.items()), flush=True)
(ROOT / "outputs" / "pooling_probe.json").write_text(json.dumps(res, indent=1))
