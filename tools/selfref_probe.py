"""Quick probe for the INP-Former idea (CVPR 2025: score a sample against normal prototypes taken from the sample
itself). Audio version, no training: per test clip, near-only residual frame spectra (256 bands, dB, 313 frames);
the clip's own reference = its median frame; score = 95th percentile over frames of the mean squared deviation from it.
If faults are local in time, this separates anomalies regardless of domain (the reference is the clip itself)."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from modulation_probe import residual, MACHINES
from dcase_ae.evaluation import compute_metrics
torch.set_num_threads(8)
tot = []
for m in MACHINES:
    wav = np.load(ROOT / "cache" / f"wav_{m}_test.npy", mmap_mode="r")
    meta = json.loads((ROOT / "cache" / f"wav_{m}_test.json").read_text()); lab, dom = np.array(meta["label"]), np.array(meta["domain"])
    S95, S50 = [], []
    with torch.no_grad():
        for i in range(0, len(wav), 20):
            r = residual(torch.from_numpy(wav[i:i+20].astype(np.float32) / 32768))
            S = torch.stft(r, 1024, 512, window=torch.hann_window(1024), return_complex=True)[:, 1:]
            L = 10 * torch.log10(S.abs().pow(2).reshape(len(r), 256, 2, -1).sum(2) + 1e-12)  # (B, 256, T)
            d = (L - L.median(-1, keepdim=True).values).pow(2).mean(1)  # (B, T)
            S95.append(torch.quantile(d, 0.95, dim=-1)); S50.append(d.median(-1).values)
    r95 = compute_metrics(torch.cat(S95).numpy(), lab, dom); r50 = compute_metrics(torch.cat(S50).numpy(), lab, dom)
    tot.append(r95["official"])
    print(f"{m:11s} self-ref p95: src {r95['auc_source']:.3f} tgt {r95['auc_target']:.3f} off {r95['official']:.3f} | p50: off {r50['official']:.3f}", flush=True)
print(f"mean p95 official {np.mean(tot):.3f}")
