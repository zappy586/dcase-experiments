"""Linear test of a principle (seconds per machine, no network): does predicting each part of the clip's machine sound
from the OTHER parts (so the predictor never sees the part it scores) beat compressing-and-reconstructing it (what our
AE does, here as its linear version, PCA)?
Data: the AE's own target (time-averaged near-only spectrum, 256 bands, dB, ae_noiseaware.Spec), standardised with all
1000 training clips. Fit on all 1000 normal training clips; scored on the test clips.
  PCA-k      : reconstruct from the top-k principal components; score = mean squared error
  crossband  : 8 contiguous blocks of 32 bands; each block predicted from the other 224 bands by ridge regression
               (alpha fixed); score = mean squared error over all 256 bands
Reported: the official-style score from dcase_ae.evaluation.compute_metrics. All settings fixed in advance, every
setting is shown, nothing is picked on test labels."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ae_noiseaware import Spec
from dcase_ae.evaluation import compute_metrics
torch.set_num_threads(8)
MACHINES = ["ToyCar", "ToyCarEmu", "bearingEmu", "fan", "gearboxEmu", "sliderEmu", "valveEmu"]

def targets(m, split):
    f = ROOT / "outputs" / "probe_cache" / f"target_{m}_{split}.npy"
    if f.exists():
        return np.load(f)
    wav = np.load(ROOT / "cache" / f"wav_{m}_{split}.npy", mmap_mode="r")
    spec, Y = Spec(torch.device("cpu")), []
    with torch.no_grad():
        for i in range(0, len(wav), 25):
            Y.append(spec(torch.from_numpy(wav[i:i+25].astype(np.float32) / 32768))[1])
    Y = torch.cat(Y).numpy()
    f.parent.mkdir(exist_ok=True); np.save(f, Y)
    return Y

def pca_err(Ytr, Yte, k):
    U, S, Vt = np.linalg.svd(Ytr, full_matrices=False)
    V = Vt[:k].T
    return ((Yte - Yte @ V @ V.T) ** 2).mean(1)

def crossband_err(Ytr, Yte, alpha, nblk=8):
    d = Ytr.shape[1]; E = np.zeros_like(Yte)
    for b in range(nblk):
        blk = np.arange(b * d // nblk, (b + 1) * d // nblk); rest = np.setdiff1d(np.arange(d), blk)
        A = Ytr[:, rest]
        W = np.linalg.solve(A.T @ A + alpha * np.eye(len(rest)), A.T @ Ytr[:, blk])
        E[:, blk] = (Yte[:, blk] - Yte[:, rest] @ W) ** 2
    return E.mean(1)

res = {}
for m in (sys.argv[1:] or MACHINES):
    Ytr, Yte = targets(m, "train"), targets(m, "test")
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-3
    Ytr, Yte = (Ytr - mu) / sd, (Yte - mu) / sd
    meta = json.loads((ROOT / "cache" / f"wav_{m}_test.json").read_text())
    lab, dom = np.array(meta["label"]), np.array(meta["domain"])
    sc = {f"PCA-{k}": pca_err(Ytr, Yte, k) for k in (8, 16, 32)}
    sc.update({f"crossband a={a}": crossband_err(Ytr, Yte, a) for a in (10, 100)})
    res[m] = {k: compute_metrics(v, lab, dom) for k, v in sc.items()}
    ref = json.loads((ROOT / "outputs/ae_noiseaware/v4_all7" / m / "summary.json").read_text())
    print(f"{m:11s} " + " | ".join(f"{k} {r['official']:.3f}" for k, r in res[m].items()), flush=True)
print("mean        " + " | ".join(f"{k} {np.mean([res[m][k]['official'] for m in res]):.3f}" for k in next(iter(res.values()))))
(ROOT / "outputs" / "crossband_probe.json").write_text(json.dumps(res, indent=1))
