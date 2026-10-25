"""Near-only spectrum + three-scorer ensemble (training-free features, seconds of training per machine).

Feature: per frequency, the near-mic power minus the part the far mic can explain,
    near_only(f) = Pn(f) - |X(f)|^2 / Pf(f),     X = time-average of near * conj(far)
estimated inside each clip, averaged over the clip, pooled to 256 bands, in dB. Background noise from distant
sources reaches both mics alike and is removed; the machine's close-range sound is kept.

Scorers (fitted on the machine's normal training clips only; no test domain labels, no other test clips):
  knn    distance to the nearest training clip / median leave-one-out NN distance of that clip's domain
  gauss  shared Ledoit-Wolf covariance, separate source and target means; smaller Mahalanobis distance
  synth  small MLP trained to tell training clips from copies with small Gaussian bumps on a few bands
Each score is normalised by its own median / interquartile range on the training clips, then summed.
"""
import glob
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

NFFT, HOP, POOL = 2048, 512, 4
EPS = 1e-12
DOMAINS = {"source": 0, "target": 1}
LABELS = {"normal": 0, "anomaly": 1}


# ---- features ---------------------------------------------------------------------------------------------
@torch.no_grad()
def clip_spectra(path, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-averaged near power, far power and near*conj(far), each (1025,). Channel 0 = near mic."""
    w, _ = sf.read(str(path), dtype="float32", always_2d=True)
    S = torch.stft(torch.from_numpy(w.T.copy()).to(device), NFFT, HOP, window=torch.hann_window(NFFT, device=device),
                   return_complex=True)
    near, far = S[0], S[1]
    return ((near.abs() ** 2).mean(-1).cpu().numpy(), (far.abs() ** 2).mean(-1).cpu().numpy(),
            (near * far.conj()).mean(-1).cpu().numpy())


def load_spectra(data_root, cache_dir, machine, split, device) -> dict:
    """Spectra + labels/domains for every clip of a split, cached in cache_dir."""
    cache = Path(cache_dir) / f"xspec_{machine}_{split}.npz"
    if cache.exists():
        return dict(np.load(cache))
    files = sorted(glob.glob(str(Path(data_root) / machine / split / "*.wav")))
    if not files:
        raise FileNotFoundError(f"No wavs in {Path(data_root) / machine / split}")
    Pn, Pf, X = zip(*(clip_spectra(f, device) for f in tqdm(files, desc=f"{machine}/{split}", leave=False)))
    parts = [Path(f).stem.split("_") for f in files]
    out = {"Pn": np.stack(Pn), "Pf": np.stack(Pf), "X": np.stack(X), "names": np.array([Path(f).name for f in files]),
           "label": np.array([LABELS[p[4]] for p in parts]), "domain": np.array([DOMAINS[p[2]] for p in parts])}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def near_only(spec: dict) -> np.ndarray:
    """(N, 256) near-only spectrum in dB."""
    lin = np.maximum(spec["Pn"] - np.abs(spec["X"]) ** 2 / (spec["Pf"] + EPS), EPS)
    n = lin.shape[-1] // POOL * POOL
    return 10 * np.log10(lin[:, :n].reshape(len(lin), -1, POOL).mean(-1))


# ---- scorers: each returns (test scores, training scores) -------------------------------------------------
def knn_scores(A, B, trdom, device):
    A, B = torch.tensor(A, device=device, dtype=torch.float32), torch.tensor(B, device=device, dtype=torch.float32)
    dBA, dAA = torch.cdist(B, A), torch.cdist(A, A)
    dAA.fill_diagonal_(float("inf"))
    loo, trd = dAA.min(1).values, torch.tensor(trdom, device=device)
    typical = torch.stack([loo[trd == k].median() for k in (0, 1)])
    v, j = dBA.min(1)
    vt, jt = dAA.min(1)
    return (v / typical[trd[j]]).cpu().numpy(), (vt / typical[trd[jt]]).cpu().numpy()


def gauss_scores(A, B, trdom):
    means = [A[trdom == k].mean(0) for k in (0, 1)]
    P = LedoitWolf().fit(np.concatenate([A[trdom == k] - means[k] for k in (0, 1)])).precision_
    maha = lambda Z: np.minimum(*[np.einsum("ij,jk,ik->i", Z - mu, P, Z - mu) for mu in means])
    return maha(B), maha(A)


def synth_scores(A, B, device, seed=0, steps=1500, batch=128):
    torch.manual_seed(seed)
    A_t = torch.tensor(A, device=device, dtype=torch.float32)
    F = A.shape[1]
    net = nn.Sequential(nn.Linear(F, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    pos = torch.arange(F, device=device).float()
    for _ in range(steps):
        x = A_t[torch.randint(len(A_t), (batch,), device=device)]
        bump = torch.zeros_like(x)
        for _ in range(3):  # up to 3 bumps: width 1-6 bands, height +-0.5..2 normal spreads
            centre = torch.randint(F, (batch, 1), device=device).float()
            width = torch.empty(batch, 1, device=device).uniform_(1, 6)
            height = torch.empty(batch, 1, device=device).uniform_(0.5, 2.0) * (torch.randint(0, 2, (batch, 1), device=device) * 2 - 1)
            on = (torch.rand(batch, 1, device=device) < 0.67).float()
            bump += on * height * torch.exp(-0.5 * ((pos[None] - centre) / width) ** 2)
        y = torch.cat([torch.zeros(batch), torch.ones(batch)]).to(device)
        loss = nn.functional.binary_cross_entropy_with_logits(net(torch.cat([x, x + bump])).squeeze(1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor(np.concatenate([B, A]), device=device, dtype=torch.float32)).squeeze(1).cpu().numpy()
    return out[: len(B)], out[len(B):]


def _train_normalised(test, train):
    q1, med, q3 = np.percentile(train, [25, 50, 75])
    return (test - med) / (q3 - q1 + 1e-9)


def anomaly_scores(train: dict, test: dict, device, seed=0) -> dict[str, np.ndarray]:
    """Per-clip scores for the test split: each scorer and their sum ('ensemble')."""
    A, B = near_only(train), near_only(test)
    mu, sd = A.mean(0), A.std(0) + 1e-6
    A, B = (A - mu) / sd, (B - mu) / sd
    trdom = train["domain"]
    raw = {"knn": knn_scores(A, B, trdom, device), "gauss": gauss_scores(A, B, trdom), "synth": synth_scores(A, B, device, seed)}
    out = {k: _train_normalised(*v) for k, v in raw.items()}
    out["ensemble"] = out["knn"] + out["gauss"] + out["synth"]
    return out
