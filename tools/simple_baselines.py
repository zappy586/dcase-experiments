"""Are the general-audio antennas more than a simple distance? Same clip-level features, same layers, same
held-out z-normalisation and layer sum as clora_gen.py; only the per-layer distance changes:
  knn1 / knn5     mean Euclidean distance to the 1 / 5 nearest normal training clips (features standardised per dim)
  knn1-cos        cosine distance to the nearest normal clip
  maha-lw         full Mahalanobis distance, Ledoit-Wolf covariance
  gen-loud        the method (32 general-audio directions)
Features are cached in cache/rows_<encoder>_<machine>.pt.
Usage: uv run python -u tools/simple_baselines.py [dasheng12b|eat]
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import hmean
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clora_gen import get_encoder  # noqa: E402
from dcase_ae.efficientad import WavStore, build_cache  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402

MACHINES = ["ToyCar", "ToyCarEmu", "bearingEmu", "fan", "gearboxEmu", "sliderEmu", "valveEmu"]
METHODS = ("knn1", "knn5", "knn1-cos", "maha-lw", "gen-loud")


@torch.no_grad()
def near_only_audio(w2, dev, nfft=2048, hop=512):
    """(B, 2, L) near/far -> (B, L) near minus the part the far mic predicts, per frequency, per clip:
    R = N - (X / Pf) F with X = time-mean N conj(F), Pf = time-mean |F|^2. The time-mean of |R|^2 equals the
    near-only feature Pn - |X|^2 / Pf exactly."""
    w2 = torch.as_tensor(w2, device=dev).float()
    B, _, L = w2.shape
    win = torch.hann_window(nfft, device=dev)
    S = torch.stft(w2.reshape(B * 2, L), nfft, hop, window=win, return_complex=True).reshape(B, 2, nfft // 2 + 1, -1)
    N, F = S[:, 0], S[:, 1]
    H = (N * F.conj()).mean(-1, keepdim=True) / ((F.abs() ** 2).mean(-1, keepdim=True) + 1e-12)
    return torch.istft(N - H * F, nfft, hop, window=win, length=L)


def main():
    enc = sys.argv[1] if len(sys.argv) > 1 else "dasheng12b"
    machines = sys.argv[2].split(",") if len(sys.argv) > 2 else MACHINES
    base = enc.removesuffix("-resid")  # "<encoder>-resid": the encoder sees the near-only residual audio
    dev = torch.device("cuda")
    LAYERS, row_features = None, None
    G = torch.load(f"cache/esc50_{base}_rows.pt")  # general audio is mono, used as is
    print(f"=== simple_baselines: encoder {enc}, machines {machines}", flush=True)
    table = {k: [] for k in METHODS}
    for m in machines:
        cp = Path(f"cache/rows_{enc}_{m}.pt")
        if cp.exists():
            d = torch.load(cp, weights_only=False)  # our own cache file
        else:
            if row_features is None:
                LAYERS, row_features = get_encoder(base, dev)
            for split in ("train", "test"):
                build_cache("data", "cache", m, split)
            tr, te = WavStore("cache", m, "train"), WavStore("cache", m, "test")
            idx = list(range(len(tr)))
            random.Random(0).shuffle(idx)
            d = {"lab": np.array(te.meta["label"]), "dom": np.array(te.meta["domain"])}
            for name, store, ids in (("fit", tr, idx[100:]), ("val", tr, idx[:100]), ("test", te, list(range(len(te))))):
                get = (lambda b: near_only_audio(store.get(b), dev)) if enc.endswith("-resid") else (lambda b: store.get(b)[:, 0])
                parts = [row_features(get(ids[i : i + 64])) for i in range(0, len(ids), 64)]
                d[name] = {L: torch.cat([p[L] for p in parts]) for L in parts[0]}
            torch.save(d, cp)
        layers = list(d["fit"].keys())
        scores = {k: {} for k in METHODS}
        for L in layers:
            X, V, T = (d[s][L].double().numpy() for s in ("fit", "val", "test"))  # (N, rows, D)
            mu = X.mean(0)
            D = X.shape[-1]
            Xf, Vf, Tf = (A.reshape(len(A), -1) for A in (X, V, T))  # clip vectors (rows concatenated)
            m_, s_ = Xf.mean(0), Xf.std(0) + 1e-6
            Xs, Vs, Ts = ((A - m_) / s_ for A in (Xf, Vf, Tf))
            Xt, Vt, Tt = (torch.tensor(A, device=dev) for A in (Xs, Vs, Ts))
            for split, Q in (("val", Vt), ("test", Tt)):
                dist = torch.cdist(Q, Xt)
                scores["knn1"][(split, L)] = dist.min(1).values.cpu().numpy()
                scores["knn5"][(split, L)] = dist.topk(5, largest=False).values.mean(1).cpu().numpy()
                Qn, Xn = torch.nn.functional.normalize(Q, dim=1), torch.nn.functional.normalize(Xt, dim=1)
                scores["knn1-cos"][(split, L)] = (1 - Qn @ Xn.T).min(1).values.cpu().numpy()
            lw = LedoitWolf().fit(Xf)
            P = lw.precision_
            for split, Q in (("val", Vf), ("test", Tf)):
                Z = Q - lw.location_
                scores["maha-lw"][(split, L)] = np.einsum("nd,de,ne->n", Z, P, Z)
            # gen-loud, as in clora_gen.py
            Xc = (X - mu).reshape(-1, D)
            Gc = (G[L].double().numpy() - mu).reshape(-1, D)
            reg = lambda S: S + 1e-3 * np.trace(S) / D * np.eye(D)
            Sn, Sg = reg(Xc.T @ Xc / len(Xc)), reg(Gc.T @ Gc / len(Gc))
            ev, U = np.linalg.eigh(Sn)
            W = U / np.sqrt(ev)
            Eg = np.linalg.eigh(W.T @ Sg @ W)[1][:, -32:]
            A = (W @ Eg)
            for split, Q in (("val", V), ("test", T)):
                scores["gen-loud"][(split, L)] = (((Q - mu) @ A) ** 2).sum(-1).max(1)
        for k in METHODS:
            s = sum((scores[k][("test", L)] - scores[k][("val", L)].mean()) / (scores[k][("val", L)].std() + 1e-9) for L in layers)
            table[k].append(compute_metrics(s, d["lab"], d["dom"]))
        print(f"{m:11s} " + "  ".join(f"{k} {table[k][-1]['auc_all']:.3f}" for k in METHODS), flush=True)
    print(f"\n{'method':10s} {'meanAUC':>8s} {'Omega':>7s}   per machine")
    for k in METHODS:
        om = hmean([x for r in table[k] for x in (r["auc_source"], r["auc_target"], r["pauc"])])
        print(f"{k:10s} {np.mean([r['auc_all'] for r in table[k]]):8.3f} {om:7.3f}   " + " ".join(f"{r['auc_all']:.3f}" for r in table[k]))


if __name__ == "__main__":
    main()
