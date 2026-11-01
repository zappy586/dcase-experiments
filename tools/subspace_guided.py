"""Synthetic-guided subspace scorer (no test labels used for scoring; labels only for the metrics).
Frozen BEATs features (768, token average, full 10 s, clean near-minus-far), standardised by the real training normals.
1. Map synthetic features onto the real distribution (none / linear / ot from tools/transport_v1.py, fitted on all 250
   synthetic normals and the 1000 real training normals).
2. Fault displacement of each mapped synthetic faulty clip = it minus the mean of its 10 nearest mapped synthetic normals.
3. PCA of the 250 displacements -> top k directions U (where the synthetic faults live).
4. Score a real test clip by its Mahalanobis distance from the real training normals inside span(U)
   (mean and shrinkage covariance of the projected real normals).
Control: the same scorer in k random directions (mean of 20 draws), so any gain must come from WHERE the synthetic faults
point, not from using few dimensions. Reported for k = 1, 2, 4, 8, 16, 32 (k = 8 is the one fixed in advance).
Usage (main env): uv run python tools/subspace_guided.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402
from transport_v1 import CLEAN, fit_linear, fit_ot  # noqa: E402

KS = (1, 2, 4, 8, 16, 32)


def maha_in(U, R, X, a=0.1):
    P, Q = R @ U, X @ U
    S = np.cov(P.T).reshape(U.shape[1], U.shape[1])
    S = (1 - a) * S + a * np.trace(S) / len(S) * np.eye(len(S))
    D = Q - P.mean(0)
    return np.einsum("ij,jk,ik->i", D, np.linalg.inv(S), D)


def maha_dom(U, R, rdom, X, mode, a=0.1):
    """Score = min(distance to source normals, distance to target normals) inside span(U).
    'pooled': target Gaussian = mean of the 10 target clips + the source covariance.
    'shrunk': target covariance = half its own (10-clip) estimate, half the source covariance."""
    P, Q = R @ U, X @ U
    k = U.shape[1]
    cov = lambda Z: np.cov(Z.T).reshape(k, k)
    Ss = cov(P[rdom == 0]); Ss = (1 - a) * Ss + a * np.trace(Ss) / k * np.eye(k)
    St = Ss if mode == "pooled" else 0.5 * cov(P[rdom == 1]) + 0.5 * Ss
    out = []
    for m_, S in ((P[rdom == 0].mean(0), Ss), (P[rdom == 1].mean(0), St)):
        D = Q - m_
        out.append(np.einsum("ij,jk,ik->i", D, np.linalg.inv(S), D))
    return np.minimum(*out)


def main():
    import json
    rdom = np.array(json.loads((ROOT / "cache" / "wav_ToyCar_train.json").read_text())["domain"])
    Fe = dict(np.load(CLEAN))
    mu, sd = Fe["real_train"].mean(0), Fe["real_train"].std(0) + 1e-6
    z = lambda X: (X - mu) / sd
    R, Rt, Gn, Gf = z(Fe["real_train"]), z(Fe["real_test"]), z(Fe["gen_normal"]), z(Fe["gen_faulty"])
    lab, dom = Fe["test_label"], Fe["test_domain"]
    rng = np.random.default_rng(0)
    ctrl = {}
    for k in KS:
        res = []
        for _ in range(20):
            U = np.linalg.qr(rng.standard_normal((R.shape[1], k)))[0]
            res.append(compute_metrics(maha_in(U, R, Rt), lab, dom)["official"])
        ctrl[k] = np.mean(res)
    A.log("subspace_guided: random-subspace control official " + " ".join(f"k{k} {v:.3f}" for k, v in ctrl.items()))
    for mode in ("pooled", "shrunk"):
        c = {k: np.mean([compute_metrics(maha_dom(np.linalg.qr(rng.standard_normal((R.shape[1], k)))[0], R, rdom, Rt, mode),
                                         lab, dom)["official"] for _ in range(20)]) for k in KS}
        A.log(f"subspace_guided: random-subspace control, per-domain {mode} official " + " ".join(f"k{k} {v:.3f}" for k, v in c.items()))
    for mname, T in (("none", lambda X: X), ("linear", fit_linear(Gn, R)), ("ot", fit_ot(Gn, R))):
        tn, tf = T(Gn), T(Gf)
        d = ((tf[:, None] - tn[None]) ** 2).sum(-1)
        nn = np.argsort(d, 1)[:, :10]
        disp = tf - tn[nn].mean(1)
        _, _, Vt = np.linalg.svd(disp - disp.mean(0), full_matrices=False)
        mean_dir = disp.mean(0) / np.linalg.norm(disp.mean(0))
        for mode in ("single", "pooled", "shrunk"):
            out = []
            for k in KS:
                U = np.linalg.qr(np.c_[mean_dir, Vt[: k - 1].T])[0] if k > 1 else mean_dir[:, None]
                s = maha_in(U, R, Rt) if mode == "single" else maha_dom(U, R, rdom, Rt, mode)
                m = compute_metrics(s, lab, dom)
                out.append(f"k{k} {m['official']:.3f} (src {m['auc_source']:.2f} tgt {m['auc_target']:.2f})")
            A.log(f"subspace_guided [{mname} / {mode}]: " + " | ".join(out))


if __name__ == "__main__":
    main()
