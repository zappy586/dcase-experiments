"""Feature-space transport: map synthetic features onto the real distribution, then use the same map on synthetic faulty.
Features: frozen BEATs iter3, token average (768 per clip), full 10 s, near-minus-far.
  clean: real clips as recorded (cache/grl_v1_beats.npz)
  rt   : real clips sent through AudioGen's EnCodec and back (cache/transport_rt_beats.npz, built here)
The map is fitted ONLY on real training normals (1000) and 200 synthetic normals. Held out: 50 synthetic normal, 50
synthetic faulty (never used to fit anything). Maps (in feature space standardised by the real training normals):
  none   : identity
  linear : whiten with the synthetic-normal mean/covariance, re-colour with the real ones (shrinkage covariances)
  ot     : entropic optimal transport (Sinkhorn) synthetic normal -> real normal, barycentric map; new points move by the
           kernel-weighted displacement of their nearest fitted synthetic points
Report per map: fresh logistic-regression probe (5-fold CV AUC, 0.5 = cannot tell apart) real test normal vs mapped held-out
synthetic normal, and real test anomaly vs mapped held-out synthetic faulty; then a classifier trained on real training
normals + mapped synthetic normals (0) vs mapped synthetic faulty (1), scored on the 200 real test clips (labels only for
the metrics).
Usage (main env): uv run python tools/transport_v1.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402

CLEAN = ROOT / "cache" / "grl_v1_beats.npz"
RT = ROOT / "cache" / "transport_rt_beats.npz"


@torch.no_grad()
def rt_features():
    if RT.exists():
        return dict(np.load(RT))
    from BEATs import BEATs, BEATsConfig
    dev = torch.device("cuda")
    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    m = BEATs(BEATsConfig(ck["cfg"])); m.load_state_dict(ck["model"]); m = m.to(dev).eval()
    out = dict(np.load(CLEAN))
    for split in ("train", "test"):
        X = np.load(ROOT / "cache" / f"toycar_rt_{split}.npy")
        E = []
        for i in range(0, len(X), 10):
            x = torch.from_numpy(X[i : i + 10].astype(np.float32)).to(dev)
            E.append(m.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0].mean(1).cpu())
        out[f"real_{split}"] = torch.cat(E).numpy()
    np.savez(RT, **out)
    return out


def shrink_cov(X, a=0.1):
    S = np.cov(X.T)
    return (1 - a) * S + a * np.trace(S) / len(S) * np.eye(len(S))


def msqrt(C, inv=False):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-8, None)
    return (V * (w ** (-0.5 if inv else 0.5))) @ V.T


def fit_linear(S, R):
    W = msqrt(shrink_cov(R)) @ msqrt(shrink_cov(S), inv=True)
    ms, mr = S.mean(0), R.mean(0)
    return lambda X: (X - ms) @ W.T + mr


def fit_ot(S, R, eps=0.05, iters=500):
    C = ((S[:, None] - R[None]) ** 2).sum(-1)
    C = C / np.median(C)
    K = np.exp(-C / eps)
    a, b = np.full(len(S), 1 / len(S)), np.full(len(R), 1 / len(R))
    u = np.ones(len(S))
    for _ in range(iters):
        v = b / (K.T @ u + 1e-300)
        u = a / (K @ v + 1e-300)
    P = u[:, None] * K * v[None]
    disp = (P @ R) / P.sum(1, keepdims=True) - S              # where each fitted synthetic point moves
    d_ss = ((S[:, None] - S[None]) ** 2).sum(-1)
    h = np.median(np.sort(d_ss, 1)[:, 1])                     # typical nearest-neighbour distance

    def T(X):
        d = ((X[:, None] - S[None]) ** 2).sum(-1)
        w = np.exp(-(d - d.min(1, keepdims=True)) / h)
        return X + (w / w.sum(1, keepdims=True)) @ disp
    return T


def cv_auc(X0, X1):
    X, y = np.r_[X0, X1], np.r_[np.zeros(len(X0)), np.ones(len(X1))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    return roc_auc_score(y, cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1])


def main():
    rng = np.random.default_rng(0)
    pn, pf = rng.permutation(250), rng.permutation(250)
    for name, Fe in (("clean", dict(np.load(CLEAN))), ("rt", rt_features())):
        mu, sd = Fe["real_train"].mean(0), Fe["real_train"].std(0) + 1e-6
        z = lambda X: (X - mu) / sd
        R, Rt = z(Fe["real_train"]), z(Fe["real_test"])
        Gn, Gf = z(Fe["gen_normal"]), z(Fe["gen_faulty"])
        Sn, Vn, Sf, Vf = Gn[pn[50:]], Gn[pn[:50]], Gf[pf[50:]], Gf[pf[:50]]
        lab, dom = Fe["test_label"], Fe["test_domain"]
        ref = cv_auc(Rt[lab == 0], Rt[lab == 1])
        # second probe without the domain mix-up: maps fitted on 800 real training normals, probe on the other 200
        # (same 99% source / 1% target mix as the training set) vs mapped held-out synthetic normals
        pr = rng.permutation(len(R)); Rf, Rh = R[pr[200:]], R[pr[:200]]
        for mname, fit in (("none", None), ("linear", fit_linear), ("ot", fit_ot)):
            T2 = (lambda X: X) if fit is None else fit(Sn, Rf)
            A.log(f"transport_v1 [{name} / {mname}]: probe held-out real TRAIN normal vs synthetic normal {cv_auc(Rh, T2(Vn)):.3f}")
        for mname, T in (("none", lambda X: X), ("linear", fit_linear(Sn, R)), ("ot", fit_ot(Sn, R))):
            p_n = cv_auc(Rt[lab == 0], T(Vn))
            p_f = cv_auc(Rt[lab == 1], T(Vf))
            X = np.r_[R, T(Sn), T(Sf)]
            y = np.r_[np.zeros(len(R) + len(Sn)), np.ones(len(Sf))]
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")).fit(X, y)
            m = compute_metrics(clf.decision_function(Rt), lab, dom)
            syn = roc_auc_score(np.r_[np.zeros(50), np.ones(50)], clf.decision_function(np.r_[T(Vn), T(Vf)]))
            A.log(f"transport_v1 [{name} / {mname}]: probe real vs synthetic NORMAL {p_n:.3f} | real anomaly vs synthetic "
                  f"faulty {p_f:.3f} | (real n vs a reference {ref:.3f}) || classifier: real test src {m['auc_source']:.3f} "
                  f"tgt {m['auc_target']:.3f} pAUC {m['pauc']:.3f} official {m['official']:.3f}, synthetic held-out AUC {syn:.3f}")


if __name__ == "__main__":
    main()
