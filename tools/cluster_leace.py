"""Cluster-conditional LEACE on BEATs' last layer: remove the real-vs-synthetic gap separately in each region of
feature space, so that a NON-linear probe can no longer separate real from synthetic normals.
(The last-layer token eraser commutes with mean pooling, so it is fitted and applied on the pooled 768-d output.)

1. global LEACE (as in tools/leace_scrub.py, one direction) -> the two domains share one mean
2. k-means (K clusters) on the fit set in that erased space, so clusters mix both domains
3. in each cluster k: gap delta_k = mean(synthetic in k) - mean(real in k)  (clusters with < --min_syn synthetic skipped)
   "span":  one global map that removes all delta_k directions (rank-K LEACE, oblique w.r.t. the covariance)
   "piece": each clip is erased with its own (nearest) cluster's rank-1 LEACE eraser (piecewise affine)
Fit: 800 real training normals + 200 synthetic normals; probe: 200 other real training normals vs 50 held-out
synthetic normals (same split as grl_v2/leace_scrub), 5-fold CV AUC, linear (logistic regression) and RBF SVM.
Also: real test normal vs anomaly (linear) and the cosine of erased to plain real features (centred).
Usage (main env): uv run python tools/cluster_leace.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A  # noqa: E402

log = lambda m: A.log(f"cluster_leace: {m}")


def cv_auc(X0, X1, kind):
    clf = (LogisticRegression(C=0.1, max_iter=3000) if kind == "lin" else SVC(C=1.0, kernel="rbf", gamma="scale"))
    X, y = np.r_[X0, X1], np.r_[np.zeros(len(X0)), np.ones(len(X1))]
    return roc_auc_score(y, cross_val_predict(make_pipeline(StandardScaler(), clf), X, y, cv=5, method="decision_function"))


def cov_bal(R, S, shrink=1e-2):
    """Covariance with both domains weighted equally, shrunk toward a scaled identity."""
    d = S.mean(0) - R.mean(0)
    c = (np.cov(R.T) + np.cov(S.T)) / 2 + np.outer(d, d) / 4
    return c + shrink * np.trace(c) / len(c) * np.eye(len(c))


def leace(R, S, sig):
    mu, d = (R.mean(0) + S.mean(0)) / 2, S.mean(0) - R.mean(0)
    v = np.linalg.solve(sig, d)
    return lambda X: X - ((X - mu) @ (v / (v @ d)))[:, None] * d


def main():
    z = np.load(ROOT / "cache" / "grl_v1_beats.npz")
    rng = np.random.default_rng(0)  # same draws as grl_v2 / leace_scrub
    perm = {c: rng.permutation(250) for c in ("normal", "faulty")}
    pid = rng.choice(1000, 200, replace=False)
    Rf, Rp = z["real_train"][np.setdiff1d(np.arange(1000), pid)], z["real_train"][pid]
    Sf, Sp = z["gen_normal"][perm["normal"][50:]], z["gen_normal"][perm["normal"][:50]]
    T, lab = z["real_test"], z["test_label"]
    mu0 = Rp.mean(0)

    def report(name, f):
        Er, Es, Et = f(Rp), f(Sp), f(T)
        cos = np.mean(np.sum((Er - mu0) * (Rp - mu0), 1) / (np.linalg.norm(Er - mu0, axis=1) * np.linalg.norm(Rp - mu0, axis=1)))
        log(f"{name:24s} real vs synthetic normal  linear {cv_auc(Er, Es, 'lin'):.3f}  RBF {cv_auc(Er, Es, 'rbf'):.3f} | "
            f"real test n vs a {cv_auc(Et[lab == 0], Et[lab == 1], 'lin'):.3f} | cos to plain {cos:.3f}")

    report("plain", lambda X: X)
    sig = cov_bal(Rf, Sf)
    g = leace(Rf, Sf, sig)
    report("global LEACE", g)
    Gr, Gs = g(Rf), g(Sf)
    sc = StandardScaler().fit(np.r_[Gr, Gs])
    for K in (4, 8, 16, 32):
        km = KMeans(K, n_init=10, random_state=0).fit(sc.transform(np.r_[Gr, Gs]))
        cr, cs = km.labels_[: len(Gr)], km.labels_[len(Gr):]
        ok = [k for k in range(K) if (cs == k).sum() >= 5 and (cr == k).sum() >= 5]
        log(f"K={K}: {len(ok)} clusters with both domains; synthetic per cluster "
            f"{sorted(int((cs == k).sum()) for k in range(K))}")
        # span: remove all local gap directions with one global oblique projection
        Dm = np.stack([Gs[cs == k].mean(0) - Gr[cr == k].mean(0) for k in ok], 1)        # 768 x K'
        sig_g = cov_bal(Gr, Gs)
        W = np.linalg.solve(sig_g, Dm)                                                  # Sigma^-1 D
        Pm = Dm @ np.linalg.solve(Dm.T @ W, W.T)                                        # D (D'S^-1 D)^-1 D'S^-1
        mug = (Gr.mean(0) + Gs.mean(0)) / 2
        report(f"K={K} span", lambda X, Pm=Pm, mug=mug: (lambda Y: Y - (Y - mug) @ Pm.T)(g(X)))
        # piece: each clip gets its own cluster's eraser (nearest centroid)
        er = {k: leace(Gr[cr == k], Gs[cs == k], sig_g) for k in ok}

        def piece(X, km=km, er=er):
            Y = g(X); lab_k = km.predict(sc.transform(Y)); out = Y.copy()
            for k, f in er.items():
                m = lab_k == k
                if m.any():
                    out[m] = f(Y[m])
            return out
        report(f"K={K} piece", piece)


if __name__ == "__main__":
    main()
