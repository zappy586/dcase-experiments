"""LEACE concept scrubbing of BEATs (Belrose et al., "LEACE: Perfect linear concept erasure in closed form",
NeurIPS 2023): make real vs synthetic NORMALS linearly unreadable inside the encoder, changing the features as little
as possible. No gradient training.

After transformer layer l a fixed affine eraser is inserted on every token:
    x -> x - ((x - mu_l) . a_l) * delta_l
    delta_l = mean(synthetic tokens) - mean(real tokens),  a_l = Sigma_l^-1 delta_l / (delta_l' Sigma_l^-1 delta_l)
Sigma_l = token covariance with both domains weighted equally. This is the closed-form LEACE eraser for a two-class
concept: it makes the two domains' means equal (so no linear classifier beats chance) and is the least-squares
smallest edit that does so. Layers are fitted in order, each on the outputs of the already-scrubbed layers below
("concept scrubbing", LEACE paper sec. 6), so later layers cannot simply rebuild the direction.

Fit set: 800 real ToyCar training normals + 200 synthetic normals (synth_v1). Probe set (never used for fitting, same
split as grl_v2): 200 other real training normals vs 50 held-out synthetic normals. After each layer is scrubbed it logs
a linear probe (logistic regression) and a non-linear probe (RBF SVM) on the mean-pooled output, 5-fold CV AUC
(0.5 = cannot tell them apart), and how far the real features moved.
Usage (main env): uv run python tools/leace_scrub.py [--layers all|last]
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from BEATs import BEATs, BEATsConfig  # noqa: E402

SYN = ROOT / "outputs" / "audiogen" / "synth_v1"


class Stop(Exception):
    pass


class Stats:
    """Per-domain token sums and outer products (float64)."""
    def __init__(self, C, dev):
        self.n = [0, 0]; self.s = [torch.zeros(C, dtype=torch.float64, device=dev) for _ in range(2)]
        self.S = [torch.zeros(C, C, dtype=torch.float64, device=dev) for _ in range(2)]
        self.label = 0

    def add(self, x):  # x: T x B x C
        t = x.reshape(-1, x.shape[-1]).double()
        k = self.label; self.n[k] += len(t); self.s[k] += t.sum(0); self.S[k] += t.T @ t

    def eraser(self, shrink):
        mu = [self.s[k] / self.n[k] for k in range(2)]
        cov = [self.S[k] / self.n[k] - torch.outer(mu[k], mu[k]) for k in range(2)]
        delta = mu[1] - mu[0]
        sig = (cov[0] + cov[1]) / 2 + torch.outer(delta, delta) / 4
        sig = sig + shrink * torch.trace(sig) / len(sig) * torch.eye(len(sig), dtype=sig.dtype, device=sig.device)
        v = torch.linalg.solve(sig, delta)
        a = v / (v @ delta)
        return ((mu[0] + mu[1]) / 2).float(), a.float(), delta.float()


class Erased(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer, self.stats, self.e = layer, None, None

    def forward(self, x, **kw):
        x, z, pb = self.layer(x, **kw)
        if self.stats is not None:
            self.stats.add(x)
            raise Stop
        if self.e is not None:
            mu, a, delta = self.e
            xf = x.float()
            x = (xf - ((xf - mu) @ a)[..., None] * delta).to(x.dtype)
        return x, z, pb


def cv_auc(X0, X1, clf):
    X, y = np.r_[X0, X1], np.r_[np.zeros(len(X0)), np.ones(len(X1))]
    p = cross_val_predict(clf, X, y, cv=5, method="decision_function")
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="all", choices=["all", "last"])
    ap.add_argument("--shrink", type=float, default=1e-2, help="covariance shrinkage toward a scaled identity")
    args = ap.parse_args()
    dev = torch.device("cuda")
    log = lambda m: A.log(f"leace_scrub [{args.layers}]: {m}")
    rng = np.random.default_rng(0)  # same draws as grl_v2 seed 0: synthetic split, then probe ids

    D = A.Data("ToyCar", dev)
    real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)])
    test = torch.cat([D.audio("test", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 200, 25)])
    lab = np.array(D.meta["test"]["label"])
    gen = {c: torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in sorted((SYN / c).glob("*.wav"))]))
           for c in ("normal", "faulty")}
    perm = {c: rng.permutation(250) for c in gen}
    syn_fit, syn_probe = gen["normal"][perm["normal"][50:]], gen["normal"][perm["normal"][:50]]
    probe_ids = rng.choice(1000, 200, replace=False)
    real_fit = real[np.setdiff1d(np.arange(1000), probe_ids)]
    real_probe = real[probe_ids]
    log(f"fit: {len(real_fit)} real + {len(syn_fit)} synthetic normals | probe: {len(real_probe)} real + {len(syn_probe)} synthetic")

    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    beats = BEATs(BEATsConfig(ck["cfg"])); beats.load_state_dict(ck["model"]); beats.eval().to(dev)
    layers = beats.encoder.layers
    for i in range(len(layers)):
        layers[i] = Erased(layers[i])

    def run(X, bs=16):
        out = []
        for i in range(0, len(X), bs):
            x = X[i : i + bs].to(dev)
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    h = beats.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0]
                out.append(h.float().mean(1).cpu())
            except Stop:
                pass
        return torch.cat(out).numpy() if out else None

    lin = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    rbf = lambda: make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale"))
    P0r, P0s, T0 = run(real_probe), run(syn_probe), run(test)
    mu0 = P0r.mean(0)

    def probe(name):
        Pr, Ps, T = run(real_probe), run(syn_probe), run(test)
        dr, ds = Pr - P0r, Ps - P0s
        cos = np.mean(np.sum((Pr - mu0) * (P0r - mu0), 1) / (np.linalg.norm(Pr - mu0, axis=1) * np.linalg.norm(P0r - mu0, axis=1)))
        rel = np.linalg.norm(dr) / np.linalg.norm(P0r - mu0)
        log(f"{name}: real normal vs synthetic normal  linear {cv_auc(Pr, Ps, lin()):.3f}  RBF {cv_auc(Pr, Ps, rbf()):.3f} | "
            f"real features moved {rel:.3f} of their spread (cos to plain {cos:.3f}), synthetic moved "
            f"{np.linalg.norm(ds) / np.linalg.norm(P0s - P0s.mean(0)):.3f} | real test normal vs anomaly linear "
            f"{cv_auc(T[lab == 0], T[lab == 1], lin()):.3f}")

    probe("plain BEATs")
    todo = range(len(layers)) if args.layers == "all" else [len(layers) - 1]
    t0 = time.time()
    saved = {}
    for l in todo:
        st = Stats(768, dev)
        layers[l].stats = st
        st.label = 0; run(real_fit, bs=32)
        st.label = 1; run(syn_fit, bs=32)
        layers[l].stats = None
        layers[l].e = st.eraser(args.shrink)
        saved[l] = [t.cpu() for t in layers[l].e]
        d = layers[l].e[2]
        log(f"layer {l + 1}: token mean gap {d.norm():.3f} ({time.time() - t0:.0f} s)")
        probe(f"after layer {l + 1}")
    torch.save({"erasers": saved, "shrink": args.shrink}, ROOT / "outputs" / f"leace_scrub_{args.layers}.pt")
    log("done")


if __name__ == "__main__":
    main()
