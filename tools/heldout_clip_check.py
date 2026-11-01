"""Which held-out synthetic normals stay detectable after CLoRA-Align, and does it depend on where they started?

Encoder: BEATs + saved CLoRA-Align adapters (default: rank 16, 480 synthetic clips, step 600; A and B loaded from file).
Per clip of the 50 held-out synthetic normals: out-of-fold RBF-SVM score (5-fold CV, real = 200 held-out real training
normals vs these 50) on aligned features = how synthetic it still looks.
Starting position, on PLAIN BEATs features (global LEACE erased + standardised, as in the coverage check):
  real share = fraction of its 10 nearest neighbours among all 1000 real training normals + the other 249 v1 synthetic
  normals that are real (0 = deep in a synthetic-only region); also k-means (K=16) cluster: synthetic-only (< 5% real)
Usage (main env): uv run python tools/heldout_clip_check.py
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from BEATs import BEATs, BEATsConfig  # noqa: E402
from clora_align import FrozenALoRA  # noqa: E402
from cluster_leace import cov_bal, leace  # noqa: E402

SYN = ROOT / "outputs" / "audiogen" / "synth_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/clora_align_r16_s1_lr0.0005_extra.pt")
    args = ap.parse_args()
    dev = torch.device("cuda")
    log = lambda m: A.log(f"heldout_clip_check: {m}")
    rng = np.random.default_rng(0)
    D = A.Data("ToyCar", dev)
    real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)])
    gen = torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in sorted((SYN / "normal").glob("*.wav"))]))
    perm = {c: rng.permutation(250) for c in ("normal", "faulty")}
    pid = rng.choice(1000, 200, replace=False)
    ho = perm["normal"][:50]

    # ---- aligned features of the probe sets
    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    beats = BEATs(BEATsConfig(ck["cfg"])); beats.load_state_dict(ck["model"]); beats.eval().to(dev)
    saved = torch.load(ROOT / args.ckpt)
    for name in sorted({n.rsplit(".", 1)[0] for n in saved}):  # e.g. encoder.layers.0.self_attn.q_proj
        parent_name, attr = name.rsplit(".", 1)
        parent = beats.get_submodule(parent_name)
        Am = saved[f"{name}.A"]
        setattr(parent, attr, FrozenALoRA(getattr(parent, attr), Am.to(dev), 1.0))
    sd = beats.state_dict(); sd.update({n: t.to(dev) for n, t in saved.items()}); beats.load_state_dict(sd)
    log(f"loaded {len(saved)} tensors ({len(saved) // 2} adapters) from {args.ckpt}")

    def pooled(X, bs=16):
        out = []
        for i in range(0, len(X), bs):
            x = X[i : i + bs].to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(beats.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0].float().mean(1).cpu())
        return torch.cat(out).numpy()

    Pr, Ps = pooled(real[pid]), pooled(gen[ho])
    X, y = np.r_[Pr, Ps], np.r_[np.zeros(len(Pr)), np.ones(len(Ps))]
    oof = cross_val_predict(make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale")), X, y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0), method="decision_function")
    auc = roc_auc_score(y, oof)
    s_real, s_syn = oof[:200], oof[200:]
    pct = np.array([(s_real < v).mean() for v in s_syn])  # share of real clips that look less synthetic than this clip
    log(f"aligned RBF probe AUC {auc:.3f}; per held-out synthetic clip: share of real clips scored below it "
        f"(1 = clearly synthetic, 0.5 = like a typical real clip): median {np.median(pct):.2f}")

    # ---- starting position on plain features
    z = np.load(ROOT / "cache" / "grl_v1_beats.npz")
    R0, S0 = z["real_train"], z["gen_normal"]
    g = leace(R0, S0, cov_bal(R0, S0))
    sc = StandardScaler().fit(np.r_[g(R0), g(S0)])
    Gr, Gs = sc.transform(g(R0)), sc.transform(g(S0))
    allp = np.r_[Gr, Gs]; isreal = np.r_[np.ones(len(Gr)), np.zeros(len(Gs))]
    share = []
    for i in ho:
        q = Gs[i]; d = np.linalg.norm(allp - q, axis=1); d[len(Gr) + i] = np.inf
        share.append(isreal[np.argsort(d)[:10]].mean())
    share = np.array(share)
    km = KMeans(16, n_init=10, random_state=0).fit(allp)
    cr, cs = km.labels_[: len(Gr)], km.labels_[len(Gr):]
    frac_real = np.array([(cr == k).sum() / max(1, (cr == k).sum() + (cs == k).sum()) for k in range(16)])
    synonly = frac_real[cs[ho]] < 0.05
    rho, p = spearmanr(share, pct)
    log(f"start position vs still-synthetic: Spearman rho {rho:+.2f} (p {p:.3f}) between real share of 10 nearest "
        f"neighbours (plain BEATs) and the per-clip score")
    for nm, m in (("synthetic-only cluster", synonly), ("mixed cluster", ~synonly)):
        if m.any():
            log(f"  {nm:22s}: {m.sum():2d} clips | real-neighbour share {share[m].mean():.2f} | still-synthetic score "
                f"median {np.median(pct[m]):.2f} | clearly synthetic (> 0.95) {(pct[m] > 0.95).mean() * 100:.0f}%")
    for lo, hi in ((0, 0.001), (0.001, 0.3), (0.3, 0.7), (0.7, 1.01)):
        m = (share >= lo) & (share < hi)
        if m.any():
            log(f"  real-neighbour share {lo:.1f}-{min(hi, 1):.1f}: {m.sum():2d} clips | still-synthetic median {np.median(pct[m]):.2f} | "
                f"> 0.95: {(pct[m] > 0.95).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
