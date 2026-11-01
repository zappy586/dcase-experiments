"""Stage 1, alignment only: tune the backbone so that real and synthetic ToyCar NORMAL clips give the same features.
Backbone: BEATs iter3 + LoRA (rank --rank on q, v, out, fc1, fc2 of all 12 layers), token average -> 768 numbers.
Loss = MMD(real normals, synthetic normals)            multi-bandwidth Gaussian kernel, bandwidths around the median
     + --anchor x mean squared change of the real clips' features relative to plain BEATs (LoRA off)
       (stops the trivial fix of collapsing every clip to the same feature).
Data (near-minus-far, machine gain): 1000 real training normals; synthetic set v1 normals, 200 train / 50 held out.
--rt: real clips sent through AudioGen's EnCodec and back. Batches: 32 real + 32 synthetic normal, random 5 s crops.
Every --eval steps (and at step 0): fresh logistic-regression probe (5-fold CV AUC, 0.5 = cannot tell apart) on full 10 s
clips not used in training: real test normals vs held-out synthetic normals; real test anomalies vs held-out synthetic
faulty; real test normal vs anomaly (reference, to see what fault information survives); plus the feature spread of the
real test clips relative to step 0 (collapse check).
Usage (main env): uv run python tools/align_v1.py --rt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A  # noqa: E402
from grl_v2 import LoRA, Net, SYN  # noqa: E402


def mmd(x, y):
    z = torch.cat([x, y])
    d = torch.cdist(z, z).pow(2)
    med = d.detach()[d.detach() > 0].median()
    k = sum(torch.exp(-d / (med * s)) for s in (0.25, 0.5, 1.0, 2.0, 4.0))
    n = len(x)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--rt", action="store_true")
    ap.add_argument("--anchor", type=float, default=1.0)
    ap.add_argument("--mmd", type=float, default=1.0, help="weight of the MMD term")
    ap.add_argument("--coral", type=float, default=0.0, help="weight of mean + covariance matching (CORAL-style)")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--eval", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    tag = (f"mmd {args.mmd:g} coral {args.coral:g} lora r{args.rank} anchor {args.anchor:g}{' rt' if args.rt else ''} "
           f"seed {args.seed}")
    log = lambda m: A.log(f"align_v1 [{tag}]: {m}")

    D = A.Data("ToyCar", dev)
    if args.rt:
        real = torch.from_numpy(np.load(ROOT / "cache" / "toycar_rt_train.npy").astype(np.float32))
        test = torch.from_numpy(np.load(ROOT / "cache" / "toycar_rt_test.npy").astype(np.float32))
    else:
        real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)])
        test = torch.cat([D.audio("test", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 200, 25)])
    lab = np.array(D.meta["test"]["label"])
    gen = {c: torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in sorted((SYN / c).glob("*.wav"))]))
           for c in ("normal", "faulty")}
    perm = {c: rng.permutation(250) for c in gen}  # same split as grl_v2 with the same seed
    tr = gen["normal"][perm["normal"][50:]]; va = {c: gen[c][perm[c][:50]] for c in gen}

    net = Net(args.rank, False).to(dev)
    loras = [m for m in net.modules() if isinstance(m, LoRA)]
    params = [p for p in net.beats.parameters() if p.requires_grad]
    log(f"trainable backbone params {sum(p.numel() for p in params) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    def feats(x):
        return net.beats.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0].mean(1).float()

    @torch.no_grad()
    def embed(X):
        net.eval(); out = []
        for i in range(0, len(X), 16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(feats(X[i : i + 16].to(dev)).cpu())
        net.train()
        return torch.cat(out).numpy()

    def cv_auc(X0, X1):
        X, y = np.r_[X0, X1], np.r_[np.zeros(len(X0)), np.ones(len(X1))]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        return roc_auc_score(y, cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1])

    spread0 = []

    def probe(step):
        H, Hn, Hf = embed(test), embed(va["normal"]), embed(va["faulty"])
        sp = H.std(0).mean()
        if not spread0:
            spread0.append(sp)
        log(f"probe step {step}: real vs synthetic NORMAL {cv_auc(H[lab == 0], Hn):.3f} | real anomaly vs synthetic faulty "
            f"{cv_auc(H[lab == 1], Hf):.3f} | real normal vs real anomaly (reference) {cv_auc(H[lab == 0], H[lab == 1]):.3f} | "
            f"feature spread {sp / spread0[0]:.2f}x step 0")

    def crop(X, n, L=80000):
        idx = torch.randint(len(X), (n,)); st = torch.randint(0, X.shape[1] - L + 1, (n,))
        return torch.stack([X[i, s : s + L] for i, s in zip(idx.tolist(), st.tolist())])

    probe(0)
    t0 = time.time()
    for s in range(1, args.steps + 1):
        xr, xs = crop(real, 32).to(dev), crop(tr, 32).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                for m in loras:
                    m.s = 0.0
                hr0 = feats(xr)
                for m in loras:
                    m.s = 1.0
            h = feats(torch.cat([xr, xs]))
        hr, hs = h[:32], h[32:]
        scale = hr0.var(0).mean()  # plain-BEATs feature variance, so the terms do not depend on feature size
        loss_m = mmd(hr, hs)
        loss_a = (hr - hr0).pow(2).mean() / scale
        loss_mu = (hr.mean(0) - hs.mean(0)).pow(2).mean() / scale
        loss_cov = (torch.cov(hr.T) - torch.cov(hs.T)).pow(2).mean() / scale ** 2
        loss = args.mmd * loss_m + args.coral * (loss_mu + loss_cov) + args.anchor * loss_a
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0:
            log(f"step {s}: mmd {loss_m.item():.4f} mean {loss_mu.item():.4f} cov {loss_cov.item():.4f} "
                f"anchor {loss_a.item():.4f}, {(time.time() - t0) / s:.2f} s/step")
        if s % args.eval == 0:
            probe(s)
    torch.save({n: p.detach().cpu() for n, p in net.beats.named_parameters() if p.requires_grad},
               ROOT / "outputs" / f"align_v1_{tag.replace(' ', '_')}.pt")


if __name__ == "__main__":
    main()
