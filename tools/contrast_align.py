"""Contrastive real/synthetic alignment of the backbone, normals only (no fault classifier yet).
Backbone: BEATs iter3 + LoRA (rank --rank on q, v, out, fc1, fc2 of all 12 layers), token average -> 768 numbers,
standardised with the plain-BEATs statistics of the real training normals, then L2-normalised (cosine space).
Per step: 32 real normal + 32 synthetic normal random 5 s crops (clean near-minus-far).
  pairing     : each synthetic crop is matched one-to-one to a real crop of the batch by optimal assignment on the
                plain-BEATs (LoRA off) features, so every synthetic clip has its own nearest real partner (no collapse).
  align loss  : InfoNCE (CAN-style, class "normal" only): a synthetic crop's tuned feature must pick out its matched real
                crop's tuned feature among the 32 real crops (temperature --tau).
  keep loss   : InfoNCE (CSG-style): a real crop's tuned feature must pick out its own plain-BEATs feature among the 32
                plain-BEATs real features -> keeps the relations between real clips, allows an overall shift.
  loss = align + --keep x keep.
Every --eval steps (and step 0): fresh logistic-regression probe (5-fold CV AUC) on full 10 s clips:
  real TRAIN normals (200 fixed) vs 50 held-out synthetic normals (should fall towards 0.5),
  real test normal vs real test anomaly (reference, should stay ~0.83), feature spread vs step 0 (collapse check).
Usage (main env): uv run python tools/contrast_align.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A  # noqa: E402
from grl_v2 import LoRA, Net, SYN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--keep", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--eval", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    tag = f"lora r{args.rank} keep {args.keep:g} tau {args.tau:g} seed {args.seed}"
    log = lambda m: A.log(f"contrast_align [{tag}]: {m}")

    D = A.Data("ToyCar", dev)
    real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)])
    test = torch.cat([D.audio("test", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 200, 25)])
    lab = np.array(D.meta["test"]["label"])
    gen = {c: torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in sorted((SYN / c).glob("*.wav"))]))
           for c in ("normal", "faulty")}
    perm = {c: rng.permutation(250) for c in gen}  # same held-out split as grl_v2
    tr, va = gen["normal"][perm["normal"][50:]], gen["normal"][perm["normal"][:50]]
    probe_ids = rng.choice(1000, 200, replace=False)

    net = Net(args.rank, False).to(dev)
    loras = [m for m in net.modules() if isinstance(m, LoRA)]
    params = [p for p in net.beats.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    log(f"trainable backbone params {sum(p.numel() for p in params) / 1e6:.2f}M")

    def lora_on(on):
        for m in loras:
            m.s = 1.0 if on else 0.0

    def feats(x):
        return net.beats.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0].mean(1).float()

    @torch.no_grad()
    def embed(X):
        net.eval(); out = []
        for i in range(0, len(X), 16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(feats(X[i : i + 16].to(dev)).cpu())
        net.train()
        return torch.cat(out)

    lora_on(False)
    base = embed(real[rng.choice(1000, 300, replace=False)])
    lora_on(True)
    mu, sd = base.mean(0).to(dev), base.std(0).to(dev) + 1e-6
    norm = lambda h: F.normalize((h - mu) / sd, dim=1)

    def cv_auc(X0, X1):
        X, y = np.r_[X0, X1], np.r_[np.zeros(len(X0)), np.ones(len(X1))]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        return roc_auc_score(y, cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1])

    spread0 = []

    def probe(step):
        Hr, Hn, H = embed(real[probe_ids]).numpy(), embed(va).numpy(), embed(test).numpy()
        sp = H.std(0).mean()
        if not spread0:
            spread0.append(sp)
        log(f"probe step {step}: real train normal vs synthetic normal {cv_auc(Hr, Hn):.3f} | real test normal vs "
            f"real test anomaly (reference) {cv_auc(H[lab == 0], H[lab == 1]):.3f} | feature spread {sp / spread0[0]:.2f}x step 0")

    def crop(X, n, L=80000):
        idx = torch.randint(len(X), (n,)); st = torch.randint(0, X.shape[1] - L + 1, (n,))
        return torch.stack([X[i, s : s + L] for i, s in zip(idx.tolist(), st.tolist())])

    probe(0)
    tgt = torch.arange(32, device=dev)
    t0 = time.time()
    for s in range(1, args.steps + 1):
        x = torch.cat([crop(real, 32), crop(tr, 32)]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                lora_on(False); f0 = norm(feats(x)); lora_on(True)
            f = norm(feats(x))
        fr0, fs0, fr, fs = f0[:32], f0[32:], f[:32], f[32:]
        _, match = linear_sum_assignment((-(fs0 @ fr0.T)).cpu().numpy())    # synthetic i <-> real match[i]
        match = torch.as_tensor(match, device=dev)
        # candidates = 32 real crops AND the 31 other synthetic crops: a shared "synthetic" offset makes the other
        # synthetic crops look closer than the real partner, so it is penalised (with real-only candidates it was not)
        cand = fs @ torch.cat([fr, fs]).T / args.tau
        cand[:, 32:] = cand[:, 32:].masked_fill(torch.eye(32, dtype=torch.bool, device=dev), float("-inf"))
        loss_align = F.cross_entropy(cand, match)
        loss_keep = F.cross_entropy(fr @ fr0.T / args.tau, tgt)
        loss = loss_align + args.keep * loss_keep
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0:
            with torch.no_grad():
                acc = (torch.argmax(cand, 1) == match).float().mean().item()
            log(f"step {s}: align {loss_align.item():.3f} (partner found {acc:.2f}) keep {loss_keep.item():.3f}, "
                f"{(time.time() - t0) / s:.2f} s/step")
        if s % args.eval == 0:
            probe(s)
    torch.save({n: p.detach().cpu() for n, p in net.beats.named_parameters() if p.requires_grad},
               ROOT / "outputs" / f"contrast_align_{tag.replace(' ', '_')}.pt")


if __name__ == "__main__":
    main()
