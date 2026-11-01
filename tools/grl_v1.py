"""v1 domain-adversarial classifier (gradient reversal, Ganin et al. JMLR 2016) on ToyCar.
Data: 1000 real training normals (near-minus-far, machine gain) + synthetic set v1 (250 normal / 250 faulty, AudioGen).
Input: frozen BEATs iter3 features, average over tokens (768 per clip, cached in cache/grl_v1_beats.npz).
Model: backbone MLP 768 -> 256 -> 128; head 1 = normal vs faulty; head 2 = real vs synthetic behind a gradient-reversal
layer (lambda ramp 2/(1+exp(-10 p)) - 1, times --lam). Head 1 sees all clips (real normals + synthetic both classes),
batches are class- and domain-balanced. Test score = head-1 logit on the 200 ToyCar test clips (labels used only for the
metrics). --lam 0 is the same model without the adversary.
Usage (main env): uv run python tools/grl_v1.py --lam 1 --seeds 3
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402

SYN = ROOT / "outputs" / "audiogen" / "synth_v1"
CACHE = ROOT / "cache" / "grl_v1_beats.npz"


@torch.no_grad()
def features(dev):
    if CACHE.exists():
        return dict(np.load(CACHE))
    from BEATs import BEATs, BEATsConfig
    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    model = BEATs(BEATsConfig(ck["cfg"]))
    model.load_state_dict(ck["model"])
    model = model.to(dev).eval()

    def emb(x):
        return model.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0].mean(1).cpu()

    D = A.Data("ToyCar", dev)
    out = {}
    for split in ("train", "test"):
        n = D.wav[split].shape[0]
        out[f"real_{split}"] = torch.cat([emb(D.audio(split, list(range(i, min(i + 10, n))))[:, 0, :160000])
                                          for i in range(0, n, 10)]).numpy()
    out["test_label"] = np.array(D.meta["test"]["label"]); out["test_domain"] = np.array(D.meta["test"]["domain"])
    for c in ("normal", "faulty"):
        w = [sf.read(f, dtype="float32")[0][:160000] for f in sorted((SYN / c).glob("*.wav"))]
        out[f"gen_{c}"] = torch.cat([emb(torch.as_tensor(np.stack(w[i : i + 10]), device=dev)) for i in range(0, len(w), 10)]).numpy()
    np.savez(CACHE, **out)
    return out


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Net(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.GELU())
        self.cls = nn.Linear(128, 1)
        self.dom = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, lam=0.0):
        h = self.backbone(x)
        return self.cls(h)[:, 0], self.dom(GradReverse.apply(h, lam))[:, 0]


def run(Fe, lam_max, seed, steps, dev, log):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mu, sd = Fe["real_train"].mean(0), Fe["real_train"].std(0) + 1e-6  # standardise with real normals only
    T = {k: torch.tensor((Fe[k] - mu) / sd, dtype=torch.float32, device=dev) for k in ("real_train", "real_test", "gen_normal", "gen_faulty")}
    # hold out 50 synthetic clips per class to see whether head 1 learned the synthetic task
    perm = {c: rng.permutation(250) for c in ("normal", "faulty")}
    tr = {c: T[f"gen_{c}"][perm[c][50:]] for c in perm}; va = {c: T[f"gen_{c}"][perm[c][:50]] for c in perm}
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    pick = lambda X, n: X[torch.randint(len(X), (n,), device=dev)]
    for s in range(steps):
        p = s / steps
        lam = lam_max * (2 / (1 + np.exp(-10 * p)) - 1)
        xr, xn, xf = pick(T["real_train"], 64), pick(tr["normal"], 32), pick(tr["faulty"], 32)
        x = torch.cat([xr, xn, xf])
        y = torch.cat([torch.zeros(96, device=dev), torch.ones(32, device=dev)])        # 1 = faulty
        d = torch.cat([torch.zeros(64, device=dev), torch.ones(64, device=dev)])        # 1 = synthetic
        w = torch.cat([torch.full((96,), 0.5 / 96, device=dev), torch.full((32,), 0.5 / 32, device=dev)])
        lc, ld = net(x, lam)
        loss = (F.binary_cross_entropy_with_logits(lc, y, reduction="none") * w).sum() + F.binary_cross_entropy_with_logits(ld, d)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        sc, dt = net(T["real_test"])
        vn, _ = net(va["normal"]); vf, _ = net(va["faulty"])
        _, dr = net(T["real_train"]); _, dg = net(torch.cat([va["normal"], va["faulty"]]))
    m = compute_metrics(sc.cpu().numpy(), Fe["test_label"], Fe["test_domain"])
    from sklearn.metrics import roc_auc_score
    syn_auc = roc_auc_score(np.r_[np.zeros(50), np.ones(50)], torch.cat([vn, vf]).cpu().numpy())
    dom_acc = ((dr < 0).float().mean().item() * 1000 + (dg > 0).float().mean().item() * 100) / 1100
    test_as_syn = (dt > 0).float().mean().item()
    log(f"lam {lam_max:g} seed {seed}: real test AUC src {m['auc_source']:.3f} tgt {m['auc_target']:.3f} pAUC {m['pauc']:.3f} "
        f"official {m['official']:.3f} | synthetic held-out AUC {syn_auc:.3f} | domain head acc {dom_acc:.3f}, "
        f"real test called synthetic {test_as_syn:.2f}")
    return m["official"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    dev = torch.device("cuda")
    Fe = features(dev)
    log = lambda m: A.log(f"grl_v1: {m}")
    for lam in args.lam:
        r = [run(Fe, lam, s, args.steps, dev, log) for s in range(args.seeds)]
        log(f"lam {lam:g}: official mean {np.mean(r):.3f} +- {np.std(r):.3f}")


if __name__ == "__main__":
    main()
