"""Within-clip consistency fine-tuning (normal clips only, same settings for every machine).

Assumption: a machine's condition does not change within one 10 s clip, while the sound's texture does.
Train Dasheng-1.2B (LoRA on attn.qkv / attn.proj / mlp.fc1 of all 40 blocks, base frozen) so that two random
2.5 s segments of the same normal clip get the same embedding and different clips stay distinct (VICReg:
invariance + variance + covariance terms, on a projection head). The embedding is pushed to keep what is constant
within a clip (machine state, operating condition, environment) and drop texture.
Input: near mic only, as recorded.
Score: full 10 s clip, time-averaged backbone tokens at layers 7/13/20/27/33/40; Ledoit-Wolf Mahalanobis and
1-NN per layer, z-normalised on 100 held-out normal clips and summed over layers. Step 0 = frozen Dasheng
(the bar). Reported at the last step; intermediate evaluations are for watching only.
Usage: uv run python -u clip_consistency.py --machine ToyCar
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from torch.utils.checkpoint import checkpoint

from clora_gen import DASHENG_LAYERS
from dcase_ae.efficientad import WavStore, build_cache
from dcase_ae.evaluation import compute_metrics

SR, SEG = 16000, 40000  # 2.5 s segments


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T.to(x.dtype)) @ self.B.T.to(x.dtype)


def vicreg(z1, z2):
    inv = F.mse_loss(z1, z2)
    std = lambda z: torch.sqrt(z.var(0) + 1e-4)
    var = F.relu(1 - std(z1)).mean() + F.relu(1 - std(z2)).mean()
    def cov(z):
        z = z - z.mean(0)
        c = (z.T @ z) / (len(z) - 1)
        return (c.pow(2).sum() - c.diagonal().pow(2).sum()) / z.shape[1]
    return 25 * inv + 25 * var + cov(z1) + cov(z2), inv.item(), var.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    out = Path("outputs/clip_consistency") / args.name / args.machine
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')}  [{args.name}/{args.machine}] {msg}", flush=True)

    log(f"=== {vars(args)}")
    from dasheng import dasheng_12B
    model = dasheng_12B(path="pretrained/dasheng/dasheng_12b.pt").to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    loras = []
    for blk in model.blocks:
        for parent, attr in ((blk.attn, "qkv"), (blk.attn, "proj"), (blk.mlp, "fc1")):
            l = LoRA(getattr(parent, attr), args.rank, 2 * args.rank)
            setattr(parent, attr, l)
            loras.append(l)
        fwd = blk.forward
        blk.forward = (lambda f: (lambda x, **kw: checkpoint(f, x, use_reentrant=False, **kw) if torch.is_grad_enabled() else f(x, **kw)))(fwd)
    grab = {}
    for L in DASHENG_LAYERS:
        model.blocks[L - 1].register_forward_hook(lambda m, a, o, L=L: grab.__setitem__(L, o))
    head = nn.Sequential(nn.Linear(1536, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, 2048)).to(dev)
    params = [p for l in loras for p in (l.A, l.B)]
    opt = torch.optim.AdamW([{"params": params, "lr": args.lr}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.0)
    log(f"LoRA parameters {sum(p.numel() for p in params):,}, head {sum(p.numel() for p in head.parameters()):,}")

    for split in ("train", "test"):
        build_cache("data", "cache", args.machine, split)
    tr, te = WavStore("cache", args.machine, "train"), WavStore("cache", args.machine, "test")
    idx = list(range(len(tr)))
    random.Random(0).shuffle(idx)
    val_idx, fit_idx = idx[:100], idx[100:]
    lab, dom = np.array(te.meta["label"]), np.array(te.meta["domain"])

    @torch.no_grad()
    def features(store, ids):
        out = {L: [] for L in DASHENG_LAYERS}
        for i in range(0, len(ids), 16):
            w = store.get(ids[i : i + 16])[:, 0].to(dev).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(w)
            for L in DASHENG_LAYERS:
                out[L].append(grab[L].float().mean(1).cpu().double().numpy())
        return {L: np.concatenate(v) for L, v in out.items()}

    def evaluate(step, final=False):
        t0 = time.time()
        Fx, Fv, Ft = features(tr, fit_idx), features(tr, val_idx), features(te, list(range(len(te))))
        res = {}
        for name in ("maha", "knn1"):
            tot = 0
            for L in DASHENG_LAYERS:
                if name == "maha":
                    lw = LedoitWolf().fit(Fx[L])
                    f = lambda Q: np.einsum("nd,de,ne->n", Q - lw.location_, lw.precision_, Q - lw.location_)
                else:
                    m_, s_ = Fx[L].mean(0), Fx[L].std(0) + 1e-6
                    X = torch.tensor((Fx[L] - m_) / s_, device=dev)
                    f = lambda Q: torch.cdist(torch.tensor((Q - m_) / s_, device=dev), X).min(1).values.cpu().numpy()
                sv, st = f(Fv[L]), f(Ft[L])
                tot = tot + (st - sv.mean()) / (sv.std() + 1e-9)
            res[name] = compute_metrics(tot, lab, dom)
        log(f"step {step:5d} | " + " | ".join(f"{k} AUC {r['auc_all']:.3f} src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} "
                                              f"pAUC {r['pauc']:.3f} official {r['official']:.3f}" for k, r in res.items())
            + f" ({time.time() - t0:.0f}s){'  (FINAL)' if final else ''}")
        return res

    hist = {"step0": evaluate(0)}
    rng = random.Random(args.seed + 1)
    t0, losses = time.time(), []
    for step in range(1, args.steps + 1):
        ids = rng.sample(fit_idx, args.batch)
        w = tr.get(ids)[:, 0].float()
        L = w.shape[1]
        segs = torch.stack([torch.stack([x[s : s + SEG] for s in (rng.randrange(0, L - SEG), rng.randrange(0, L - SEG))]) for x in w])
        segs = segs.to(dev).reshape(-1, SEG)  # (2B, SEG), pairs adjacent
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(segs).mean(1)
        z = head(h.float()).reshape(args.batch, 2, -1)
        loss, inv, var = vicreg(z[:, 0], z[:, 1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append((loss.item(), inv, var))
        if step % 25 == 0:
            a = np.mean(losses[-25:], 0)
            log(f"step {step:5d} loss {a[0]:.3f} (invariance {a[1]:.4f}, variance {a[2]:.3f}) "
                f"{(time.time() - t0) / step:.2f}s/step, GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if step % args.eval_every == 0 and step != args.steps:
            hist[f"step{step}"] = evaluate(step)
    hist["final"] = evaluate(args.steps, final=True)
    (out / "summary.json").write_text(json.dumps(hist, indent=1))
    log("=== done")


if __name__ == "__main__":
    main()
