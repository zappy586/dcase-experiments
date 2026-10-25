"""Dinomaly-style feature reconstruction (Guo et al., CVPR 2025) for machine sound - quick test.

Frozen encoder: AudioSet ConvNeXt-Tiny (as in ae_convnext.py) on the near-only residual waveform (near - H(f) far,
least-squares H per frequency, STFT 1024/512). Features: stage-3 output (384 x 63 x 14) average-pooled 2x2 to
31 x 7 and stage-4 output (768 x 31 x 7), concatenated -> 217 tokens of 1152 numbers (cached once, fp16).
Reconstructor (trained on all 1000 normal training clips, no test data):
  noisy bottleneck: dropout(0.2) -> per-token MLP 1152 -> 256 (Dinomaly: dropout as the noise that stops identity mapping)
  decoder: 2 transformer layers, width 256, LINEAR attention (cannot focus on the token itself) + learned position
  output: 256 -> 1152, reconstruct the frozen features of every token ("loose" group: both stages at once)
  loss: 1 - cosine per token, with Dinomaly-style hard mining: tokens below the batch's 90th percentile of loss get
        their gradient scaled by 0.1
Scores (fixed in advance): cosine distance per token -> mean over all tokens ("cos-mean") and mean of the top 10% ("cos-top").
Usage: uv run python -u dinomaly_ad.py --machine ToyCar
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ae_convnext import ConvNeXtEncoder  # noqa: E402
from dcase_ae.efficientad import WavStore  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402

LOG = ROOT / "outputs" / "live.log"


def residual(w):
    """w (B, 2, L) on GPU -> near-only residual waveform (B, L)"""
    win = torch.hann_window(1024, device=w.device)
    S = torch.stft(w.reshape(-1, w.shape[-1]), 1024, 512, window=win, return_complex=True).reshape(len(w), 2, 513, -1)
    N, Fm = S[:, 0], S[:, 1]
    H = (N * Fm.conj()).mean(-1, keepdim=True) / (Fm.abs().pow(2).mean(-1, keepdim=True) + 1e-12)
    return torch.istft(N - H * Fm, 1024, 512, window=win, length=w.shape[-1])


@torch.no_grad()
def cache_features(machine, split, dev):
    f = ROOT / "cache" / f"dino_cnx_{machine}_{split}.pt"
    if f.exists():
        return torch.load(f)
    enc = ConvNeXtEncoder("frozen", 8, dev)
    m = enc.m
    st = WavStore("cache", machine, split)
    out = []
    for i in range(0, len(st), 16):
        r = residual(st.get(list(range(i, min(i + 16, len(st))))).to(dev))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = torchaudio.functional.resample(r, 16000, 32000)
            x = m.logmel_extractor(m.spectrogram_extractor(x))
            x = m.bn0(x.transpose(1, 3)).transpose(1, 3)
            for k in range(3):
                x = m.stages[k](m.downsample_layers[k](x))
            s3 = x
            s4 = m.stages[3](m.downsample_layers[3](x))
        s3 = F.adaptive_avg_pool2d(s3.float(), s4.shape[-2:])
        tok = torch.cat([s3, s4.float()], 1).flatten(2).transpose(1, 2)  # (B, 217, 1152)
        out.append(tok.half().cpu())
    out = torch.cat(out)
    torch.save(out, f)
    return out


class LinearAttention(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.h, self.qkv, self.o = heads, nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        kv = torch.einsum("bhtd,bhte->bhde", k, v)
        z = 1 / (torch.einsum("bhtd,bhd->bht", q, k.sum(2)) + 1e-6)
        y = torch.einsum("bhtd,bhde,bht->bhte", q, kv, z)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n1, self.att, self.n2 = nn.LayerNorm(d), LinearAttention(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.mlp(self.n2(x))


class Recon(nn.Module):
    def __init__(self, c=1152, d=256, layers=2, tokens=217, drop=0.2):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.bottleneck = nn.Sequential(nn.Linear(c, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, tokens, d))
        self.blocks = nn.ModuleList(Block(d) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, c))

    def forward(self, x):
        h = self.bottleneck(self.drop(x)) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.out(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  [dinomaly {args.machine}] {msg}"
        print(line, flush=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    t0 = time.time()
    Ftr, Fte = cache_features(args.machine, "train", dev), cache_features(args.machine, "test", dev)
    log(f"features train {tuple(Ftr.shape)} test {tuple(Fte.shape)} ready in {time.time() - t0:.0f}s")
    meta = json.loads((ROOT / "cache" / f"wav_{args.machine}_test.json").read_text())
    lab, dom = np.array(meta["label"]), np.array(meta["domain"])
    Ftr = Ftr.to(dev)
    model = Recon().to(dev)
    log(f"reconstructor parameters {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)

    @torch.no_grad()
    def scores(Fx):
        model.eval()
        D = []
        for i in range(0, len(Fx), 50):
            x = Fx[i : i + 50].to(dev).float()
            D.append((1 - F.cosine_similarity(model(x), x, dim=-1)).cpu())
        model.train()
        D = torch.cat(D)
        k = max(1, D.shape[1] // 10)
        return {"cos-mean": D.mean(1), "cos-top": D.topk(k, dim=1).values.mean(1)}

    def evaluate(step):
        tr_s, te_s = scores(Ftr[:200]), scores(Fte)
        res = {k: compute_metrics(v.numpy(), lab, dom) for k, v in te_s.items()}
        n = lab == 0
        log(f"step {step:5d} | train cos-mean {tr_s['cos-mean'].mean():.4f} test normal {te_s['cos-mean'][n].mean():.4f} "
            f"anomaly {te_s['cos-mean'][~n].mean():.4f} | "
            + " | ".join(f"{k} src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} pAUC {r['pauc']:.3f} off {r['official']:.3f}"
                         for k, r in res.items()))
        return res

    losses = []
    for step in range(1, args.steps + 1):
        x = Ftr[torch.randint(len(Ftr), (args.batch,), device=dev)].float()
        d = 1 - F.cosine_similarity(model(x), x, dim=-1)  # (B, tokens)
        thr = torch.quantile(d.detach().flatten(), 0.9)
        w = torch.where(d.detach() >= thr, 1.0, 0.1)
        loss = (d * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(d.mean().item())
        if step % 100 == 0:
            log(f"step {step:5d} loss {np.mean(losses[-100:]):.4f} {(time.time() - t0) / step:.3f}s/step")
        if step % args.eval_every == 0:
            res = evaluate(step)
    out = ROOT / "outputs" / "dinomaly" / args.machine
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    log("=== done")


if __name__ == "__main__":
    main()
