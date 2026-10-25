"""Noise-aware autoencoder that reconstructs the WHOLE near-minus-far spectrogram (every frame), not its time average.

Same as ae_noiseaware.py (reference v4_all7 settings) except the target and the decoder:
  input  : near + far log-power spectrograms (256 bands x 313 frames, STFT 1024/512), standardised per mic and band
           70% of training clips get extra background (far mic of another clip, 0..-15 dB at the far mic, 3-10 dB
           quieter at the near mic); the target always comes from the recording before augmentation
  target : near-minus-far power per frame: |N - H(f) F|^2 with one least-squares H(f) per clip (as in the reference,
           but kept per frame), dB, standardised per band with all training clips -> 256 x 313
  encoder: the reference's 4 conv blocks (frequency /16, time /8) -> 32 numbers per time step (~0.26 s)
  decoder: per time step 32 -> 512 -> 512 (kernel 3 over time) -> 256 bands x 8 frames
Scores (fixed in advance): mean squared error over the whole spectrogram ('mse-mean'); mean of the 10% worst frames
('mse-topframes'). All 1000 training clips used. Usage: uv run python -u ae_spec.py --machine ToyCar
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dcase_ae.efficientad import WavStore, build_cache
from dcase_ae.evaluation import compute_metrics

ROOT = Path(__file__).resolve().parent
NFFT, HOP, NB = 1024, 512, 256


class Spec:
    def __init__(self, dev):
        self.win = torch.hann_window(NFFT, device=dev)

    def __call__(self, w):
        """w (B, 2, L) -> input dB (B, 2, NB, T) and near-minus-far dB per frame (B, NB, T)"""
        B = len(w)
        S = torch.stft(w.reshape(2 * B, -1), NFFT, HOP, window=self.win, return_complex=True)[:, 1:]
        S = S.reshape(B, 2, NB, 2, -1)
        x = 10 * torch.log10(S.abs().pow(2).sum(3) + 1e-10)
        N, F = S[:, 0], S[:, 1]
        H = (N * F.conj()).mean(-1, keepdim=True) / (F.abs().pow(2).mean(-1, keepdim=True) + 1e-12)
        R = (N - H * F).abs().pow(2).sum(2)  # (B, NB, T)
        return x, 10 * torch.log10(R + 1e-10)


class AE(nn.Module):
    def __init__(self, code=32, chans=(32, 64, 128, 128), hidden=512):
        super().__init__()
        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.BatchNorm2d(o), nn.GELU())
        c1, c2, c3, c4 = chans
        self.enc = nn.Sequential(blk(2, c1, (2, 1)), blk(c1, c2, (2, 2)), blk(c2, c3, (2, 2)), blk(c3, c4, (2, 2)))
        self.squeeze = nn.Conv1d(c4 * (NB // 16), code, 1)
        self.dec = nn.Sequential(nn.Conv1d(code, hidden, 1), nn.GELU(), nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(hidden, NB * 8, 1))

    def forward(self, x):
        T = x.shape[-1]
        h = self.enc(x)  # (B, C, NB/16, T/8)
        z = self.squeeze(h.flatten(1, 2))  # (B, code, T/8)
        y = self.dec(z)  # (B, NB*8, T/8)
        B, _, Tq = y.shape
        y = y.reshape(B, NB, 8, Tq).permute(0, 1, 3, 2).reshape(B, NB, Tq * 8)
        return y[..., :T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v15_spec")
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--code", type=int, default=32)
    ap.add_argument("--aug_p", type=float, default=0.7)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device("cuda")
    out = ROOT / "outputs" / "ae_spec" / args.name / args.machine
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  [ae-spec {args.name}/{args.machine}] {msg}"
        print(line, flush=True)
        with open(ROOT / "outputs" / "live.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"=== {vars(args)}")
    for split in ("train", "test"):
        build_cache("data", "cache", args.machine, split)
    tr, te = WavStore("cache", args.machine, "train"), WavStore("cache", args.machine, "test")
    fit_idx = list(range(len(tr)))
    lab, dom = np.array(te.meta["label"]), np.array(te.meta["domain"])
    spec = Spec(dev)

    @torch.no_grad()
    def feats(store, ids):
        X, Y = [], []
        for i in range(0, len(ids), 32):
            x, y = spec(store.get(ids[i : i + 32]).to(dev))
            X.append(x.half().cpu())
            Y.append(y.half().cpu())
        return torch.cat(X), torch.cat(Y)

    Xf, Yf = feats(tr, fit_idx)
    Xt, Yt = feats(te, list(range(len(te))))
    xm, xs = Xf.float().mean(dim=(0, 3), keepdim=True)[0], Xf.float().std(dim=(0, 3), keepdim=True)[0] + 1e-3
    ym, ys = Yf.float().mean(dim=(0, 2))[:, None], Yf.float().std(dim=(0, 2))[:, None] + 1e-3
    nx = lambda X: ((X.float() - xm.to(X.device)) / xs.to(X.device))
    ny = lambda Y: ((Y.float() - ym.to(Y.device)) / ys.to(Y.device))
    model = AE(args.code).to(dev)
    log(f"input {tuple(Xf.shape[1:])}, target {tuple(Yf.shape[1:])}; parameters {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)

    @torch.no_grad()
    def errors(X, Y):
        model.eval()
        E = [(model(nx(X[i : i + 32].to(dev))) - ny(Y[i : i + 32].to(dev))).pow(2).mean(1).cpu() for i in range(0, len(X), 32)]
        model.train()
        return torch.cat(E)  # (N, T) per-frame error

    def evaluate(step):
        et, ef = errors(Xt, Yt), errors(Xf[:200], Yf[:200])
        k = max(1, et.shape[1] // 10)
        sc = {"mse-mean": et.mean(1), "mse-topframes": et.topk(k, dim=1).values.mean(1)}
        res = {n: compute_metrics(s.numpy(), lab, dom) for n, s in sc.items()}
        log(f"step {step:5d} | train err {ef.mean():.3f} test normal {et[lab == 0].mean():.3f} anomaly {et[lab == 1].mean():.3f} | "
            + " | ".join(f"{n} AUC {r['auc_all']:.3f} src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} pAUC {r['pauc']:.3f} "
                         f"off {r['official']:.3f}" for n, r in res.items()))
        return res

    losses, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        ids = rng.sample(fit_idx, args.batch)
        w = tr.get(ids).to(dev)
        with torch.no_grad():
            _, y = spec(w)
            wa = w.clone()
            for b in range(len(ids)):
                if rng.random() < args.aug_p:
                    other = tr.get([rng.choice(fit_idx)])[0, 1].to(dev)
                    other = torch.roll(other, rng.randrange(len(other)))
                    g_far = wa[b, 1].pow(2).mean().sqrt() / (other.pow(2).mean().sqrt() + 1e-9) * 10 ** (-rng.uniform(0, 15) / 20)
                    g_near = g_far * 10 ** (-rng.uniform(3, 10) / 20)
                    wa[b, 1] += g_far * other
                    wa[b, 0] += g_near * other
            x, _ = spec(wa)
        loss = (model(nx(x)) - ny(y)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 100 == 0:
            log(f"step {step:5d} loss {np.mean(losses[-100:]):.4f} {(time.time() - t0) / step:.2f}s/step "
                f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if step % args.eval_every == 0 and step != args.steps:
            evaluate(step)
    res = evaluate(args.steps)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    torch.save(model.state_dict(), out / "model.pt")
    log("=== done (FINAL is the last evaluation line)")


if __name__ == "__main__":
    main()
