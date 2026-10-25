"""Partial-diffusion reconstruction for anomaly detection (normal clips only, same settings for every machine).

Train an unconditional diffusion model (DDPM, epsilon prediction, cosine schedule, small UNet) on the machine's
normal near-mic log-mel crops (128 mel x 256 frames = 2.56 s, each mel bin standardised with training statistics).
Test: for a clip, noise it to level t0 (fixed in advance: 100, 250, 500 of 1000), denoise back with DDIM
(20-step grid, deterministic, fixed noise seed per clip). The model knows only normal sound, so the result is the
nearest normal-looking version of the clip at that noise level. Full clip = 4 non-overlapping crops.
Scores per t0:
  mse     mean squared difference between clip and its reconstruction
  steady  signed difference averaged over time per mel bin (texture cancels, steady offsets add up), squared, summed
'all t'   z-scores (on 100 held-out normal clips) summed over the three noise levels; reported at the last step.
Usage: uv run python -u diffusion_ad.py --machine ToyCar
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel

from clora_ad import fbank
from dcase_ae.efficientad import WavStore, build_cache
from dcase_ae.evaluation import compute_metrics

CROP, NCROP = 256, 4
T0S = (100, 250, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval_every", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    out = Path("outputs/diffusion_ad") / args.name / args.machine
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')}  [diffusion {args.name}/{args.machine}] {msg}", flush=True)

    log(f"=== {vars(args)}")
    for split in ("train", "test"):
        build_cache("data", "cache", args.machine, split)
    tr, te = WavStore("cache", args.machine, "train"), WavStore("cache", args.machine, "test")
    idx = list(range(len(tr)))
    random.Random(0).shuffle(idx)
    val_idx, fit_idx = idx[:100], idx[100:]
    lab, dom = np.array(te.meta["label"]), np.array(te.meta["domain"])

    @torch.no_grad()
    def logmels(store, ids):
        out = []
        for i in range(0, len(ids), 32):
            out += [fbank(w).T.cpu() for w in store.get(ids[i : i + 32])[:, 0].to(dev).float()]  # (128, 1024)
        return torch.stack(out)

    t0 = time.time()
    Xfit, Xval, Xte = logmels(tr, fit_idx), logmels(tr, val_idx), logmels(te, list(range(len(te))))
    mu, sd = Xfit.mean(dim=(0, 2), keepdim=True)[0], Xfit.std(dim=(0, 2), keepdim=True)[0] + 1e-4  # per mel bin
    norm = lambda X: ((X - mu) / sd).float()
    Xfit, Xval, Xte = norm(Xfit).to(dev), norm(Xval), norm(Xte)
    log(f"log-mels ready in {time.time() - t0:.0f}s: fit {tuple(Xfit.shape)}")

    unet = UNet2DModel(sample_size=(128, CROP), in_channels=1, out_channels=1, layers_per_block=2,
                       block_out_channels=(64, 128, 256, 256),
                       down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                       up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")).to(dev)
    ema = copy.deepcopy(unet).eval()
    unet.enable_gradient_checkpointing()
    for p in ema.parameters():
        p.requires_grad_(False)
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    ddim = DDIMScheduler.from_config(sched.config)
    ddim.set_timesteps(20)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=0.0)
    log(f"UNet parameters {sum(p.numel() for p in unet.parameters()):,}")

    @torch.no_grad()
    def reconstruct(x0, t0, gen):
        """x0 (B,1,128,CROP) -> DDIM reconstruction from noise level t0."""
        ts = [int(t) for t in ddim.timesteps if t <= t0]
        noise = torch.randn(x0.shape, generator=gen, device=dev)
        x = sched.add_noise(x0, noise, torch.full((len(x0),), ts[0], device=dev, dtype=torch.long))
        for t in ts:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                eps = ema(x, t).sample.float()
            x = ddim.step(eps, t, x).prev_sample
        return x

    @torch.no_grad()
    def scores(X):
        """X (N,128,1024) -> {(t0, kind): (N,)}"""
        res = {(t0, k): np.zeros(len(X)) for t0 in T0S for k in ("mse", "steady")}
        crops = torch.stack([X[:, :, i * CROP : (i + 1) * CROP] for i in range(NCROP)], 1)  # (N, NCROP, 128, CROP)
        for i in range(0, len(X), 16):
            xb = crops[i : i + 16].reshape(-1, 1, 128, CROP).to(dev)
            for t0 in T0S:
                gen = torch.Generator(device=dev).manual_seed(1000 * t0 + i)
                r = reconstruct(xb, t0, gen)
                d = (xb - r).reshape(-1, NCROP, 128, CROP)
                res[(t0, "mse")][i : i + len(d)] = d.pow(2).mean(dim=(1, 2, 3)).cpu().numpy()
                res[(t0, "steady")][i : i + len(d)] = d.mean(dim=(1, 3)).pow(2).sum(-1).cpu().numpy()
        return res

    def evaluate(step, final=False):
        t1 = time.time()
        sv, st = scores(Xval), scores(Xte)
        out = {}
        for k in ("mse", "steady"):
            for t0 in T0S:
                out[f"{k} t{t0}"] = compute_metrics(st[(t0, k)], lab, dom)
            z = sum((st[(t0, k)] - sv[(t0, k)].mean()) / (sv[(t0, k)].std() + 1e-9) for t0 in T0S)
            out[f"{k} all-t"] = compute_metrics(z, lab, dom)
        log(f"step {step:5d} ({time.time() - t1:.0f}s){'  (FINAL)' if final else ''}")
        for k in ("mse", "steady"):
            log("    " + " | ".join(f"{name} AUC {r['auc_all']:.3f} off {r['official']:.3f}" for name, r in out.items() if name.startswith(k)))
        return out

    rng = random.Random(args.seed + 1)
    hist, losses, t0 = {}, [], time.time()
    ck = out / "resume.pt"
    start = 1
    if ck.exists():  # resume after a crash
        s = torch.load(ck, weights_only=False)  # our own checkpoint
        unet.load_state_dict(s["unet"]), ema.load_state_dict(s["ema"]), opt.load_state_dict(s["opt"])
        rng.setstate(s["rng"]), torch.cuda.set_rng_state(s["cuda_rng"])
        start, hist = s["step"] + 1, s["hist"]
        log(f"resumed from step {s['step']}")
    for step in range(start, args.steps + 1):
        ids = torch.tensor([rng.randrange(len(Xfit)) for _ in range(args.batch)], device=dev)
        offs = [rng.randrange(0, 1024 - CROP + 1) for _ in range(args.batch)]
        x0 = torch.stack([Xfit[i, :, o : o + CROP] for i, o in zip(ids.tolist(), offs)])[:, None]
        noise = torch.randn_like(x0)
        t = torch.randint(0, 1000, (args.batch,), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(sched.add_noise(x0, noise, t), t).sample
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), unet.parameters()):
                pe.lerp_(p, 1 - 0.999)
        losses.append(loss.item())
        if step % 100 == 0:
            log(f"step {step:5d} loss {np.mean(losses[-100:]):.4f} {(time.time() - t0) / step:.2f}s/step "
                f"GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if step % args.eval_every == 0 and step != args.steps:
            hist[f"step{step}"] = evaluate(step)
        if step % 500 == 0:
            torch.save({"unet": unet.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                        "rng": rng.getstate(), "cuda_rng": torch.cuda.get_rng_state(), "hist": hist}, ck)
    hist["final"] = evaluate(args.steps, final=True)
    (out / "summary.json").write_text(json.dumps(hist, indent=1))
    torch.save(ema.state_dict(), out / "ema.pt")
    log("=== done")


if __name__ == "__main__":
    main()
