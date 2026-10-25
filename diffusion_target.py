"""Synthetic target-domain samples from a domain-conditioned diffusion model (per machine, normal clips only).

Representation (what the noise-aware autoencoder uses, all magnitudes, so no phase is needed):
  3 aligned log-power spectrograms, 256 bands (STFT 1024, hop 512, adjacent bins paired), 313 frames:
    ch0 near mic, ch1 far mic, ch2 near minus far per frame: |N - H F|^2 with H = mean(N F*) / mean(|F|^2) per bin
  (the time-average of ch2 is exactly the autoencoder's target, Pn - |X|^2 / Pf)
Model: DDPM (cosine schedule, epsilon prediction), small UNet with a domain label (0 source, 1 target) as class
condition, label dropped 15% of the time (classifier-free guidance). Trained on random 128-frame crops of all 1000
normal training clips, standardised per channel and band.
Generation: start from a random 128-frame crop of one of the 10 real target clips, noise it to a random level
t0 ~ U(200, 600) of 1000, denoise with DDIM (25-step grid) as 'target domain' (guidance 2). Saved in dB.
Check: per-band average of synthetic vs real target vs real source (synthetic should sit with real target).
Usage: uv run python -u diffusion_target.py --machine ToyCar
"""
import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel

from dcase_ae.efficientad import WavStore, build_cache

NFFT, HOP, NB, CROP = 1024, 512, 256, 128


@torch.no_grad()
def three_channel(w, win):
    """w (B, 2, L) -> (B, 3, NB, T) dB: near, far, near minus far per frame."""
    B = len(w)
    S = torch.stft(w.reshape(2 * B, -1), NFFT, HOP, window=win, return_complex=True)[:, 1:].reshape(B, 2, 512, -1)
    N, Fr = S[:, 0], S[:, 1]
    H = (N * Fr.conj()).mean(-1, keepdim=True) / (Fr.abs().pow(2).mean(-1, keepdim=True) + 1e-12)
    P = torch.stack([N.abs().pow(2), Fr.abs().pow(2), (N - H * Fr).abs().pow(2)], 1)  # (B, 3, 512, T)
    P = P.reshape(B, 3, NB, 2, -1).sum(3)
    return 10 * torch.log10(P + 1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_synth", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=200, help="quick target-likeness check every N steps (16 samples)")
    ap.add_argument("--load", action="store_true", help="skip training, load the saved model")
    ap.add_argument("--gen", default="100-300:4,200-400:3",
                    help="generation settings 'tmin-tmax:guidance' (comma-separated); the first one is saved for training")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device("cuda")

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')}  [diffusion-target {args.machine}] {msg}", flush=True)

    log(f"=== {vars(args)}")
    build_cache("data", "cache", args.machine, "train")
    tr = WavStore("cache", args.machine, "train")
    dom = np.array(tr.meta["domain"])
    win = torch.hann_window(NFFT, device=dev)
    t0 = time.time()
    X = torch.cat([three_channel(tr.get(list(range(i, min(i + 32, len(tr))))).to(dev), win).half().cpu()
                   for i in range(0, len(tr), 32)])  # (1000, 3, NB, 313) dB
    mu, sd = X.float().mean(dim=(0, 3), keepdim=True)[0], X.float().std(dim=(0, 3), keepdim=True)[0] + 1e-3  # (3, NB, 1)
    Xn = ((X.float() - mu) / sd).half()
    T = X.shape[-1]
    tgt_ids, src_ids = np.flatnonzero(dom == 1).tolist(), np.flatnonzero(dom == 0).tolist()
    log(f"3-channel spectrograms of {len(X)} clips ({len(src_ids)} source, {len(tgt_ids)} target) in {time.time() - t0:.0f}s")

    unet = UNet2DModel(sample_size=(NB, CROP), in_channels=3, out_channels=3, layers_per_block=2,
                       block_out_channels=(32, 64, 128, 128), num_class_embeds=3,  # 0 source, 1 target, 2 = no label
                       down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                       up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")).to(dev)
    ema = copy.deepcopy(unet).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=0.0)
    log(f"UNet parameters {sum(p.numel() for p in unet.parameters()):,}")

    def crops(ids):
        out = []
        for i in ids:
            o = rng.randrange(0, T - CROP + 1)
            out.append(Xn[i, :, :, o : o + CROP])
        return torch.stack(out).float().to(dev)

    ddim = DDIMScheduler.from_config(sched.config)
    ddim.set_timesteps(25)
    mpath = Path(f"cache/diffusion_target_{args.machine}.pt")
    if args.load:
        ema.load_state_dict(torch.load(mpath))
        log(f"loaded {mpath}")
    losses, t1 = [], time.time()
    for step in range(1, 0 if args.load else args.steps + 1):
        # every batch: 3/4 random clips, 1/4 target clips, so the target label is learned from more than a handful of steps
        n_t = args.batch // 4
        ids = rng.sample(src_ids, args.batch - n_t) + [rng.choice(tgt_ids) for _ in range(n_t)]
        x0 = crops(ids)
        y = torch.tensor(dom[ids], device=dev, dtype=torch.long)
        y = torch.where(torch.rand(len(y), device=dev) < 0.15, torch.full_like(y, 2), y)  # label dropout
        noise = torch.randn_like(x0)
        t = torch.randint(0, 1000, (len(x0),), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(sched.add_noise(x0, noise, t), t, class_labels=y).sample
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            decay = min(0.999, (1 + step) / (10 + step))  # EMA warm-up, so early checks use a meaningful average
            for pe, p in zip(ema.parameters(), unet.parameters()):
                pe.lerp_(p, 1 - decay)
        losses.append(loss.item())
        if step % 200 == 0:
            log(f"step {step:5d} loss {np.mean(losses[-200:]):.4f} {(time.time() - t1) / step:.2f}s/step")
        if step % args.eval_every == 0:
            tmin, tmax, g = parse_gen(args.gen.split(",")[0])
            S, X0 = generate(ddim, sched, ema, crops, rng, tgt_ids, dev, mu, sd, 16, tmin, tmax, g)
            tl, ch = target_likeness(S, X0, X, tgt_ids, src_ids)
            log(f"step {step:5d} CHECK target-likeness {tl:+.2f} (+1 = like real target, -1 = like source) | "
                f"changed {ch:.2f} dB from its starting clip")

    if not args.load:
        torch.save(ema.state_dict(), mpath)
        log(f"saved model to {mpath}")

    # ---- generation: partial noising of real target crops, denoised as 'target domain' ------------------------
    for gi, setting in enumerate(args.gen.split(",")):
        tmin, tmax, g = parse_gen(setting)
        S, X0 = generate(ddim, sched, ema, crops, rng, tgt_ids, dev, mu, sd, args.n_synth, tmin, tmax, g)
        tl, ch = target_likeness(S, X0, X, tgt_ids, src_ids)
        log(f"--- generation: noise level {tmin}-{tmax} of 1000, guidance {g}: target-likeness {tl:+.2f}, "
            f"changed {ch:.2f} dB from its starting clip")
        band = lambda Z: Z.float().mean(-1)
        real_t, real_s, syn = band(X[tgt_ids]).mean(0), band(X[src_ids]).mean(0), band(S).mean(0)
        for c, name in enumerate(("near", "far", "near-minus-far")):
            log(f"    {name:15s}: real target vs source {(real_t[c] - real_s[c]).abs().mean():.2f} dB | synthetic vs real "
                f"target {(syn[c] - real_t[c]).abs().mean():.2f} dB | synthetic vs real source {(syn[c] - real_s[c]).abs().mean():.2f} dB")
        if gi == 0:
            path = Path(f"cache/synth_target_{args.machine}.pt")
            torch.save({"spec_db": S.half(), "info": f"near, far, near-minus-far log-power dB, 256 bands x 128 frames; "
                        f"noise {tmin}-{tmax}, guidance {g}"}, path)
            log(f"saved {len(S)} synthetic target samples to {path}")


def parse_gen(setting):
    rng_tag, g_str = setting.split(":")
    tmin, tmax = map(int, rng_tag.split("-"))
    return tmin, tmax, float(g_str)


def target_likeness(S, X0, X, tgt_ids, src_ids):
    """+1: synthetic per-band averages match real target clips; -1: they match source clips (mean over 3 spectrograms).
    Also: mean |difference| (dB) between each synthetic sample and the real target crop it started from."""
    band = lambda Z: Z.float().mean(-1)
    real_t, real_s, syn = band(X[tgt_ids]).mean(0), band(X[src_ids]).mean(0), band(S).mean(0)
    sc = []
    for c in range(3):
        d_ts = (real_t[c] - real_s[c]).abs().mean()
        sc.append((((syn[c] - real_s[c]).abs().mean() - (syn[c] - real_t[c]).abs().mean()) / d_ts).item())
    return float(np.mean(sc)), (S.float() - X0.float()).abs().mean().item()


def generate(ddim, sched, ema, crops, rng, tgt_ids, dev, mu, sd, n_total, tmin, tmax, guidance):
    """returns synthetic samples and the real target crops they started from, both in dB (N, 3, NB, CROP)"""
    out, starts = [], []
    with torch.no_grad():
        for i in range(0, n_total, 32):
            n = min(32, n_total - i)
            x0 = crops([rng.choice(tgt_ids) for _ in range(n)])
            starts.append(x0.cpu())
            t0s = [rng.randint(tmin, tmax) for _ in range(n)]
            xs = []
            for b in range(n):  # each sample from its own noise level
                ts = [int(t) for t in ddim.timesteps if t <= t0s[b]]
                x = sched.add_noise(x0[b : b + 1], torch.randn_like(x0[b : b + 1]), torch.tensor([ts[0]], device=dev))
                for tt in ts:
                    xin = torch.cat([x, x])
                    lab = torch.tensor([1, 2], device=dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        e = ema(xin, tt, class_labels=lab).sample.float()
                    e = e[1:] + guidance * (e[:1] - e[1:])  # classifier-free guidance toward 'target'
                    x = ddim.step(e, tt, x).prev_sample
                xs.append(x)
            out.append(torch.cat(xs).cpu())
    return torch.cat(out) * sd + mu, torch.cat(starts) * sd + mu


if __name__ == "__main__":
    main()
