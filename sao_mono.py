"""Stable Audio Open on the machine's own sound only (one channel), so no cross-mic relationship has to be preserved.

Machine sound = near mic minus the part the far mic predicts (per frequency, per clip; tools/simple_baselines.py
near_only_audio), 16 kHz -> 44.1 kHz, the same signal on both of the model's stereo channels.
Stages:
  cache    encode all training clips' machine sound -> cache/sao_nmf_latents_<machine>.pt;
           control: round-trip (encode -> decode) band-level error, dB
  decoder  fine-tune the VAE decoder (encoder frozen, latent space unchanged) on the machine sound;
           loss = multi-resolution STFT log-magnitude L1 + band-level L1 (dB, weight --band_w);
           CHECK every --eval_every steps: round-trip band-level error on 16 held-out clips;
           saved to cache/sao_nmf_decoder_<machine>.pt
Band levels: 16 kHz, STFT 1024 / hop 512, adjacent bins paired -> 256 bands, power averaged over the clip, dB.
Usage: uv run python -u sao_mono.py --machine ToyCar --stage cache
       uv run python -u sao_mono.py --machine ToyCar --stage decoder
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.checkpoint import checkpoint

from dcase_ae.efficientad import WavStore, build_cache
from sao_decoder_ft import mrstft
from sao_target import WAV_LEN, load
from tools.simple_baselines import near_only_audio

CROP = 32


def up(w16):  # (B, L) 16 kHz mono -> (B, 2, WAV_LEN) 44.1 kHz, same signal on both channels
    w = torchaudio.functional.resample(w16, 16000, 44100)
    w = torch.nn.functional.pad(w, (0, max(0, WAV_LEN - w.shape[-1])))[..., :WAV_LEN]
    return w[:, None].repeat(1, 2, 1)


def down(w44):  # (B, 2, L) -> (B, L) 16 kHz mono (average of the two channels)
    return torchaudio.functional.resample(w44.mean(1), 44100, 16000)


def band_levels(w16):  # (B, L) -> (B, 256) dB, power averaged over the clip
    win = torch.hann_window(1024, device=w16.device)
    S = torch.stft(w16, 1024, 512, window=win, return_complex=True)[:, 1:]
    P = S.abs().pow(2).reshape(len(w16), 256, 2, -1).sum(2).mean(-1)
    return 10 * torch.log10(P + 1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--stage", choices=["cache", "decoder"], required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--band_w", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device("cuda")

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')}  [sao-mono {args.machine}] {msg}", flush=True)

    log(f"=== {vars(args)}")
    build_cache("data", "cache", args.machine, "train")
    tr = WavStore("cache", args.machine, "train")
    pipe = load(dev)
    del pipe.transformer, pipe.text_encoder
    torch.cuda.empty_cache()
    vae = pipe.vae
    lat_path = Path(f"cache/sao_nmf_latents_{args.machine}.pt")

    def machine_sound(ids):  # (B, L) 16 kHz near-minus-far audio of training clips
        return near_only_audio(tr.get(ids), dev)

    idx = list(range(len(tr)))
    random.Random(0).shuffle(idx)
    held, fit = idx[:100], idx[100:]

    if args.stage == "cache":
        t0 = time.time()
        with torch.no_grad():
            Z = torch.cat([vae.encode(up(machine_sound(list(range(i, min(i + 8, len(tr))))))).latent_dist.mean.cpu().half()
                           for i in range(0, len(tr), 8)])
        torch.save(Z, lat_path)
        log(f"machine-sound latents {tuple(Z.shape)} saved to {lat_path} in {time.time() - t0:.0f}s")
        with torch.no_grad():
            ids = held[:16]
            rec = down(vae.decode(Z[ids].float().to(dev)).sample)[:, :160000]
            e = (band_levels(rec) - band_levels(machine_sound(ids))).abs().mean()
        log(f"CHECK original decoder round-trip band-level error on held-out clips: {e:.2f} dB")
        return

    Z = torch.load(lat_path).float()
    for blk in vae.decoder.block:
        f = blk.forward
        blk.forward = (lambda f: (lambda x: checkpoint(f, x, use_reentrant=False) if torch.is_grad_enabled() else f(x)))(f)
    for p in vae.parameters():
        p.requires_grad_(False)
    params = list(vae.decoder.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    ref_all = {}

    @torch.no_grad()
    def check(step):
        vae.eval()
        ids = held[:16]
        errs = []
        for i in range(0, 16, 4):
            b = ids[i : i + 4]
            rec = down(vae.decode(Z[b].to(dev)).sample)[:, :160000]
            errs.append((band_levels(rec) - band_levels(machine_sound(b))).abs().mean().item())
        log(f"step {step:5d} CHECK round-trip band-level error on held-out clips: {np.mean(errs):.2f} dB")
        vae.train()

    log(f"decoder parameters {sum(p.numel() for p in params):,} (encoder frozen)")
    check(0)
    vae.train()
    losses, t0 = [], time.time()
    hop = 2048
    for step in range(1, args.steps + 1):
        ids = rng.sample(fit, args.batch)
        offs = [rng.randrange(0, Z.shape[-1] - CROP + 1) for _ in ids]
        z = torch.stack([Z[i, :, o : o + CROP] for i, o in zip(ids, offs)]).to(dev)
        with torch.no_grad():
            full = up(machine_sound(ids))
            ref = torch.stack([full[k, :, o * hop : (o + CROP) * hop] for k, o in enumerate(offs)])
        rec = vae.decode(z).sample[..., : ref.shape[-1]]
        l_stft = mrstft(rec, ref)
        l_band = (band_levels(down(rec)) - band_levels(down(ref))).abs().mean()
        loss = l_stft + args.band_w * l_band
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append((l_stft.item(), l_band.item()))
        if step % 50 == 0:
            a = np.mean(losses[-50:], 0)
            log(f"step {step:5d} loss: spectrogram {a[0]:.4f}, band level {a[1]:.2f} dB | {(time.time() - t0) / step:.2f}s/step "
                f"GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if step % args.eval_every == 0:
            check(step)
    out = Path(f"cache/sao_nmf_decoder_{args.machine}.pt")
    torch.save(vae.decoder.state_dict(), out)
    log(f"saved decoder to {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
