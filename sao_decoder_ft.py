"""Fine-tune Stable Audio Open's VAE decoder on our recordings (encoder frozen, so the latent space the diffusion model
knows stays exactly the same).

Data: one machine's normal training clips, near mic = left, far mic = right, 44.1 kHz; latents from
sao_target.py --stage cache. 900 clips for fine-tuning, 100 held out for the round-trip check.
Training on random 64-frame latent crops (~3 s of audio):
  loss = multi-resolution STFT loss per channel at 44.1 kHz (log-magnitude L1, FFT 512/1024/2048)
       + band-level loss: |difference| of time-averaged band levels at 16 kHz (dB, 256 bands) for near, far and
         near minus far, i.e. exactly what the realism check and the autoencoder use
Check every --eval_every steps on 16 held-out clips (full 10 s): time-averaged band level error, dB, per spectrogram.
Saves the decoder to cache/sao_decoder_<machine>.pt.
Usage: uv run python -u sao_decoder_ft.py --machine ToyCar
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dcase_ae.efficientad import WavStore
from diffusion_target import three_channel
from sao_target import load, to_16k, to_44k

CROP = 32  # latent frames per training crop (~1.5 s)


def mrstft(a, b):
    """multi-resolution log-magnitude L1 between waveforms (B, 2, L)"""
    loss = 0
    for n in (512, 1024, 2048):
        win = torch.hann_window(n, device=a.device)
        A = torch.stft(a.reshape(-1, a.shape[-1]), n, n // 4, window=win, return_complex=True).abs()
        B = torch.stft(b.reshape(-1, b.shape[-1]), n, n // 4, window=win, return_complex=True).abs()
        loss = loss + (torch.log(A + 1e-5) - torch.log(B + 1e-5)).abs().mean()
    return loss / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--band_w", type=float, default=0.1, help="weight of the band-level loss (per dB)")
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device("cuda")

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')}  [sao-decoder {args.machine}] {msg}", flush=True)

    log(f"=== {vars(args)}")
    pipe = load(dev)
    del pipe.transformer, pipe.text_encoder
    torch.cuda.empty_cache()
    vae = pipe.vae
    tr = WavStore("cache", args.machine, "train")
    Z = torch.load(f"cache/sao_latents_{args.machine}.pt").float()
    idx = list(range(len(tr)))
    random.Random(0).shuffle(idx)
    held, fit = idx[:100], idx[100:]
    win16 = torch.hann_window(1024, device=dev)
    hop = 2048

    def band_levels(w16):  # (B, 2, L) 16 kHz -> (B, 3, 256) time-averaged dB
        return three_channel(w16, win16).mean(-1)

    @torch.no_grad()
    def check(step):
        vae.eval()
        ids = held[:16]
        errs = []
        for i in range(0, len(ids), 4):
            b = ids[i : i + 4]
            rec = to_16k(vae.decode(Z[b].to(dev)).sample)
            errs.append((band_levels(rec) - band_levels(tr.get(b).to(dev))).abs().mean(dim=(0, 2)).cpu())
        e = torch.stack(errs).mean(0)
        log(f"step {step:5d} CHECK round-trip band-level error on held-out clips: near {e[0]:.2f} dB, far {e[1]:.2f} dB, "
            f"near-minus-far {e[2]:.2f} dB")
        vae.train()
        return e

    from torch.utils.checkpoint import checkpoint
    for blk in vae.decoder.block:  # recompute each upsampling block in the backward pass instead of storing it
        f = blk.forward
        blk.forward = (lambda f: (lambda x: checkpoint(f, x, use_reentrant=False) if torch.is_grad_enabled() else f(x)))(f)
    for p in vae.parameters():
        p.requires_grad_(False)
    params = list(vae.decoder.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    log(f"decoder parameters {sum(p.numel() for p in params):,} (encoder frozen)")
    check(0)
    vae.train()
    losses, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        ids = rng.sample(fit, args.batch)
        offs = [rng.randrange(0, Z.shape[-1] - CROP + 1) for _ in ids]
        z = torch.stack([Z[i, :, o : o + CROP] for i, o in zip(ids, offs)]).to(dev)
        with torch.no_grad():
            full = to_44k(tr.get(ids).to(dev))
            ref = torch.stack([full[k, :, o * hop : (o + CROP) * hop] for k, o in enumerate(offs)])
        rec = vae.decode(z).sample[..., : ref.shape[-1]]
        l_stft = mrstft(rec, ref)
        l_band = (band_levels(to_16k_any(rec)) - band_levels(to_16k_any(ref))).abs().mean()
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
    torch.save(vae.decoder.state_dict(), Path(f"cache/sao_decoder_{args.machine}.pt"))
    log(f"saved decoder to cache/sao_decoder_{args.machine}.pt")


def to_16k_any(w):
    import torchaudio
    return torchaudio.functional.resample(w, 44100, 16000)


if __name__ == "__main__":
    main()
