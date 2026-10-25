"""Target-domain samples from a pretrained stereo diffusion model (Stable Audio Open 1.0), personalised on the 10
target-domain training clips, used as a reference-based style transfer of source clips.

Stereo layout: near mic = left, far mic = right (16 kHz -> 44.1 kHz). Clips are 10 s = 215 latent frames
(VAE hop 2048), duration conditioning 0-10 s.
Two personalisation methods (model weights frozen unless stated):
  ti  textual inversion: one new word '<tgt>' (T5 input embedding, initialised from 'unusual') learned on the 10 target
      clips with prompts '<machine description>, <tgt>'
  db  DreamBooth with LoRA (rank 16 on the DiT attention): target clips with '..., recorded in sks condition', plus
      source clips with the same prompts without the phrase (prior preservation, half of every batch)
Prompts: several detailed descriptions per machine (to use what the model already knows), drawn at random.
Training objective = Stable Audio Open's own: v-prediction on the cosine schedule (x_t = cos(pi t/2) x + sin(pi t/2) e,
target cos e - sin x, timestep input t), 10% of steps with empty conditioning (keeps guidance working).
Editing (style transfer): a source clip is encoded, noised to sigma_start (fixed in advance: 0.8 and 2.0), and
denoised with the personalised prompt (guidance 4). Control: the same edit with the prompt WITHOUT the target word.
Check: target-likeness of the edits (+1 like real target, -1 like source, from 16 kHz near/far/near-minus-far band
levels), and how much each edit changed its source clip (dB). Edits with the target word are saved as
cache/sao_target_<machine>_<method>.pt (same format as diffusion_target.py, usable by ae_noiseaware.py --synth).
Usage:
  uv run python -u sao_target.py --machine ToyCar --stage cache
  uv run python -u sao_target.py --machine ToyCar --stage train --method ti
  uv run python -u sao_target.py --machine ToyCar --stage edit --method ti
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from dcase_ae.efficientad import WavStore, build_cache
from diffusion_target import three_channel, target_likeness

REPO = "stabilityai/stable-audio-open-1.0"
SR_IN, SR = 16000, 44100
FRAMES = 215
WAV_LEN = FRAMES * 2048
DESCR = {
    "ToyCar": [
        "close-up stereo recording of a small battery-powered toy car running on a roller, whirring electric motor and plastic gears",
        "a miniature toy car with a buzzing DC motor and meshing gears, recorded at close range in a factory hall",
        "high-pitched whine of a toy car motor and spinning wheels on a test rig, mechanical, steady",
        "field recording of a little electric toy vehicle, motor hum, gearbox rattle and rolling tires",
    ],
    "fan": [
        "close-up stereo recording of an industrial ventilation fan, steady whoosh of air and spinning blades",
        "a large factory exhaust fan running at constant speed, rumbling motor and broadband air noise",
        "humming electric fan motor with rotating blades moving air, mechanical, steady",
        "field recording of an industrial blower fan in a factory, airflow noise and motor hum",
    ],
}
TOKEN, DB_PHRASE = "<tgt>", "recorded in sks condition"


def load(dev):
    from diffusers import StableAudioPipeline
    pipe = StableAudioPipeline.from_pretrained(REPO, torch_dtype=torch.float32)
    pipe.vae.to(dev).eval()
    pipe.text_encoder.to(dev).eval()
    pipe.projection_model.to(dev).eval()
    pipe.transformer.to(dev, dtype=torch.bfloat16).eval()
    for m in (pipe.vae, pipe.text_encoder, pipe.projection_model, pipe.transformer):
        for p in m.parameters():
            p.requires_grad_(False)
    return pipe


def conditioning(pipe, prompts, dev, empty=None):
    """prompts -> (encoder_hidden_states, global_hidden_states); rows where empty[i] are zeroed (unconditional)."""
    tok = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True,
                         return_tensors="pt")
    ids, mask = tok.input_ids.to(dev), tok.attention_mask.to(dev)
    h = pipe.text_encoder(ids, attention_mask=mask)[0]
    h = pipe.projection_model(text_hidden_states=h).text_hidden_states * mask[..., None].to(h.dtype)
    n = len(prompts)
    pr = pipe.projection_model(start_seconds=torch.zeros(n, device=dev), end_seconds=torch.full((n,), 10.0, device=dev))
    ss, se = pr.seconds_start_hidden_states, pr.seconds_end_hidden_states
    enc = torch.cat([h, ss, se], dim=1)
    glob = torch.cat([ss, se], dim=2)
    if empty is not None:
        enc = torch.where(empty[:, None, None], torch.zeros_like(enc), enc)
    return enc, glob


def rotary(pipe, dev):
    from diffusers.models.embeddings import get_1d_rotary_pos_embed
    r = get_1d_rotary_pos_embed(pipe.transformer.config.attention_head_dim // 2, FRAMES + 1, use_real=True,
                                repeat_interleave_real=False)
    return tuple(x.to(dev) for x in r)


def to_44k(w):  # (B, 2, L) 16 kHz near/far -> (B, 2, WAV_LEN) 44.1 kHz, left = near, right = far
    w = torchaudio.functional.resample(w, SR_IN, SR)
    return F.pad(w, (0, max(0, WAV_LEN - w.shape[-1])))[..., :WAV_LEN]


def to_16k(w):
    return torchaudio.functional.resample(w, SR, SR_IN)[..., :160000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--stage", choices=["cache", "train", "edit"], required=True)
    ap.add_argument("--method", choices=["ti", "db"], default="ti")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_edit", type=int, default=200)
    ap.add_argument("--sigmas", default="0.8,2.0", help="edit strengths (start noise level), fixed in advance")
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device("cuda")

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')}  [sao-{args.method} {args.machine}] {msg}", flush=True)

    log(f"=== {vars(args)}")
    build_cache("data", "cache", args.machine, "train")
    tr = WavStore("cache", args.machine, "train")
    dom = np.array(tr.meta["domain"])
    src_ids, tgt_ids = np.flatnonzero(dom == 0).tolist(), np.flatnonzero(dom == 1).tolist()
    pipe = load(dev)
    lat_path = Path(f"cache/sao_latents_{args.machine}.pt")
    win = torch.hann_window(1024, device=dev)

    if args.stage == "cache":
        t0 = time.time()
        Z = []
        with torch.no_grad():
            for i in range(0, len(tr), 8):
                w = to_44k(tr.get(list(range(i, min(i + 8, len(tr))))).to(dev))
                Z.append(pipe.vae.encode(w).latent_dist.mean.cpu().half())
        Z = torch.cat(Z)
        torch.save(Z, lat_path)
        log(f"latents {tuple(Z.shape)} saved to {lat_path} in {time.time() - t0:.0f}s, std {Z.float().std():.2f}")
        # control: what the model's own compressor does to a clip (encode -> decode, no diffusion)
        with torch.no_grad():
            ids = src_ids[:16]
            orig = tr.get(ids).to(dev)
            rec = to_16k(pipe.vae.decode(Z[ids].float().to(dev)).sample)
            a, b = three_channel(orig, win), three_channel(rec, win)
        log(f"compressor round trip (encode -> decode) changes the 16 kHz spectrograms by {(a - b).abs().mean():.2f} dB "
            f"(near {(a[:, 0] - b[:, 0]).abs().mean():.2f}, far {(a[:, 1] - b[:, 1]).abs().mean():.2f}, "
            f"near-minus-far {(a[:, 2] - b[:, 2]).abs().mean():.2f})")
        return

    Z = torch.load(lat_path).float()
    descr = DESCR[args.machine]
    tok, te = pipe.tokenizer, pipe.text_encoder
    tok.add_tokens([TOKEN])
    te.resize_token_embeddings(len(tok))
    tid = tok.convert_tokens_to_ids(TOKEN)
    rot = rotary(pipe, dev)
    tr_net = pipe.transformer
    out_dir = Path("cache")

    def tgt_prompt(d):
        return f"{d}, {TOKEN}" if args.method == "ti" else f"{d}, {DB_PHRASE}"

    if args.stage == "train":
        with torch.no_grad():
            init = tok("unusual", add_special_tokens=False).input_ids[0]
            te.shared.weight[tid] = te.shared.weight[init].clone()
        if args.method == "ti":
            te.shared.weight.requires_grad_(True)
            params = [te.shared.weight]
            lr = 5e-3
        else:
            from peft import LoraConfig
            tr_net.add_adapter(LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian",
                                          target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
            params = [p for p in tr_net.parameters() if p.requires_grad]
            for p in params:
                p.data = p.data.float()
            lr = 1e-4
        tr_net.enable_gradient_checkpointing()
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        log(f"training {args.method}: {sum(p.numel() for p in params):,} parameters "
            f"({'one word' if args.method == 'ti' else 'LoRA'}), lr {lr}")
        losses, t0 = [], time.time()
        for step in range(1, args.steps + 1):
            if args.method == "ti":
                ids = [rng.choice(tgt_ids) for _ in range(args.batch)]
                prompts = [tgt_prompt(rng.choice(descr)) for _ in ids]
            else:  # half target (with the phrase), half source (without): prior preservation
                ids = [rng.choice(tgt_ids) for _ in range(args.batch // 2)] + rng.sample(src_ids, args.batch - args.batch // 2)
                prompts = [tgt_prompt(rng.choice(descr)) if i in tgt_ids else rng.choice(descr) for i in ids]
            x0 = Z[ids].to(dev)
            empty = torch.rand(len(ids), device=dev) < 0.1
            enc, glob = conditioning(pipe, prompts, dev, empty)
            t = torch.rand(len(ids), device=dev).clamp(1e-3, 1 - 1e-3)
            al, si = torch.cos(t * np.pi / 2)[:, None, None], torch.sin(t * np.pi / 2)[:, None, None]
            e = torch.randn_like(x0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = tr_net(al * x0 + si * e, t, encoder_hidden_states=enc, global_hidden_states=glob,
                           rotary_embedding=rot, return_dict=False)[0]
            loss = F.mse_loss(v.float(), al * e - si * x0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.method == "ti":  # only the new word's row may change
                g = te.shared.weight.grad
                keep = torch.zeros_like(g)
                keep[tid] = g[tid]
                te.shared.weight.grad = keep
            opt.step()
            losses.append(loss.item())
            if step % 50 == 0:
                log(f"step {step:5d} loss {np.mean(losses[-50:]):.4f} {(time.time() - t0) / step:.2f}s/step "
                    f"GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if args.method == "ti":
            torch.save({"embedding": te.shared.weight[tid].detach().cpu()}, out_dir / f"sao_ti_{args.machine}.pt")
        else:
            from peft import get_peft_model_state_dict
            torch.save(get_peft_model_state_dict(tr_net), out_dir / f"sao_db_{args.machine}.pt")
        log(f"saved {args.method} weights to cache/sao_{args.method}_{args.machine}.pt")
        return

    # ---- edit: style transfer of source clips ---------------------------------------------------------------------
    if args.method == "ti":
        with torch.no_grad():
            te.shared.weight[tid] = torch.load(out_dir / f"sao_ti_{args.machine}.pt")["embedding"].to(dev)
    else:
        from peft import LoraConfig, set_peft_model_state_dict
        tr_net.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        set_peft_model_state_dict(tr_net, torch.load(out_dir / f"sao_db_{args.machine}.pt"))
        tr_net.to(dtype=torch.bfloat16)
    sched = pipe.scheduler
    X16 = torch.cat([three_channel(tr.get(list(range(i, min(i + 32, len(tr))))).to(dev), win).cpu()
                     for i in range(0, len(tr), 32)])  # (1000, 3, 256, 313) dB, real training clips

    @torch.no_grad()
    def edit(ids, prompts, sigma_start):
        sched.set_timesteps(50, device=dev)
        k = int((sched.sigmas[:-1] - sigma_start).abs().argmin())
        sched.set_begin_index(k)
        lat = Z[ids].to(dev)
        lat = lat + sched.sigmas[k] * torch.randn_like(lat)
        enc, glob = conditioning(pipe, prompts, dev)
        enc2 = torch.cat([torch.zeros_like(enc), enc])
        glob2 = torch.cat([glob, glob])
        for t in sched.timesteps[k:]:
            inp = sched.scale_model_input(torch.cat([lat, lat]), t)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = tr_net(inp, t.unsqueeze(0), encoder_hidden_states=enc2, global_hidden_states=glob2,
                           rotary_embedding=rot, return_dict=False)[0].float()
            vu, vc = v.chunk(2)
            lat = sched.step(vu + args.guidance * (vc - vu), t, lat).prev_sample
        return to_16k(pipe.vae.decode(lat).sample)

    for s_i, sig in enumerate(float(x) for x in args.sigmas.split(",")):
        for with_word in (True, False):
            S, X0 = [], []
            for i in range(0, args.n_edit, 8):
                ids = [rng.choice(src_ids) for _ in range(min(8, args.n_edit - i))]
                prompts = [tgt_prompt(rng.choice(descr)) if with_word else rng.choice(descr) for _ in ids]
                w = edit(ids, prompts, sig)
                S.append(three_channel(w, win).cpu())
                X0.append(X16[ids])
            S, X0 = torch.cat(S), torch.cat(X0)
            tl, ch = target_likeness(S, X0, X16, tgt_ids, src_ids)
            name = "WITH target word" if with_word else "control, no target word"
            log(f"edit sigma {sig}: {name:24s} target-likeness {tl:+.2f} (+1 = like real target, -1 = like source) | "
                f"changed {ch:.2f} dB from its source clip")
            if with_word and s_i == 0:
                p = out_dir / f"sao_target_{args.machine}_{args.method}.pt"
                torch.save({"spec_db": S.half(), "info": f"Stable Audio Open {args.method} edits of source clips, "
                            f"sigma {sig}, guidance {args.guidance}; near, far, near-minus-far dB, 256 bands x 313 frames"}, p)
                log(f"saved {len(S)} edits to {p}")


if __name__ == "__main__":
    main()
