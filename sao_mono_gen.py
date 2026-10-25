"""Target-style machine sound from Stable Audio Open, personalised on the 10 target-domain clips (one-channel version).

Signal: the machine's own sound (near minus far-predicted part), latents from sao_mono.py --stage cache, decoded with
the fine-tuned decoder from sao_mono.py --stage decoder.
Stage train (diffusion model only; VAE untouched):
  ti  textual inversion: new word '<tgt>' (T5 input embedding, init 'unusual'), trained on the 10 target clips
  db  DreamBooth LoRA (rank 16, DiT attention): target clips with '..., recorded in sks condition', source clips with
      the same prompts without it (half of every batch, prior preservation)
  objective: Stable Audio Open's v-prediction on the cosine schedule; 10% empty conditioning; several detailed prompts
Stage edit: source clips' latents noised to sigma (fixed in advance: 0.8, 2.0), denoised with guidance 4, prompt WITH
the target word, and the control WITHOUT it. Decoded, back to 16 kHz.
  target-likeness: (|syn - source| - |syn - target|) / |target - source| on clip-averaged band levels (256 bands, dB),
  +1 = like the real target clips, -1 = like source; 'changed' = mean |band level change| vs its source clip.
  Saved for the autoencoder (sigma 0.8 only): two-mic inputs rebuilt with real background: far = a real far-mic
  recording of another training clip, near = generated machine sound + that background 3-10 dB quieter; stored as
  near / far / near-minus-far spectrograms (diffusion_target.three_channel, the format ae_noiseaware.py --synth reads):
    cache/sao_mono_<method>_<machine>_word.pt     (target-style)
    cache/sao_mono_<method>_<machine>_noword.pt   (control: source content through the same model and decoder)
Usage: uv run python -u sao_mono_gen.py --machine ToyCar --stage train --method ti
       uv run python -u sao_mono_gen.py --machine ToyCar --stage edit --method ti
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dcase_ae.efficientad import WavStore, build_cache
from diffusion_target import three_channel
from sao_mono import band_levels, down
from sao_target import DB_PHRASE, DESCR, FRAMES, TOKEN, conditioning, load, rotary
from tools.simple_baselines import near_only_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--stage", choices=["train", "edit", "listen", "sweep"], required=True)
    ap.add_argument("--sweep", default="0.25:3.0:0.25", help="sweep: sigma from:to:step")
    ap.add_argument("--method", choices=["ti", "db"], required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_edit", type=int, default=200)
    ap.add_argument("--sigmas", default="0.8,2.0")
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--decoder", choices=["original", "finetuned"], default="original",
                    help="VAE decoder for audio output: Stable Audio's own, or ours (sao_mono.py --stage decoder)")
    ap.add_argument("--sample_every", type=int, default=200, help="train: log edits to the sample viewer every N steps")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank (alpha = rank)")
    ap.add_argument("--lr", type=float, default=None, help="default: 5e-3 (ti), 1e-4 (db)")
    ap.add_argument("--grad_ckpt", type=int, default=1, help="gradient checkpointing in the diffusion transformer")
    ap.add_argument("--tag", default="", help="appended to the weights file and viewer run name")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--only_word", action="store_true", help="edit: only generate WITH the target word (no control)")
    ap.add_argument("--only_noword", action="store_true", help="edit: only generate WITHOUT the target word")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device("cuda")

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')}  [sao-mono-{args.method} {args.machine}] {msg}", flush=True)

    log(f"=== {vars(args)}")
    build_cache("data", "cache", args.machine, "train")
    tr = WavStore("cache", args.machine, "train")
    dom = np.array(tr.meta["domain"])
    src_ids, tgt_ids = np.flatnonzero(dom == 0).tolist(), np.flatnonzero(dom == 1).tolist()
    pipe = load(dev)
    Z = torch.load(f"cache/sao_nmf_latents_{args.machine}.pt").float()
    descr = DESCR[args.machine]
    tok, te, net = pipe.tokenizer, pipe.text_encoder, pipe.transformer
    tok.add_tokens([TOKEN])
    te.resize_token_embeddings(len(tok))
    tid = tok.convert_tokens_to_ids(TOKEN)
    rot = rotary(pipe, dev)
    tag = f"_{args.tag}" if args.tag else ""
    wpath = Path(f"cache/sao_mono_{args.method}_{args.machine}{tag}.pt")

    def tgt_prompt(d):
        return f"{d}, {TOKEN}" if args.method == "ti" else f"{d}, {DB_PHRASE}"

    sched = pipe.scheduler
    if args.decoder == "finetuned":
        pipe.vae.decoder.load_state_dict(torch.load(f"cache/sao_nmf_decoder_{args.machine}.pt"))
    log(f"decoder: {args.decoder}")

    @torch.no_grad()
    def edit(ids, prompts, sigma, n_grid=50, noise=None, used=None):
        """noise: fixed standard-normal latent noise (optional); used: list that receives the sigma actually used"""
        sched.set_timesteps(n_grid, device=dev)
        k = int((sched.sigmas[:-1] - sigma).abs().argmin())
        sched.set_begin_index(k)
        if used is not None:
            used.append(float(sched.sigmas[k]))
        lat = Z[ids].to(dev)
        lat = lat + sched.sigmas[k] * (torch.randn_like(lat) if noise is None else noise.to(dev))
        enc, glob = conditioning(pipe, prompts, dev)
        enc2, glob2 = torch.cat([torch.zeros_like(enc), enc]), torch.cat([glob, glob])
        for t in sched.timesteps[k:]:
            inp = sched.scale_model_input(torch.cat([lat, lat]), t)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = net(inp, t.unsqueeze(0), encoder_hidden_states=enc2, global_hidden_states=glob2,
                        rotary_embedding=rot, return_dict=False)[0].float()
            vu, vc = v.chunk(2)
            lat = sched.step(vu + args.guidance * (vc - vu), t, lat).prev_sample
        m = torch.cat([down(pipe.vae.decode(lat[j : j + 1]).sample)[:, :160000] for j in range(len(lat))])  # one clip at a time
        return F.pad(m, (0, 160000 - m.shape[-1]))


    if args.stage == "train":
        from peft import LoraConfig, get_peft_model_state_dict
        with torch.no_grad():
            te.shared.weight[tid] = te.shared.weight[tok("unusual", add_special_tokens=False).input_ids[0]].clone()
        if args.method == "ti":
            te.shared.weight.requires_grad_(True)
            params, lr = [te.shared.weight], args.lr or 5e-3
        else:
            from peft import inject_adapter_in_model
            inject_adapter_in_model(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                               target_modules=["to_q", "to_k", "to_v", "to_out.0"]), net)
            params = [p for n, p in net.named_parameters() if "lora_" in n]
            for p in params:
                p.requires_grad_(True)
            for p in params:
                p.data = p.data.float()
            lr = args.lr or 1e-4
        if args.grad_ckpt:
            net.enable_gradient_checkpointing()
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        n_tr = te.shared.weight.shape[1] if args.method == "ti" else sum(p.numel() for p in params)
        log(f"training {args.method}: {n_tr:,} trainable values, lr {lr}, rank {args.rank}, batch {args.batch}, "
            f"gradient checkpointing {'on' if args.grad_ckpt else 'off'}")
        from tools.viewer import SampleLog
        view = SampleLog(f"sao-{args.method}-{args.machine}-{args.decoder}-decoder{tag}", description=(
            f"Stable Audio Open, {'DreamBooth LoRA' if args.method == 'db' else 'textual inversion'} on {args.machine}'s "
            f"machine sound (near minus far). Every {args.sample_every} steps: two fixed source clips edited at sigma 2.0, "
            f"guidance {args.guidance}, with the target phrase and without it (control), decoded with the {args.decoder} "
            "decoder. References: the two source clips and two real target-domain clips (machine sound)."))
        view_ids = random.Random(1).sample(src_ids, 2)
        with torch.no_grad():
            ref = near_only_audio(tr.get(view_ids + tgt_ids[:2]), dev).cpu()
        for j, i in enumerate(view_ids):
            view.reference(f"source clip {i}", ref[j], note="machine sound of the source clip being edited")
        for j, i in enumerate(tgt_ids[:2]):
            view.reference(f"REAL target clip {i}", ref[2 + j], note="real target-domain clip, machine sound")
        view_prompt = descr[0]

        def log_samples(step):
            net.eval()
            g = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            torch.manual_seed(1234)  # same noise every time, so steps are comparable
            for i in view_ids:
                for with_word in (True, False):
                    p = tgt_prompt(view_prompt) if with_word else view_prompt
                    w = edit([i], [p], 2.0)[0]
                    view.sample(step, f"clip {i} | sigma 2.0 | {'WITH phrase' if with_word else 'no phrase (control)'}", w,
                                note=f"prompt: '{p}'")
            torch.random.set_rng_state(g[0]); torch.cuda.set_rng_state(g[1])
            net.train()

        if args.sample_every:
            log_samples(0)
        losses, t0 = [], time.time()
        for step in range(1, args.steps + 1):
            if args.method == "ti":
                ids = [rng.choice(tgt_ids) for _ in range(args.batch)]
            else:
                ids = [rng.choice(tgt_ids) for _ in range(args.batch // 2)] + rng.sample(src_ids, args.batch - args.batch // 2)
            prompts = [tgt_prompt(rng.choice(descr)) if i in tgt_ids else rng.choice(descr) for i in ids]
            x0 = Z[ids].to(dev)
            enc, glob = conditioning(pipe, prompts, dev, torch.rand(len(ids), device=dev) < 0.1)
            t = torch.rand(len(ids), device=dev).clamp(1e-3, 1 - 1e-3)
            al, si = torch.cos(t * np.pi / 2)[:, None, None], torch.sin(t * np.pi / 2)[:, None, None]
            e = torch.randn_like(x0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = net(al * x0 + si * e, t, encoder_hidden_states=enc, global_hidden_states=glob,
                        rotary_embedding=rot, return_dict=False)[0]
            loss = F.mse_loss(v.float(), al * e - si * x0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.method == "ti":
                g = te.shared.weight.grad
                keep = torch.zeros_like(g)
                keep[tid] = g[tid]
                te.shared.weight.grad = keep
            opt.step()
            losses.append(loss.item())
            if step % 50 == 0:
                log(f"step {step:5d} loss {np.mean(losses[-50:]):.4f} {(time.time() - t0) / step:.2f}s/step "
                    f"GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
            if args.sample_every and step % args.sample_every == 0:
                log_samples(step)
                log(f"step {step:5d} logged samples to the viewer")
            if args.save_every and step % args.save_every == 0 and step != args.steps:
                ck = wpath.with_name(wpath.stem + f"_step{step}.pt")
                torch.save({"embedding": te.shared.weight[tid].detach().cpu()} if args.method == "ti"
                           else get_peft_model_state_dict(net), ck)
                log(f"step {step:5d} checkpoint saved to {ck}")
        torch.save({"embedding": te.shared.weight[tid].detach().cpu()} if args.method == "ti"
                   else get_peft_model_state_dict(net), wpath)
        log(f"saved {wpath} ({wpath.stat().st_size / 1e6:.1f} MB)")
        return

    # ---- edit -------------------------------------------------------------------------------------------------
    if args.method == "ti":
        with torch.no_grad():
            te.shared.weight[tid] = torch.load(wpath)["embedding"].to(dev)
    else:
        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
        inject_adapter_in_model(LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=["to_q", "to_k", "to_v", "to_out.0"]), net)
        res = set_peft_model_state_dict(net, torch.load(wpath))
        assert not res.unexpected_keys, res.unexpected_keys
        net.to(dtype=torch.bfloat16)
    log(f"loaded {wpath}")
    with torch.no_grad():
        machine = torch.cat([near_only_audio(tr.get(list(range(i, min(i + 32, len(tr))))), dev).cpu()
                             for i in range(0, len(tr), 32)])  # (1000, L) 16 kHz machine sound
        B = torch.cat([band_levels(machine[i : i + 64].to(dev)).cpu() for i in range(0, len(machine), 64)])  # (1000, 256)
    real_t, real_s = B[tgt_ids].mean(0), B[src_ids].mean(0)
    d_ts = (real_t - real_s).abs().mean()
    log(f"real machine sound: target vs source clip-averaged band levels differ by {d_ts:.2f} dB on average")

    win = torch.hann_window(1024, device=dev)
    if args.stage == "sweep":  # one edit per sigma for a few fixed source clips, shown in the sample viewer
        from tools.viewer import SampleLog
        a, b, c = (float(x) for x in args.sweep.split(":"))
        sigmas = [round(a + i * c, 4) for i in range(int(round((b - a) / c)) + 1)]
        view = SampleLog(f"sigma-sweep-{args.method}-{args.machine}{tag}-{args.decoder}-decoder", axis="sigma", description=(
            f"Edits of 3 fixed {args.machine} source clips (machine sound) WITH the target phrase at increasing noise level "
            f"sigma (requested {sigmas[0]}..{sigmas[-1]} step {c}; the card shows the sigma actually used, 200-step "
            f"schedule, minimum 0.3). Same noise for a clip at every sigma. Weights {wpath.name}, guidance {args.guidance}, "
            f"{args.decoder} decoder. sigma 0 = decoder reconstruction (no diffusion). References: each source clip and 2 "
            "real target clips."))
        clips = random.Random(1).sample(src_ids, 3)
        prompt = tgt_prompt(descr[0])
        for i in clips:
            view.reference(f"source clip {i}", machine[i], note="machine sound of the source clip (input to the edits)")
        for i in tgt_ids[:2]:
            view.reference(f"REAL target clip {i}", machine[i], note="real target-domain clip, machine sound")
        for i in clips:
            with torch.no_grad():
                rec = down(pipe.vae.decode(Z[[i]].to(dev)).sample)[0, :160000]
            view.sample(0.0, f"clip {i}", rec, note=f"decoder reconstruction ({args.decoder} decoder), no diffusion")
            noise = torch.randn(Z[[i]].shape, generator=torch.Generator().manual_seed(1000 + i))
            for sg in sigmas:
                used = []
                w = edit([i], [prompt], sg, n_grid=200, noise=noise, used=used)[0]
                view.sample(round(used[0], 3), f"clip {i}", w, note=f"requested sigma {sg}, used {used[0]:.3f}; prompt '{prompt}'")
                log(f"clip {i}: sigma {sg} (used {used[0]:.3f}) done")
        log(f"sweep done: {len(clips)} clips x {len(sigmas)} sigmas in the viewer")
        return
    if args.stage == "listen":  # audio + spectrograms on the comparison page (outputs/sam_compare)
        import json
        import soundfile as sf
        import sys
        sys.path.insert(0, "tools")
        from filter_compare import mel_db, save_png
        out = Path("outputs/sam_compare")
        man = json.loads((out / "manifest.json").read_text())
        mach = f"{args.machine} SAO edits"
        dec_tag = f"{args.method}, {args.decoder} decoder"

        def put(clip, label, domain, file, variant, wav, note, vmax):
            stem = f"sao-{args.method}-{args.decoder}-{clip}-{variant}".replace(" ", "-").replace(".", "p").replace("(", "").replace(")", "")
            sf.write(out / "audio" / f"{stem}.wav", wav.astype(np.float32), 16000)
            save_png(mel_db(wav.astype(np.float32)), out / "img" / f"{stem}.png", len(wav) / 16000, vmax - 70, vmax)
            man["items"] = [i for i in man["items"] if not (i["clip"] == clip and i["variant"] == variant)]
            man["items"].append({"clip": clip, "machine": mach, "label": label, "domain": domain, "file": file,
                                 "mic": f"machine sound ({dec_tag})", "variant": variant, "audio": f"audio/{stem}.wav",
                                 "img": f"img/{stem}.png", "note": note})

        names = tr.meta["names"]
        pick = random.Random(1).sample(src_ids, 3)
        for i in pick:
            orig = machine[i].numpy()
            vmax = float(mel_db(orig.astype(np.float32)).max())
            clip = f"{args.machine}-src-{i}"
            put(clip, "normal", "source", names[i], "original", orig, "machine sound = near minus far-predicted part", vmax)
            with torch.no_grad():
                rec = down(pipe.vae.decode(Z[[i]].to(dev)).sample)[0, :160000].cpu().numpy()
            put(clip, "normal", "source", names[i], "decoder reconstruction", rec, f"encode -> {args.decoder} decoder, no diffusion", vmax)
            for sig in (float(x) for x in args.sigmas.split(",")):
                for with_word in (True, False):
                    d = rng.choice(descr)
                    w = edit([i], [tgt_prompt(d) if with_word else d], sig)[0].cpu().numpy()
                    put(clip, "normal", "source", names[i], f"edit {sig} {'with word' if with_word else 'no word'}", w,
                        f"sigma {sig}, guidance {args.guidance}, prompt: '{tgt_prompt(d) if with_word else d}'", vmax)
        for i in tgt_ids[:2]:
            orig = machine[i].numpy()
            put(f"{args.machine}-tgt-{i}", "normal", "target", names[i], "original", orig,
                "REAL target-domain clip, machine sound, for reference", float(mel_db(orig.astype(np.float32)).max()))
        (out / "manifest.json").write_text(json.dumps(man, indent=1))
        log(f"added {len(pick)} source clips (original, reconstruction, 4 edits) and 2 real target clips to the page")
        return
    for s_i, sig in enumerate(float(x) for x in args.sigmas.split(",")):
        for with_word in ((True,) if args.only_word else (False,) if args.only_noword else (True, False)):
            Bs, Bsrc, pairs = [], [], []
            for i in range(0, args.n_edit, 8):
                ids = [rng.choice(src_ids) for _ in range(min(8, args.n_edit - i))]
                prompts = [tgt_prompt(rng.choice(descr)) if with_word else rng.choice(descr) for _ in ids]
                m = edit(ids, prompts, sig)  # (b, L) generated machine sound, 16 kHz
                Bs.append(band_levels(m).cpu())
                Bsrc.append(B[ids])
                if s_i == 0:  # rebuild two-mic input with real background
                    far = tr.get([rng.choice(src_ids + tgt_ids) for _ in ids])[:, 1].to(dev)
                    g = (m.pow(2).mean(-1, keepdim=True).sqrt() / (far.pow(2).mean(-1, keepdim=True).sqrt() + 1e-9)
                         * torch.tensor([10 ** (-rng.uniform(3, 10) / 20) for _ in ids], device=dev)[:, None])
                    pairs.append(three_channel(torch.stack([m + g * far, far], 1), win).half().cpu())
            Bs, Bsrc = torch.cat(Bs), torch.cat(Bsrc)
            syn = Bs.mean(0)
            tl = (((syn - real_s).abs().mean() - (syn - real_t).abs().mean()) / d_ts).item()
            ch = (Bs - Bsrc).abs().mean().item()
            name = "WITH target word" if with_word else "control, no target word"
            log(f"edit sigma {sig}: {name:24s} target-likeness {tl:+.2f} (+1 = like real target, -1 = like source) | "
                f"changed {ch:.2f} dB from its source clip")
            if s_i == 0:
                p = Path(f"cache/sao_mono_{args.method}_{args.machine}{tag}_{'word' if with_word else 'noword'}.pt")
                torch.save({"spec_db": torch.cat(pairs), "info": f"{name}, sigma {sig}, guidance {args.guidance}; "
                            "two-mic input rebuilt with real far-mic background; near, far, near-minus-far dB"}, p)
                log(f"saved {len(Bs)} samples to {p}")


if __name__ == "__main__":
    main()
