"""Fine-tune SAM-Audio as a 'normality separator' for one machine (LoRA on DiT self-attention + span table).

Training example (near mic): [5 s normal reference, marked '+'] + [5 s of another normal clip]. With probability 0.5 a
synthetic anomaly n (narrow-band gain, weak tone, wobble, or band-limited noise burst) is added to the second part.
Ground truth: target = reference + clean second part, residual = n (silence for clean examples).
Loss: flow matching on SAM-Audio's latents (noise at t=0 -> [target, residual] latents at t=1); the residual half is
weighted --res_weight times. Fake anomalies start loud and become subtle over training (curriculum).

Validation (at step 0 and every --eval_every steps):
  synthetic : held-out normal clips, half with a fake anomaly    -> AUC of the residual-share score
  real      : machine test clips (normal vs anomaly), each paired with 1 random normal reference, both 5 s halves
Scores (both over the test part): latent = distance of the residual latent from the silence latent (test minus
reference frames); share = residual's share of the decoded energy. Higher = more 'not like the reference'.

Runs in the SAM-Audio environment (PYTHONPATH=tools):
  envs/samaudio/Scripts/python tools/sam_normality_ft.py --machine fan --steps 400
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from torchdiffeq import odeint

from sam_audio_demo import machine_phrase

SR_IN, SEG = 16000, 5.0
MODEL_ID = "facebook/sam-audio-small-tv"
EVAL_ODE = {"method": "midpoint", "options": {"step_size": 1 / 8}}  # 8 steps (default 16); same for every evaluation


# ---- LoRA ----------------------------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=16, alpha=16):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.a = nn.Parameter(torch.randn(rank, base.in_features, device=dev) / math.sqrt(base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.a.t().to(x.dtype) @ self.b.t().to(x.dtype)) * self.scale


def add_lora(transformer, rank):
    n = 0
    for name, mod in transformer.named_modules():
        if name.endswith(".attention"):  # self-attention only (not cross_attention)
            for p in ("wq", "wk", "wv", "wo"):
                setattr(mod, p, LoRALinear(getattr(mod, p), rank))
                n += 1
    return n


# ---- synthetic anomalies (16 kHz, on GPU) -------------------------------------------------------------------
# (subtle range, loud range) per fake type; `loud` in [0, 1] interpolates between them (curriculum)
RANGES = {"gain_db": ((2, 6), (6, 15)), "tone_db": ((-35, -20), (-20, -8)), "wobble": ((0.1, 0.3), (0.3, 0.6)),
          "burst_db": ((-30, -15), (-15, -5))}


def rng_range(name, loud):
    (a0, b0), (a1, b1) = RANGES[name]
    return a0 + loud * (a1 - a0), b0 + loud * (b1 - b0)


def fake_anomaly(x, g, loud=0.0, kind=None):
    """Returns n such that x + n is x with one random fault-like change (loud=0: subtle, loud=1: loud)."""
    L, dev = len(x), x.device
    t = torch.arange(L, device=dev) / SR_IN
    rms = x.pow(2).mean().sqrt()
    kind = g.randrange(4) if kind is None else kind  # 0 gain, 1 tone, 2 wobble, 3 burst
    if kind == 0:  # narrow-band gain change
        X = torch.stft(x, 2048, 512, window=torch.hann_window(2048, device=dev), return_complex=True)
        f = torch.fft.rfftfreq(2048, 1 / SR_IN).to(dev)
        f0, bw, db = g.uniform(200, 7000), g.uniform(50, 300), g.choice([-1, 1]) * g.uniform(*rng_range("gain_db", loud))
        gain = 1 + (10 ** (db / 20) - 1) * torch.exp(-0.5 * ((f - f0) / bw) ** 2)
        y = torch.istft(X * gain[:, None], 2048, 512, window=torch.hann_window(2048, device=dev), length=L)
        return y - x
    if kind == 1:  # weak tone
        return rms * 10 ** (g.uniform(*rng_range("tone_db", loud)) / 20) * math.sqrt(2) * torch.sin(2 * math.pi * g.uniform(150, 7000) * t)
    if kind == 2:  # amplitude wobble
        return x * g.uniform(*rng_range("wobble", loud)) * torch.sin(2 * math.pi * g.uniform(4, 20) * t)
    # band-limited noise burst
    noise = torch.randn(L, device=dev)
    lo = g.uniform(100, 6000)
    noise = torchaudio.functional.bandpass_biquad(noise, SR_IN, lo + 250, Q=2.0)
    a, d = g.uniform(0, SEG - 0.5), g.uniform(0.5, 3.0)
    env = ((t >= a) & (t < a + d)).float()
    return env * noise / (noise.pow(2).mean().sqrt() + 1e-8) * rms * 10 ** (g.uniform(*rng_range("burst_db", loud)) / 20)


def loudness_at(step, steps):
    """Curriculum: loud for the first third, linearly to subtle over the middle third, subtle for the last third."""
    p = step / steps
    return 1.0 if p < 1 / 3 else 0.0 if p > 2 / 3 else 1.0 - (p - 1 / 3) * 3


def crop(x, g):
    n = int(SEG * SR_IN)
    s = g.randrange(0, len(x) - n + 1)
    return x[s : s + n]


# ---- model helpers --------------------------------------------------------------------------------------------
class Runner:
    def __init__(self, model, proc, text, device):
        self.model, self.proc, self.text, self.dev = model, proc, text, device
        self.sr = proc.audio_sampling_rate

    def up(self, w):  # 16 kHz -> codec rate
        return torchaudio.functional.resample(w, SR_IN, self.sr)

    def batch(self, mixtures):
        return self.proc(audios=[self.up(m)[None].cpu() for m in mixtures], descriptions=[self.text] * len(mixtures),
                         anchors=[[["+", 0.0, SEG]]] * len(mixtures)).to(self.dev)

    def latents(self, wavs):  # (B, L) @16k -> (B, T, C)
        return self.model.audio_codec(self.up(wavs)[:, None]).transpose(1, 2)

    def loss(self, mixtures, targets, residuals, res_weight=1.0):
        """Flow-matching loss; returns (weighted loss with v = x1 - noise, unweighted loss with the reverse direction)."""
        b = self.batch(mixtures)
        with torch.no_grad():
            args = self.model._get_forward_args(b)
            x1 = torch.cat([self.latents(targets), self.latents(residuals)], dim=2)
        T = min(x1.shape[1], args["audio_features"].shape[1])
        x1 = x1[:, :T]
        for k in ("audio_features", "anchor_alignment", "audio_pad_mask"):
            args[k] = args[k][:, :T]
        args["masked_video_features"] = args["masked_video_features"][..., :T]
        noise = torch.randn_like(x1)
        t = torch.rand(len(x1), device=self.dev)
        xt = (1 - t[:, None, None]) * noise + t[:, None, None] * x1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.model(noisy_audio=xt, time=t, **args).float()
        C = x1.shape[2] // 2  # first half = target latents, second half = residual latents
        v = x1 - noise
        weighted = F.mse_loss(pred[..., :C], v[..., :C]) + res_weight * F.mse_loss(pred[..., C:], v[..., C:])
        return weighted, F.mse_loss(pred, noise - x1)

    @torch.inference_mode()
    def score_many(self, pairs, seed=0, bs=4):
        """Separate [ref | test] for many pairs, bs at a time; returns (N, 2): latent score, decoded residual share,
        both over the test part. Latent score = mean distance of the residual latent from the 'silence' latent on the
        test frames minus the same on the reference frames (the reference part should have a silent residual)."""
        g = torch.Generator(device=self.dev).manual_seed(seed)
        out = []
        for i in range(0, len(pairs), bs):
            chunk = pairs[i : i + bs]
            b = self.batch([torch.cat([r, t]) for r, t in chunk])
            args = self.model._get_forward_args(b)
            noise = torch.randn(args["audio_features"].shape, device=self.dev, generator=g)
            field = lambda t, x: self.model(noisy_audio=x, time=t.expand(x.size(0)), **args)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = odeint(field, noise, torch.tensor([0.0, 1.0], device=self.dev), **EVAL_ODE)[-1].float()
            B, T, C2 = gen.shape
            C = C2 // 2
            split = int(round(SEG * self.sr / self.model.audio_codec.hop_length))
            d = (gen[:, :, C:] - self.silence).pow(2).sum(-1)  # (B, T)
            latent = d[:, split:].mean(1) - d[:, :split].mean(1)
            lat = gen.transpose(1, 2).reshape(2 * B, C, T)
            wav = torch.cat([self.model.audio_codec.decode(lat[i : i + 1]).float() for i in range(2 * B)]).view(B, 2, -1)
            s = int(SEG * self.sr)
            tg, rs = wav[:, 0, s:], wav[:, 1, s:]
            share = rs.pow(2).sum(1) / (rs.pow(2).sum(1) + tg.pow(2).sum(1) + 1e-12)
            out += list(zip(latent.cpu().tolist(), share.cpu().tolist()))
        return np.array(out)

    @torch.no_grad()
    def set_silence(self):
        """Mean codec latent of silence (the latent a silent residual should have)."""
        self.silence = self.latents(torch.zeros(1, int(2 * SEG * SR_IN), device=self.dev))[0].mean(0)


def load_near(path):
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR_IN
    return torch.from_numpy(w[: 10 * SR_IN, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="fan")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_steps", default="", help="comma-separated steps to evaluate at (overrides --eval_every)")
    ap.add_argument("--n_real", type=int, default=10, help="per (label, domain) group of real test clips")
    ap.add_argument("--res_weight", type=float, default=10.0, help="weight of the residual half of the loss")
    ap.add_argument("--no_curriculum", action="store_true", help="subtle fakes from the start")
    args = ap.parse_args()
    dev = torch.device("cuda")
    g = random.Random(0)
    torch.manual_seed(0)

    from sam_audio import SAMAudio, SAMAudioProcessor

    model = SAMAudio.from_pretrained(MODEL_ID, visual_ranker=None, text_ranker=None, span_predictor=None).eval()
    model.vision_encoder.model.to("cpu")
    for name, mod in model.named_children():
        if name != "vision_encoder":
            mod.to(dev)
    model.transformer.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    n_lora = add_lora(model.transformer, args.rank)
    for p in model.embed_anchors.parameters():
        p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA on {n_lora} attention layers; trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.2f}M")
    proc = SAMAudioProcessor.from_pretrained(MODEL_ID)
    run = Runner(model, proc, f"{machine_phrase(args.machine)} running", dev)
    run.set_silence()

    # data: normal training clips (near mic); hold out 40 for the synthetic validation
    files = sorted(Path(f"data/{args.machine}/train").glob("*.wav"))
    g.shuffle(files)
    clips = torch.stack([load_near(f) for f in files]).to(dev)
    tr_idx, val_idx = list(range(40, len(files))), list(range(40))
    syn_val = []  # (ref, test, label)
    for k, i in enumerate(val_idx):
        ref = crop(clips[g.choice(tr_idx)], g)
        test = crop(clips[i], g)
        lab = k % 2
        syn_val.append((ref, test + fake_anomaly(test, g) if lab else test, lab))
    # real validation: test clips, 2 random references each, both halves
    real = []
    for lab in ("normal", "anomaly"):
        for dom in ("source", "target"):
            pool = sorted(Path(f"data/{args.machine}/test").glob(f"*{dom}_test_{lab}*.wav"))
            for f in g.sample(pool, args.n_real):
                x = load_near(f).to(dev)
                halves = [x[: int(SEG * SR_IN)], x[int(SEG * SR_IN): int(2 * SEG * SR_IN)]]
                refs = [crop(clips[g.choice(tr_idx)], g)]
                real.append((refs, halves, int(lab == "anomaly"), dom))

    out = Path(f"outputs/sam_normality/{args.machine}")
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def evaluate(step):
        model.eval()
        t0 = time.time()
        syn = run.score_many([(r, x) for r, x, _ in syn_val])  # (N, 2): latent score, share
        lab_syn = [l for *_, l in syn_val]
        pairs = [(r, h) for refs, halves, *_ in real for r in refs for h in halves]
        per_clip = len(real[0][0]) * len(real[0][1])
        rl = run.score_many(pairs).reshape(len(real), per_clip, 2).mean(1)
        lab_real = np.array([lab for *_, lab, _ in real])
        dom_real = np.array([dom for *_, dom in real])
        row = {"step": step}
        for j, name in ((0, "latent"), (1, "share")):
            row[f"syn_{name}"] = roc_auc_score(lab_syn, syn[:, j])
            row[f"real_{name}"] = roc_auc_score(lab_real, rl[:, j])
            for d in ("source", "target"):
                row[f"real_{name}_{d}"] = roc_auc_score(lab_real[dom_real == d], rl[dom_real == d, j])
        row["share_normal"], row["share_anomaly"] = float(rl[lab_real == 0, 1].mean()), float(rl[lab_real == 1, 1].mean())
        log.append(row)
        print(f"[eval step {step}] latent score: synthetic AUC {row['syn_latent']:.3f} | real AUC {row['real_latent']:.3f} "
              f"(src {row['real_latent_source']:.3f} / tgt {row['real_latent_target']:.3f})   || decoded share: synthetic AUC "
              f"{row['syn_share']:.3f} | real AUC {row['real_share']:.3f} | share normal {row['share_normal']:.3f} vs anomaly "
              f"{row['share_anomaly']:.3f} | {time.time() - t0:.0f}s", flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1))

    # sanity check of the flow direction with the pretrained weights
    with torch.no_grad():
        mix, tgt, rsd = [], [], []
        for _ in range(args.batch):
            r, x = crop(clips[g.choice(tr_idx)], g), crop(clips[g.choice(tr_idx)], g)
            mix.append(torch.cat([r, x]))
            tgt.append(torch.cat([r, x]))
            rsd.append(torch.zeros_like(mix[-1]))
        l_fwd, l_rev = run.loss(mix, torch.stack(tgt), torch.stack(rsd))
    print(f"flow-direction check (pretrained): loss with v = x1 - noise: {l_fwd.item():.3f} | v = noise - x1: {l_rev.item():.3f}")

    eval_steps = {int(x) for x in args.eval_steps.split(",") if x}
    evaluate(0)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.transformer.train()  # only the trained part; frozen encoders stay in eval mode (no T5 dropout)
        mix, tgt, rsd = [], [], []
        for _ in range(args.batch):
            r, x = crop(clips[g.choice(tr_idx)], g), crop(clips[g.choice(tr_idx)], g)
            n = fake_anomaly(x, g, 0.0 if args.no_curriculum else loudness_at(step, args.steps)) if g.random() < 0.5 else torch.zeros_like(x)
            mix.append(torch.cat([r, x + n]))
            tgt.append(torch.cat([r, x]))
            rsd.append(torch.cat([torch.zeros_like(r), n]))
        loss, _ = run.loss(mix, torch.stack(tgt), torch.stack(rsd), args.res_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"step {step:4d} loss {loss.item():.4f} fake loudness {0.0 if args.no_curriculum else loudness_at(step, args.steps):.2f} ({(time.time() - t0) / step:.2f}s/step)", flush=True)
        if (step in eval_steps) if eval_steps else (step % args.eval_every == 0 or step == args.steps):
            evaluate(step)
    torch.save({k: v for k, v in model.state_dict().items() if ".a" in k[-2:] or ".b" in k[-2:] or k.startswith("embed_anchors")},
               out / "adapter.pt")


if __name__ == "__main__":
    main()
