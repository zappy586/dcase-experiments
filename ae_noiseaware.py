"""Noise-aware autoencoder, trained from scratch on one machine's normal clips (same settings for every machine).

Input: both mics (near, far) as log-power spectrograms, 256 frequency bands (STFT 1024, hop 512, adjacent bins
paired), standardised per band with the machine's training statistics -> (2, 256, 313).
Training augmentation (70% of clips): extra real background is added to both mics: the far-mic recording of
another random normal clip of the same machine, at 0 to -15 dB relative to the far mic, and 3-10 dB quieter in
the near mic than in the far mic (background is closer to the far mic).
Target: the clip's own machine sound, from the ORIGINAL (un-augmented) recording: per band, near-mic power minus
the part the far mic explains (Pn - |Xnf|^2 / Pf, time-averaged), in dB, standardised per band.
Model: conv encoder over frequency and time -> average over the whole clip -> 32-number code (the squeeze)
-> MLP decoder -> 256-band spectrum.
Score (test clip, no augmentation): gap between the clip's own target and the prediction:
  gap-mean  mean squared gap over bands
  gap-max   largest band gap, each band z-scored on 100 held-out normal clips
Reported at the last step; evaluations every --eval_every steps are for watching only.
Usage: uv run python -u ae_noiseaware.py --machine ToyCar
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn as nn

from dcase_ae.efficientad import WavStore, build_cache
from dcase_ae.evaluation import compute_metrics

NFFT, HOP, NB = 1024, 512, 256


class Spec:
    def __init__(self, dev):
        self.win = torch.hann_window(NFFT, device=dev)

    def __call__(self, w):
        """w (B, 2, L) -> input log-power (B, 2, NB, T) and target near-only spectrum (B, NB), both in dB."""
        B = len(w)
        S = torch.stft(w.reshape(2 * B, -1), NFFT, HOP, window=self.win, return_complex=True)[:, 1:]  # drop DC
        S = S.reshape(B, 2, NB, 2, -1)
        P = S.abs().pow(2).sum(3)  # (B, 2, NB, T), adjacent bins paired
        x = 10 * torch.log10(P + 1e-10)
        N, F = S[:, 0], S[:, 1]
        Pn, Pf = N.abs().pow(2).mean(-1), F.abs().pow(2).mean(-1)  # (B, NB, 2) time-averaged
        X = (N * F.conj()).mean(-1)
        resid = (Pn - X.abs().pow(2) / (Pf + 1e-12)).clamp_min(1e-12).sum(-1)  # (B, NB)
        return x, 10 * torch.log10(resid)


class AE(nn.Module):
    """memory > 0: MemAE-style bottleneck (Gong et al. 2019). The code is a query over `memory` learned slots
    (cosine similarity, softmax), weights below `shrink` are zeroed (hard shrinkage) and the rest renormalised;
    the decoder only sees the sparse combination of slots. self.entropy (mean attention entropy) is added to the loss."""

    def __init__(self, code=32, memory=0, shrink=None, retrieve_k=0, chans=(32, 64, 128, 128), double=False, dec_hidden=512,
                 bf16=False, grad_ckpt=False):
        super().__init__()
        self.bf16, self.grad_ckpt = bf16, grad_ckpt
        self.retrieve_k = retrieve_k
        if retrieve_k:
            self.rdec = RetDecoder(code)
        self.memory = memory
        if memory:
            self.mem = nn.Parameter(torch.randn(memory, code) / code**0.5)
            self.shrink = shrink if shrink is not None else 2.0 / memory
        self.entropy, self.active, self.att = torch.zeros(()), torch.zeros(()), None
        def blk(i, o, s):
            layers = [nn.Conv2d(i, o, 3, stride=s, padding=1), nn.BatchNorm2d(o), nn.GELU()]
            if double:  # second convolution at the same resolution
                layers += [nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU()]
            return nn.Sequential(*layers)
        c1, c2, c3, c4 = chans
        self.enc = nn.Sequential(blk(2, c1, (2, 1)), blk(c1, c2, (2, 2)), blk(c2, c3, (2, 2)), blk(c3, c4, (2, 2)))
        self.squeeze = nn.Linear(c4 * (NB // 16), code)  # after averaging over time
        h = dec_hidden
        self.dec = nn.Sequential(nn.Linear(code, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, NB))

    def forward(self, x, return_z=False, bank=None, exclude=None, drop=None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):  # encoder in bf16 (optional)
            h = x
            for blk in self.enc:
                if self.grad_ckpt and torch.is_grad_enabled():
                    h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False)
                else:
                    h = blk(h)
        h = h.float().mean(-1)  # (B, C, NB/16): average over the whole clip
        z = self.squeeze(h.flatten(1))
        z_enc = z
        if self.memory:
            att = torch.softmax(nn.functional.normalize(z, dim=-1) @ nn.functional.normalize(self.mem, dim=-1).T * 10.0, -1)
            att = nn.functional.relu(att - self.shrink) * att / ((att - self.shrink).abs() + 1e-12)  # hard shrinkage
            att = att / (att.sum(-1, keepdim=True) + 1e-12)
            self.entropy = -(att * torch.log(att + 1e-12)).sum(-1).mean()
            self.active = (att > 0).float().sum(-1).mean()
            self.att = att
            z = att @ self.mem
        if self.retrieve_k and bank is not None:
            idx = retrieve(z, bank[0], self.retrieve_k, exclude)
            out = self.rdec(z, bank[0][idx], bank[1][idx], drop)
        else:
            out = self.dec(z)
        return (out, z_enc) if return_z else out


class RetDecoder(nn.Module):
    """Decoder that sees the clip's own code (query) and K retrieved normal training clips (code + spectrum) through
    cross-attention. drop[b] = True hides the neighbours for that sample (the decoder must use the code alone)."""

    def __init__(self, code=32, d=128):
        super().__init__()
        self.q = nn.Linear(code, d)
        self.kv = nn.Linear(code + NB, d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(2 * d, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, NB))

    def forward(self, z, nz, ny_, drop=None):
        q = self.q(z)[:, None]
        kv = self.kv(torch.cat([nz, ny_], -1))
        a = self.attn(q, kv, kv)[0][:, 0]
        if drop is not None:
            a = a * (~drop)[:, None].to(a.dtype)
        return self.out(torch.cat([q[:, 0], a], -1))


def retrieve(z, bank_z, k, exclude=None):
    """indices of the k bank clips with the most similar codes (cosine); exclude[b] = bank position to skip"""
    sim = nn.functional.normalize(z, dim=-1) @ nn.functional.normalize(bank_z, dim=-1).T
    if exclude is not None:
        sim[torch.arange(len(z), device=z.device), exclude] = -float("inf")
    return sim.topk(k, dim=-1).indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--code", type=int, default=32)
    ap.add_argument("--aug_p", type=float, default=0.7)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--target_frac", type=float, default=0.2, help="share of each batch drawn from target-domain clips")
    ap.add_argument("--shift_p", type=float, default=0.0,
                    help="chance that a source training clip is moved a random fraction toward the target domain's tone")
    ap.add_argument("--shift_smooth", type=float, default=4.0, help="Gaussian smoothing of the domain shift across bands")
    ap.add_argument("--domain_norm", type=int, default=1, help="standardise scores with held-out normals of the clip's own domain")
    ap.add_argument("--holdout", type=int, default=1,
                    help="1: hold out 95 source + half the target clips; 0: train on all clips (score statistics in-sample)")
    ap.add_argument("--synth", default="", help="synthetic target samples (diffusion_target.py output)")
    ap.add_argument("--synth_p", type=float, default=0.1, help="share of each batch replaced by synthetic target samples")
    ap.add_argument("--memory", type=int, default=0, help="MemAE memory slots (0 = plain bottleneck)")
    ap.add_argument("--shrink", type=float, default=None, help="hard-shrinkage threshold (default 2/memory)")
    ap.add_argument("--mem_entropy", type=float, default=2e-4, help="weight of the attention-entropy loss")
    ap.add_argument("--reset_every", type=int, default=0, help="memory: every N steps move dead slots to the codes of the "
                    "worst-reconstructed training clips (0 = off)")
    ap.add_argument("--reset_until", type=float, default=0.8, help="memory: last reset at this fraction of training")
    ap.add_argument("--reset_max", type=int, default=8, help="memory: at most this many slots moved per reset round")
    ap.add_argument("--dead_share", type=float, default=0.1, help="memory: dead = running usage below this share of 1/memory")
    ap.add_argument("--usage_w", type=float, default=0.0, help="memory: weight of the usage-balance term "
                    "(KL of batch slot usage from uniform)")
    ap.add_argument("--retrieve", type=int, default=0, help="retrieval-augmented decoder: K nearest training clips (0 = off)")
    ap.add_argument("--ret_drop", type=float, default=0.2, help="share of training samples with neighbours hidden")
    ap.add_argument("--ret_refresh", type=int, default=250, help="recompute the training clips' codes every N steps")
    ap.add_argument("--chans", default="32,64,128,128", help="encoder channels of the 4 blocks")
    ap.add_argument("--double", type=int, default=0, help="two convolutions per encoder block")
    ap.add_argument("--dec_hidden", type=int, default=512)
    ap.add_argument("--bf16", type=int, default=0, help="encoder in bf16 autocast")
    ap.add_argument("--grad_ckpt", type=int, default=0, help="gradient checkpointing per encoder block")
    ap.add_argument("--cache_dir", default="cache", help="waveform cache directory")
    ap.add_argument("--export_dir", default="", help="write anomaly_score / decision_result CSVs (official evaluator format)")
    ap.add_argument("--crop_min", type=float, default=0.0, help="> 0: train on a random crop of each clip, length uniform "
                    "in [crop_min, 1] x clip (one length per batch); the target comes from the same crop")
    ap.add_argument("--mix_p", type=float, default=0.0, help="share of training clips replaced by a power-preserving mix of "
                    "two normal recordings (both mics); target computed from the mix")
    ap.add_argument("--sam", type=float, default=0.0, help="> 0: sharpness-aware minimisation with this radius rho")
    ap.add_argument("--consist", type=float, default=0.0, help="weight of the code-consistency loss: code of the clip with "
                    "added background vs code of the clean clip (stop-gradient), divided by the code variance")
    ap.add_argument("--domain_lora", type=int, default=0, help="> 0: target-domain expert = the trained AE plus a LoRA of "
                    "this rank on the decoder's linear layers (and trainable decoder biases), trained on the target-domain "
                    "training clips only; score = min over the two experts' gaps (no test domain labels needed)")
    ap.add_argument("--lora_steps", type=int, default=500)
    ap.add_argument("--load", default="", help="load a trained AE (model.pt) instead of training; use with --steps 0")
    ap.add_argument("--lora_lr", type=float, default=1e-3)
    ap.add_argument("--whiten", type=float, default=0.0, help="> 0: whitened target (principal components of the training "
                    "targets, each divided by sqrt(eigenvalue + this floor)); 0 = per-band standardisation only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device("cuda")
    out = Path("outputs/ae_noiseaware") / args.name / args.machine
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')}  [ae-noiseaware {args.name}/{args.machine}] {msg}", flush=True)

    log(f"=== {vars(args)}")
    for split in ("train", "test"):
        build_cache("data", args.cache_dir, args.machine, split)
    tr, te = WavStore(args.cache_dir, args.machine, "train"), WavStore(args.cache_dir, args.machine, "test")
    trdom = np.array(tr.meta["domain"])
    src_ids, tgt_ids = [i for i in range(len(tr)) if trdom[i] == 0], [i for i in range(len(tr)) if trdom[i] == 1]
    random.Random(0).shuffle(src_ids)
    random.Random(0).shuffle(tgt_ids)
    if args.holdout:
        n_tv = len(tgt_ids) // 2  # half of the target clips held out (for per-domain score statistics)
        val_idx = src_ids[:95] + tgt_ids[:n_tv]
        fit_src, fit_tgt = src_ids[95:], tgt_ids[n_tv:]
    else:  # train on everything; score statistics come from (in-sample) training clips
        fit_src, fit_tgt = src_ids, tgt_ids
        val_idx = src_ids[:190] + tgt_ids
    fit_idx = fit_src + fit_tgt
    vdom = trdom[val_idx]
    lab, dom = np.array(te.meta["label"]), np.array(te.meta["domain"])
    spec = Spec(dev)

    @torch.no_grad()
    def feats(store, ids):
        X, Y = [], []
        for i in range(0, len(ids), 32):
            x, y = spec(store.get(ids[i : i + 32]).to(dev))
            X.append(x.cpu())
            Y.append(y.cpu())
        return torch.cat(X), torch.cat(Y)

    t0 = time.time()
    Xf, Yf = feats(tr, fit_idx)
    Xv, Yv = feats(tr, val_idx)
    Xt, Yt = feats(te, list(range(len(te))))
    xm, xs = Xf.mean(dim=(0, 3), keepdim=True)[0], Xf.std(dim=(0, 3), keepdim=True)[0] + 1e-3  # (2, NB, 1)
    ym, ys = Yf.mean(0), Yf.std(0) + 1e-3
    nx = lambda X: ((X - xm.to(X.device)) / xs.to(X.device)).float()
    ny = lambda Y: ((Y - ym.to(Y.device)) / ys.to(Y.device)).float()
    if args.whiten > 0:  # rotate onto the principal components of the training targets, divide each by sqrt(eigenvalue + floor)
        Z = (Yf - ym) / ys
        lam, V = torch.linalg.eigh(Z.T @ Z / len(Z))
        Wt = (V / (lam.clamp_min(0) + args.whiten).sqrt()).float()
        log(f"whitened target: eigenvalues {lam.max():.2f} .. {lam.min():.2e}, floor {args.whiten}; "
            f"{int((lam > args.whiten).sum())} of {NB} principal components above the floor")
        ny = lambda Y: (((Y - ym.to(Y.device)) / ys.to(Y.device)) @ Wt.to(Y.device)).float()
    log(f"features ready in {time.time() - t0:.0f}s: input {tuple(Xf.shape[1:])}; train source {len(fit_src)} "
        f"target {len(fit_tgt)} (target share per batch {args.target_frac:.0%}); held out source {int((vdom == 0).sum())} target {int((vdom == 1).sum())}")

    model = AE(args.code, args.memory, args.shrink, args.retrieve, tuple(int(c) for c in args.chans.split(",")),
               bool(args.double), args.dec_hidden, bool(args.bf16), bool(args.grad_ckpt)).to(dev)
    pos_of = {i: p for p, i in enumerate(fit_idx)}  # dataset index -> position in Xf (= position in the bank)
    bank = None

    @torch.no_grad()
    def refresh_bank():
        model.eval()
        zs = [model(nx(Xf[i : i + 64].to(dev)), return_z=True)[1] for i in range(0, len(Xf), 64)]
        model.train()
        return torch.cat(zs), ny(Yf).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=max(args.steps, 1), pct_start=0.05)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=dev))
        log(f"loaded trained AE from {args.load}")
    log(f"parameters {sum(p.numel() for p in model.parameters()):,}")
    if args.synth:
        Sy = torch.cat([torch.load(f)["spec_db"].float() for f in args.synth.split(",")])  # (N, 3, NB, frames) dB
        Xsyn = Sy[:, :2]
        Ysyn = 10 * torch.log10((10 ** (Sy[:, 2] / 10)).mean(-1))  # time-average of near-minus-far power, dB
        log(f"synthetic target samples: {len(Sy)} from {args.synth}, {args.synth_p:.0%} of each batch")

    # domain shift: average tone difference (dB per band, per mic) target minus source, from the training clips,
    # smoothed across bands so only the broad tone change is kept (not the 5 clips' narrow-band details)
    fdom = trdom[fit_idx]
    tone = Xf.mean(-1)  # (N, 2, NB) time-averaged dB
    shift_db = tone[torch.tensor(fdom == 1)].mean(0) - tone[torch.tensor(fdom == 0)].mean(0)  # (2, NB)
    from scipy.ndimage import gaussian_filter1d
    shift_db = torch.tensor(gaussian_filter1d(shift_db.numpy(), args.shift_smooth, axis=1, mode="nearest")).float().to(dev)
    log(f"domain shift (target - source, smoothed): near mic mean {shift_db[0].mean():+.1f} dB, range "
        f"{shift_db[0].min():+.1f}..{shift_db[0].max():+.1f} dB; far mic mean {shift_db[1].mean():+.1f} dB; "
        f"applied to {args.shift_p:.0%} of source clips at a random fraction 0-100%")
    win = torch.hann_window(NFFT, device=dev)

    def apply_shift(x2, frac):
        """x2 (2, L) waveform -> same recording with frac x the domain tone shift applied per band, per mic"""
        S = torch.stft(x2, NFFT, HOP, window=win, return_complex=True)  # (2, 513, T)
        g = torch.ones(2, NFFT // 2 + 1, device=dev)
        g[:, 1:] = (10 ** (frac * shift_db / 20)).repeat_interleave(2, dim=1)  # two bins per band, DC unchanged
        return torch.istft(S * g[..., None], NFFT, HOP, window=win, length=x2.shape[-1])

    @torch.no_grad()
    def predict(X, excl=None, hide=False):
        """excl: bank positions of X's clips (training clips must not retrieve themselves); hide: neighbours hidden"""
        model.eval()
        P = []
        for i in range(0, len(X), 64):
            kw = {}
            if args.retrieve:
                n = len(X[i : i + 64])
                kw = dict(bank=bank, exclude=None if excl is None else torch.tensor(excl[i : i + 64], device=dev),
                          drop=torch.full((n,), hide, dtype=torch.bool, device=dev))
            P.append(model(nx(X[i : i + 64].to(dev)), **kw).cpu())
        model.train()
        return torch.cat(P)

    v_excl = [pos_of[i] for i in val_idx] if args.retrieve else None

    def evaluate(step, final=False, hide=False):
        gv = (predict(Xv, v_excl, hide) - ny(Yv)) ** 2
        gt = (predict(Xt, None, hide) - ny(Yt)) ** 2
        val_mse = gv.mean().item()
        src_v = torch.tensor(vdom == 0)
        sv = {"gap-mean": gv.mean(1), "gap-max": ((gv - gv[src_v].mean(0)) / (gv[src_v].std(0) + 1e-9)).max(1).values}
        st = {"gap-mean": gt.mean(1), "gap-max": ((gt - gv[src_v].mean(0)) / (gv[src_v].std(0) + 1e-9)).max(1).values}
        res = {}
        for k in sv:
            s_t = st[k].numpy().copy()
            if args.domain_norm:
                for d in (0, 1):
                    v = sv[k].numpy()[vdom == d]
                    s_t[dom == d] = (s_t[dom == d] - v.mean()) / (v.std() + 1e-9)
            res[k] = compute_metrics(s_t, lab, dom)
        a, n = lab == 1, lab == 0
        if hide:
            log(f"step {step:5d} | CHECK with neighbours HIDDEN: test gap normal {gt[n].mean():.3f} vs anomaly {gt[a].mean():.3f} | "
                + " | ".join(f"{k} AUC {r['auc_all']:.3f} src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} "
                             f"off {r['official']:.3f}" for k, r in res.items()))
            return res
        log(f"step {step:5d} | val gap {val_mse:.3f} | test gap normal {gt[n].mean():.3f} vs anomaly {gt[a].mean():.3f} | "
            + " | ".join(f"{k} AUC {r['auc_all']:.3f} src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} "
                         f"pAUC {r['pauc']:.3f} off {r['official']:.3f}" for k, r in res.items())
            + ("  (FINAL)" if final else ""))
        if final and args.export_dir and not hide:
            names = te.meta["names"]
            for k in sv:
                d = Path(args.export_dir) / k
                d.mkdir(parents=True, exist_ok=True)
                s_t = st[k].numpy()
                thr = np.percentile(sv[k].numpy(), 90)  # decision threshold: 90th percentile of training-clip scores
                (d / f"anomaly_score_{args.machine}_section_00_test.csv").write_text(
                    "".join(f"{n},{v}" + chr(10) for n, v in zip(names, s_t)))
                (d / f"decision_result_{args.machine}_section_00_test.csv").write_text(
                    "".join(f"{n},{int(v > thr)}" + chr(10) for n, v in zip(names, s_t)))
            log(f"exported anomaly scores to {args.export_dir}/{{gap-mean,gap-max}}")
        return {"val_gap": val_mse, **res}

    hist, losses, t0 = {}, [], time.time()
    usage = torch.zeros(max(args.memory, 1), device=dev)  # running slot usage (starts at 0: never-used slots stay 0)

    @torch.no_grad()
    def reset_dead(step):
        nonlocal usage
        dead = torch.nonzero(usage < args.dead_share / args.memory).flatten()
        if not len(dead):
            log(f"step {step:5d} slot reset: no dead slots")
            return
        model.eval()
        Z, G = [], []
        for i in range(0, len(Xf), 64):
            out, z = model(nx(Xf[i : i + 64].to(dev)), return_z=True)
            Z.append(z)
            G.append((out - ny(Yf[i : i + 64].to(dev))).pow(2).mean(1))
        Z, G = torch.cat(Z), torch.cat(G)
        dead = dead[: args.reset_max]  # move only a few slots per round, so the decoder can adapt
        worst = torch.argsort(G, descending=True)[: len(dead)]
        live = torch.ones(args.memory, dtype=torch.bool, device=dev)
        live[dead] = False
        scale = model.mem.data[live].norm(dim=-1).mean() if live.any() else model.mem.data.norm(dim=-1).mean()
        new = nn.functional.normalize(Z[worst], dim=-1) * scale  # same size as the slots in use
        model.mem.data[dead] = new + 0.01 * scale * torch.randn_like(new)
        usage[dead] = 1.0 / args.memory
        n_tgt = int((trdom[np.asarray(fit_idx)[worst.cpu().numpy()]] == 1).sum())
        model.train()
        log(f"step {step:5d} slot reset: {len(dead)} dead slots moved to the worst-reconstructed training clips "
            f"({n_tgt} of them target-domain clips; their gap {G[worst].mean():.3f} vs average {G.mean():.3f})")
    for step in range(1, args.steps + 1):
        if args.target_frac > 0:
            n_t = sum(rng.random() < args.target_frac for _ in range(args.batch))
            ids = rng.sample(fit_src, args.batch - n_t) + [rng.choice(fit_tgt) for _ in range(n_t)]
        else:
            ids = rng.sample(fit_idx, args.batch)  # natural share: target clips appear as often as any other
        w = tr.get(ids).to(dev)  # (B, 2, L), original recording
        if args.mix_p > 0:
            with torch.no_grad():
                for b in range(len(ids)):
                    if rng.random() < args.mix_p:
                        lam = rng.uniform(0.2, 0.8)
                        w2 = tr.get([rng.choice(fit_idx)])[0].to(dev)
                        w[b] = lam ** 0.5 * w[b] + (1 - lam) ** 0.5 * w2
        if args.crop_min > 0:
            L = int(w.shape[-1] * rng.uniform(args.crop_min, 1.0)) // HOP * HOP
            st = [rng.randrange(w.shape[-1] - L + 1) for _ in ids]
            w = torch.stack([w[b, :, s0 : s0 + L] for b, s0 in enumerate(st)])
        if args.shift_p > 0:
            with torch.no_grad():
                for b, i in enumerate(ids):
                    if trdom[i] == 0 and rng.random() < args.shift_p:
                        w[b] = apply_shift(w[b], rng.random())
        with torch.no_grad():
            _, y = spec(w)  # target from the recording before background augmentation (after any domain shift)
            wa = w.clone()
            for b in range(len(ids)):
                if rng.random() < args.aug_p:
                    other = tr.get([rng.choice(fit_idx)])[0, 1].to(dev)  # far mic of another normal clip
                    other = torch.roll(other, rng.randrange(len(other)))[: wa.shape[-1]]
                    g_far = wa[b, 1].pow(2).mean().sqrt() / (other.pow(2).mean().sqrt() + 1e-9) * 10 ** (-rng.uniform(0, 15) / 20)
                    g_near = g_far * 10 ** (-rng.uniform(3, 10) / 20)
                    wa[b, 1] += g_far * other
                    wa[b, 0] += g_near * other
            x, _ = spec(wa)
        if args.retrieve:
            if bank is None or step % args.ret_refresh == 1:
                bank = refresh_bank()
            drop = torch.rand(len(ids), device=dev) < args.ret_drop
            pred = model(nx(x), bank=bank, exclude=torch.tensor([pos_of[i] for i in ids], device=dev), drop=drop)
            loss = (pred - ny(y)).pow(2).mean()
        else:
            def loss_fn():
                out, z = model(nx(x), return_z=True)
                l = (out - ny(y)).pow(2).mean()
                if args.consist > 0:
                    with torch.no_grad():
                        z_c = model(nx(x_clean), return_z=True)[1]
                    l = l + args.consist * (z - z_c).pow(2).mean() / (z_c.var(0).mean() + 1e-6)
                return l
            if args.consist > 0:
                with torch.no_grad():
                    x_clean, _ = spec(w)
            loss = loss_fn()
            if args.sam > 0:  # SAM: gradient at the worst point within radius rho, applied at the current weights
                opt.zero_grad(set_to_none=True)
                loss.backward()
                ps = [q for q in model.parameters() if q.grad is not None]
                gn = torch.norm(torch.stack([q.grad.norm() for q in ps])) + 1e-12
                eps = [args.sam * q.grad / gn for q in ps]
                with torch.no_grad():
                    for q, e in zip(ps, eps):
                        q.add_(e)
                loss2 = loss_fn()
                opt.zero_grad(set_to_none=True)
                loss2.backward()
                with torch.no_grad():
                    for q, e in zip(ps, eps):
                        q.sub_(e)
                opt.step()
                sched.step()
                losses.append(loss.item())
                loss = None
        if args.memory:
            loss = loss + args.mem_entropy * model.entropy
            p_use = model.att.mean(0)  # slot usage in this batch
            if args.usage_w:
                loss = loss + args.usage_w * (p_use * torch.log(p_use * args.memory + 1e-12)).sum()
            usage = 0.99 * usage + 0.01 * p_use.detach()
        if args.synth:
            n_s = sum(rng.random() < args.synth_p for _ in range(args.batch))
            if n_s:
                j = torch.tensor([rng.randrange(len(Xsyn)) for _ in range(n_s)])
                ls = (model(nx(Xsyn[j].to(dev))) - ny(Ysyn[j].to(dev))).pow(2).mean()
                loss = (loss * (args.batch - n_s) + ls * n_s) / args.batch
        if loss is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        if step % 50 == 0:
            log(f"step {step:5d} loss {np.mean(losses[-50:]):.4f} {(time.time() - t0) / step:.2f}s/step "
                f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
                + (f" | memory slots active per clip {model.active.item():.1f} of {args.memory}" if args.memory else ""))
        if step % args.eval_every == 0 and step != args.steps:
            hist[f"step{step}"] = evaluate(step)
            if args.retrieve:
                hist[f"step{step}_hidden"] = evaluate(step, hide=True)
        if args.memory and args.reset_every and step % args.reset_every == 0 and step <= args.reset_until * args.steps:
            reset_dead(step)
    def augment(w):
        wa = w.clone()
        for b in range(len(w)):
            if rng.random() < args.aug_p:
                other = tr.get([rng.choice(fit_idx)])[0, 1].to(dev)
                other = torch.roll(other, rng.randrange(len(other)))[: wa.shape[-1]]
                g_far = wa[b, 1].pow(2).mean().sqrt() / (other.pow(2).mean().sqrt() + 1e-9) * 10 ** (-rng.uniform(0, 15) / 20)
                g_near = g_far * 10 ** (-rng.uniform(3, 10) / 20)
                wa[b, 1] += g_far * other
                wa[b, 0] += g_near * other
        return wa

    def domain_expert(base):
        """Target-domain expert: copy of the trained AE + rank-r LoRA on the decoder, trained on target clips only."""
        import copy

        class LoRALinear(nn.Module):
            def __init__(self, lin, r):
                super().__init__()
                self.lin = lin
                self.A = nn.Parameter(torch.randn(r, lin.in_features, device=lin.weight.device) / lin.in_features ** 0.5)
                self.B = nn.Parameter(torch.zeros(lin.out_features, r, device=lin.weight.device))

            def forward(self, x):
                return self.lin(x) + x @ self.A.T @ self.B.T

        tgt_clips = [i for i in fit_idx if trdom[i] == 1]

        def make_expert(clips, quiet=False):
            expert = copy.deepcopy(base)
            for p_ in expert.parameters():
                p_.requires_grad_(False)
            for i, layer in enumerate(expert.dec):
                if isinstance(layer, nn.Linear):
                    layer.bias.requires_grad_(True)
                    expert.dec[i] = LoRALinear(layer, args.domain_lora)
            train_p = [p_ for p_ in expert.parameters() if p_.requires_grad]
            expert.eval()  # BatchNorm statistics of the shared encoder stay fixed
            opt_e = torch.optim.AdamW(train_p, lr=args.lora_lr, weight_decay=0.0)
            if not quiet:
                log(f"domain expert: LoRA rank {args.domain_lora} on the decoder, {sum(p_.numel() for p_ in train_p):,} "
                    f"trainable parameters, {len(clips)} target clips, {args.lora_steps} steps")
            for st in range(1, args.lora_steps + 1):
                w = tr.get(clips).to(dev)
                with torch.no_grad():
                    _, y = spec(w)
                    x, _ = spec(augment(w))
                l = (expert(nx(x)) - ny(y)).pow(2).mean()
                opt_e.zero_grad(set_to_none=True)
                l.backward()
                opt_e.step()
                if st % 100 == 0 and not quiet:
                    log(f"domain expert step {st:4d} loss {l.item():.4f}")
            return expert

        expert = make_expert(tgt_clips)

        @torch.no_grad()
        def gap(net, X, Y):
            net.eval()
            return torch.cat([(net(nx(X[i : i + 64].to(dev))).cpu() - ny(Y[i : i + 64])).pow(2).mean(1) for i in range(0, len(X), 64)])
        gs, ge = gap(base, Xt, Yt), gap(expert, Xt, Yt)
        # scales: each expert's typical gap on its own domain's training clips. Source expert: in-sample (the shared AE
        # saw them). Target expert: in-sample, and leave-one-out (LoRA fit on 9 target clips, gap on the 10th).
        pos = {i: k for k, i in enumerate(fit_idx)}
        src_pos = [pos[i] for i in fit_idx if trdom[i] == 0][:300]
        tgt_pos = [pos[i] for i in tgt_clips]
        c_s = gap(base, Xf[src_pos], Yf[src_pos]).median()
        c_t_in = gap(expert, Xf[tgt_pos], Yf[tgt_pos]).median()
        loo = []
        for j, i in enumerate(tgt_clips):
            e_j = make_expert([c for c in tgt_clips if c != i], quiet=True)
            loo.append(gap(e_j, Xf[tgt_pos[j] : tgt_pos[j] + 1], Yf[tgt_pos[j] : tgt_pos[j] + 1])[0])
        c_t_loo = torch.stack(loo).median()
        log(f"domain expert scales: source expert in-sample {c_s:.4f} | target expert in-sample {c_t_in:.4f}, "
            f"leave-one-out {c_t_loo:.4f}")
        base.train()
        res = {}
        for k, sc in (("source-expert", gs), ("target-expert", ge), ("min raw", torch.minimum(gs, ge)),
                      ("min scaled in-sample", torch.minimum(gs / c_s, ge / c_t_in)),
                      ("min scaled LOO", torch.minimum(gs / c_s, ge / c_t_loo))):
            res[k] = compute_metrics(sc.numpy(), lab, dom)
        picks = (ge / c_t_loo < gs / c_s).numpy()
        log("domain experts | " + " | ".join(f"{k} src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} pAUC {r['pauc']:.3f} "
                                              f"off {r['official']:.3f}" for k, r in res.items())
            + f" | with LOO scaling the target expert is chosen for {picks[dom == 1].mean():.0%} of target-domain and {picks[dom == 0].mean():.0%} "
              f"of source-domain test clips  (FINAL)")
        return {"domain_experts": res}

    hist["final"] = evaluate(args.steps, final=True)
    if args.retrieve:
        hist["final_hidden"] = evaluate(args.steps, final=True, hide=True)
    if args.domain_lora > 0:
        hist.update(domain_expert(model))
    (out / "summary.json").write_text(json.dumps(hist, indent=1))
    torch.save(model.state_dict(), out / "model.pt")
    log("=== done")


if __name__ == "__main__":
    main()
