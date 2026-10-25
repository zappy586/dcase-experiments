"""Embedding-space denoising with a pretrained Audio-JEPA (normal clips only, same settings for every machine).

Audio-JEPA (ltuncay/Audio-JEPA, ICME 2025): ViT-B context encoder, EMA target encoder, 6-layer predictor; input
10 s at 32 kHz -> kaldi log-mel 256 frames x 128 mel -> 16x16 patches -> 16 time x 8 frequency = 128 tokens.
Encoders frozen; only the predictor trains.

Training: context tokens of a normal clip are pushed with noise ("synthetic anomalies" in embedding space):
  row    one steady random vector added to every time step of 1-2 random frequency rows
  clip   one steady random vector added to every token
  token  independent noise per token
strength ~ U(0.2, 1.5) x the token RMS; 25% of context tokens are also hidden (JEPA masking). The predictor must
output the clean target-encoder embeddings at all 128 positions (Audio-JEPA's own loss: cosine distance to
per-token standardised targets).
Validation:
  pushback  on held-out normal clips, loss with a pushed context vs with the clean context (closer = pushed back)
  test      real test clips, clean full context; score = how far the predictor moves the clip from its own
            target embedding:
              push-mean    mean per-token loss
              push-steady  per frequency row, signed difference averaged over time, squared; z-scored on held-out
                           normal clips; max over rows
Reported at the last step; step 0 = the pretrained predictor untouched.
Usage: uv run python -u jepa_ad.py --machine ToyCar
"""
import argparse
import json
import random
import sys
import time
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "third_party" / "Audio-JEPA"
sys.path.insert(0, str(ROOT))
from dcase_ae.efficientad import WavStore, build_cache  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402

TT, FF, D = 16, 8, 768  # time patches, frequency patches, dim (BEATs: 62 x 8 x 768)


def load_beats(dev):
    """Frozen BEATs (iter3) as both context and target encoder; a fresh predictor of Audio-JEPA's architecture."""
    import glob
    sys.path.insert(0, str(ROOT / "third_party" / "beats"))
    from BEATs import BEATs, BEATsConfig
    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)  # Microsoft BEATs release
    m = BEATs(BEATsConfig(ck["cfg"]))
    m.load_state_dict(ck["model"])
    m = m.to(dev).eval()
    VisionTransformerPredictor = _ajepa_classes()[1]
    pred = VisionTransformerPredictor(num_patches_h=62, num_patches_w=8, embed_dim=D, predictor_embed_dim=384,
                                      depth=6, num_heads=12, mlp_ratio=4.0, use_flash_attn=False).to(dev)
    return m, pred


def _ajepa_classes():
    class MHA(nn.Module):  # torch-native stand-in for flash_attn MHA, same parameter names
        def __init__(self, embed_dim, num_heads, dropout=0.0, qkv_proj_bias=True, **_):
            super().__init__()
            self.h, self.qkv = num_heads, nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_proj_bias)
            self.proj = nn.Linear(embed_dim, embed_dim)

        def forward(self, x):
            b, n, d = x.shape
            q, k, v = self.qkv(x).reshape(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4).unbind(0)
            return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))

    def mod(name):
        m = types.ModuleType(name)
        m.__spec__ = ModuleSpec(name, loader=None)
        return m
    fa, fam, mha = mod("flash_attn"), mod("flash_attn.modules"), mod("flash_attn.modules.mha")
    mha.MHA = MHA
    sys.modules.update({"flash_attn": fa, "flash_attn.modules": fam, "flash_attn.modules.mha": mha})
    for name in ("src", "src.utils", "src.models", "src.models.components", "src.masks", "src.masks.components"):
        m = types.ModuleType(name)
        m.__path__ = [str(SRC / name.replace(".", "/"))]
        sys.modules[name] = m
    sys.path.insert(0, str(SRC))
    from src.models.components.vision_transformer import VisionTransformer, VisionTransformerPredictor
    return VisionTransformer, VisionTransformerPredictor


def load_jepa(dev):
    VisionTransformer, VisionTransformerPredictor = _ajepa_classes()
    sd = torch.load(ROOT / "pretrained/audio_jepa/JEPA.ckpt", map_location="cpu", weights_only=False)  # MIT release
    sd = sd.get("state_dict", sd)
    enc_kw = dict(input_size=(256, 128), patch_size=(16, 16), in_chans=1, embed_dim=D, depth=12, num_heads=12,
                  mlp_ratio=4.0, use_flash_attn=False)
    enc, tgt = VisionTransformer(**enc_kw), VisionTransformer(**enc_kw)
    pred = VisionTransformerPredictor(num_patches_h=TT, num_patches_w=FF, embed_dim=D, predictor_embed_dim=384,
                                      depth=6, num_heads=12, mlp_ratio=4.0, use_flash_attn=False)
    for name, m in (("encoder", enc), ("target_encoder", tgt), ("predictor", pred)):
        part = {k[len(name) + 1 :]: v for k, v in sd.items() if k.startswith(name + ".")}
        m.load_state_dict(part, strict=True)
    return enc.to(dev).eval(), tgt.to(dev).eval(), pred.to(dev)


def melspec(w16k, dev):
    """(B, L) 16 kHz near-mic -> (B, 1, 256, 128) as Audio-JEPA's pipeline (peak-normalised, 32 kHz, kaldi fbank)."""
    w = torchaudio.functional.resample(w16k.to(dev).float(), 16000, 32000)
    out = []
    for x in w:
        x = x - x.mean()
        x = x / x.abs().max() + 1e-6
        s = torchaudio.compliance.kaldi.fbank(x[None] - x.mean(), sample_frequency=32000, frame_length=97.65625,
                                              frame_shift=39.0625, num_mel_bins=128, low_freq=20, high_freq=16000,
                                              use_log_fbank=True, window_type="hanning")
        s = F.pad(s, (0, 0, 0, max(0, 256 - len(s))))[:256]
        out.append(s)
    return torch.stack(out)[:, None]


def std_target(t):
    return (t - t.mean(-1, keepdim=True)) / (t.var(-1, keepdim=True) + 1e-6).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder", default="audiojepa", choices=["audiojepa", "beats"])
    ap.add_argument("--push", default="random", choices=["random", "general"],
                    help="random: isotropic noise; general: random mixes of the directions real-world sounds (ESC-50) "
                         "use and this machine's normal tokens do not, at the spread real sounds have along them")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--analyze_only", action="store_true", help="report fault overlap with the push directions, then exit")
    ap.add_argument("--clean_frac", type=float, default=0.4, help="share of training clips left completely unpushed")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    global TT
    out = ROOT / "outputs" / "jepa_ad" / f"{args.name}_{args.encoder}_{args.push}" / args.machine
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')}  [jepa-{args.encoder} {args.name}/{args.machine}] {msg}", flush=True)

    log(f"=== {vars(args)}")
    if args.encoder == "beats":
        enc, pred = load_beats(dev)
        tgt, TT = enc, 62
    else:
        enc, tgt, pred = load_jepa(dev)
    for p in list(enc.parameters()) + list(tgt.parameters()):
        p.requires_grad_(False)
    for split in ("train", "test"):
        build_cache("data", "cache", args.machine, split)
    tr, te = WavStore("cache", args.machine, "train"), WavStore("cache", args.machine, "test")
    idx = list(range(len(tr)))
    random.Random(0).shuffle(idx)
    val_idx, fit_idx = idx[:100], idx[100:]
    lab, dom = np.array(te.meta["label"]), np.array(te.meta["domain"])

    @torch.no_grad()
    def embed(store, ids, get=None, targets=True):
        """stored on the CPU; batches are moved to the GPU when used"""
        H, T = [], []
        for i in range(0, len(ids), 32):
            w = store.get(ids[i : i + 32])[:, 0] if get is None else get(ids[i : i + 32])
            if args.encoder == "beats":
                h = enc.extract_features(w.to(dev).float(), padding_mask=torch.zeros(w.shape, dtype=torch.bool, device=dev))[0].float()
                h = h[:, : TT * FF]
                H.append(h.cpu())
                if targets:
                    T.append(std_target(h).cpu())
            else:
                x = melspec(w, dev)
                H.append(enc(x).float().cpu())
                if targets:
                    T.append(std_target(tgt(x).float()).cpu())
        return torch.cat(H), (torch.cat(T) if targets else None)  # (N, 128, D) each

    t0 = time.time()
    Hf, Tf = embed(tr, fit_idx)
    Hv, Tv = embed(tr, val_idx)
    Ht, Tt = embed(te, list(range(len(te))))
    rms = Hf.pow(2).mean().sqrt().item()
    log(f"embeddings ready in {time.time() - t0:.0f}s, context token RMS {rms:.3f}")

    all_idx = torch.arange(TT * FF, device=dev)

    if args.push == "general":
        esc = torch.from_numpy(np.load("cache/esc50_16k.npz")["wav"]).float().repeat(1, 2)  # 2000 clips, looped to 10 s
        He, _ = embed(None, list(range(len(esc))), get=lambda ids: esc[ids], targets=False)
        mu = Hf.reshape(len(Hf), TT, FF, D).mean(dim=(0, 1)).to(dev)  # (FF, D) normal mean per frequency row
        Sn = torch.zeros(D, D, device=dev, dtype=torch.float64)
        Sg = torch.zeros(D, D, device=dev, dtype=torch.float64)
        for X, S in ((Hf, Sn), (He, Sg)):
            for i in range(0, len(X), 64):
                Z = (X[i : i + 64].to(dev).reshape(-1, TT, FF, D) - mu.to(dev)).reshape(-1, D).double()
                S += Z.T @ Z
            S /= len(X) * TT * FF
        Sn += 1e-3 * torch.trace(Sn) / D * torch.eye(D, device=dev, dtype=torch.float64)
        ev, U = torch.linalg.eigh(Sn)
        W = U / ev.sqrt()  # W^T Sn W = I
        lam, E = torch.linalg.eigh(W.T @ Sg @ W)
        A = (W @ E[:, -args.rank :]).T  # (K, D), a^T Sn a = 1, a^T Sg a = lam
        lam_k = lam[-args.rank :].float().clamp_min(0)
        # push vectors: b_k = Sn a_k, so a push c_k b_k changes the projection a_k^T x by exactly c_k
        # (c_k ~ N(0, lam_k) at strength 1 = the spread real sounds have along a_k)
        Bvec = (A @ Sn).float()
        # -- overlap of the real fault shift with the push directions (test labels: analysis only) --------------
        rowavg = lambda X: X.reshape(len(X), TT, FF, D).mean(1)  # (N, FF, D) time-averaged per frequency row
        Rt = rowavg(Ht).double().to(dev) - mu.double()
        an, nm = torch.tensor(lab == 1), torch.tensor(lab == 0)
        perm = torch.randperm(int(nm.sum()), generator=torch.Generator().manual_seed(0))
        nidx = torch.nonzero(nm).squeeze(1)[perm]
        shifts = {"anomaly vs normal": Rt[an].mean(0) - Rt[nm].mean(0),
                  "normal half vs half (noise)": Rt[nidx[: len(nidx) // 2]].mean(0) - Rt[nidx[len(nidx) // 2 :]].mean(0)}
        n_gt1 = int((lam > 1).sum())
        for name, dsh in shifts.items():
            dw = dsh @ W  # whitened by normal token covariance
            tot = dw.pow(2).sum()
            parts = []
            for K in (32, 64, 128, 256, n_gt1):
                parts.append(f"top-{K}: {(dw @ E[:, -K:]).pow(2).sum() / tot * 100:.1f}% (random {K / D * 100:.1f}%)")
            log(f"overlap, {name} (whitened size {tot.sqrt():.2f}): " + " | ".join(parts))
        log(f"directions where real sounds spread more than normal (lambda > 1): {n_gt1} of {D}")
        # per clip: each test clip's own deviation from the normal mean (time-averaged per row, whitened);
        # energy inside the push directions vs inside random directions; AUC = faulty vs normal clips
        from sklearn.metrics import roc_auc_score
        Zw = Rt @ W  # (N, FF, D) whitened per-clip deviations
        tot_e = Zw.pow(2).sum(dim=(1, 2)).cpu().numpy()
        def sub_e(Q):  # energy in subspace with orthonormal (whitened) basis Q (D, K)
            return (Zw @ Q).pow(2).sum(dim=(1, 2)).cpu().numpy()
        y = lab.astype(int)
        res = {"all directions": roc_auc_score(y, tot_e)}
        for K in (32, 128):
            ep = sub_e(E[:, -K:])
            res[f"push top-{K}"] = roc_auc_score(y, ep)
            res[f"push top-{K} share of clip's deviation"] = roc_auc_score(y, ep / tot_e)
            g = torch.Generator(device="cpu").manual_seed(0)
            ra = [roc_auc_score(y, sub_e(torch.linalg.qr(torch.randn(D, K, generator=g, dtype=torch.float64))[0].to(dev)))
                  for _ in range(10)]
            res[f"random-{K} (mean of 10)"] = float(np.mean(ra))
        log("per-clip AUC (faulty vs normal test clips): " + " | ".join(f"{k} {v:.3f}" for k, v in res.items()))
        if args.analyze_only:
            return
        del He, esc, Sg
        torch.cuda.empty_cache()  # ESC-50 tokens are only needed for the directions
        log(f"general-audio push directions: {args.rank}, real-sound / normal spread along them "
            f"{lam_k.sqrt().mean():.1f}x (min {lam_k.sqrt().min():.1f}x)")

    def vec(shape, s, rng_t):
        """one push vector per leading index: random (isotropic) or a random mix of general-audio directions"""
        if args.push == "general":
            c = torch.randn(*shape, args.rank, device=dev, generator=rng_t) * lam_k.sqrt()
            return s * (c @ Bvec)
        return s * rms * torch.randn(*shape, D, device=dev, generator=rng_t)

    def push(H, rng_t, kind=None, strength=None):
        """add embedding-space 'anomalies' to context tokens H (B, 128, D)"""
        B = len(H)
        Hn = H.clone().reshape(B, TT, FF, D)
        for b in range(B):
            if kind is None and random.random() < args.clean_frac:
                continue  # clean normal clip: the predictor must leave it unchanged
            k = kind or random.choice(["row", "clip", "token"])
            if strength is not None:
                s = strength
            elif args.push == "general":
                s = float(np.exp(random.uniform(np.log(0.02), np.log(0.3))))  # subtle, mostly small
            else:
                s = random.uniform(0.2, 1.5)
            if k == "row":
                for r in random.sample(range(FF), random.choice([1, 2])):
                    Hn[b, :, r] += vec((), s, rng_t)
            elif k == "clip":
                Hn[b] += vec((), s, rng_t)
            else:
                Hn[b] += vec((TT, FF), s, rng_t)
        return Hn.reshape(B, TT * FF, D)

    EVAL_S = 0.1 if args.push == "general" else 1.0  # push strength used for the pushback check
    with torch.no_grad():
        tok = Hv[:8].to(dev)
        rel = {k: ((push(tok, torch.Generator(device=dev).manual_seed(0), k, EVAL_S) - tok).norm(dim=-1).mean() / tok.norm(dim=-1).mean()).item()
               for k in ("clip", "token")}
    log(f"pushback-check push size relative to token size: clip {rel['clip']:.3f}, token {rel['token']:.3f}; "
        f"clean training clips {args.clean_frac:.0%}")

    def predict_all(H, keep=None):
        """predictor output for all 128 positions from context tokens H (optionally only the `keep` positions)"""
        B = len(H)
        if keep is None:
            keep = all_idx[None].expand(B, -1)
        ctx = torch.gather(H, 1, keep[..., None].expand(-1, -1, D))
        return pred(ctx, [keep], [all_idx[None].expand(B, -1)])

    def loss_fn(p, t):  # Audio-JEPA norm_mse: cosine distance per token
        return 2 - 2 * (F.normalize(p, dim=-1) * F.normalize(t, dim=-1)).sum(-1)

    @torch.no_grad()
    def evaluate(step, final=False):
        pred.eval()
        g = torch.Generator(device=dev).manual_seed(123)
        random.seed(123)
        clean = loss_fn(predict_all(Hv.to(dev)), Tv.to(dev)).mean().item()
        pb = {}
        for k in ("row", "clip", "token"):
            pushed = loss_fn(predict_all(push(Hv.to(dev), g, k, EVAL_S)), Tv.to(dev)).mean().item()
            pb[k] = pushed
        res = {}

        def scores(H, T):
            P = predict_all(H.to(dev))
            l = loss_fn(P, T.to(dev))  # (N, 128)
            d = (F.normalize(P, dim=-1) - F.normalize(T.to(dev), dim=-1)).reshape(len(H), TT, FF, D)
            steady = d.mean(1).pow(2).sum(-1)  # (N, FF)
            return l.mean(1).cpu().numpy(), steady.cpu().numpy()
        mv, sv = scores(Hv, Tv)
        mt, st = scores(Ht, Tt)
        res["push-mean"] = compute_metrics(mt, lab, dom)
        res["push-steady"] = compute_metrics(((st - sv.mean(0)) / (sv.std(0) + 1e-9)).max(1), lab, dom)
        a, n = lab == 1, lab == 0
        log(f"step {step:5d} | pushback (held-out normal, val loss): clean {clean:.4f} | pushed row {pb['row']:.4f} "
            f"clip {pb['clip']:.4f} token {pb['token']:.4f}{'  (FINAL)' if final else ''}")
        log(f"           test: mean push normal {mt[n].mean():.4f} vs anomaly {mt[a].mean():.4f} | "
            + " | ".join(f"{k} AUC {r['auc_all']:.3f} src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} "
                         f"pAUC {r['pauc']:.3f} off {r['official']:.3f}" for k, r in res.items()))
        pred.train()
        return {"clean": clean, "pushback": pb, **res}

    opt = torch.optim.AdamW(pred.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    hist = {"step0": evaluate(0)}
    g = torch.Generator(device=dev).manual_seed(args.seed)
    random.seed(args.seed)
    losses, t0 = [], time.time()
    pred.train()
    for step in range(1, args.steps + 1):
        b = torch.randint(0, len(Hf), (args.batch,))
        H, T = Hf[b].to(dev), Tf[b].to(dev)
        Hn = push(H, g)
        keep = torch.stack([torch.randperm(TT * FF, device=dev, generator=g)[: int(0.75 * TT * FF)].sort().values
                            for _ in range(args.batch)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P = predict_all(Hn, keep)
        loss = loss_fn(P.float(), T).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            log(f"step {step:5d} loss {np.mean(losses[-100:]):.4f} {(time.time() - t0) / step:.3f}s/step")
        if step % args.eval_every == 0 and step != args.steps:
            hist[f"step{step}"] = evaluate(step)
    hist["final"] = evaluate(args.steps, final=True)
    (out / "summary.json").write_text(json.dumps(hist, indent=1))
    log("=== done")


if __name__ == "__main__":
    main()
