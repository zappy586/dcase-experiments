"""Band-masked AudioMAE reconstruction for anomaly detection (normal clips only, same settings for every machine).

Model: AudioMAE ViT-B (AudioSet-2M pretrained, encoder + 16-block Swin decoder), input 1024 frames x 128 mel,
16x16 patches -> 64 time x 8 frequency rows.
Masking: one whole frequency row (16 mel bins) hidden over the full 10 s; the encoder sees the other 7 rows and the
decoder predicts the hidden row. A steady narrow-band fault cannot leak in from other time frames.
Target: plain log-mel (AudioSet normalisation), not per-patch normalised, so level shifts in a band count.
Training: random row per clip; decoder fully trained, encoder LoRA (rank 8, qkv/proj/fc1), lr 1e-4.
Score: hide each row in turn (8 passes). Per row, the signed error (target - prediction) is averaged over time
per mel bin (random texture errors cancel, a steady offset adds up), squared and summed over the row's 16 bins.
Each row's score is z-normalised on 100 held-out normal clips; clip score = max over rows ('band-max').
Also reported: standard MAE score = mean squared error over all hidden patches ('mse').
Reported at the last step; intermediate evaluations are for watching only.
Usage: uv run python -u mae_band.py --machine ToyCar
"""
import argparse
import functools
import importlib.util
import json
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
np.float = float  # AudioMAE code uses the removed alias
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
AMAE = ROOT / "third_party" / "AudioMAE"

# --- make AudioMAE's code importable with current timm / torch -----------------------------------------------
_src = (AMAE / "timm_patch" / "swin_transformer.py").read_text(encoding="utf-8").replace("from .layers import", "from timm.layers import")
_swin = types.ModuleType("timm.models.swin_transformer")
exec(compile(_src, str(AMAE / "timm_patch" / "swin_transformer.py"), "exec"), _swin.__dict__)
sys.modules["timm.models.swin_transformer"] = _swin
_misc = types.ModuleType("util.misc")
_misc.concat_all_gather = lambda x: x
sys.modules["util.misc"] = _misc
sys.path.insert(0, str(AMAE))
import models_mae  # noqa: E402
sys.path.remove(str(AMAE))
_Block = models_mae.Block
models_mae.Block = lambda *a, qk_scale=None, **k: _Block(*a, **k)  # newer timm dropped qk_scale
sys.path.insert(0, str(ROOT))
from clora_ad import fbank  # noqa: E402
from dcase_ae.efficientad import WavStore, build_cache  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402

TR, FR, P = 64, 8, 16


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r=8):
        super().__init__()
        self.base, self.scale = base, 2.0
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T.to(x.dtype)) @ self.B.T.to(x.dtype)


def build_model(dev):
    from safetensors.torch import load_file
    m = models_mae.MaskedAutoencoderViT(
        img_size=(1024, 128), patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=16, decoder_num_heads=16, mlp_ratio=4,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False, audio_exp=True, decoder_mode=1)
    missing, unexpected = m.load_state_dict(load_file(str(ROOT / "pretrained/audiomae/pretrained.safetensors")), strict=False)
    assert not [k for k in missing if "relative_position_index" not in k and "attn_mask" not in k], missing
    return m.to(dev)


def to_input(fbs):
    return torch.stack([(f + 4.2677393) / (4.5689974 * 2) for f in fbs])[:, None]  # (B, 1, 1024, 128)


def reconstruct_row(model, x, row):
    """x (B,1,1024,128); hide frequency row `row` (tensor of B ints) -> pred, target (B, TR, P*P) for that row."""
    B = len(x)
    tok = model.patch_embed(x) + model.pos_embed[:, 1:, :]  # (B, 512, D), index = t*8 + f
    f_of = torch.arange(TR * FR, device=x.device) % FR
    hidden = f_of[None] == row[:, None]  # (B, 512)
    ids_shuffle = torch.argsort(hidden.int(), dim=1, stable=True)  # visible first, hidden last
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    n_vis = TR * FR - TR
    vis = torch.gather(tok, 1, ids_shuffle[:, :n_vis, None].expand(-1, -1, tok.shape[-1]))
    cls = (model.cls_token + model.pos_embed[:, :1, :]).expand(B, -1, -1)
    h = torch.cat([cls, vis], 1)
    for blk in model.blocks:
        h = blk(h)
    h = model.norm(h)
    pred, _, _ = model.forward_decoder(h, ids_restore)  # (B, 512, 256)
    target = model.patchify(x)
    sel = hidden.nonzero()[:, 1].reshape(B, TR)  # token ids of the hidden row, in time order
    g = lambda a: torch.gather(a, 1, sel[..., None].expand(-1, -1, a.shape[-1]))
    return g(pred), g(target)


def band_errors(pred, target):
    """(B, TR, 256) row patches -> signed error averaged over time per mel bin, squared and summed; and plain MSE."""
    e = (target - pred).float().reshape(len(pred), TR, P, P)  # (B, time patch, time-in-patch, mel-in-patch)
    steady = e.mean(dim=(1, 2)).pow(2).sum(-1)  # (B,)
    return steady, e.pow(2).mean(dim=(1, 2, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    out = ROOT / "outputs" / "mae_band" / args.name / args.machine
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')}  [mae_band {args.name}/{args.machine}] {msg}", flush=True)

    log(f"=== {vars(args)}")
    model = build_model(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    loras = []
    for blk in model.blocks:
        for parent, attr in ((blk.attn, "qkv"), (blk.attn, "proj"), (blk.mlp, "fc1")):
            l = LoRA(getattr(parent, attr))
            setattr(parent, attr, l)
            loras.append(l)
    dec = [p for n, p in model.named_parameters() if n.startswith("decoder") or n == "mask_token"]
    for p in dec:
        p.requires_grad_(True)
    params = dec + [p for l in loras for p in (l.A, l.B)]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    log(f"trainable parameters {sum(p.numel() for p in params):,}")

    for split in ("train", "test"):
        build_cache("data", "cache", args.machine, split)
    tr, te = WavStore("cache", args.machine, "train"), WavStore("cache", args.machine, "test")
    idx = list(range(len(tr)))
    random.Random(0).shuffle(idx)
    val_idx, fit_idx = idx[:100], idx[100:]
    lab, dom = np.array(te.meta["label"]), np.array(te.meta["domain"])

    @torch.no_grad()
    def row_scores(store, ids):
        model.eval()
        S, M = np.zeros((len(ids), FR)), np.zeros((len(ids), FR))
        for i in range(0, len(ids), 16):
            x = to_input([fbank(w) for w in store.get(ids[i : i + 16])[:, 0].to(dev).float()])
            for r in range(FR):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred, tgt = reconstruct_row(model, x, torch.full((len(x),), r, device=dev))
                s, m = band_errors(pred, tgt)
                S[i : i + len(x), r], M[i : i + len(x), r] = s.cpu().numpy(), m.cpu().numpy()
        return S, M

    def evaluate(step, final=False):
        t0 = time.time()
        (Sv, Mv), (St, Mt) = row_scores(tr, val_idx), row_scores(te, list(range(len(te))))
        z = lambda a, v: (a - v.mean(0)) / (v.std(0) + 1e-9)
        res = {"band-max": compute_metrics(z(St, Sv).max(1), lab, dom),
               "band-mean": compute_metrics(z(St, Sv).mean(1), lab, dom),
               "mse": compute_metrics(Mt.mean(1), lab, dom)}
        log(f"step {step:5d} | " + " | ".join(f"{k} AUC {r['auc_all']:.3f} src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} "
                                              f"pAUC {r['pauc']:.3f} off {r['official']:.3f}" for k, r in res.items())
            + f" | val mse {Mv.mean():.4f} ({time.time() - t0:.0f}s){'  (FINAL)' if final else ''}")
        return res

    hist = {"step0": evaluate(0)}
    rng = random.Random(args.seed + 1)
    t0, losses = time.time(), []
    for step in range(1, args.steps + 1):
        model.train()
        ids = rng.sample(fit_idx, args.batch)
        with torch.no_grad():
            x = to_input([fbank(w) for w in tr.get(ids)[:, 0].to(dev).float()])
        rows = torch.tensor([rng.randrange(FR) for _ in ids], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, tgt = reconstruct_row(model, x, rows)
        loss = (pred.float() - tgt.float()).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0:
            log(f"step {step:5d} loss {np.mean(losses[-50:]):.4f} {(time.time() - t0) / step:.2f}s/step "
                f"GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if step % args.eval_every == 0 and step != args.steps:
            hist[f"step{step}"] = evaluate(step)
    hist["final"] = evaluate(args.steps, final=True)
    (out / "summary.json").write_text(json.dumps(hist, indent=1))
    log("=== done")


if __name__ == "__main__":
    main()
