"""Noise-aware autoencoder with a pretrained AudioSet ConvNeXt-Tiny encoder (Pellegrini et al. 2023,
third_party/audioset-convnext-inf, checkpoint pretrained/convnext_audioset/convnext_tiny_471mAP.pth).

Everything except the encoder is as in ae_noiseaware.py (reference run v4_all7):
  training clips: all 1000 normal clips; 70% get extra background (far mic of another clip, 0..-15 dB at the far mic,
  3-10 dB quieter at the near mic); target = the clip's own machine sound (near minus far-predicted part, time-averaged,
  256 bands, standardised); 32-number code; decoder 32 -> 512 -> 512 -> 256; score = squared gap (gap-mean, gap-max).
Encoder: near and far waveforms (16 -> 32 kHz) each through the ConvNeXt's own front end (log-mel 224 bands, 50 Hz -
  14 kHz, pretrained per-band normalisation) and 4 stages -> last stage 768 x (time) x 7 frequency regions -> average
  over time -> both mics concatenated (2 x 768 x 7) -> LayerNorm -> code layer.
  --mode frozen  the whole encoder is frozen (only LayerNorm, code layer and decoder train)
  --mode lora    + LoRA (rank --rank) on pwconv1 / pwconv2 of stages 3 and 4, and their LayerNorms; stages 1-2 run
                 without gradients
Encoder in bf16 autocast. Reported at the last step; evaluations every --eval_every steps are for watching only.
Usage: uv run python -u ae_convnext.py --machine ToyCar --mode frozen
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party" / "audioset-convnext-inf" / "src"))
from ae_noiseaware import NB, Spec  # noqa: E402
from dcase_ae.efficientad import WavStore, build_cache  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402


class ConvNeXtEncoder(nn.Module):
    def __init__(self, mode, rank, dev):
        super().__init__()
        from audioset_convnext_inf.pytorch.convnext import convnext_tiny
        m = convnext_tiny(pretrained=False, strict=False, drop_path_rate=0.0, after_stem_dim=[252, 56],
                          use_speed_perturb=False)
        ck = torch.load(ROOT / "pretrained/convnext_audioset/convnext_tiny_471mAP.pth", map_location="cpu", weights_only=False)
        r = m.load_state_dict(ck.get("model", ck), strict=False)
        assert not r.missing_keys and not r.unexpected_keys, r
        del m.head_audioset
        for p in m.parameters():
            p.requires_grad_(False)
        self.trainable = []
        if mode == "lora":
            from peft import LoraConfig, inject_adapter_in_model
            targets = [n for n, _ in m.named_modules() if n.startswith(("stages.2.", "stages.3.")) and n.endswith(("pwconv1", "pwconv2"))]
            inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=rank, target_modules=targets), m)
            for n, p in m.named_parameters():
                in_late = n.startswith(("stages.2.", "stages.3."))
                if "lora_" in n or (in_late and ".norm." in n):
                    p.requires_grad_(True)
                    self.trainable.append(p)
        self.m = m.to(dev).eval()
        self.mode = mode

    def train(self, mode=True):  # the pretrained network always stays in eval mode (frozen normalisation statistics)
        super().train(mode)
        self.m.eval()
        return self

    def forward(self, w16):
        """w16 (B, L) 16 kHz -> (B, 768, 7) last-stage features averaged over time"""
        m = self.m
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                x = torchaudio.functional.resample(w16.float(), 16000, 32000)
                x = m.logmel_extractor(m.spectrogram_extractor(x))
                x = m.bn0(x.transpose(1, 3)).transpose(1, 3)
                for i in range(2):
                    x = m.stages[i](m.downsample_layers[i](x))
                if self.mode == "frozen":
                    for i in range(2, 4):
                        x = m.stages[i](m.downsample_layers[i](x))
            if self.mode == "lora":
                for i in range(2, 4):
                    x = m.stages[i](m.downsample_layers[i](x))
        return x.float().mean(2)  # (B, 768, time, 7) -> average over time


class Head(nn.Module):
    def __init__(self, code=32, feat=2 * 768 * 7):
        super().__init__()
        self.norm = nn.LayerNorm(feat)
        self.squeeze = nn.Linear(feat, code)
        self.dec = nn.Sequential(nn.Linear(code, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, NB))

    def forward(self, f):
        return self.dec(self.squeeze(self.norm(f.flatten(1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v11_convnext")
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--mode", choices=["frozen", "lora"], default="frozen")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--aug_p", type=float, default=0.7)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--export_dir", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device("cuda")
    out = ROOT / "outputs" / "ae_convnext" / f"{args.name}_{args.mode}" / args.machine
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')}  [ae-convnext {args.name}_{args.mode}/{args.machine}] {msg}", flush=True)

    log(f"=== {vars(args)}")
    for split in ("train", "test"):
        build_cache("data", args.cache_dir, args.machine, split)
    tr, te = WavStore(args.cache_dir, args.machine, "train"), WavStore(args.cache_dir, args.machine, "test")
    trdom = np.array(tr.meta["domain"])
    src_ids, tgt_ids = [i for i in range(len(tr)) if trdom[i] == 0], [i for i in range(len(tr)) if trdom[i] == 1]
    random.Random(0).shuffle(src_ids)
    fit_idx = list(range(len(tr)))
    val_idx = src_ids[:190] + tgt_ids  # score statistics from (in-sample) training clips, as in the reference run
    vdom = trdom[val_idx]
    lab, dom = np.array(te.meta["label"]), np.array(te.meta["domain"])
    spec = Spec(dev)

    with torch.no_grad():  # targets: the clip's own machine sound (same as the reference)
        Yf = torch.cat([spec(tr.get(fit_idx[i : i + 32]).to(dev))[1].cpu() for i in range(0, len(fit_idx), 32)])
    ym, ys = Yf.mean(0), Yf.std(0) + 1e-3
    ny = lambda Y: ((Y - ym.to(Y.device)) / ys.to(Y.device)).float()

    enc = ConvNeXtEncoder(args.mode, args.rank, dev)
    head = Head().to(dev)
    groups = [{"params": head.parameters(), "lr": args.lr}]
    if enc.trainable:
        groups.append({"params": enc.trainable, "lr": args.lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=args.steps, pct_start=0.05)
    log(f"encoder {sum(p.numel() for p in enc.parameters()) / 1e6:.1f} M parameters, trainable in encoder "
        f"{sum(p.numel() for p in enc.trainable):,}; head {sum(p.numel() for p in head.parameters()):,}")

    def features(w):  # (B, 2, L) -> (B, 2, 768, 7)
        B = len(w)
        f = enc(w.reshape(2 * B, -1))
        return f.reshape(B, 2, *f.shape[1:])

    @torch.no_grad()
    def gaps(store, ids):
        head.eval()
        G = []
        for i in range(0, len(ids), 32):
            w = store.get(ids[i : i + 32]).to(dev)
            G.append((head(features(w)) - ny(spec(w)[1])).pow(2).cpu())
        head.train()
        return torch.cat(G)

    def evaluate(step, final=False):
        gv, gt = gaps(tr, val_idx), gaps(te, list(range(len(te))))
        src_v = torch.tensor(vdom == 0)
        mu, sd = gv[src_v].mean(0), gv[src_v].std(0) + 1e-9
        sv = {"gap-mean": gv.mean(1), "gap-max": ((gv - mu) / sd).max(1).values}
        st = {"gap-mean": gt.mean(1), "gap-max": ((gt - mu) / sd).max(1).values}
        res = {k: compute_metrics(st[k].numpy(), lab, dom) for k in st}
        a, n = lab == 1, lab == 0
        log(f"step {step:5d} | val gap {gv.mean():.3f} | test gap normal {gt[n].mean():.3f} vs anomaly {gt[a].mean():.3f} | "
            + " | ".join(f"{k} AUC {r['auc_all']:.3f} src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} "
                         f"pAUC {r['pauc']:.3f} off {r['official']:.3f}" for k, r in res.items())
            + ("  (FINAL)" if final else ""))
        if final and args.export_dir:
            names = te.meta["names"]
            for k in st:
                d = Path(args.export_dir) / k
                d.mkdir(parents=True, exist_ok=True)
                thr = np.percentile(sv[k].numpy(), 90)
                (d / f"anomaly_score_{args.machine}_section_00_test.csv").write_text(
                    "".join(f"{nm},{v}" + chr(10) for nm, v in zip(names, st[k].numpy())))
                (d / f"decision_result_{args.machine}_section_00_test.csv").write_text(
                    "".join(f"{nm},{int(v > thr)}" + chr(10) for nm, v in zip(names, st[k].numpy())))
        return {"val_gap": float(gv.mean()), **res}

    hist, losses, t0 = {}, [], time.time()
    for step in range(1, args.steps + 1):
        ids = rng.sample(fit_idx, args.batch)
        w = tr.get(ids).to(dev)
        with torch.no_grad():
            _, y = spec(w)  # target from the recording before background augmentation
            wa = w.clone()
            for b in range(len(ids)):
                if rng.random() < args.aug_p:
                    other = tr.get([rng.choice(fit_idx)])[0, 1].to(dev)
                    other = torch.roll(other, rng.randrange(len(other)))
                    g_far = wa[b, 1].pow(2).mean().sqrt() / (other.pow(2).mean().sqrt() + 1e-9) * 10 ** (-rng.uniform(0, 15) / 20)
                    g_near = g_far * 10 ** (-rng.uniform(3, 10) / 20)
                    wa[b, 1] += g_far * other
                    wa[b, 0] += g_near * other
        loss = (head(features(wa)) - ny(y)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 50 == 0:
            log(f"step {step:5d} loss {np.mean(losses[-50:]):.4f} {(time.time() - t0) / step:.2f}s/step "
                f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if step % args.eval_every == 0 and step != args.steps:
            hist[f"step{step}"] = evaluate(step)
    hist["final"] = evaluate(args.steps, final=True)
    (out / "summary.json").write_text(json.dumps(hist, indent=1))
    torch.save({"head": head.state_dict(), "encoder_trainable": [p.detach().cpu() for p in enc.trainable]}, out / "model.pt")
    log("=== done")


if __name__ == "__main__":
    main()
