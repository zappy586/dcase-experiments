"""CLoRA fine-tuning of EAT for anomaly detection (per machine, same settings for every machine).

Translation of contrastive LoRA (OB-SAE/contrastive_lora) to anomaly detection:
  general text      -> this machine's normal training clips
  task text         -> the change a synthetic anomaly causes: input(bumped clip) - input(same clip)
  LoRA              -> on every block's attn.qkv, attn.proj and mlp.fc1 (fc2 skipped, like down_proj)
  A (rank r)        -> frozen, computed per module from the base model's input activations x:
                       clora       top solutions of S_delta a = lambda (S_norm + eps I) a   (anomaly-loud, normal-silent)
                       clora-null  bottom eigenvectors of S_norm                            (normal-silent)
                       lora        ordinary LoRA, A trainable (control)
                       head        no adapter, only the head trains (control)
                       S_norm = E[x x^T] over normal tokens, S_delta = E[d d^T], d = paired difference
  B                 -> zero-initialised, trained (so the model starts identical to EAT)
  objective         -> classify clean vs bumped copy of the same normal clip; head = per frequency row, time-averaged
                       last-layer token minus the machine's normal row mean (/ std) -> shared MLP -> max over rows
  anomaly score     -> head logit on the test clip; reported at the last step (no checkpoint selection)

Usage: uv run python -u clora_ft.py --name v1 --machines ToyCar --modes head,lora,clora,clora-null
Log:   outputs/clora_ft/<name>/train.log
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn

from clora_ad import bump, fbank, load_eat
from dcase_ae.efficientad import WavStore, build_cache
from dcase_ae.evaluation import compute_metrics

DIM = 768
TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, A: torch.Tensor, train_a: bool, scale: float):
        super().__init__()
        self.base, self.scale = base, scale
        self.A = nn.Parameter(A.clone(), requires_grad=train_a)
        self.B = nn.Parameter(torch.zeros(base.out_features, A.shape[0], device=A.device))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T.to(x.dtype)) @ self.B.T.to(x.dtype)


class EscMixer:
    """Mixes a random ESC-50 recording into near-mic waveforms, 15-35 dB below the clip.
    70%: the 5 s recording placed once at a random time; 30%: looped over the whole clip.
    ESC-50 folds 1-4 for 'train', fold 5 for 'val' (sounds never used in training)."""

    def __init__(self, dev, root="data_ext/ESC-50-master", cache="cache/esc50_16k.npz"):
        cache = Path(cache)
        if not cache.exists():
            import csv
            import soundfile as sf
            import torchaudio
            rows = list(csv.DictReader(open(Path(root) / "meta" / "esc50.csv")))
            wavs, folds = [], []
            for r in rows:
                w, sr = sf.read(Path(root) / "audio" / r["filename"], dtype="float32")
                w = torchaudio.functional.resample(torch.from_numpy(w), sr, 16000).numpy()
                wavs.append(np.pad(w, (0, max(0, 80000 - len(w))))[:80000])
                folds.append(int(r["fold"]))
            np.savez(cache, wav=np.stack(wavs).astype(np.float16), fold=np.array(folds))
        d = np.load(cache)
        self.wav = torch.from_numpy(d["wav"]).to(dev)
        self.ids = {"train": np.flatnonzero(d["fold"] != 5).tolist(), "val": np.flatnonzero(d["fold"] == 5).tolist()}

    @staticmethod
    def _active_rms(x):
        fr = x[: len(x) // 400 * 400].reshape(-1, 400).pow(2).mean(1)
        return fr[fr > 0.01 * fr.max()].mean().sqrt() + 1e-8

    def __call__(self, w, rng, split):
        out = []
        for x in w:
            e = self.wav[rng.choice(self.ids[split])].float()
            L = len(x)
            if rng.random() < 0.7:
                y = torch.zeros(L, device=x.device)
                t0 = rng.randrange(0, L - len(e))
                y[t0 : t0 + len(e)] = e
            else:
                y = torch.roll(e.repeat(L // len(e) + 1)[:L], rng.randrange(0, len(e)))
            snr = rng.uniform(15, 35)
            g = x.pow(2).mean().sqrt() / self._active_rms(e) * 10 ** (-snr / 20)
            out.append(x + g * y)
        return out


class RowHead(nn.Module):
    """Per frequency row: time-averaged last-layer token, standardised by this machine's normal row statistics,
    -> shared MLP -> row score; clip logit = max over the 8 rows (nonlinear, so raised and lowered bands both count)."""

    def __init__(self, mu, sd, hidden=256):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.mlp = nn.Sequential(nn.Linear(DIM, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, tok):  # tok (B, 512, DIM) without the extra token
        rows = tok.float().reshape(len(tok), 64, 8, DIM).mean(1)  # (B, 8, DIM)
        return self.mlp((rows - self.mu) / self.sd).squeeze(-1).amax(1)


@torch.no_grad()
def row_stats(model, store, ids, dev):
    rows = []
    for i in range(0, len(ids), 8):
        w = store.get(ids[i : i + 8])[:, 0].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok = model.encode(to_input([fbank(x) for x in w]))[:, 1:]
        rows.append(tok.float().reshape(len(w), 64, 8, DIM).mean(1))
    rows = torch.cat(rows)
    return rows.mean(0), rows.std(0) + 1e-4


def modules_of(model):
    out = {}
    for i, blk in enumerate(model.blocks):
        for t in TARGETS:
            parent, attr = blk, t.split(".")
            for a in attr[:-1]:
                parent = getattr(parent, a)
            out[f"{i}.{t}"] = (parent, attr[-1])
    return out


def to_input(fbs):
    return torch.stack([(f + 4.268) / (4.569 * 2) for f in fbs])[:, None]


@torch.no_grad()
def input_moments(model, store, ids, rng, dev, perturb):
    """Per adapted module: S_norm = E[x x^T] over normal tokens, S_delta = E[d d^T] over paired differences."""
    mods = modules_of(model)
    grab, hooks = {}, []
    for k, (parent, attr) in mods.items():
        hooks.append(getattr(parent, attr).register_forward_hook(lambda m, a, o, k=k: grab.__setitem__(k, a[0])))
    acc = {k: [torch.zeros(DIM, DIM, device=dev, dtype=torch.float64) for _ in range(2)] + [0] for k in mods}
    for i in range(0, len(ids), 8):
        w = store.get(ids[i : i + 8])[:, 0].to(dev)
        clean = [fbank(x) for x in w]
        pert = perturb(w, clean, rng, "train")
        model.encode(to_input(clean))
        xc = {k: v.double().reshape(-1, DIM) for k, v in grab.items()}
        model.encode(to_input(pert))
        for k in mods:
            x = xc[k]
            d = grab[k].double().reshape(-1, DIM) - x
            acc[k][0] += x.T @ x
            acc[k][1] += d.T @ d
            acc[k][2] += len(x)
    for h in hooks:
        h.remove()
    return {k: (v[0] / v[2], v[1] / v[2]) for k, v in acc.items()}


def make_A(S, Sd, kind, rank, rng_np):
    S, Sd = S.cpu().numpy(), Sd.cpu().numpy()
    Sr = S + 1e-3 * np.trace(S) / DIM * np.eye(DIM)
    if kind == "clora":
        lam, V = scipy.linalg.eigh(Sd, Sr, subset_by_index=[DIM - rank, DIM - 1])
        A, ratio = np.linalg.qr(V)[0].T, float(lam.mean())  # orthonormal rows, as in CLoRA
    elif kind == "clora-null":
        A, ratio = np.linalg.eigh(Sr)[1][:, :rank].T, float("nan")
    else:  # lora: ordinary random init (kaiming-uniform-like), trained
        A, ratio = rng_np.uniform(-1, 1, (rank, DIM)) / np.sqrt(DIM), float("nan")
    return torch.tensor(A, dtype=torch.float32), ratio


def energy_share(A, S):
    """fraction of normal-token energy along A (orthonormal rows)"""
    return float(torch.trace(A.double() @ S @ A.double().T) / torch.trace(S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--machines", default="ToyCar")
    ap.add_argument("--modes", default="head,lora,clora,clora-null")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16, help="clean clips per step (each also gets a bumped copy)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--basis_clips", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anomaly", default="bump", choices=["bump", "esc50"],
                    help="bump: Gaussian mel bump; esc50: real ESC-50 sound mixed into the near-mic waveform")
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = Path("outputs/clora_ft") / args.name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"=== run '{args.name}': {vars(args)}")
    summary = {}
    for m in args.machines.split(","):
        for split in ("train", "test"):
            build_cache("data", "cache", m, split)
        tr, te = WavStore("cache", m, "train"), WavStore("cache", m, "test")
        rng = random.Random(args.seed)
        idx = list(range(len(tr)))
        rng.shuffle(idx)
        val_idx, fit_idx = idx[:100], idx[100:]
        lab, dom = np.array(te.meta["label"]), np.array(te.meta["domain"])

        model, _ = load_eat(dev)
        for p in model.parameters():
            p.requires_grad_(False)
        t0 = time.time()
        if args.anomaly == "esc50":
            mixer = EscMixer(dev)
            perturb = lambda w, clean, r, split: [fbank(x) for x in mixer(w, r, split)]
        else:
            perturb = lambda w, clean, r, split: [bump(f, r) for f in clean]
        moments = input_moments(model, tr, fit_idx[: args.basis_clips], random.Random(args.seed + 1), dev, perturb)
        log(f"--- {m}: input moments of {len(moments)} modules from {args.basis_clips} clips in {time.time() - t0:.0f}s")
        mu, sd = row_stats(model, tr, fit_idx[: args.basis_clips], dev)  # base-model normal row statistics

        # fixed validation bumps: synthetic-detection accuracy on held-out normal clips
        vrng = random.Random(123)
        v_clean, v_pert = [], []
        for i in range(0, len(val_idx), 8):
            vw = tr.get(val_idx[i : i + 8])[:, 0].to(dev)
            c = [fbank(x) for x in vw]
            v_clean += [f.cpu() for f in c]
            v_pert += [f.cpu() for f in perturb(vw, c, vrng, "val")]
        t_fb = [fbank(x.to(dev)).cpu() for x in te.get(list(range(len(te))))[:, 0]]

        summary[m] = {}
        for mode in args.modes.split(","):
            torch.manual_seed(args.seed)
            model, _ = load_eat(dev)
            for p in model.parameters():
                p.requires_grad_(False)
            rng_np = np.random.default_rng(args.seed)
            loras, ratios, shares = [], [], []
            if mode != "head":
                for k, (parent, attr) in modules_of(model).items():
                    S, Sd = moments[k]
                    A, ratio = make_A(S, Sd, mode, args.rank, rng_np)
                    ratios.append(ratio)
                    shares.append(energy_share(A.to(dev), S) if mode != "lora" else float("nan"))
                    lin = LoRALinear(getattr(parent, attr), A.to(dev), train_a=(mode == "lora"), scale=args.scale)
                    setattr(parent, attr, lin)
                    loras.append(lin)
                log(f"  [{mode}] {len(loras)} adapters, rank {args.rank}; bump/normal ratio in A {np.nanmean(ratios):.2f}x; "
                    f"normal-energy share in A {np.nanmean(shares) * 100:.3f}% (random {args.rank / DIM * 100:.2f}%)")
            head = RowHead(mu, sd).to(dev)
            params = list(head.parameters()) + [p for l in loras for p in l.parameters() if p.requires_grad]
            log(f"  [{mode}] trainable parameters: {sum(p.numel() for p in params):,}")
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
            warm = max(1, args.steps // 20)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / args.steps))))

            def logits(fbs, train=False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tok = model.encode(to_input(fbs))[:, 1:]
                return head(tok)

            @torch.no_grad()
            def score(fbs):
                return torch.cat([logits([f.to(dev) for f in fbs[i : i + 16]]) for i in range(0, len(fbs), 16)]).cpu().numpy()

            def evaluate(step, final=False):
                sv_c, sv_p = score(v_clean), score(v_pert)
                acc = float(((sv_p > 0).mean() + (sv_c <= 0).mean()) / 2)
                st = score(t_fb)
                r = compute_metrics(st, lab, dom)
                log(f"  [{mode}] step {step:5d} | synthetic val acc {acc:.3f} | test AUC {r['auc_all']:.3f} src {r['auc_source']:.3f} "
                    f"tgt {r['auc_target']:.3f} pAUC {r['pauc']:.3f} official {r['official']:.3f}{'  (FINAL)' if final else ''}")
                return r

            brng = random.Random(args.seed + 2)
            evaluate(0)
            t0, losses = time.time(), []
            for step in range(1, args.steps + 1):
                ids = brng.sample(fit_idx, args.batch)
                w = tr.get(ids)[:, 0].to(dev)
                with torch.no_grad():
                    clean = [fbank(x) for x in w]
                    pert = perturb(w, clean, brng, "train")
                y = torch.cat([torch.zeros(len(clean)), torch.ones(len(pert))]).to(dev)
                loss = nn.functional.binary_cross_entropy_with_logits(logits(clean + pert, train=True), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                losses.append(loss.item())
                if step % 50 == 0:
                    log(f"  [{mode}] step {step:5d} loss {np.mean(losses[-50:]):.4f} ({(time.time() - t0) / step:.2f}s/step)")
                if step % args.eval_every == 0 and step != args.steps:
                    evaluate(step)
            summary[m][mode] = evaluate(args.steps, final=True)
            del model, head, opt
            torch.cuda.empty_cache()
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log("=== final-step results (AUC / official):")
    for m, r in summary.items():
        log(f"  {m:11s} " + "  ".join(f"{k} {v['auc_all']:.3f}/{v['official']:.3f}" for k, v in r.items()))
    log("=== done")


if __name__ == "__main__":
    main()
