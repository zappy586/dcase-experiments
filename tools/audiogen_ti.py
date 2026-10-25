"""AudioGen-medium (facebook/audiogen-medium) + textual inversion of a new token for ToyCar's NORMAL machine sound.

Audio everywhere: near-minus-far (near-only residual, as tools/encodec_knn.py), one gain for the whole machine
(training-clip median brought to -23 dBFS), 16 kHz (AudioGen's rate).
  train  : add token <toycar> to AudioGen's T5 text encoder; ONLY its input embedding is trained (everything else
           frozen) to minimise AudioGen's next-token loss on the 1000 normal training clips' EnCodec tokens
           (random 5 s crops, prompts like 'the sound of <toycar>'). Saves outputs/audiogen/toycar_token.pt
  recon  : 'reconstruction' of test clips = continuation: first 3 s of the real clip + text '<toycar>' -> AudioGen writes
           the next 7 s. Also the plain codec round trip. Added to the comparison page next to the clip's other columns.
  anomaly: generate 5 s from prompts built on <toycar> (normal, and subtle-fault wordings), 2 samples each -> page
  score  : AudioGen's average surprise (negative log-likelihood per token) for each test clip under '<toycar>',
           AUC normal vs anomaly (label-free score; labels only to measure)
Runs in envs/audiocraft:  envs/audiocraft/Scripts/python tools/audiogen_ti.py train|recon|anomaly|score
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchaudio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "audiogen"
PAGE = ROOT / "outputs" / "sam_compare"
TOKEN = "<toycar>"  # shown on the page as [toycar] (page reads < > as HTML)
TEMPLATES = ["the sound of {}", "a recording of {}", "{}", "{} running", "audio of {}"]
ANOMALY_PROMPTS = {
    "normal: '<toycar>'": "the sound of <toycar>",
    "broken, subtle deviations": "a broken <toycar> with subtle deviations from a working <toycar>",
    "faint intermittent rattle": "a <toycar> with a faint intermittent rattle",
    "slightly worn grinding gear": "a <toycar> with a slightly worn, grinding gear",
}
LOG = ROOT / "outputs" / "live.log"
LORA = {"rank": 0, "name": ""}  # set from --lora; > 0: LoRA on the LM's attention output and feed-forward layers
LORA_TARGETS = ["out_proj", "linear1", "linear2"]


def tag():
    return f" (LoRA r{LORA['rank']}{' ' + LORA['name'] if LORA['name'] else ''})" if LORA["rank"] else ""


def files():
    r, n = LORA["rank"], ("_" + LORA["name"]) if LORA["name"] else ""
    return (OUT / ("toycar_token.pt" if not r else f"toycar_token_lora_r{r}{n}.pt"), OUT / f"lora_r{r}{n}.pt")


def log(msg):
    line = f"{time.strftime('%H:%M:%S')}  [audiogen-ti] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def residual(w):
    win = torch.hann_window(1024, device=w.device)
    S = torch.stft(w.reshape(-1, w.shape[-1]), 1024, 512, window=win, return_complex=True).reshape(len(w), 2, 513, -1)
    N, Fm = S[:, 0], S[:, 1]
    H = (N * Fm.conj()).mean(-1, keepdim=True) / (Fm.abs().pow(2).mean(-1, keepdim=True) + 1e-12)
    return torch.istft(N - H * Fm, 1024, 512, window=win, length=w.shape[-1])


class Data:
    def __init__(self, machine, dev):
        self.dev = dev
        self.wav = {s: np.load(ROOT / "cache" / f"wav_{machine}_{s}.npy", mmap_mode="r") for s in ("train", "test")}
        self.meta = {s: json.loads((ROOT / "cache" / f"wav_{machine}_{s}.json").read_text()) for s in ("train", "test")}
        with torch.no_grad():
            r = residual(self.raw("train", list(range(100))))
            self.gain = float(10 ** (-23 / 20) / r.pow(2).mean(-1).sqrt().median())

    def raw(self, split, ids):
        return torch.from_numpy(np.stack([self.wav[split][i] for i in ids]).astype(np.float32) / 32768).to(self.dev)

    @torch.no_grad()
    def audio(self, split, ids):  # (B, 1, L) near-minus-far, machine gain, clipped to [-1, 1]
        return (residual(self.raw(split, ids)) * self.gain).clamp(-1, 1)[:, None]


def load_model(dev):
    from audiocraft.models import AudioGen
    m = AudioGen.get_pretrained("facebook/audiogen-medium", device=str(dev))
    cond = m.lm.condition_provider.conditioners["description"]
    tok, t5 = cond.t5_tokenizer, cond.__dict__["t5"]
    if TOKEN not in tok.get_vocab():
        tok.add_tokens([TOKEN])
        t5.resize_token_embeddings(len(tok))
    tid = tok.convert_tokens_to_ids(TOKEN)
    f_tok, f_lora = files()
    if LORA["rank"] and not f_tok.exists():
        f_tok = OUT / "toycar_token.pt"  # LoRA training starts from the textual-inversion token
    if f_tok.exists():
        with torch.no_grad():
            t5.get_input_embeddings().weight[tid] = torch.load(f_tok).to(dev)
    if LORA["rank"]:
        from peft import LoraConfig, inject_adapter_in_model
        names = [n for n, mod in m.lm.transformer.named_modules() if isinstance(mod, torch.nn.Linear) and n.split(".")[-1] in LORA_TARGETS]
        inject_adapter_in_model(LoraConfig(r=LORA["rank"], lora_alpha=LORA["rank"], target_modules=names), m.lm.transformer)
        m.lm.to(dev)
        for n, prm in m.lm.transformer.named_parameters():  # peft creates them in 16-bit here; Adam needs 32-bit
            if "lora_" in n:
                prm.data = prm.data.float()
        if f_lora.exists():
            missing = m.lm.transformer.load_state_dict(torch.load(f_lora), strict=False)
            assert not missing.unexpected_keys, missing.unexpected_keys[:3]
    return m, cond, t5, tid


def cond_tensors(cond, t5, texts):
    """T5 text conditioning WITH gradients (AudioGen's own forward runs it without)."""
    inputs = cond.tokenize(texts)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        emb = t5(**inputs).last_hidden_state
    emb = cond.output_proj(emb.float())
    mask = inputs["attention_mask"]
    return {"description": (emb * mask.unsqueeze(-1), mask)}


def lm_nll(m, codes, ct):
    """mean next-token loss per clip (B,) for codes (B, K, T)"""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m.lm.compute_predictions(codes, [], ct)
    nan = torch.isnan(out.logits).any(-1)
    if (nan & out.mask).float().mean() > 0.01:
        raise FloatingPointError(f"NaN logits at {(nan & out.mask).float().mean():.1%} of valid positions")
    mask = out.mask & ~nan
    logits = torch.nan_to_num(out.logits.float())  # invalid positions (delay pattern) come back as NaN
    ce = F.cross_entropy(logits.permute(0, 3, 1, 2), codes, reduction="none")  # (B, K, T)
    return (ce * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)


@torch.no_grad()
def encode(m, wav):
    codes, _ = m.compression_model.encode(wav)
    return codes


def train(args, dev):
    m, cond, t5, tid = load_model(dev)
    D = Data(args.machine, dev)
    log(f"=== {'LoRA r%d + token' % LORA['rank'] if LORA['rank'] else 'textual inversion of ' + TOKEN} on {args.machine} "
        f"near-minus-far, gain {20 * np.log10(D.gain):+.1f} dB")
    lora_p = [p for n, p in m.lm.transformer.named_parameters() if "lora_" in n]
    for p in list(m.lm.parameters()) + list(t5.parameters()):
        p.requires_grad_(False)
    for p in lora_p:
        p.requires_grad_(True)
    emb = t5.get_input_embeddings().weight
    if not LORA["rank"]:
        with torch.no_grad():  # start from the average of 'toy', 'car', 'motor'
            ids = [i for w in ("toy", "car", "motor") for i in cond.t5_tokenizer(w, add_special_tokens=False).input_ids]
            emb[tid] = emb[ids].mean(0)
    vec = torch.nn.Parameter(emb[tid].clone())
    hook = t5.get_input_embeddings().register_forward_hook(
        lambda mod, inp, out: torch.where((inp[0] == tid)[..., None], vec.to(out.dtype), out))
    groups = [{"params": [vec], "lr": args.lr}] + ([{"params": lora_p, "lr": args.lora_lr}] if lora_p else [])
    opt = torch.optim.Adam(groups)
    if lora_p:
        log(f"LoRA parameters {sum(p.numel() for p in lora_p):,} on {len(lora_p) // 2} layers; token starts from the "
            f"textual-inversion token")
    n_train = len(D.wav["train"])
    rng = random.Random(0)
    held = list(range(0, n_train, 50))[:16]  # 16 training clips for a fixed loss check (same clips every time)

    def check(tag):
        with torch.no_grad():
            codes = encode(m, D.audio("train", held))[..., :250]
            l_tok = lm_nll(m, codes, cond_tensors(cond, t5, [f"the sound of {TOKEN}"] * len(held))).mean().item()
            l_txt = lm_nll(m, codes, cond_tensors(cond, t5, ["the sound of a toy car motor running"] * len(held))).mean().item()
            f_tok = lm_nll(m, codes[..., :25], cond_tensors(cond, t5, [f"the sound of {TOKEN}"] * len(held))).mean().item()
            f_txt = lm_nll(m, codes[..., :25], cond_tensors(cond, t5, ["the sound of a toy car motor running"] * len(held))).mean().item()
        log(f"{tag}: loss over 5 s with '{TOKEN}' {l_tok:.4f} vs plain text {l_txt:.4f} | first 0.5 s: {f_tok:.4f} vs {f_txt:.4f}")

    check("step 0")
    t0, losses = time.time(), []
    for step in range(1, args.steps + 1):
        ids = rng.sample(range(n_train), args.batch)
        codes = encode(m, D.audio("train", ids))
        s = rng.randrange(codes.shape[-1] - args.crop + 1)
        codes = codes[..., s : s + args.crop]
        texts = [rng.choice(TEMPLATES).format(TOKEN) for _ in ids]
        loss = lm_nll(m, codes, cond_tensors(cond, t5, texts)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 25 == 0:
            log(f"step {step:4d} loss {np.mean(losses[-25:]):.4f} {(time.time() - t0) / step:.2f}s/step "
                f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if step % 200 == 0 or step == args.steps:
            check(f"step {step}")
            OUT.mkdir(parents=True, exist_ok=True)
            f_tok, f_lora = files()
            torch.save(vec.detach().cpu(), f_tok)
            if lora_p:
                torch.save({n: p.detach().cpu() for n, p in m.lm.transformer.named_parameters() if "lora_" in n}, f_lora)
    hook.remove()
    log("=== done")


# ---- comparison page ----
def mel_db(wav, sr=16000):
    spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=2048, hop_length=sr // 100, n_mels=128, f_max=8000)(torch.from_numpy(wav))
    return 10 * torch.log10(spec + 1e-10).numpy()


def save_png(db, vmax, dur, path):
    fig, ax = plt.subplots(figsize=(4.6, 2.0), constrained_layout=True)
    ax.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=vmax - 70, vmax=vmax, extent=[0, dur, 0, 128])
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("mel bin (0-8 kHz)", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def page_add(entries):
    mpath = PAGE / "manifest.json"
    man = json.loads(mpath.read_text())
    keys = {(e["clip"], e["mic"], e["variant"]) for e in entries}
    man["items"] = [i for i in man["items"] if (i["clip"], i["mic"], i["variant"]) not in keys] + entries
    mpath.write_text(json.dumps(man, indent=1))


def page_entry(base, variant, wav, ref_vmax, note, mic="near"):
    br = lambda s: s.replace("<", "[").replace(">", "]")  # the page reads < > as HTML
    base, variant, note = {**base, "clip": br(base["clip"])}, br(variant), br(note)
    stem = "".join(c if c.isalnum() else "-" for c in f"{base['clip']}_{mic}_{variant}")
    sf.write(PAGE / "audio" / f"{stem}.wav", wav, 16000)
    save_png(mel_db(wav), ref_vmax, len(wav) / 16000, PAGE / "img" / f"{stem}.png")
    return {**{k: base[k] for k in ("clip", "machine", "label", "file", "domain") if k in base}, "mic": mic,
            "variant": variant, "audio": f"audio/{stem}.wav", "img": f"img/{stem}.png", "note": note}


def recon(args, dev):
    m, cond, t5, tid = load_model(dev)
    D = Data(args.machine, dev)
    man = json.loads((PAGE / "manifest.json").read_text())
    clips = [i for i in man["items"] if i["machine"] == args.machine and i["mic"] == "near" and i["variant"] == "original"]
    names = D.meta["test"]["names"]
    m.set_generation_params(duration=10.0, cfg_coef=3.0)
    entries = []
    for it in clips:
        idx = names.index(it["file"])
        x = D.audio("test", [idx])
        ref = x[0, 0].cpu().numpy()
        vmax = float(mel_db(ref).max())
        torch.manual_seed(0)
        with torch.no_grad():
            rt = m.compression_model.decode(*m.compression_model.encode(x))[0, 0].cpu().numpy()
            gen = m.generate_continuation(x[..., : 3 * 16000], 16000, [f"the sound of {TOKEN}"])[0, 0].cpu().numpy()
        n = min(len(gen), len(ref))
        err = float(np.mean((mel_db(gen[3 * 16000 : n]) - mel_db(ref[3 * 16000 : n])) ** 2))
        entries.append(page_entry(it, "AudioGen codec round trip", rt, vmax, "near-minus-far through AudioGen's EnCodec and back"))
        entries.append(page_entry(it, f"AudioGen continuation ({TOKEN}){tag()}", gen[:n], vmax,
                                  f"first 3 s = the real clip, next 7 s written by AudioGen with '{TOKEN}'; "
                                  f"mel error vs the real 3-10 s: {err:.1f} dB^2"))
        log(f"recon {it['clip']:26s} ({it['label']}): continuation mel error {err:.1f} dB^2")
    page_add(entries)


def anomaly(args, dev):
    m, cond, t5, tid = load_model(dev)
    m.set_generation_params(duration=5.0, cfg_coef=args.cfg)
    entries = []
    for s in range(2):
        run = tag().strip(" ()") or "token only"
        base = {"clip": f"AudioGen {TOKEN}{tag()} sample {s + 1}", "machine": f"{args.machine} AudioGen ({run})",
                "label": "generated", "file": f"{run}, sample {s + 1}", "domain": ""}
        wavs = {}
        for name, prompt in ANOMALY_PROMPTS.items():
            torch.manual_seed(100 + s)
            with torch.no_grad():
                wavs[name] = m.generate([prompt])[0, 0].cpu().numpy()
        vmax = float(mel_db(wavs["normal: '<toycar>'"]).max())
        for name, w in wavs.items():
            entries.append(page_entry(base, name, w, vmax, f"prompt: '{ANOMALY_PROMPTS[name]}' (seed {100 + s}, cfg {args.cfg})",
                                      mic="near minus far (generated)"))
        log(f"anomaly prompts sample {s + 1} done")
    page_add(entries)


def score(args, dev):
    sys.path.insert(0, str(ROOT))
    from dcase_ae.evaluation import compute_metrics
    m, cond, t5, tid = load_model(dev)
    D = Data(args.machine, dev)
    lab, dom = np.array(D.meta["test"]["label"]), np.array(D.meta["test"]["domain"])
    res = {}
    for name, text in ((f"'{TOKEN}'", f"the sound of {TOKEN}"), ("plain text", "the sound of a toy car motor running")):
        S = []
        with torch.no_grad():
            for i in range(0, len(lab), 8):
                ids = list(range(i, min(i + 8, len(lab))))
                S.append(lm_nll(m, encode(m, D.audio("test", ids)), cond_tensors(cond, t5, [text] * len(ids))).cpu())
        s = torch.cat(S).numpy()
        r = compute_metrics(s, lab, dom)
        res[name] = r
        name = name + tag()
        log(f"score with {name}: surprise normal {s[lab == 0].mean():.4f} anomaly {s[lab == 1].mean():.4f} | "
            f"AUC src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} pAUC {r['pauc']:.3f} official {r['official']:.3f}")
    (OUT / f"score_{args.machine}{tag().strip().replace(' ', '_')}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "recon", "anomaly", "score"])
    ap.add_argument("--machine", default="ToyCar")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--crop", type=int, default=250, help="training crop in tokens (50 per second)")
    ap.add_argument("--lora", type=int, default=0, help="LoRA rank on the LM (0 = token only)")
    ap.add_argument("--lora_name", default="", help="suffix for the LoRA files, e.g. full10")
    ap.add_argument("--lora_lr", type=float, default=1e-4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LORA["rank"] = args.lora
    LORA["name"] = args.lora_name
    dev = torch.device("cuda")
    {"train": train, "recon": recon, "anomaly": anomaly, "score": score}[args.cmd](args, dev)


if __name__ == "__main__":
    main()
