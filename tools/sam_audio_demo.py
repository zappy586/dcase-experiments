"""Separate a few clips with SAM-Audio (text prompt) and build a page to listen to original / machine / residual.

Runs in the separate environment:  envs/samaudio/Scripts/python tools/sam_audio_demo.py
Writes outputs/sam_audio_demo/{index.html, audio/*.wav, img/*.png}.
"""
import argparse
import html
import random
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from sam_audio import SAMAudio, SAMAudioProcessor  # noqa: E402

OUT = Path("outputs/sam_audio_demo")


def machine_phrase(machine: str) -> str:
    """ToyCarEmu -> 'toy car'; same rule for every machine type."""
    name = machine.removesuffix("Emu")
    return " ".join(w.lower() for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", name))


def spectrogram_png(wav, sr, path, title):
    spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=2048, hop_length=sr // 100, n_mels=128, f_max=8000)(torch.from_numpy(wav))
    db = 10 * torch.log10(spec + 1e-10).numpy()
    fig, ax = plt.subplots(figsize=(5, 2.2), constrained_layout=True)
    ax.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=db.max() - 70, vmax=db.max(),
              extent=[0, len(wav) / sr, 0, 128])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("mel bin (0-8 kHz)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/sam-audio-small-tv")
    ap.add_argument("--machines", default="fan,valveEmu,ToyCar")
    ap.add_argument("--per_label", type=int, default=2)
    args = ap.parse_args()

    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    # skip the reranking models and the automatic span predictor: unused with reranking_candidates=1 and no spans
    model = SAMAudio.from_pretrained(args.model, visual_ranker=None, text_ranker=None, span_predictor=None).cuda().eval()
    proc = SAMAudioProcessor.from_pretrained(args.model)
    sr = proc.audio_sampling_rate
    print(f"loaded {args.model}, codec sample rate {sr} Hz, GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    rng = random.Random(0)
    sections = []
    for m in args.machines.split(","):
        prompt = f"{machine_phrase(m)} running"
        clips = [(f, "normal") for f in rng.sample(sorted(Path(f"data/{m}/test").glob("*source_test_normal*.wav")), args.per_label)]
        clips += [(f, "anomaly") for f in rng.sample(sorted(Path(f"data/{m}/test").glob("*source_test_anomaly*.wav")), args.per_label)]
        rows = []
        for f, lab in clips:
            w, in_sr = sf.read(f, dtype="float32", always_2d=True)
            near = torch.from_numpy(w[:, 0])
            x = torchaudio.functional.resample(near, in_sr, sr)[None]  # (1, L) @ codec rate
            batch = proc(audios=[x], descriptions=[prompt]).to("cuda")
            t0 = time.time()
            with torch.inference_mode():
                res = model.separate(batch, predict_spans=False, reranking_candidates=1)
            dt = time.time() - t0
            stem = f"{m}_{lab}_{f.stem.split('_')[5]}"
            tracks = {"original": x[0].numpy(), "machine": res.target[0].float().cpu().numpy().reshape(-1),
                      "residual": res.residual[0].float().cpu().numpy().reshape(-1)}
            energy = {k: float(np.mean(v**2)) for k, v in tracks.items()}
            for k, v in tracks.items():
                sf.write(OUT / "audio" / f"{stem}_{k}.wav", v, sr)
                spectrogram_png(v, sr, OUT / "img" / f"{stem}_{k}.png", f"{k}")
            share = energy["machine"] / (energy["machine"] + energy["residual"])
            rows.append((stem, lab, f.name, share, dt))
            print(f"{m:10s} {lab:7s} {f.name}: separated in {dt:.1f}s, machine share of energy {share:.2f}", flush=True)
        sections.append((m, prompt, rows))

    parts = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
             "<title>SAM-Audio Separations</title><style>body{font:14px system-ui,sans-serif;margin:0;background:#f7f7f5;color:#1d1d1b}",
             "main{max-width:1150px;margin:0 auto;padding:20px 16px}h1{font-size:20px}h2{font-size:16px;margin-top:28px}",
             ".clip{background:#fff;border:1px solid #e3e3de;border-radius:10px;padding:12px;margin:10px 0}",
             ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.grid img{width:100%}audio{width:100%;height:32px}",
             ".tag{font-weight:600}.normal{color:#2f5fd0}.anomaly{color:#b4570f}small{color:#6b6b66}",
             "@media(max-width:700px){.grid{grid-template-columns:1fr}}",
             "@media(prefers-color-scheme:dark){body{background:#161615;color:#ecece8}.clip{background:#1f1f1d;border-color:#33332f}}</style></head><body><main>",
             f"<h1>SAM-Audio separations ({html.escape(args.model)})</h1>",
             "<p><small>Near mic only. Text prompt '&lt;machine&gt; running'. 'machine' = what SAM-Audio isolated, 'residual' = everything else. "
             "Labels are shown on purpose (qualitative check).</small></p>"]
    for m, prompt, rows in sections:
        parts.append(f"<h2>{html.escape(m)} &nbsp;<small>prompt: \"{html.escape(prompt)}\"</small></h2>")
        for stem, lab, name, share, dt in rows:
            parts.append(f"<div class='clip'><div><span class='tag {lab}'>{lab}</span> <small>{html.escape(name)} | "
                         f"machine share of energy {share:.2f} | {dt:.1f}s</small></div><div class='grid'>")
            for k in ("original", "machine", "residual"):
                parts.append(f"<div><img src='img/{stem}_{k}.png' alt='{k} spectrogram'><audio controls preload='none' src='audio/{stem}_{k}.wav'></audio></div>")
            parts.append("</div></div>")
    parts.append("</main></body></html>")
    (OUT / "index.html").write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
