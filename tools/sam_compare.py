"""Separate clips with SAM-Audio and add the results to the comparison page (outputs/sam_compare/).

Each --variant is "NAME=PROMPT:KEEP": PROMPT is the text prompt ({machine} is replaced by the machine phrase) and KEEP
says which output is the cleaned machine sound: 'target' (the prompt describes the machine) or 'residual'
(the prompt describes the noise). Results are appended to manifest.json; already-done items are skipped, so adding
a new variant is just another run. The page (copied from tools/compare_page.html) shows every variant as a column.
All spectrograms of one clip and mic share the original's colour scale.

Runs in the separate environment:
  envs/samaudio/Scripts/python tools/sam_compare.py --variant "noise removed=background noise:residual"
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from sam_audio_demo import machine_phrase  # noqa: E402

OUT = Path("outputs/sam_compare")
MICS = {"near": 0, "far": 1}
DEFAULT_VARIANTS = ["machine isolated={machine} running:target",
                    "background removed=background noise:residual",
                    "factory noise removed=factory noise:residual"]


def pick_clips(machines, per_label, seed=0):
    """Same draw as tools/sam_audio_demo.py."""
    rng = random.Random(seed)
    out = []
    for m in machines:
        for lab in ("normal", "anomaly"):
            out += [(m, lab, f) for f in rng.sample(sorted(Path(f"data/{m}/test").glob(f"*source_test_{lab}*.wav")), per_label)]
    return out


def mel_db(wav, sr):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", help='"NAME=PROMPT:target|residual"; repeatable (default: 3 built-in)')
    ap.add_argument("--machines", default="fan,valveEmu,ToyCar")
    ap.add_argument("--mics", default="near,far")
    ap.add_argument("--per_label", type=int, default=2)
    ap.add_argument("--model", default="facebook/sam-audio-small-tv")
    args = ap.parse_args()

    for d in ("audio", "img"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    shutil.copy("tools/compare_page.html", OUT / "index.html")
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"items": []}
    manifest["description"] = ("SAM-Audio separations. 'original' = recording; other columns = the cleaned machine sound "
                               "for each prompt. Shared colour scale per clip and mic. Notes: share of energy kept.")
    done = {(i["clip"], i["mic"], i["variant"]) for i in manifest["items"]}

    variants = []
    for v in args.variant or DEFAULT_VARIANTS:
        name, rest = v.split("=", 1)
        prompt, keep = rest.rsplit(":", 1)
        assert keep in ("target", "residual"), v
        variants.append((name.strip(), prompt.strip(), keep))

    model = proc = None
    for m, lab, f in pick_clips(args.machines.split(","), args.per_label):
        clip = f"{m}_{f.stem.split('_')[4]}_{f.stem.split('_')[5]}"
        w, in_sr = sf.read(f, dtype="float32", always_2d=True)
        for mic in args.mics.split(","):
            todo = [v for v in variants if (clip, mic, v[0]) not in done]
            if not todo and (clip, mic, "original") in done:
                continue
            if model is None:
                from sam_audio import SAMAudio, SAMAudioProcessor
                model = SAMAudio.from_pretrained(args.model, visual_ranker=None, text_ranker=None, span_predictor=None).cuda().eval()
                proc = SAMAudioProcessor.from_pretrained(args.model)
            sr = proc.audio_sampling_rate
            x = torchaudio.functional.resample(torch.from_numpy(w[:, MICS[mic]]), in_sr, sr)
            orig = x.numpy()
            odb = mel_db(orig, sr)
            vmax, dur = float(odb.max()), len(orig) / sr

            def add(variant, wav, note=""):
                stem = f"{clip}_{mic}_{variant.replace(' ', '-')}"
                sf.write(OUT / "audio" / f"{stem}.wav", wav, sr)
                save_png(mel_db(wav, sr), vmax, dur, OUT / "img" / f"{stem}.png")
                manifest["items"].append({"clip": clip, "machine": m, "label": lab, "file": f.name, "mic": mic,
                                          "variant": variant, "audio": f"audio/{stem}.wav", "img": f"img/{stem}.png", "note": note})
                done.add((clip, mic, variant))

            if (clip, mic, "original") not in done:
                add("original", orig)
            for name, prompt, keep in todo:
                batch = proc(audios=[x[None]], descriptions=[prompt.format(machine=machine_phrase(m))]).to("cuda")
                t0 = time.time()
                with torch.inference_mode():
                    res = model.separate(batch, predict_spans=False, reranking_candidates=1)
                tgt = res.target[0].float().cpu().numpy().reshape(-1)
                rsd = res.residual[0].float().cpu().numpy().reshape(-1)
                kept = tgt if keep == "target" else rsd
                share = float(np.mean(kept**2) / (np.mean(tgt**2) + np.mean(rsd**2) + 1e-12))
                add(name, kept, f"keeps {share:.0%} of the energy | {time.time() - t0:.1f}s")
            print(f"{clip} {mic}: done", flush=True)
            mpath.write_text(json.dumps(manifest, indent=1))
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest['items'])} items in {mpath}")


if __name__ == "__main__":
    main()
