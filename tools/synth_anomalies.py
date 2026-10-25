"""Generate candidate synthetic anomalies from normal clips and put them on the comparison page (outputs/sam_compare/).

A. Edit (AudioLDM2-large, zero-shot DDPM-inversion editing, github.com/HilaManor/AudioEditingCode):
   near-mic clip -> invert once with a neutral source prompt -> regenerate with each target prompt at start steps
   50 / 100 / 150 of 200 (subtle -> strong). Controls: 'codec only' (encode/decode, no diffusion) and the neutral
   prompt at the same strengths, so normal and anomalous versions go through the same generator.
B. Add (TangoFlux): generate a 10 s fault sound per prompt, mix it into the clip at -20 dB and -10 dB relative to the
   clip, 6 dB quieter on the far mic than on the near mic.

Runs in the audiogen environment:  envs/audiogen/Scripts/python tools/synth_anomalies.py [--machines fan,ToyCar]
"""
import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "AudioEditingCode" / "code"))
sys.path.insert(0, str(ROOT / "tools"))
from filter_compare import OUT, SR, mel_db, save_png  # noqa: E402


def _sf_load(path, *args, **kwargs):
    """torchaudio.load replacement (new torchaudio needs torchcodec + FFmpeg for loading)."""
    w, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(w.T.copy()), sr


torchaudio.load = _sf_load

FAULTS = {
    "grinding": "{m} with a worn bearing grinding",
    "rattling": "{m} with a loose part rattling",
    "squealing": "{m} with a squealing belt",
    "knocking": "{m} running unevenly with a knocking sound",
}
NEUTRAL = "{m} running normally"
STRENGTHS = (50, 100, 150)
ADD_PROMPTS = {"grinding": "metal grinding noise", "rattling": "rattling of a loose metal part",
               "squealing": "high-pitched squealing belt", "knocking": "irregular mechanical knocking"}
PHRASE = {"ToyCar": "toy car", "ToyCarEmu": "toy car", "bearingEmu": "bearing", "fan": "fan", "gearboxEmu": "gearbox",
          "sliderEmu": "slider", "valveEmu": "valve"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", default="fan,ToyCar")
    ap.add_argument("--per_machine", type=int, default=2)
    ap.add_argument("--skip_edit", action="store_true")
    ap.add_argument("--skip_add", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text())
    done = {(i["clip"], i["mic"], i["variant"]) for i in manifest["items"]}
    rng = random.Random(0)

    clips = []
    for m in args.machines.split(","):
        for f in rng.sample(sorted((ROOT / "data" / m / "train").glob("*source_train_normal*.wav")), args.per_machine):
            w, sr = sf.read(f, dtype="float32", always_2d=True)
            assert sr == SR
            clips.append((m, f, w[: 10 * SR].T.copy()))  # (2, L)

    def add(m, f, mic, variant, wav, ref, note=""):
        clip = f"synth-{m}-{f.stem.split('_')[5]}"
        if (clip, mic, variant) in done:
            return
        stem = f"{clip}_{mic}_{''.join(c if c.isalnum() else '-' for c in variant)}"
        wav = np.asarray(wav, dtype=np.float32)[: len(ref)]
        sf.write(OUT / "audio" / f"{stem}.wav", wav, SR)
        vmax = float(mel_db(ref).max())
        save_png(mel_db(wav), OUT / "img" / f"{stem}.png", len(ref) / SR, vmax - 70, vmax)
        manifest["items"].append({"clip": clip, "machine": f"{m} synthetic", "label": "normal", "file": f.name, "mic": mic,
                                  "variant": variant, "audio": f"audio/{stem}.wav", "img": f"img/{stem}.png", "note": note})
        done.add((clip, mic, variant))

    for m, f, w in clips:
        add(m, f, "near", "original", w[0], w[0])
        add(m, f, "far", "original", w[1], w[1])

    if not args.skip_edit:
        from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
        from models import load_model
        from utils import load_audio, set_reproducability

        set_reproducability(0, extreme=False)
        ldm = load_model("cvssp/audioldm2-large", dev, 200)
        for m, f, w in clips:
            phrase = PHRASE.get(m, m)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "in.wav"
                sf.write(p, w[0], SR)
                x0, sr, duration = load_audio(str(p), ldm.get_fn_STFT(), device=dev, stft=True, model_sr=ldm.get_sr())
            with torch.inference_mode():
                w0 = ldm.vae_encode(x0)
                dec = ldm.vae_decode(w0)
                dec = dec[None] if dec.dim() < 4 else dec
                add(m, f, "near", "codec only", ldm.decode_to_mel(dec).cpu().numpy().reshape(-1), w[0], "AudioLDM2 encode/decode, no diffusion")
                _, zs, wts, extra = inversion_forward_process(ldm, w0, etas=1.0, prompts=[NEUTRAL.format(m=phrase)],
                                                              cfg_scales=[3.0], prog_bar=False, num_inference_steps=200,
                                                              cutoff_points=None, numerical_fix=True, duration=duration)
                for name, tmpl in [("neutral", NEUTRAL)] + list(FAULTS.items()):
                    for ts in STRENGTHS:
                        variant = f"edit {name} t{ts}"
                        clip = f"synth-{m}-{f.stem.split('_')[5]}"
                        if (clip, "near", variant) in done:
                            continue
                        out, _ = inversion_reverse_process(ldm, xT=wts, tstart=torch.tensor([ts], dtype=torch.int),
                                                           fix_alpha=0.1, etas=1.0, prompts=[tmpl.format(m=phrase)],
                                                           neg_prompts=[""], cfg_scales=[12.0], prog_bar=False,
                                                           zs=zs[:ts], cutoff_points=None, duration=duration, extra_info=extra)
                        dec = ldm.vae_decode(out)
                        dec = dec[None] if dec.dim() < 4 else dec
                        add(m, f, "near", variant, ldm.decode_to_mel(dec).cpu().numpy().reshape(-1), w[0],
                            f"prompt '{tmpl.format(m=phrase)}', start step {ts}/200")
                        print(f"{m} {f.stem.split('_')[5]} {variant} done", flush=True)
                mpath.write_text(json.dumps(manifest, indent=1))
        del ldm
        torch.cuda.empty_cache()

    if not args.skip_add:
        from tangoflux import TangoFluxInference

        tf = TangoFluxInference(name="declare-lab/TangoFlux", device="cuda")
        faults = {}
        for name, prompt in ADD_PROMPTS.items():
            a = tf.generate(prompt, steps=50, duration=10)  # (ch, n) at 44.1 kHz
            a = torchaudio.functional.resample(a.float().mean(0), 44100, SR).cpu().numpy()
            faults[name] = a / (np.sqrt(np.mean(a**2)) + 1e-9)
            print(f"TangoFlux '{prompt}' generated", flush=True)
        for m, f, w in clips:
            rms = np.sqrt(np.mean(w[0] ** 2))
            for name, a in faults.items():
                a = np.pad(a, (0, max(0, w.shape[1] - len(a))))[: w.shape[1]]
                for db in (-20, -10):
                    g = rms * 10 ** (db / 20)
                    add(m, f, "near", f"add {name} {db} dB", w[0] + g * a, w[0], f"TangoFlux '{ADD_PROMPTS[name]}' mixed at {db} dB")
                    add(m, f, "far", f"add {name} {db} dB", w[1] + 0.5 * g * a, w[1], f"same sound 6 dB quieter on the far mic")
        mpath.write_text(json.dumps(manifest, indent=1))
    mpath.write_text(json.dumps(manifest, indent=1))
    print("done")


if __name__ == "__main__":
    main()
