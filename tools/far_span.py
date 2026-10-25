"""SAM-Audio with the FAR mic as the example of what to remove.

Input per clip (48 kHz after resampling): [far mic, up to 10 s] then [near mic, same part of the recording].
The far part is marked '-' (target absent), the near part is unmarked, so SAM-Audio is asked to separate what the
near mic contains but the far mic does not: ideally the machine's own sound, with the shared background removed.
Two prompts: no text, and the machine phrase. Kept: the target over the near part.
Results go on the comparison page (outputs/sam_compare) next to 'original' and 'near minus far' for the same clip.

Runs in the SAM-Audio environment (PYTHONPATH=tools):
  envs/samaudio/Scripts/python tools/far_span.py --clips fan_normal_0024 fan_anomaly_0016 ToyCar_anomaly_0081
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from neg_span import load_model, separate
from sam_compare import OUT, mel_db, save_png

PHRASE = {"fan": "industrial fan", "ToyCar": "toy car motor", "ToyCarEmu": "toy car motor", "gearboxEmu": "gearbox",
          "bearingEmu": "rotating bearing", "sliderEmu": "slide rail", "valveEmu": "solenoid valve"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--precision", default="bf16-autocast")
    ap.add_argument("--max_s", type=float, default=10.0)
    args = ap.parse_args()
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text())
    info = {i["clip"]: i for i in manifest["items"] if i["mic"] == "near" and i["variant"] == "original"}
    model, proc = load_model(args.precision)
    sr = proc.audio_sampling_rate
    for clip in args.clips:
        it = info[clip]
        m = it["machine"]
        w, in_sr = sf.read(Path("data") / m / "test" / it["file"], dtype="float32", always_2d=True)
        w = torch.from_numpy(w[: int(args.max_s * in_sr)].T.copy())  # (2, L): channel 0 near, 1 far
        near, far = (torchaudio.functional.resample(c, in_sr, sr) for c in w)
        x = torch.cat([far, near])
        split, dur = len(far), len(far) / sr
        anchors = [[["-", 0.0, dur]]]
        for name, text in (("no text", ""), ("machine text", f"{PHRASE[m]} sound")):
            t0 = time.time()
            tgt, rsd = separate(model, proc, x, text, anchors, args.precision)
            tn, rn = tgt[split:], rsd[split:]
            e = lambda a: float((a ** 2).sum())
            share = e(tn) / (e(tn) + e(rn) + 1e-12)
            leak = e(tgt[:split]) / (e(tgt[:split]) + e(rsd[:split]) + 1e-12)
            variant = f"far-span target ({name})"
            stem = f"{clip}_near_{''.join(c if c.isalnum() else '-' for c in variant)}"
            sf.write(OUT / "audio" / f"{stem}.wav", tn.numpy(), sr)
            save_png(mel_db(tn.numpy(), sr), float(mel_db(near.numpy(), sr).max()), len(tn) / sr, OUT / "img" / f"{stem}.png")
            manifest["items"] = [i for i in manifest["items"] if not (i["clip"] == clip and i["variant"] == variant)]
            manifest["items"].append({"clip": clip, "machine": m, "label": it["label"], "file": it["file"], "mic": "near",
                                      "variant": variant, "audio": f"audio/{stem}.wav", "img": f"img/{stem}.png",
                                      "note": f"far mic first, marked '-'; text '{text}' | kept on the near part: "
                                              f"{share:.0%} of energy; target on the far part (should be ~0): {leak:.0%}"})
            mpath.write_text(json.dumps(manifest, indent=1))
            print(f"{clip:22s} {name:12s} kept {share:5.1%} of the near part | target on far part {leak:5.1%} | "
                  f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
