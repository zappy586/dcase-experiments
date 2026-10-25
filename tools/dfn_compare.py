"""Denoise every clip on the comparison page (outputs/sam_compare/) with DeepFilterNet3 and add it as columns.

  DeepFilterNet3            : default, full noise suppression
  DeepFilterNet3 (max 12 dB): suppression capped at 12 dB (gentler)
DeepFilterNet works at 48 kHz: 16 kHz -> 48 kHz -> denoise -> 16 kHz. Same spectrogram style / colour scale as
tools/filter_compare.py.

Runs in its own environment:  envs/dfn/Scripts/python tools/dfn_compare.py
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from df.enhance import enhance, init_df

from filter_compare import MICS, OUT, SR, mel_db, save_png

VARIANTS = {"DeepFilterNet3": None, "DeepFilterNet3 (max 12 dB)": 12.0}


def main():
    model, state, _ = init_df()
    df_sr = state.sr()
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text())
    done = {(i["clip"], i["mic"], i["variant"]) for i in manifest["items"]}
    clips = sorted({(i["clip"], i["machine"], i["label"], i["file"]) for i in manifest["items"]})
    for clip, machine, lab, fname in clips:
        w, sr = sf.read(Path("data") / machine / "test" / fname, dtype="float32", always_2d=True)
        for mic, ch in MICS.items():
            x = w[:, ch]
            odb = mel_db(x)
            vmax, dur = float(odb.max()), len(x) / SR
            x48 = torchaudio.functional.resample(torch.from_numpy(x)[None], SR, df_sr)
            for name, lim in VARIANTS.items():
                if (clip, mic, name) in done:
                    continue
                y = enhance(model, state, x48, atten_lim_db=lim)
                y = torchaudio.functional.resample(y, df_sr, SR)[0].numpy()[: len(x)]
                kept = float(np.mean(y**2) / np.mean(x**2))
                stem = f"{clip}_{mic}_{name.replace(' ', '-').replace('(', '').replace(')', '')}"
                sf.write(OUT / "audio" / f"{stem}.wav", y, SR)
                save_png(mel_db(y), OUT / "img" / f"{stem}.png", dur, vmax - 70, vmax)
                manifest["items"].append({"clip": clip, "machine": machine, "label": lab, "file": fname, "mic": mic,
                                          "variant": name, "audio": f"audio/{stem}.wav", "img": f"img/{stem}.png",
                                          "note": f"keeps {kept:.0%} of the energy"})
        print(clip, "done", flush=True)
        mpath.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
