"""Add BEATs' own input spectrogram to the comparison page (outputs/sam_compare/).

Exactly BEATs.preprocess: waveform x 2^15 -> Kaldi fbank (128 mel bins, 25 ms window, 10 ms shift, Povey window,
pre-emphasis 0.97, 20 Hz - 8 kHz, natural log of power) -> (fbank - 15.41663) / (2 x 6.55582).
Two variants per clip: 'BEATs input (near)' from the near-mic original, 'BEATs input (near minus far)' from the page's
near-minus-far audio. Colour range: the image's maximum down to the equivalent of 70 dB below it (same range as the
page's own spectrograms). Audio column = the same audio as the source variant.
Usage: uv run python tools/beats_fbank_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from filter_compare import OUT, SR, save_png  # noqa: E402

RANGE = 70 / (10 / np.log(10)) / (2 * 6.55582)  # 70 dB in BEATs' normalised units


def beats_fbank(wav):
    fb = ta_kaldi.fbank(torch.from_numpy(wav)[None] * 2**15, num_mel_bins=128, sample_frequency=16000,
                        frame_length=25, frame_shift=10)
    return ((fb - 15.41663) / (2 * 6.55582)).T.numpy()  # (128, frames)


def main():
    mpath = ROOT / OUT / "manifest.json"
    man = json.loads(mpath.read_text())
    items = man["items"]
    done = {(i["clip"], i["variant"]) for i in items}
    src = {("original", "BEATs input (near)"), ("near minus far", "BEATs input (near minus far)")}
    new = []
    for it in items:
        if it["mic"] != "near":
            continue
        for v_src, v_new in src:
            if it["variant"] != v_src or (it["clip"], v_new) in done:
                continue
            wav, sr = sf.read(ROOT / OUT / it["audio"], dtype="float32")
            if wav.ndim != 1:
                continue
            if sr != SR:  # page originals are stored at 48 kHz (for SAM-Audio); BEATs takes 16 kHz
                wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
            img = beats_fbank(wav)
            stem = f"{it['clip']}_near_{''.join(c if c.isalnum() else '-' for c in v_new)}"
            vmax = float(img.max())
            save_png(img, ROOT / OUT / "img" / f"{stem}.png", len(wav) / SR, vmax - RANGE, vmax)
            new.append({**{k: it[k] for k in ("clip", "machine", "label", "file", "domain") if k in it}, "mic": "near",
                        "variant": v_new, "audio": it["audio"], "img": f"img/{stem}.png",
                        "note": f"BEATs' own input (Kaldi fbank, 25 ms, pre-emphasis, normalised); range "
                                f"{img.min():.2f}..{vmax:.2f}, mean {img.mean():.2f}"})
            done.add((it["clip"], v_new))
    items.extend(new)
    mpath.write_text(json.dumps(man, indent=1))
    print(f"added {len(new)} BEATs input images")


if __name__ == "__main__":
    main()
