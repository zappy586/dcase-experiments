"""Audio examples + spectrograms for a dated update page (showcase/updates/<date>.html), in the same style as
tools/build_showcase.py: mono, 16 kHz, up to 10 s, -23 dBFS RMS, 128-band mel 0-8 kHz, 70 dB range.
Output: showcase/assets/updates/<date>/{audio,img}/<name>.{wav,png}
Usage: uv run python tools/build_update_assets.py 2026-11-02
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "dcase_gens"
UPDATES = {
    "2026-11-02": {
        "real_normal": ROOT / "showcase" / "assets" / "audio" / "toycar_real_normal.wav",
        "syn_normal": G / "normal" / "0000.wav",
        "syn_broken": G / "faulty" / "0501.wav",
        "syn_rattle": G / "faulty" / "0502.wav",
        "syn_grinding": G / "faulty" / "0503.wav",
    },
}


def main(date):
    out = ROOT / "showcase" / "assets" / "updates" / date
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(parents=True, exist_ok=True)
    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=160, n_mels=128, f_max=8000)
    for name, src in UPDATES[date].items():
        w, sr = sf.read(src, dtype="float32", always_2d=True)
        w = torch.from_numpy(w.mean(1))
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        w = w[: 10 * 16000]
        w = w / (w.pow(2).mean().sqrt() + 1e-9) * 10 ** (-23 / 20)
        w = w.clamp(-1, 1).numpy()
        sf.write(out / "audio" / f"{name}.wav", w, 16000, subtype="PCM_16")
        db = 10 * np.log10(mel(torch.from_numpy(w)).numpy() + 1e-10)
        fig, ax = plt.subplots(figsize=(5.2, 1.9), dpi=110)
        ax.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=db.max() - 70, vmax=db.max(),
                  extent=[0, len(w) / 16000, 0, 8])
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_ylabel("kHz", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.tight_layout(pad=0.3)
        fig.savefig(out / "img" / f"{name}.png")
        plt.close(fig)
        print(f"{name}: {len(w) / 16000:.1f} s <- {src.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1])
