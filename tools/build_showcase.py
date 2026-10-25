"""Copy the audio examples used by the showcase page (showcase/index.html) into showcase/assets/ and draw their
spectrograms in one style. Every clip: mono, 16 kHz, up to 10 s, loudness-normalised to -23 dBFS RMS (so the page
compares sounds, not recording levels). Spectrogram: 128-band mel, 0-8 kHz, 70 dB range below the clip's maximum.
Usage: uv run python tools/build_showcase.py
"""
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "outputs" / "sam_compare" / "audio"
AG = ROOT / "outputs" / "audiogen"
OUT = ROOT / "showcase" / "assets"

CLIPS = {
    # near vs far vs near-minus-far (real recordings)
    "fan_normal_near": PAGE / "fan_normal_0024_near_original.wav",
    "fan_normal_far": PAGE / "fan_normal_0024_far_original.wav",
    "fan_normal_nmf": PAGE / "fan_normal_0024_near_near-minus-far.wav",
    "fan_anomaly_nmf": PAGE / "fan_anomaly_0026_near_near-minus-far.wav",
    # real ToyCar (near-minus-far)
    "toycar_real_normal": PAGE / "ToyCar_normal_0077_near_near-minus-far.wav",
    "toycar_real_anomaly": PAGE / "ToyCar_anomaly_0081_near_near-minus-far.wav",
    # AudioGen + LoRA + [toycar]
    "gen_normal_5s": PAGE / "AudioGen--toycar---LoRA-r16--sample-1-near-minus-far--generated--normal----toycar--.wav",
    "gen_rattle_5s": PAGE / "AudioGen--toycar---LoRA-r16--sample-1-near-minus-far--generated--faint-intermittent-rattle.wav",
    "gen_grinding_5s": PAGE / "AudioGen--toycar---LoRA-r16--sample-1-near-minus-far--generated--slightly-worn-grinding-gear.wav",
    "gen_normal_full10": AG / "full_length_full10" / "gen_00.wav",
    "gen_opening_normal": AG / "prefix" / "p0_00.wav",
    "gen_opening_rattle": AG / "prefix" / "p2_00.wav",
}


def main():
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=160, n_mels=128, f_max=8000)
    for name, src in CLIPS.items():
        w, sr = sf.read(src, dtype="float32", always_2d=True)
        w = torch.from_numpy(w.mean(1))
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        w = w[: 10 * 16000]
        w = w / (w.pow(2).mean().sqrt() + 1e-9) * 10 ** (-23 / 20)
        w = w.clamp(-1, 1).numpy()
        sf.write(OUT / "audio" / f"{name}.wav", w, 16000, subtype="PCM_16")
        db = 10 * np.log10(mel(torch.from_numpy(w)).numpy() + 1e-10)
        fig, ax = plt.subplots(figsize=(5.2, 1.9), dpi=110)
        ax.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=db.max() - 70, vmax=db.max(),
                  extent=[0, len(w) / 16000, 0, 8])
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_ylabel("kHz", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.tight_layout(pad=0.3)
        fig.savefig(OUT / "img" / f"{name}.png")
        plt.close(fig)
        print(f"{name}: {len(w) / 16000:.1f} s")
    shutil.copy(Path.home() / "Desktop" / "Audio Spectrogram Anomaly Detection Pipeline.png", OUT / "img" / "architecture.png")


if __name__ == "__main__":
    main()
