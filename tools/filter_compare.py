"""Add Butterworth high-pass filter (BHPF) variants of clips already on the comparison page (outputs/sam_compare/).

  BHPF audio <fc>  : 4th-order Butterworth high-pass on the waveform (zero-phase), cutoff fc Hz
  BHPF image       : 2nd-order Butterworth high-pass on the log-mel spectrogram IMAGE (2-D FFT), cutoff D0;
                     removes smooth large-scale shading, keeps fine lines/edges (audio column = original)

Usage: uv run python tools/filter_compare.py [--machine fan]
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

OUT = Path("outputs/sam_compare")
MICS = {"near": 0, "far": 1}
SR = 16000
SMOOTH = (0.7, 20)  # Gaussian sigma: (mel bins, time frames of 10 ms)


def mel_db(wav):
    spec = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=SR // 100, n_mels=128)(torch.from_numpy(wav))
    return 10 * torch.log10(spec + 1e-10).numpy()


def save_png(img, path, dur, vmin, vmax, cmap="magma"):
    fig, ax = plt.subplots(figsize=(4.6, 2.0), constrained_layout=True)
    ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, dur, 0, 128])
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("mel bin (0-8 kHz)", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def bhpf_image(img, d0=8.0, order=2):
    """Butterworth high-pass in the 2-D Fourier domain of the spectrogram image."""
    F, T = img.shape
    u = np.fft.fftfreq(F) * F
    v = np.fft.fftfreq(T) * T
    D = np.sqrt(u[:, None] ** 2 + (v[None, :] * F / T) ** 2)  # distance in 'frequency-bin' units
    H = 1.0 / (1.0 + (d0 / np.maximum(D, 1e-6)) ** (2 * order))
    return np.real(np.fft.ifft2(np.fft.fft2(img - img.mean()) * H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="fan")
    args = ap.parse_args()
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text())
    clips = {(i["clip"], i["file"], i["label"]) for i in manifest["items"] if i["machine"] == args.machine}
    if not clips:  # machine not on the page yet: same draw as the SAM-Audio pages (2 normal + 2 anomaly, seed 0)
        rng = random.Random(0)
        for lab in ("normal", "anomaly"):
            for f in rng.sample(sorted(Path(f"data/{args.machine}/test").glob(f"*source_test_{lab}*.wav")), 2):
                clips.add((f"{args.machine}_{lab}_{f.stem.split('_')[5]}", f.name, lab))
    done = {(i["clip"], i["mic"], i["variant"]) for i in manifest["items"]}

    for clip, fname, lab in sorted(clips):
        w, sr = sf.read(Path("data") / args.machine / "test" / fname, dtype="float32", always_2d=True)
        assert sr == SR
        for mic, ch in MICS.items():
            x = w[:, ch]
            odb = mel_db(x)
            vmax, dur = float(odb.max()), len(x) / SR

            def add(variant, wav, img, vmin, vmax_, cmap="magma", note=""):
                if (clip, mic, variant) in done:
                    return
                stem = f"{clip}_{mic}_{variant.replace(' ', '-')}"
                if wav is not None:
                    sf.write(OUT / "audio" / f"{stem}.wav", wav, SR)
                audio = f"audio/{stem}.wav" if wav is not None else next(
                    i["audio"] for i in manifest["items"] if i["clip"] == clip and i["mic"] == mic and i["variant"] == "original")
                save_png(img, OUT / "img" / f"{stem}.png", dur, vmin, vmax_, cmap)
                manifest["items"].append({"clip": clip, "machine": args.machine, "label": lab, "file": fname, "mic": mic,
                                          "variant": variant, "audio": audio, "img": f"img/{stem}.png", "note": note})

            add("original", x, odb, vmax - 70, vmax)
            for fc in (200, 1000):
                y = sosfiltfilt(butter(4, fc, "highpass", fs=SR, output="sos"), x).astype(np.float32)
                kept = float(np.mean(y**2) / np.mean(x**2))
                add(f"BHPF audio {fc} Hz", y, mel_db(y), vmax - 70, vmax, note=f"keeps {kept:.0%} of the energy")
                if fc == 200:  # smoothing mostly along time: removes frame-to-frame grain, keeps thin frequency lines
                    ydb = mel_db(y)
                    smooth = gaussian_filter(ydb, SMOOTH)
                    add("BHPF 200 Hz + smoothed", y, smooth, vmax - 70, vmax,
                        note=f"Gaussian smoothing, sigma {SMOOTH[1] * 10} ms in time, {SMOOTH[0]} mel bins")
                    detail = ydb - smooth
                    lim = float(np.percentile(np.abs(detail), 99))
                    add("BHPF 200 Hz - smoothed", y, detail, -lim, lim, cmap="coolwarm",
                        note=f"BHPF 200 Hz spectrogram minus its smoothed version (dB); colour range +-{lim:.1f} dB")
            hp = bhpf_image(odb)
            lim = float(np.percentile(np.abs(hp), 99))
            add("BHPF image", None, hp, -lim, lim, cmap="coolwarm", note="spectrogram image high-passed (D0=8, order 2); audio = original")
            hs = gaussian_filter(hp, SMOOTH)
            lim = float(np.percentile(np.abs(hs), 99))
            add("BHPF image + smoothed", None, hs, -lim, lim, cmap="coolwarm",
                note=f"BHPF image, then Gaussian smoothing ({SMOOTH[1] * 10} ms, {SMOOTH[0]} mel bins); audio = original")
        print(clip, "done")
    mpath.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
