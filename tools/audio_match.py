"""Remove simple level/colour differences between synthetic and real ToyCar audio, then check plain BEATs.
Real: 1000 training normals (near-minus-far, machine gain, 10 s). Synthetic: set v1, normals split 200 fit / 50 held out
(same split as grl_v2 / contrast_align); faulty clips get the same corrections (fitted on normals only).
Corrections (all fitted on real training normals vs the 200 synthetic fit normals):
  loud : quantile mapping of clip loudness (RMS dB): a synthetic clip at the q-th loudness quantile of the synthetic
         normals is scaled to the q-th quantile of the real normals (fixes both the 2 dB offset and the too-small spread)
  eq   : one fixed EQ curve = sqrt(mean real power spectrum / mean synthetic power spectrum) per STFT bin (n_fft 1024),
         smoothed over 9 bins, applied to every frame of every synthetic clip (magnitude only, phase kept)
Probe (plain BEATs iter3, token average, full 10 s; fresh logistic regression, 5-fold CV AUC, 0.5 = cannot tell apart):
  200 fixed real training normals vs the 50 held-out synthetic normals; plus the same on time-averaged log-mel.
Writes the loud+eq version to outputs/audiogen/synth_v1_matched/{normal,faulty}/NNN.wav.
Usage (main env): uv run python tools/audio_match.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from BEATs import BEATs, BEATsConfig  # noqa: E402

SYN = ROOT / "outputs" / "audiogen" / "synth_v1"
OUT = ROOT / "outputs" / "audiogen" / "synth_v1_matched"
NFFT, HOP = 1024, 256


def rms_db(X):
    return 10 * np.log10((X.astype(np.float64) ** 2).mean(1) + 1e-12)


def power_spec(X):
    S = torch.stft(torch.from_numpy(X), NFFT, HOP, window=torch.hann_window(NFFT), return_complex=True)
    return (S.abs() ** 2).mean((0, 2)).numpy()


def apply_eq(X, g):
    w = torch.hann_window(NFFT)
    S = torch.stft(torch.from_numpy(X), NFFT, HOP, window=w, return_complex=True)
    S = S * torch.from_numpy(g).float()[None, :, None]
    return torch.istft(S, NFFT, HOP, window=w, length=X.shape[1]).numpy()


def main():
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    D = A.Data("ToyCar", dev)
    real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)]).numpy()
    gen = {c: np.stack([sf.read(f, dtype="float32")[0][:160000] for f in sorted((SYN / c).glob("*.wav"))]) for c in ("normal", "faulty")}
    pn = rng.permutation(250)  # same as grl_v2 (seed 0: normal permutation drawn first)
    fit, held = gen["normal"][pn[50:]], gen["normal"][pn[:50]]
    probe_ids = np.random.default_rng(1).choice(1000, 200, replace=False)

    r_db, s_db = rms_db(real), rms_db(fit)
    A.log(f"audio_match: loudness real {r_db.mean():.1f} +- {r_db.std():.1f} dB, synthetic {s_db.mean():.1f} +- {s_db.std():.1f} dB")
    qs = np.sort(s_db); qr = np.sort(r_db)

    def loud(X):
        d = rms_db(X)
        q = np.interp(d, qs, np.linspace(0, 1, len(qs)))
        target = np.interp(q, np.linspace(0, 1, len(qr)), qr)
        return (X * 10 ** ((target - d) / 20)[:, None]).astype(np.float32)

    g = np.sqrt(power_spec(real[rng.choice(1000, 300, replace=False)]) / (power_spec(loud(fit)) + 1e-20))
    g = np.convolve(g, np.ones(9) / 9, mode="same")
    f_hz = np.arange(len(g)) * 16000 / NFFT
    A.log("audio_match: EQ gain dB at 100/500/1k/2k/4k/7k Hz: " +
          " ".join(f"{20 * np.log10(g[np.argmin(abs(f_hz - f))]):+.1f}" for f in (100, 500, 1000, 2000, 4000, 7000)))
    eq = lambda X: apply_eq(X, g).astype(np.float32)

    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    m = BEATs(BEATsConfig(ck["cfg"])); m.load_state_dict(ck["model"]); m = m.to(dev).eval()
    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=160, n_mels=64, f_max=8000)

    @torch.no_grad()
    def beats(X):
        out = []
        for i in range(0, len(X), 10):
            x = torch.from_numpy(np.clip(X[i : i + 10], -1, 1)).to(dev)
            out.append(m.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0].mean(1).cpu())
        return torch.cat(out).numpy()

    logmel = lambda X: np.log(mel(torch.from_numpy(X)).mean(-1).numpy() + 1e-8)

    def cv_auc(X0, X1):
        X, y = np.r_[X0, X1], np.r_[np.zeros(len(X0)), np.ones(len(X1))]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        return roc_auc_score(y, cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1])

    R = real[probe_ids]
    Rb, Rm = beats(R), logmel(R)
    for name, fn in (("none", lambda X: X), ("loud", loud), ("eq", eq), ("loud+eq", lambda X: eq(loud(X)))):
        H = fn(held)
        A.log(f"audio_match [{name:7s}]: probe real vs synthetic normal  BEATs {cv_auc(Rb, beats(H)):.3f} | "
              f"log-mel {cv_auc(Rm, logmel(H)):.3f} | held-out synthetic loudness {rms_db(H).mean():.1f} +- {rms_db(H).std():.1f} dB")

    for c in ("normal", "faulty"):
        (OUT / c).mkdir(parents=True, exist_ok=True)
        Y = eq(loud(gen[c]))
        for i, y in enumerate(Y):
            sf.write(OUT / c / f"{i:03d}.wav", np.clip(y, -1, 1), 16000)
    A.log(f"audio_match: wrote {OUT}")


if __name__ == "__main__":
    main()
