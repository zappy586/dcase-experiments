"""What exactly differs between AudioGen normal-prompt clips (LoRA r16, cfg 3; outputs/audiogen/realism/p0_*) and real
ToyCar normals (near-minus-far, middle 5 s, round-tripped through AudioGen's EnCodec)? Per 64-band log-mel (dB):
  tone       : time-averaged level, generated minus real (mean level removed first)
  variety    : spread ACROSS clips of the time-averaged level, generated / real
  steadiness : spread OVER TIME within a clip (frame-to-frame std, averaged over clips), generated / real
  also: overall level, and the strongest periodic modulation of the broadband envelope (motor rhythm).
Runs in envs/audiocraft."""
import sys
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A
dev = torch.device("cuda")
A.LORA["rank"] = 0
m, *_ = A.load_model(dev)
D = A.Data("ToyCar", dev)
lab = np.array(D.meta["test"]["label"])
real = []
with torch.no_grad():
    for i in np.where(lab == 0)[0]:
        x = D.audio("test", [int(i)]); mid = x.shape[-1] // 2; x = x[..., mid - 40000 : mid + 40000]
        real.append(m.compression_model.decode(*m.compression_model.encode(x))[0, 0, :80000].cpu().numpy())
gen = [sf.read(f, dtype="float32")[0][:80000] for f in sorted((ROOT / "outputs/audiogen/realism").glob("p0_*_r16.wav"))]
melT = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=256, n_mels=64, f_max=8000)
fc = torchaudio.functional.melscale_fbanks(513, 0, 8000, 64, 16000).argmax(0).numpy() * 16000 / 1024  # band centre, Hz
def stats(ws):
    S = np.stack([10 * np.log10(melT(torch.as_tensor(w)).numpy() + 1e-10) for w in ws])  # (N, 64, T)
    avg = S.mean(-1)
    env = 10 * np.log10((10 ** (S / 10)).sum(1))  # broadband envelope dB per frame
    e = env - env.mean(1, keepdims=True)
    spec = np.abs(np.fft.rfft(e, axis=1)) ** 2; f = np.fft.rfftfreq(e.shape[1], 256 / 16000)
    band = (f > 1) & (f < 25)
    pk = f[band][spec[:, band].mean(0).argmax()]
    return avg - avg.mean(1, keepdims=True), avg.mean(1), S.std(-1).mean(0), pk, spec[:, band].mean(0), f[band]
r_shape, r_lvl, r_t, r_pk, r_ms, fm = stats(real)
g_shape, g_lvl, g_t, g_pk, g_ms, _ = stats(gen)
tone = g_shape.mean(0) - r_shape.mean(0); var = g_shape.std(0) / (r_shape.std(0) + 1e-6); steady = g_t / r_t
print(f"overall level: generated {g_lvl.mean() - r_lvl.mean():+.1f} dB vs real; level spread across clips: generated {g_lvl.std():.1f} dB, real {r_lvl.std():.1f} dB")
print(f"spread of the tone ACROSS clips (median over bands): generated / real = {np.median(var):.2f}")
print(f"fluctuation OVER TIME within a clip (median over bands): generated / real = {np.median(steady):.2f}")
print(f"strongest envelope rhythm 1-25 Hz: real {r_pk:.1f} Hz, generated {g_pk:.1f} Hz; rhythm strength generated/real {g_ms.sum() / r_ms.sum():.2f}")
print("\nband (Hz)    tone gen-real (dB)  variety gen/real  over-time gen/real")
for b in list(range(0, 64, 6)) + [63]:
    print(f"{fc[b]:8.0f}     {tone[b]:+8.1f}            {var[b]:6.2f}            {steady[b]:6.2f}")
