"""Do AudioGen (LoRA r16 + [toycar]) clips from the normal prompt look like REAL normal ToyCar clips, and clips from the
fault prompts like REAL anomalous ones?

Generated: N clips per prompt (tools/audiogen_ti.py ANOMALY_PROMPTS), 5 s, cached in outputs/audiogen/realism/.
Real: the 200 ToyCar test clips (100 normal, 100 anomalous), near-minus-far with the same machine gain as training,
middle 5 s (the generated clips are steady 5 s pieces).
Feature: time-averaged 64-band log-mel spectrum (dB); 'shape' = with each clip's mean level removed (so loudness of
the generator output does not decide), 'with level' = as is.
Checks (labels of REAL clips are used only to build the reference, never to tune anything on the generated clips):
  1. classifier: logistic regression real normal vs real anomaly (its own 5-fold AUC on real clips says how much it
     can be trusted), then the share of each prompt's generated clips it calls anomalous
  2. nearest real clips: share of anomalous clips among the 5 nearest real clips
  3. distance to the real-normal average vs the real-anomaly average
Runs in envs/audiocraft:  envs/audiocraft/Scripts/python tools/audiogen_realism.py --n 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A  # noqa: E402

CACHE = ROOT / "outputs" / "audiogen" / "realism"


def feat(w):
    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=256, n_mels=64, f_max=8000)(torch.as_tensor(w))
    return (10 * torch.log10(mel + 1e-10)).mean(-1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--lora", type=int, default=16)
    args = ap.parse_args()
    A.LORA["rank"] = args.lora
    dev = torch.device("cuda")
    CACHE.mkdir(parents=True, exist_ok=True)
    gen = {}
    keyof = lambda name: f"p{list(A.ANOMALY_PROMPTS).index(name)}"
    need = [k for k in A.ANOMALY_PROMPTS if len(list(CACHE.glob(f"{keyof(k)}_*_r{args.lora}.wav"))) < args.n]
    if need:
        m, cond, t5, tid = A.load_model(dev)
        m.set_generation_params(duration=5.0, cfg_coef=3.0)
    for name, prompt in A.ANOMALY_PROMPTS.items():
        key = keyof(name)
        if name in need:
            for b in range(0, args.n, 10):
                torch.manual_seed(1000 + b)
                with torch.no_grad():
                    w = m.generate([prompt] * min(10, args.n - b)).cpu().numpy()[:, 0]
                for j, x in enumerate(w):
                    sf.write(CACHE / f"{key}_{b + j:03d}_r{args.lora}.wav", x, 16000)
            print(f"generated {args.n} for {name!r}", flush=True)
        gen[name] = np.stack([feat(sf.read(f, dtype="float32")[0]) for f in sorted(CACHE.glob(f"{key}_*_r{args.lora}.wav"))[: args.n]])

    D = A.Data("ToyCar", dev)
    lab, dom = np.array(D.meta["test"]["label"]), np.array(D.meta["test"]["domain"])
    R = []
    for i in range(0, len(lab), 20):
        a = D.audio("test", list(range(i, min(i + 20, len(lab)))))[:, 0].cpu().numpy()
        mid = a.shape[1] // 2
        R += [feat(x[mid - 40000 : mid + 40000]) for x in a]
    R = np.stack(R)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    for mode in ("shape", "with level"):
        tf = (lambda X: X - X.mean(1, keepdims=True)) if mode == "shape" else (lambda X: X)
        Xr = tf(R)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        cv = cross_val_predict(clf, Xr, lab, cv=5, method="predict_proba")[:, 1]
        clf.fit(Xr, lab)
        mu_n, mu_a = Xr[lab == 0].mean(0), Xr[lab == 1].mean(0)
        sd = Xr.std(0) + 1e-6
        print(f"\n== feature: time-averaged 64-band log-mel, {mode} ==")
        print(f"reference classifier on REAL clips (5-fold): AUC {roc_auc_score(lab, cv):.3f}; real normals called anomalous "
              f"{(cv[lab == 0] > 0.5).mean():.0%}, real anomalies called anomalous {(cv[lab == 1] > 0.5).mean():.0%}")
        lvl = lambda X: X.mean()
        print(f"{'':30s} {'called anomalous':>17s} {'anomalous among 5 nearest real':>31s} {'closer to anomaly average':>26s}"
              f"   level vs real normals")
        for name, G in gen.items():
            Xg = tf(G)
            p = clf.predict_proba(Xg)[:, 1]
            d = (((Xg[:, None] - Xr[None]) / sd) ** 2).sum(-1)
            nn5 = lab[np.argsort(d, 1)[:, :5]].mean()
            closer = ((((Xg - mu_a) / sd) ** 2).sum(1) < (((Xg - mu_n) / sd) ** 2).sum(1)).mean()
            print(f"{name:30s} {(p > 0.5).mean():17.0%} {nn5:31.0%} {closer:26.0%}   {G.mean() - R[lab == 0].mean():+6.1f} dB")
        print(f"{'(real normals, for scale)':30s} {(cv[lab == 0] > 0.5).mean():17.0%}")
        print(f"{'(real anomalies, for scale)':30s} {(cv[lab == 1] > 0.5).mean():17.0%}")


if __name__ == "__main__":
    main()
