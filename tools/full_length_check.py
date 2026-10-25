"""Full-length check: 10 generated FULL-LENGTH normal clips (AudioGen LoRA r16, 'the sound of <toycar>', cfg 3, same length
as the real clips) vs the 100 real ToyCar normal test clips, whole clips, near-minus-far, through AudioGen's EnCodec.
Reports: real-vs-generated AUC (5-fold logistic regression; log-mel shape, EnCodec latent; lower = closer), level and
low-band tone differences, and how both change second by second. 3 generated clips go on the comparison page.
Runs in envs/audiocraft:  envs/audiocraft/Scripts/python tools/full_length_check.py
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A  # noqa: E402
from audiogen_realism import feat  # noqa: E402

OUT = ROOT / "outputs" / "audiogen" / "full_length"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_name", default="", help="which LoRA (suffix), e.g. full10")
    args = ap.parse_args()
    global OUT
    OUT = OUT.parent / ("full_length" + ("_" + args.lora_name if args.lora_name else ""))
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    A.LORA["rank"] = 16
    A.LORA["name"] = args.lora_name
    m, cond, t5, tid = A.load_model(dev)
    D = A.Data("ToyCar", dev)
    lab = np.array(D.meta["test"]["label"])
    L = D.wav["test"].shape[-1]
    files = sorted(OUT.glob("gen_*.wav"))
    if len(files) < 10:
        m.set_generation_params(duration=L / 16000, cfg_coef=3.0)
        for b in range(2):
            torch.manual_seed(700 + b)
            with torch.no_grad():
                w = m.generate([f"the sound of {A.TOKEN}"] * 5).cpu().numpy()[:, 0, :L]
            for j, x in enumerate(w):
                sf.write(OUT / f"gen_{5 * b + j:02d}.wav", x, 16000)
        files = sorted(OUT.glob("gen_*.wav"))
    gen = [sf.read(f, dtype="float32")[0][:L] for f in files]
    print(f"clip length {L / 16000:.1f} s; generated {len(gen)} clips of {min(len(g) for g in gen) / 16000:.1f} s")
    real = []
    with torch.no_grad():
        for i in np.where(lab == 0)[0]:
            x = D.audio("test", [int(i)])
            real.append(m.compression_model.decode(*m.compression_model.encode(x))[0, 0, :L].cpu().numpy())

    @torch.no_grad()
    def F(ws):
        x = torch.as_tensor(np.stack(ws), device=dev)[:, None]
        enc = torch.cat([m.compression_model.encoder(x[i : i + 8]).mean(-1).cpu() for i in range(0, len(x), 8)]).numpy()
        ls = np.stack([feat(w) for w in ws])
        return ls, enc

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lr, er = F(real)
    lg, eg = F(gen)
    shape = lambda X: X - X.mean(1, keepdims=True)
    for name, R, G in (("log-mel shape", shape(lr), shape(lg)), ("EnCodec latent", er, eg)):
        X = np.r_[R, G]
        y = np.r_[np.zeros(len(R)), np.ones(len(G))]
        p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)), X, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        print(f"full clips, real normal vs generated, {name}: AUC {roc_auc_score(y, p):.3f}  (5 s middle pieces earlier: "
              f"{'0.910' if 'log' in name else '0.915'})")
    print(f"level generated - real {lg.mean(1).mean() - lr.mean(1).mean():+.1f} dB; level spread across clips: generated "
          f"{lg.mean(1).std():.1f} dB, real {lr.mean(1).std():.1f} dB; low-band (<~800 Hz) tone generated - real "
          f"{(lg[:, :8].mean(1) - lg.mean(1)).mean() - (lr[:, :8].mean(1) - lr.mean(1)).mean():+.1f} dB")
    print("\nsecond by second (average over clips):   real level / low-band tone   |   generated level / low-band tone")
    for t in range(int(L / 16000)):
        fr = np.stack([feat(w[t * 16000 : (t + 1) * 16000]) for w in real])
        fg = np.stack([feat(w[t * 16000 : (t + 1) * 16000]) for w in gen])
        print(f"  {t:2d}-{t + 1:2d} s:   {fr.mean():7.1f} dB / {(fr[:, :8].mean(1) - fr.mean(1)).mean():+5.1f} dB   |   "
              f"{fg.mean():7.1f} dB / {(fg[:, :8].mean(1) - fg.mean(1)).mean():+5.1f} dB")

    ref_vmax = float(np.median([A.mel_db(w).max() for w in real[:10]]))
    entries = []
    for j, w in enumerate(gen[:3]):
        base = {"clip": f"AudioGen full-length {j + 1}{A.tag()}", "machine": f"ToyCar AudioGen{A.tag()}", "label": "generated",
                "file": f"full {L / 16000:.0f} s, sample {j + 1}", "domain": ""}
        entries.append(A.page_entry(base, "full-length normal [toycar]", w, ref_vmax,
                                    f"'the sound of [toycar]', {L / 16000:.0f} s, cfg 3, seed {700 + j // 5}",
                                    mic="near minus far (generated)"))
    A.page_add(entries)


if __name__ == "__main__":
    main()
