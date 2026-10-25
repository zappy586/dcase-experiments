"""Which sampling settings make AudioGen (LoRA r16, prompt 'the sound of <toycar>') closest to REAL ToyCar normals?
Real reference: the 100 real normal test clips (near-minus-far, middle 5 s), sent through AudioGen's EnCodec so both
sides carry the same codec. For each setting: 20 generated 5 s clips; AUC of a classifier real-normal vs generated
(5-fold, lower = more realistic; 0.5 = indistinguishable) in log-mel shape, EnCodec latent and AudioGen-LM layer 24.
Settings: guidance strength (cfg) 3 (default) / 2 / 1 (none), top-k 250 (default) vs pure sampling (top-k 0).
Runs in envs/audiocraft:  envs/audiocraft/Scripts/python tools/sampling_realism.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A  # noqa: E402
from audiogen_realism import feat  # noqa: E402

SETTINGS = {"cfg 3, top-k 250 (used so far)": dict(cfg_coef=3.0, top_k=250), "cfg 2, top-k 250": dict(cfg_coef=2.0, top_k=250),
            "cfg 1 (no guidance), top-k 250": dict(cfg_coef=1.0, top_k=250), "cfg 1, pure sampling": dict(cfg_coef=1.0, top_k=0)}


def main():
    dev = torch.device("cuda")
    A.LORA["rank"] = 16
    m, cond, t5, tid = A.load_model(dev)
    D = A.Data("ToyCar", dev)
    lab = np.array(D.meta["test"]["label"])
    real = []
    with torch.no_grad():
        for i in np.where(lab == 0)[0]:
            x = D.audio("test", [int(i)])
            mid = x.shape[-1] // 2
            x = x[..., mid - 40000 : mid + 40000]
            real.append(m.compression_model.decode(*m.compression_model.encode(x))[0, 0, :80000].cpu().numpy())
    gen = {}
    for name, kw in SETTINGS.items():
        m.set_generation_params(duration=5.0, **kw)
        w = []
        for b in range(2):
            torch.manual_seed(500 + b)
            with torch.no_grad():
                w += list(m.generate([f"the sound of {A.TOKEN}"] * 10).cpu().numpy()[:, 0, :80000])
        gen[name] = w
        print(f"generated 20 with {name}", flush=True)

    hid = {}
    h = m.lm.transformer.layers[23].register_forward_hook(lambda mod, inp, out: hid.__setitem__(0, out))
    A.LORA["rank"] = 0

    @torch.no_grad()
    def F(ws):
        x = torch.as_tensor(np.stack(ws), device=dev)[:, None]
        lm, enc = [], []
        for i in range(0, len(x), 8):
            xb = x[i : i + 8]
            enc.append(m.compression_model.encoder(xb).mean(-1).cpu())
            ct = A.cond_tensors(cond, t5, ["the sound of a machine"] * len(xb))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m.lm.compute_predictions(A.encode(m, xb), [], ct)
            lm.append(hid[0].float().mean(1).cpu())
        ls = np.stack([feat(w) for w in ws])
        return {"log-mel shape": ls - ls.mean(1, keepdims=True), "EnCodec latent": torch.cat(enc).numpy(),
                "AudioGen LM layer 24 (with LoRA)": torch.cat(lm).numpy()}

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Fr = F(real)
    print(f"\n{'setting':34s} " + " ".join(f"{k:>34s}" for k in Fr) + "   level vs real (dB)")
    for name, w in gen.items():
        Fg = F(w)
        aucs = []
        for k in Fr:
            X = np.r_[Fr[k], Fg[k]]
            y = np.r_[np.zeros(len(Fr[k])), np.ones(len(Fg[k]))]
            p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)), X, y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
            aucs.append(roc_auc_score(y, p))
        lvl = 10 * np.log10(np.mean([np.mean(np.square(x)) for x in w]) / np.mean([np.mean(np.square(x)) for x in real]))
        print(f"{name:34s} " + " ".join(f"{a:34.3f}" for a in aucs) + f"   {lvl:+.1f}")
    h.remove()


if __name__ == "__main__":
    main()
