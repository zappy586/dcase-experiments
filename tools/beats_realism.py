"""Same realism check as tools/audiogen_realism.py, but in BEATs embedding space (BEATs iter3, final layer, average over
all tokens -> 768 numbers per clip). Generated clips: outputs/audiogen/realism/p{0..3}_*_r16.wav. Real: the 200 ToyCar
test clips, near-minus-far, machine gain, middle 5 s.
Checks: logistic-regression reference (real normal vs real anomaly; its 5-fold AUC shown), share of each prompt called
anomalous, share of anomalous clips among the 5 nearest real clips (cosine), cosine similarity to the real-normal and
real-anomaly averages.
Usage (main env): uv run python tools/beats_realism.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from BEATs import BEATs, BEATsConfig  # noqa: E402

GEN = ROOT / "outputs" / "audiogen" / "realism"
NAMES = list(A.ANOMALY_PROMPTS)


@torch.no_grad()
def embed(model, wavs, dev):
    out = []
    for i in range(0, len(wavs), 16):
        x = torch.as_tensor(np.stack(wavs[i : i + 16]), dtype=torch.float32, device=dev)
        f = model.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0]
        out.append(f.mean(1).cpu())
    return torch.cat(out).numpy()


def main():
    dev = torch.device("cuda")
    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    model = BEATs(BEATsConfig(ck["cfg"]))
    model.load_state_dict(ck["model"])
    model = model.to(dev).eval()

    gen = {}
    for k, name in enumerate(NAMES):
        files = sorted(GEN.glob(f"p{k}_*_r16.wav"))
        gen[name] = embed(model, [sf.read(f, dtype="float32")[0][:80000] for f in files], dev)
    D = A.Data("ToyCar", dev)
    lab = np.array(D.meta["test"]["label"])
    real = []
    for i in range(0, len(lab), 20):
        a = D.audio("test", list(range(i, min(i + 20, len(lab)))))[:, 0].cpu().numpy()
        mid = a.shape[1] // 2
        real += [x[mid - 40000 : mid + 40000] for x in a]
    R = embed(model, real, dev)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    cv = cross_val_predict(clf, R, lab, cv=5, method="predict_proba")[:, 1]
    clf.fit(R, lab)
    nrm = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    Rn = nrm(R)
    mu_n, mu_a = nrm(R[lab == 0].mean(0, keepdims=True))[0], nrm(R[lab == 1].mean(0, keepdims=True))[0]
    print(f"BEATs reference classifier on REAL clips (5-fold): AUC {roc_auc_score(lab, cv):.3f}; real normals called "
          f"anomalous {(cv[lab == 0] > 0.5).mean():.0%}, real anomalies {(cv[lab == 1] > 0.5).mean():.0%}")
    print(f"cosine similarity of the real-normal and real-anomaly averages to each other: {mu_n @ mu_a:.4f}\n")
    print(f"{'':30s} {'called anomalous':>16s} {'faulty among 5 nearest':>23s} {'cos to normal avg':>18s} {'cos to faulty avg':>18s}"
          f" {'cos to nearest real':>20s}")
    for name, G in list(gen.items()) + [("(real normals)", R[lab == 0]), ("(real anomalies)", R[lab == 1])]:
        Gn = nrm(G)
        p = clf.predict_proba(G)[:, 1] if not name.startswith("(") else cv[lab == (0 if "normals" in name else 1)]
        sim = Gn @ Rn.T
        if name.startswith("("):  # exclude each real clip itself
            idx = np.where(lab == (0 if "normals" in name else 1))[0]
            sim[np.arange(len(idx)), idx] = -2
        nn5 = lab[np.argsort(-sim, 1)[:, :5]].mean()
        print(f"{name:30s} {(p > 0.5).mean():16.0%} {nn5:23.0%} {(Gn @ mu_n).mean():18.4f} {(Gn @ mu_a).mean():18.4f} "
              f"{sim.max(1).mean():20.4f}")


if __name__ == "__main__":
    main()
