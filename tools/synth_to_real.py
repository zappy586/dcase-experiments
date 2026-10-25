"""Train on GENERATED clips only (AudioGen LoRA r16: normal prompt = 0, the three fault prompts = 1), test on REAL ToyCar
test clips (100 normal, 100 anomalous). Features: time-averaged 64-band log-mel shape (mean level removed), and BEATs
(iter3, final layer, token average). Classifier: standardise + logistic regression (C=0.1). Real labels only measure.
Also the same with balanced classes (20 normal-prompt clips vs 20 drawn from the fault prompts).
Usage: uv run python tools/synth_to_real.py"""
import glob, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A
from audiogen_realism import feat
from beats_realism import embed, GEN
from dcase_ae.evaluation import compute_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(ROOT / "third_party" / "beats"))
from BEATs import BEATs, BEATsConfig
dev = torch.device("cuda")
gw, gy = [], []
for k in range(4):
    for f in sorted(GEN.glob(f"p{k}_*_r16.wav")):
        gw.append(sf.read(f, dtype="float32")[0][:80000]); gy.append(int(k > 0))
gy = np.array(gy)
D = A.Data("ToyCar", dev)
lab, dom = np.array(D.meta["test"]["label"]), np.array(D.meta["test"]["domain"])
rw = []
for i in range(0, len(lab), 20):
    a = D.audio("test", list(range(i, min(i + 20, len(lab)))))[:, 0].cpu().numpy(); mid = a.shape[1] // 2
    rw += [x[mid - 40000 : mid + 40000] for x in a]
ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0], map_location="cpu", weights_only=False)
bm = BEATs(BEATsConfig(ck["cfg"])); bm.load_state_dict(ck["model"]); bm = bm.to(dev).eval()
shape = lambda X: X - X.mean(1, keepdims=True)
feats = {"log-mel shape": (shape(np.stack([feat(w) for w in gw])), shape(np.stack([feat(w) for w in rw]))),
         "BEATs": (embed(bm, gw, dev), embed(bm, rw, dev))}
rng = np.random.default_rng(0)
bal = np.r_[np.where(gy == 0)[0], rng.choice(np.where(gy == 1)[0], 20, replace=False)]
for name, (G, R) in feats.items():
    for tag, idx in (("20 normal-prompt vs 60 fault-prompt", np.arange(len(gy))), ("balanced 20 vs 20", bal)):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(G[idx], gy[idx])
        p = clf.predict_proba(R)[:, 1]
        r = compute_metrics(p, lab, dom)
        print(f"{name:14s} {tag:36s} -> real clips: AUC {r['auc_all']:.3f} src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} "
              f"pAUC {r['pauc']:.3f} official {r['official']:.3f} | real normals called faulty {(p[lab == 0] > .5).mean():.0%}, "
              f"real anomalies {(p[lab == 1] > .5).mean():.0%}")
