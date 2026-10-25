"""Generated (AudioGen LoRA r16, 4 prompts x 20) vs real ToyCar test clips (100 normal, 100 anomalous, near-minus-far,
middle 5 s) seen through different encoders:
  AudioGen LM (no LoRA)  : hidden state of layer 24 and of the last layer (48), averaged over time; the clip's EnCodec
                           tokens are fed with the fixed neutral prompt 'the sound of a machine'
  EnCodec (AudioGen's)   : 16 kHz EnCodec encoder output before quantisation (128 x 50 per s), averaged over time
  BEATs                  : iter3 final layer, token average
  log-mel shape          : time-averaged 64-band log-mel, mean level removed
Tests (5-fold CV inside each where training and testing use the same group):
  1 real vs generated    : can a classifier tell generated clips from real ones (AUC; 0.5 = indistinguishable)
  2 where prompts land   : classifier trained on REAL normal vs REAL anomaly -> share of each prompt called anomalous
  3 generated -> real    : classifier trained on generated normal-prompt vs fault-prompt clips (20 vs 20) -> AUC on real
Runs in envs/audiocraft:  envs/audiocraft/Scripts/python tools/encoder_realism.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from audiogen_realism import feat  # noqa: E402

GEN = ROOT / "outputs" / "audiogen" / "realism"
NAMES = list(A.ANOMALY_PROMPTS)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrip", action="store_true", help="send the REAL clips through AudioGen's EnCodec (encode + "
                    "decode) first, like the generated ones; also reports real vs round-tripped real")
    args = ap.parse_args()
    dev = torch.device("cuda")
    gw, gk = [], []
    for k in range(4):
        for f in sorted(GEN.glob(f"p{k}_*_r16.wav")):
            gw.append(sf.read(f, dtype="float32")[0][:80000]); gk.append(k)
    gk = np.array(gk)
    D = A.Data("ToyCar", dev)
    lab, dom = np.array(D.meta["test"]["label"]), np.array(D.meta["test"]["domain"])
    rw = []
    for i in range(0, len(lab), 20):
        a = D.audio("test", list(range(i, min(i + 20, len(lab)))))[:, 0].cpu().numpy(); mid = a.shape[1] // 2
        rw += [x[mid - 40000 : mid + 40000] for x in a]
    feats = {}
    A.LORA["rank"] = 0
    m, cond, t5, tid = A.load_model(dev)  # plain AudioGen (token file only affects the <toycar> word, not used here)
    raw = list(rw)
    if args.roundtrip:
        with torch.no_grad():
            rw = [m.compression_model.decode(*m.compression_model.encode(torch.as_tensor(w, device=dev)[None, None]))[0, 0, :80000].cpu().numpy()
                  for w in rw]
        print("real clips sent through AudioGen's EnCodec (encode + decode)")
    allw = rw + gw + (raw if args.roundtrip else [])
    hid = {}
    hooks = [m.lm.transformer.layers[i].register_forward_hook(lambda mod, inp, out, i=i: hid.__setitem__(i, out)) for i in (23, 47)]
    L24, L48, E = [], [], []
    with torch.no_grad():
        for i in range(0, len(allw), 8):
            x = torch.as_tensor(np.stack(allw[i : i + 8]), device=dev)[:, None]
            E.append(m.compression_model.encoder(x).mean(-1).cpu())
            codes = A.encode(m, x)
            ct = A.cond_tensors(cond, t5, ["the sound of a machine"] * len(x))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m.lm.compute_predictions(codes, [], ct)
            L24.append(hid[23].float().mean(1).cpu()); L48.append(hid[47].float().mean(1).cpu())
    for h in hooks:
        h.remove()
    feats["AudioGen LM layer 24"] = torch.cat(L24).numpy()
    feats["AudioGen LM layer 48 (last)"] = torch.cat(L48).numpy()
    feats["EnCodec 16 kHz (AudioGen's)"] = torch.cat(E).numpy()
    del m
    torch.cuda.empty_cache()
    from BEATs import BEATs, BEATsConfig
    from beats_realism import embed
    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    bm = BEATs(BEATsConfig(ck["cfg"])); bm.load_state_dict(ck["model"]); bm = bm.to(dev).eval()
    feats["BEATs"] = embed(bm, allw, dev)
    X = np.stack([feat(w) for w in allw])
    feats["log-mel shape"] = X - X.mean(1, keepdims=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    cvk = StratifiedKFold(5, shuffle=True, random_state=0)
    nr = len(rw)
    rng = np.random.default_rng(0)
    bal = np.r_[np.where(gk == 0)[0], rng.choice(np.where(gk > 0)[0], 20, replace=False)]
    print(f"{'encoder':30s} {'1 real vs generated':>20s} | {'2 real n/a ref AUC':>18s} " + " ".join(f"{n[:14]:>14s}" for n in NAMES)
          + f" {'real n':>7s} {'real a':>7s} | {'3 gen->real AUC':>15s}")
    for name, F in feats.items():
        if args.roundtrip:
            Fr = np.r_[F[:nr], F[nr + len(gk):]]
            y = np.r_[np.ones(nr), np.zeros(nr)]
            a = roc_auc_score(y, cross_val_predict(clf(), Fr, y, cv=cvk, method="predict_proba")[:, 1])
            print(f"   {name}: real vs the same real clips round-tripped through the codec: AUC {a:.3f}")
            F = F[: nr + len(gk)]
        R, G = F[:nr], F[nr:]
        y_rg = np.r_[np.zeros(nr), np.ones(len(G))]
        auc_rg = roc_auc_score(y_rg, cross_val_predict(clf(), F, y_rg, cv=cvk, method="predict_proba")[:, 1])
        cv = cross_val_predict(clf(), R, lab, cv=cvk, method="predict_proba")[:, 1]
        ref = clf().fit(R, lab)
        pg = ref.predict_proba(G)[:, 1]
        shares = [(pg[gk == k] > 0.5).mean() for k in range(4)]
        g2r = clf().fit(G[bal], (gk[bal] > 0).astype(int)).predict_proba(R)[:, 1]
        print(f"{name:30s} {auc_rg:20.3f} | {roc_auc_score(lab, cv):18.3f} " + " ".join(f"{s:14.0%}" for s in shares)
              + f" {(cv[lab == 0] > .5).mean():7.0%} {(cv[lab == 1] > .5).mean():7.0%} | {roc_auc_score(lab, g2r):15.3f}")


if __name__ == "__main__":
    main()
