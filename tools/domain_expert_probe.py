"""Linear check of the domain-expert idea (seconds). Data: the AE's target (256-band near-only spectrum), standardised.
  baseline   : PCA-36 subspace of all 1000 training clips (linear analogue of our AE); score = residual energy
  experts    : source expert = PCA-36 of the 990 source clips, centred on the source mean
               target expert = source PCA-(36-r) + r extra directions from the 10 target clips' residuals (low-rank
               update, the linear analogue of a rank-r LoRA), centred on the target mean
               each expert's error divided by its typical error on its own domain's training clips (target: leave-one-out)
               score = min over the two experts (no test domain labels needed)
Every r is shown; nothing is picked on test labels."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from crossband_probe import targets, MACHINES
from dcase_ae.evaluation import compute_metrics
K = 36

def pcs(X, k):
    return np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:k].T

def err(Y, mu, V):
    Z = Y - mu
    return ((Z - Z @ V @ V.T) ** 2).mean(1)

def target_expert(S, T, r):
    Vs = pcs(S, K - r); mt = T.mean(0)
    R = (T - mt) - (T - mt) @ Vs @ Vs.T
    U = np.linalg.svd(R, full_matrices=False)[2][:r].T
    V = np.linalg.qr(np.hstack([Vs, U]))[0]
    return mt, V

rows = {}
for m in MACHINES:
    Ytr, Yte = targets(m, "train"), targets(m, "test"); mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-3; Ytr, Yte = (Ytr - mu) / sd, (Yte - mu) / sd
    trdom = np.array(json.loads((ROOT / "cache" / f"wav_{m}_train.json").read_text())["domain"])
    meta = json.loads((ROOT / "cache" / f"wav_{m}_test.json").read_text()); lab, dom = np.array(meta["label"]), np.array(meta["domain"])
    S, T = Ytr[trdom == 0], Ytr[trdom == 1]
    res = {"PCA-36 all": compute_metrics(err(Yte, Ytr.mean(0), pcs(Ytr, K)), lab, dom)}
    ms, Vs = S.mean(0), pcs(S, K); es = err(Yte, ms, Vs) / np.median(err(S, ms, Vs))
    for r in (2, 4, 8):
        loo = [err(T[j:j+1], *target_expert(S, np.delete(T, j, 0), r))[0] for j in range(len(T))]
        et = err(Yte, *target_expert(S, T, r)) / np.median(loo)
        res[f"experts r={r}"] = compute_metrics(np.minimum(es, et), lab, dom)
    rows[m] = res
    print(f"{m:11s} " + " | ".join(f"{k}: src {v['auc_source']:.3f} tgt {v['auc_target']:.3f} off {v['official']:.3f}" for k, v in res.items()), flush=True)
print("mean official " + " | ".join(f"{k} {np.mean([rows[m][k]['official'] for m in rows]):.3f}" for k in rows["ToyCar"]))
print("mean tgt AUC  " + " | ".join(f"{k} {np.mean([rows[m][k]['auc_target'] for m in rows]):.3f}" for k in rows["ToyCar"]))
