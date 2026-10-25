"""General-audio antennas with a better-estimated normal covariance (Dasheng-1.2B on near-only residual audio).

Problem: 900 normal clip vectors cannot fill Dasheng's 1536 dimensions, so the antennas partly point at directions
where normal clips were simply never sampled. Fix, with no new hyperparameter:
  S_norm = S_clip + S_within / K
  S_clip   covariance of the normal clip averages (900 vectors, as before)
  S_within within-clip covariance of 1-second window averages (K = 10 windows per clip, 9000 vectors, full rank)
  S_within / K is the noise a clip average carries from within-clip variation, which fills the unsampled directions.
Scoring is unchanged (whole-clip averages), layers z-normalised on 100 held-out normal clips and summed.
Reported, same features: gen-loud and Mahalanobis, each with the old (clip-only, constant regulariser) and the new
covariance; Ledoit-Wolf Mahalanobis as before.
Usage: uv run python -u tools/window_cov.py [machines]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import hmean
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clora_gen import DASHENG_LAYERS  # noqa: E402
from dcase_ae.efficientad import WavStore  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402
from tools.simple_baselines import MACHINES, near_only_audio  # noqa: E402

K, RANK = 10, 32


def main():
    machines = sys.argv[1].split(",") if len(sys.argv) > 1 else MACHINES
    dev = torch.device("cuda")
    from dasheng import dasheng_12B
    model = dasheng_12B(path="pretrained/dasheng/dasheng_12b.pt").to(dev).eval()
    grab = {}
    for L in DASHENG_LAYERS:
        model.blocks[L - 1].register_forward_hook(lambda m, a, o, L=L: grab.__setitem__(L, o))
    G = torch.load("cache/esc50_dasheng12b_rows.pt")
    print(f"=== window_cov: machines {machines}", flush=True)
    methods = ("gen-loud old", "gen-loud new", "maha old", "maha new", "maha-lw")
    table = {k: [] for k in methods}
    for m in machines:
        d = torch.load(f"cache/rows_dasheng12b-resid_{m}.pt", weights_only=False)  # our own cache
        wcache = Path(f"cache/within_dasheng12b-resid_{m}.pt")
        if wcache.exists():
            Sw = torch.load(wcache)
        else:
            import random
            tr = WavStore("cache", m, "train")
            idx = list(range(len(tr)))
            random.Random(0).shuffle(idx)  # same split as the clip cache
            fit = idx[100:]
            D = 1536
            acc = {L: torch.zeros(D, D, device=dev, dtype=torch.float64) for L in DASHENG_LAYERS}
            n = 0
            with torch.no_grad():
                for i in range(0, len(fit), 8):
                    w = near_only_audio(tr.get(fit[i : i + 8]), dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        model(w)
                    for L in DASHENG_LAYERS:
                        t = grab[L].double()  # (B, T, D)
                        T = t.shape[1] // K * K
                        win = t[:, :T].reshape(len(t), K, -1, D).mean(2)  # (B, K, D) 1-second averages
                        c = (win - win.mean(1, keepdim=True)).reshape(-1, D)
                        acc[L] += c.T @ c
                    n += len(w) * (K - 1)
            Sw = {L: (acc[L] / n).cpu() for L in DASHENG_LAYERS}
            torch.save(Sw, wcache)
        lab, dom = d["lab"], d["dom"]
        scores = {k: {} for k in methods}
        for L in DASHENG_LAYERS:
            X, V, T = (d[s][L].double().numpy()[:, 0] for s in ("fit", "val", "test"))  # (N, D) clip averages
            D = X.shape[1]
            mu = X.mean(0)
            Sc = (X - mu).T @ (X - mu) / len(X)
            Gc = G[L].double().numpy()[:, 0] - mu
            Sg = Gc.T @ Gc / len(Gc)
            reg = lambda S: S + 1e-3 * np.trace(S) / D * np.eye(D)
            Sns = {"old": reg(Sc), "new": Sc + Sw[L].numpy() / K}
            Sg = reg(Sg)
            for tag, Sn in Sns.items():
                ev, U = np.linalg.eigh(Sn)
                W = U / np.sqrt(np.maximum(ev, 1e-12))
                Eg = np.linalg.eigh(W.T @ Sg @ W)[1][:, -RANK:]
                A = W @ Eg
                for split, Q in (("val", V), ("test", T)):
                    Z = Q - mu
                    scores[f"gen-loud {tag}"][(split, L)] = ((Z @ A) ** 2).sum(1)
                    scores[f"maha {tag}"][(split, L)] = ((Z @ W) ** 2).sum(1)
            lw = LedoitWolf().fit(X)
            for split, Q in (("val", V), ("test", T)):
                Z = Q - lw.location_
                scores["maha-lw"][(split, L)] = np.einsum("nd,de,ne->n", Z, lw.precision_, Z)
        for k in methods:
            s = sum((scores[k][("test", L)] - scores[k][("val", L)].mean()) / (scores[k][("val", L)].std() + 1e-9)
                    for L in DASHENG_LAYERS)
            table[k].append(compute_metrics(s, lab, dom))
        print(f"{m:11s} " + "  ".join(f"{k} {table[k][-1]['auc_all']:.3f}" for k in methods), flush=True)
    print(f"\n{'method':13s} {'meanAUC':>8s} {'Omega':>7s}   per machine")
    for k in methods:
        om = hmean([x for r in table[k] for x in (r["auc_source"], r["auc_target"], r["pauc"])])
        print(f"{k:13s} {np.mean([r['auc_all'] for r in table[k]]):8.3f} {om:7.3f}   " + " ".join(f"{r['auc_all']:.3f}" for r in table[k]), flush=True)


if __name__ == "__main__":
    main()
