"""Controls for the general-audio antennas (clora_gen.py), frozen EAT, clip-level row features, rank 32.

1. Random-direction control. Same scoring as the headline (max over rows, layers z-normalised on held-out normals
   and summed), with antennas:
     gen-loud      top solutions of S_gen a = lambda S_norm a (the method), scaled so a^T S_norm a = 1
     null          bottom eigenvectors of S_norm (as in clora_ad.py)
     rand-whiten   random 32-dim slice of the S_norm-whitened space (a = S_norm^-1/2 q), 20 draws
     rand-low      random 32-dim subspace of the 128 lowest-variance normal directions, whitened, 20 draws
     gen-quiet     bottom solutions of S_gen a = lambda S_norm a (directions general audio uses least): reverse control
2. Overlap. Shift d = mean(anomalous test clips) - mean(normal test clips), per row, measured in the S_norm-whitened
   metric. Reported: share of |S_norm^-1/2 d|^2 that falls inside each antenna set (expected share for a random
   32-dim set = 32/768 = 4.2%). Noise reference: the same share for the shift between two random halves of the normal
   test clips. Uses test labels: analysis only, never for choosing settings.

Usage: uv run python -u tools/clora_controls.py [eat|dasheng12b]
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clora_gen import get_encoder  # noqa: E402
from dcase_ae.efficientad import WavStore, build_cache  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402
from scipy.stats import hmean  # noqa: E402

MACHINES = ["ToyCar", "ToyCarEmu", "bearingEmu", "fan", "gearboxEmu", "sliderEmu", "valveEmu"]
RANK, DRAWS = 32, 20


def main():
    enc = sys.argv[1] if len(sys.argv) > 1 else "eat"
    dev = torch.device("cuda")
    out = Path("outputs/clora_gen/controls" + ("" if enc == "eat" else f"_{enc}"))
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "w", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    LAYERS, row_features = get_encoder(enc, dev)
    G = torch.load(f"cache/esc50_{enc}_rows.pt")
    rng = np.random.default_rng(0)
    summary = {}
    for m in MACHINES:
        for split in ("train", "test"):
            build_cache("data", "cache", m, split)
        tr, te = WavStore("cache", m, "train"), WavStore("cache", m, "test")
        idx = list(range(len(tr)))
        random.Random(0).shuffle(idx)
        val_idx, fit_idx = idx[:100], idx[100:]
        feats = {}
        for name, store, ids in (("fit", tr, fit_idx), ("val", tr, val_idx), ("test", te, list(range(len(te))))):
            parts = [row_features(store.get(ids[i : i + 64])[:, 0]) for i in range(0, len(ids), 64)]
            feats[name] = {L: torch.cat([p[L] for p in parts]).double().numpy() for L in LAYERS}
        lab, dom = np.array(te.meta["label"]), np.array(te.meta["domain"])
        normal = np.flatnonzero(lab == 0)
        halves = rng.permutation(normal)
        h1, h2 = halves[: len(halves) // 2], halves[len(halves) // 2 :]

        # bases per layer; every basis is S_norm-orthonormal-ish so that energy = whitened energy along it
        bases = {L: {} for L in LAYERS}
        overlap = {}
        for L in LAYERS:
            X = feats["fit"][L]
            mu = X.mean(0)
            D = X.shape[-1]
            Xc, Gc = (X - mu).reshape(-1, D), (G[L].double().numpy() - mu).reshape(-1, D)
            reg = lambda S: S + 1e-3 * np.trace(S) / D * np.eye(D)
            Sn, Sg = reg(Xc.T @ Xc / len(Xc)), reg(Gc.T @ Gc / len(Gc))
            ev, U = np.linalg.eigh(Sn)
            W = U / np.sqrt(ev)  # whitening: W^T Sn W = I
            # generalized problem in whitened space: M = W^T Sg W, eigenvectors give S_norm-orthonormal directions
            lg, Eg = np.linalg.eigh(W.T @ Sg @ W)
            b = bases[L]
            b["gen-loud"] = (W @ Eg[:, -RANK:]).T
            b["gen-quiet"] = (W @ Eg[:, :RANK]).T
            b["null"] = U[:, :RANK].T  # raw energy, as in clora_ad.py
            b["null-whitened"] = W[:, :RANK].T
            for r in range(DRAWS):
                q = np.linalg.qr(rng.standard_normal((D, RANK)))[0]
                b[f"rand-whiten#{r}"] = (W @ q).T
                q = np.linalg.qr(rng.standard_normal((128, RANK)))[0]
                b[f"rand-low#{r}"] = (W[:, :128] @ q).T
            # overlap of the real anomaly shift with each set, in the whitened metric
            Z = feats["test"][L] - mu  # (N, rows, D)
            for tag, (a, n) in {"anomaly": (np.flatnonzero(lab == 1), normal), "noise": (h1, h2)}.items():
                d = Z[a].mean(0) - Z[n].mean(0)  # (rows, D)
                dw = d @ W  # whitened coordinates (rows, D)
                tot = (dw**2).sum()
                for k in ("gen-loud", "gen-quiet", "null-whitened"):
                    E = {"gen-loud": Eg[:, -RANK:], "gen-quiet": Eg[:, :RANK], "null-whitened": np.eye(D)[:, :RANK]}[k]
                    overlap.setdefault((tag, k), []).append(float(((dw @ E) ** 2).sum() / tot))

        def score(name, split):
            return {L: (np.einsum("nfd,kd->nfk", feats[split][L] - feats["fit"][L].mean(0), bases[L][name]) ** 2).sum(-1).max(1)
                    for L in LAYERS}

        def evaluate(name):
            sv, st = score(name, "val"), score(name, "test")
            s = sum((st[L] - sv[L].mean()) / (sv[L].std() + 1e-9) for L in LAYERS)
            return compute_metrics(s, lab, dom)

        res = {k: evaluate(k) for k in ("gen-loud", "null", "null-whitened", "gen-quiet")}
        for fam in ("rand-whiten", "rand-low"):
            res[fam] = [evaluate(f"{fam}#{r}") for r in range(DRAWS)]
        summary[m] = {"res": res, "overlap": {f"{t}|{k}": float(np.mean(v)) for (t, k), v in overlap.items()}}
        rw = np.array([r["auc_all"] for r in res["rand-whiten"]])
        rl = np.array([r["auc_all"] for r in res["rand-low"]])
        log(f"--- {m}: AUC gen-loud {res['gen-loud']['auc_all']:.3f} | null {res['null']['auc_all']:.3f} | "
            f"null-whitened {res['null-whitened']['auc_all']:.3f} | gen-quiet {res['gen-quiet']['auc_all']:.3f} | "
            f"rand-whiten {rw.mean():.3f}+-{rw.std():.3f} (max {rw.max():.3f}) | rand-low {rl.mean():.3f}+-{rl.std():.3f} (max {rl.max():.3f})")
        o = summary[m]["overlap"]
        log(f"    share of whitened shift inside 32 directions (random = {RANK / feats["fit"][LAYERS[0]].shape[-1] * 100:.1f}%): "
            + " | ".join(f"{k}: anomaly {o['anomaly|' + k] * 100:.1f}% vs noise {o['noise|' + k] * 100:.1f}%"
                         for k in ("gen-loud", "gen-quiet", "null-whitened")))

    def omega(get):
        return hmean([x for m in MACHINES for r in [get(m)] for x in (r["auc_source"], r["auc_target"], r["pauc"])])

    log("=== overall (mean AUC / Omega):")
    for k in ("gen-loud", "null", "null-whitened", "gen-quiet"):
        log(f"  {k:14s} {np.mean([summary[m]['res'][k]['auc_all'] for m in MACHINES]):.3f} / {omega(lambda m: summary[m]['res'][k]):.3f}")
    for fam in ("rand-whiten", "rand-low"):
        aucs = np.array([[summary[m]["res"][fam][r]["auc_all"] for m in MACHINES] for r in range(DRAWS)]).mean(1)
        oms = np.array([omega(lambda m: summary[m]["res"][fam][r]) for r in range(DRAWS)])
        log(f"  {fam:14s} {aucs.mean():.3f}+-{aucs.std():.3f} (best draw {aucs.max():.3f}) / {oms.mean():.3f}+-{oms.std():.3f} (best {oms.max():.3f})")
    log("  overlap, mean over machines: " + " | ".join(
        f"{k}: anomaly {np.mean([summary[m]['overlap']['anomaly|' + k] for m in MACHINES]) * 100:.1f}% vs noise "
        f"{np.mean([summary[m]['overlap']['noise|' + k] for m in MACHINES]) * 100:.1f}%" for k in ("gen-loud", "gen-quiet", "null-whitened")))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
