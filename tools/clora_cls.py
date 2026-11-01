"""Part 2 of CLoRA-Align: a normal-vs-faulty head on top of the aligned BEATs encoder, scored on real ToyCar test clips.

Encoder: BEATs + the CLoRA-Align adapters from tools/clora_align.py (step 600; A rebuilt from the same clips and
verified by re-running the held-out real-vs-synthetic probes; frozen here). Control: the same head on plain BEATs.
Features: mean-pooled last layer of full 10 s clips.
Head training data (identical for both encoders):
  normal: all 1000 real training normals + 200 synthetic normals (training split)
  faulty: 200 synthetic faulty (training split)
  "real+syn": both normal groups, weights real normal 1/4, synthetic normal 1/4, synthetic faulty 1/2
  "syn only": synthetic normal vs synthetic faulty, 1/2 each
Heads (fixed settings, no test-label selection): logistic regression (C=0.1) and MLP 768-128-1 (GELU, AdamW lr 1e-3,
wd 1e-2, 300 full-batch epochs; 5 seeds, mean +- sd). Anomaly score = head logit.
Also logs, per encoder: held-out synthetic normal vs faulty AUC (does fault content survive?), and RBF probes
real test anomaly vs synthetic faulty and real training normal vs synthetic normal (0.5 = indistinguishable).
Usage (main env): uv run python tools/clora_cls.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from BEATs import BEATs, BEATsConfig  # noqa: E402
from clora_align import FrozenALoRA, probes  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402

SYN = ROOT / "outputs" / "audiogen" / "synth_v1"
CKPT = ROOT / "outputs" / "clora_align_r16_s1_lr0.0005.pt"
FEAT = ROOT / "cache" / "clora_cls_aligned.npz"
log = lambda m: A.log(f"clora_cls: {m}")


def aligned_features(real, test, gen, real_fit, syn_fit, real_probe, syn_probe, dev):
    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    beats = BEATs(BEATsConfig(ck["cfg"])); beats.load_state_dict(ck["model"]); beats.eval().to(dev)
    for p in beats.parameters():
        p.requires_grad_(False)

    def pooled(X, bs=16):
        out = []
        for i in range(0, len(X), bs):
            x = X[i : i + bs].to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(beats.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0].float().mean(1).cpu())
        return torch.cat(out)

    groups = []
    for layer in beats.encoder.layers:
        at = layer.self_attn
        groups += [(at.q_proj, ["q_proj", "k_proj", "v_proj"], at), (at.out_proj, ["out_proj"], at), (layer.fc1, ["fc1"], layer)]

    def moments(X):
        acc = [[torch.zeros(768, 768, dtype=torch.float64, device=dev), 0] for _ in groups]
        def grab(j):
            def hook(mod, a, o):
                t = a[0].reshape(-1, 768).double(); acc[j][0] += t.T @ t; acc[j][1] += len(t)
            return hook
        hs = [g[0].register_forward_hook(grab(j)) for j, g in enumerate(groups)]
        pooled(X)
        for h in hs:
            h.remove()
        return [(a / n).cpu().numpy() for a, n in acc]

    Mr, Ms = moments(real_fit[:200]), moments(syn_fit)  # same clips and order as clora_align.py
    for (_, names, parent), Sr, Ss in zip(groups, Mr, Ms):
        Srr = Sr + 1e-3 * np.trace(Sr) / 768 * np.eye(768)
        _, V = scipy.linalg.eigh(Ss, Srr, subset_by_index=[768 - 16, 767])
        A_ = torch.tensor(np.linalg.qr(V)[0].T, dtype=torch.float32, device=dev)
        for n in names:
            setattr(parent, n, FrozenALoRA(getattr(parent, n), A_, 1.0))
    saved = torch.load(CKPT)
    sd = beats.state_dict()
    for n, t in saved.items():
        sd[n] = t.to(dev)
    beats.load_state_dict(sd)
    log(f"loaded {len(saved)} tensors from {CKPT.name}")
    ho = probes(pooled(real_probe).numpy(), pooled(syn_probe).numpy())
    log("check (step 600 was linear 0.762 RBF 0.905 MLP 0.860 kNN 0.788): " + "  ".join(f"{k} {v:.3f}" for k, v in ho.items()))
    out = {"real_train": pooled(real), "real_test": pooled(test), "gen_normal": pooled(gen["normal"]), "gen_faulty": pooled(gen["faulty"])}
    out = {k: v.numpy() for k, v in out.items()}
    np.savez(FEAT, **out)
    return out


def mlp_scores(Xtr, ytr, wtr, Xev, seed, dev):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.GELU(), nn.Linear(128, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    X, y, w = (torch.tensor(a, dtype=torch.float32, device=dev) for a in (Xtr, ytr, wtr))
    for _ in range(300):
        loss = (F.binary_cross_entropy_with_logits(net(X)[:, 0], y, reduction="none") * w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return [net(torch.tensor(e, dtype=torch.float32, device=dev))[:, 0].cpu().numpy() for e in Xev]


def main():
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)  # same draws as clora_align
    D = A.Data("ToyCar", dev)
    lab, dom = np.array(D.meta["test"]["label"]), np.array(D.meta["test"]["domain"])
    perm = {c: rng.permutation(250) for c in ("normal", "faulty")}
    pid = rng.choice(1000, 200, replace=False)
    if FEAT.exists():
        F_al = dict(np.load(FEAT))
        log(f"aligned features from {FEAT.name}")
    else:
        real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)])
        test = torch.cat([D.audio("test", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 200, 25)])
        gen = {c: torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in sorted((SYN / c).glob("*.wav"))]))
               for c in ("normal", "faulty")}
        F_al = aligned_features(real, test, gen, real[np.setdiff1d(np.arange(1000), pid)], gen["normal"][perm["normal"][50:]],
                                real[pid], gen["normal"][perm["normal"][:50]], dev)
    F_pl = dict(np.load(ROOT / "cache" / "grl_v1_beats.npz"))

    for enc, Fe in (("plain BEATs", F_pl), ("CLoRA-Align", F_al)):
        R, T = Fe["real_train"], Fe["real_test"]
        Sn_tr, Sn_va = Fe["gen_normal"][perm["normal"][50:]], Fe["gen_normal"][perm["normal"][:50]]
        Sf_tr, Sf_va = Fe["gen_faulty"][perm["faulty"][50:]], Fe["gen_faulty"][perm["faulty"][:50]]
        p_rn = probes(R[pid], Sn_va)["RBF"]
        p_af = probes(T[lab == 1], Sf_va)["RBF"]
        p_nf = probes(np.r_[Sn_tr, Sn_va], np.r_[Sf_tr, Sf_va])
        log(f"[{enc}] RBF probes: real train normal vs synthetic normal {p_rn:.3f} | real test anomaly vs synthetic faulty "
            f"{p_af:.3f} | synthetic normal vs faulty linear {p_nf['linear']:.3f} RBF {p_nf['RBF']:.3f}")
        for variant in ("real+syn", "syn only"):
            if variant == "real+syn":
                Xtr = np.r_[R, Sn_tr, Sf_tr]
                ytr = np.r_[np.zeros(len(R) + len(Sn_tr)), np.ones(len(Sf_tr))]
                wtr = np.r_[np.full(len(R), 0.25 / len(R)), np.full(len(Sn_tr), 0.25 / len(Sn_tr)), np.full(len(Sf_tr), 0.5 / len(Sf_tr))]
            else:
                Xtr = np.r_[Sn_tr, Sf_tr]
                ytr = np.r_[np.zeros(len(Sn_tr)), np.ones(len(Sf_tr))]
                wtr = np.r_[np.full(len(Sn_tr), 0.5 / len(Sn_tr)), np.full(len(Sf_tr), 0.5 / len(Sf_tr))]
            sc = StandardScaler().fit(Xtr)
            Xs, Ts, Vs = sc.transform(Xtr), sc.transform(T), sc.transform(np.r_[Sn_va, Sf_va])
            yva = np.r_[np.zeros(50), np.ones(50)]
            lr = LogisticRegression(C=0.1, max_iter=5000).fit(Xs, ytr, sample_weight=wtr * len(ytr))
            m = compute_metrics(lr.decision_function(Ts), lab, dom)
            log(f"[{enc} | {variant} | logistic] real test src {m['auc_source']:.3f} tgt {m['auc_target']:.3f} "
                f"pAUC {m['pauc']:.3f} official {m['official']:.3f} | held-out synthetic AUC "
                f"{roc_auc_score(yva, lr.decision_function(Vs)):.3f}")
            res, syn = [], []
            for seed in range(5):
                st, sv = mlp_scores(Xs, ytr, wtr, [Ts, Vs], seed, dev)
                res.append(compute_metrics(st, lab, dom)); syn.append(roc_auc_score(yva, sv))
            ms = lambda k: f"{np.mean([r[k] for r in res]):.3f}+-{np.std([r[k] for r in res]):.3f}"
            log(f"[{enc} | {variant} | MLP x5] real test src {ms('auc_source')} tgt {ms('auc_target')} pAUC {ms('pauc')} "
                f"official {ms('official')} | held-out synthetic AUC {np.mean(syn):.3f}")


if __name__ == "__main__":
    main()
