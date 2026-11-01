"""CLoRA-Align: move synthetic ToyCar normals onto the real normal distribution INSIDE BEATs, leaving real clips untouched.

Adapters: in all 12 layers on q_proj, k_proj, v_proj (shared input), out_proj and fc1, LoRA y = W x + s * B (A x) with
  A (rank --rank) FROZEN = top solutions of S_syn a = lambda (S_real + eps I) a for that module's input tokens
      (CLoRA, clora_ft.py: directions synthetic normals excite and real clips leave silent; tools/clora_align_basis.py
       measured 0.067% of held-out real-normal energy there, 1.63% of synthetic, 2.08% for a random subspace)
  B zero-initialised (start = plain BEATs), trained.
Since A x ~ 0 for real inputs at every layer, real features (normals and anomalies) stay ~ plain BEATs by construction.
Loss (one-sided distribution matching, non-linear): unbiased multi-bandwidth Gaussian-kernel MMD^2 between the
mean-pooled outputs of 16 synthetic training normals (10 s, with adapter) and 128 clips drawn from a fixed bank of the
800 real fit normals (plain BEATs = the target). Features standardised with the real bank's statistics.
Split as grl_v2 / leace_scrub: probe = 200 real training normals + 50 synthetic normals never used for fitting.
Every --eval steps: linear / RBF-SVM / MLP / kNN probes (5-fold CV AUC, 0.5 = cannot tell real from synthetic),
the same RBF probe on the fit sets, real-feature drift, and real test normal vs anomaly (linear probe).
v2 (--crop --keep W): random 5 s crops + gain on synthetic, crop-level real bank, and a real-keep loss (W x mean
squared change of real crops' pooled features vs adapters off), so real drift is held near 0 explicitly.
Paired codec invariance (--pair W --basis both): real fit clips round-tripped through AudioGen's EnCodec must map onto
plain-BEATs features of the same original (per-clip target, same content, so only the codec signature differs); basis
'both' adds the top S_roundtrip vs S_real directions (codec only) to the S_syn vs S_real ones.
Usage (main env): uv run python tools/clora_align.py --steps 600
       paired:    uv run python tools/clora_align.py --extra --pair 1 --basis both --steps 600
       v2:        uv run python tools/clora_align.py --rank 8 --crop --keep 10 --steps 1500
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from BEATs import BEATs, BEATsConfig  # noqa: E402

SYN = ROOT / "outputs" / "audiogen" / "synth_v1"


class FrozenALoRA(nn.Module):
    def __init__(self, lin, A_, s):
        super().__init__()
        self.lin, self.s = lin, s
        self.register_buffer("A", A_)                                   # r x in, orthonormal rows, frozen
        self.B = nn.Parameter(torch.zeros(lin.out_features, A_.shape[0], device=A_.device))

    def forward(self, x):
        return self.lin(x) + self.s * ((x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype))


def probes(R, S):
    X, y = np.r_[R, S], np.r_[np.zeros(len(R)), np.ones(len(S))]
    clfs = {"linear": LogisticRegression(C=0.1, max_iter=3000), "RBF": SVC(C=1.0, kernel="rbf", gamma="scale"),
            "MLP": MLPClassifier((128,), alpha=1e-2, max_iter=1000, random_state=0), "kNN": KNeighborsClassifier(10)}
    out = {}
    for n, c in clfs.items():
        meth = "decision_function" if hasattr(c, "decision_function") else "predict_proba"
        p = cross_val_predict(make_pipeline(StandardScaler(), c), X, y, cv=5, method=meth)
        out[n] = roc_auc_score(y, p if p.ndim == 1 else p[:, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", action="store_true", help="v2: train on random circular time shifts (length --crop_len) with "
                    "+-2 dB gain on synthetic clips; MMD target = bank of 2 shifted crops per real fit clip")
    ap.add_argument("--keep", type=float, default=0.0, help="v2: weight of the real-keep loss = mean squared change "
                    "(standardised) of 8 real crops' pooled features vs plain BEATs (adapters off)")
    ap.add_argument("--crop_len", type=int, default=160000, help="crop length in samples (160000 = full 10 s clip, "
                    "circularly shifted; 80000 = 5 s, which aligned crops but not 10 s clips)")
    ap.add_argument("--extra", action="store_true", help="add outputs/audiogen/synth_v1_extra/normal (tools/gen_synth_extra.py) "
                    "to the synthetic training clips")
    ap.add_argument("--mmd", type=float, default=1.0, help="weight of the output MMD loss (0 = off)")
    ap.add_argument("--loud", type=float, default=0.0, help="weight of the per-layer loudness loss: in every adapted "
                    "module, squared log ratio of synthetic-batch to real energy along each watched direction")
    ap.add_argument("--monitor", type=int, default=32, help="watched directions per module for the loudness loss")
    ap.add_argument("--pair", type=float, default=0.0, help="weight of the paired codec-invariance loss: real fit clips sent "
                    "through EnCodec and back (cache/toycar_rt_train.npy) must map, with adapters, onto plain-BEATs features "
                    "of the same original clip (mean squared standardised difference of pooled features, 8 pairs per step)")
    ap.add_argument("--basis", default="syn", choices=["syn", "rt", "both"], help="CLoRA basis: S_syn vs S_real (syn), "
                    "S_roundtrip vs S_real (rt: same content, so only codec directions), or both stacked (rank x 2)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    tag = (f"r{args.rank} s{args.scale:g} lr{args.lr:g}{f' crop{args.crop_len // 16000}s' if args.crop else ''}"
           f"{f' keep{args.keep:g}' if args.keep else ''}{' extra' if args.extra else ''}"
           f"{f' mmd{args.mmd:g}' if args.mmd != 1 else ''}{f' loud{args.loud:g}' if args.loud else ''}"
           f"{f' pair{args.pair:g}' if args.pair else ''}{f' basis-{args.basis}' if args.basis != 'syn' else ''}")
    log = lambda m: A.log(f"clora_align [{tag}]: {m}")
    rng = np.random.default_rng(0)  # same draws as grl_v2 / leace_scrub

    D = A.Data("ToyCar", dev)
    real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)])
    test = torch.cat([D.audio("test", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 200, 25)])
    lab = np.array(D.meta["test"]["label"])
    gen = {c: torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in sorted((SYN / c).glob("*.wav"))]))
           for c in ("normal", "faulty")}
    perm = {c: rng.permutation(250) for c in gen}
    pid = rng.choice(1000, 200, replace=False)
    real_fit, real_probe = real[np.setdiff1d(np.arange(1000), pid)], real[pid]
    syn_fit, syn_probe = gen["normal"][perm["normal"][50:]], gen["normal"][perm["normal"][:50]]
    rt = torch.from_numpy(np.load(ROOT / "cache" / "toycar_rt_train.npy").astype(np.float32))[:, :160000]
    rt_fit, rt_probe = rt[np.setdiff1d(np.arange(1000), pid)], rt[pid]  # row i pairs with real_fit[i] / real_probe[i]
    if args.extra:  # extra synthetic normals join the training side only; the 50 held-out clips are unchanged
        ex = sorted((ROOT / "outputs" / "audiogen" / "synth_v1_extra" / "normal").glob("*.wav"))
        syn_fit = torch.cat([syn_fit, torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in ex]))])
        log(f"+{len(ex)} extra synthetic normals -> {len(syn_fit)} for training")

    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    beats = BEATs(BEATsConfig(ck["cfg"])); beats.load_state_dict(ck["model"]); beats.eval().to(dev)
    for p in beats.parameters():
        p.requires_grad_(False)

    def pooled(X, bs=16, grad=False):
        out = []
        for i in range(0, len(X), bs):
            x = X[i : i + bs].to(dev)
            with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
                h = beats.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0].float().mean(1)
            out.append(h if grad else h.cpu())
        return torch.cat(out)

    # ---- CLoRA bases from plain-BEATs input moments (200 real fit + 200 synthetic fit clips)
    groups = []  # (input-hook module, [modules sharing that input])
    for layer in beats.encoder.layers:
        at = layer.self_attn
        groups += [(at.q_proj, ["q_proj", "k_proj", "v_proj"], at), (at.out_proj, ["out_proj"], at)]
        if isinstance(layer.fc1, nn.Linear):
            groups.append((layer.fc1, ["fc1"], layer))

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

    t0 = time.time()
    Mr, Ms = moments(real_fit[:200]), moments(syn_fit)
    Mt = moments(rt_fit[:200]) if args.basis != "syn" else Ms
    n_ad = 0
    for (_, names, parent), Sr, Ss, St in zip(groups, Mr, Ms, Mt):
        Srr = Sr + 1e-3 * np.trace(Sr) / 768 * np.eye(768)
        top = lambda S: scipy.linalg.eigh(S, Srr, subset_by_index=[768 - args.rank, 767])[1]
        V = {"syn": lambda: top(Ss), "rt": lambda: top(St), "both": lambda: np.c_[top(Ss), top(St)]}[args.basis]()
        A_ = torch.tensor(np.linalg.qr(V)[0].T, dtype=torch.float32, device=dev)
        for n in names:
            setattr(parent, n, FrozenALoRA(getattr(parent, n), A_, args.scale)); n_ad += 1
    # ---- loudness monitor (--loud): per module the top --monitor generalized eigenvectors (wider than the adapter's rank,
    # so energy cannot just move to unwatched real-silent directions) and the real clips' energy along each of them
    Mon, logE_real = [], []
    for Sr, Ss in zip(Mr, Ms):
        Srr = Sr + 1e-3 * np.trace(Sr) / 768 * np.eye(768)
        lam_, V = scipy.linalg.eigh(Ss, Srr, subset_by_index=[768 - args.monitor, 767])
        V = V / np.linalg.norm(V, axis=0, keepdims=True)
        Mon.append(torch.tensor(V, dtype=torch.float32, device=dev))
        logE_real.append(torch.tensor(np.log(np.einsum("ik,ij,jk->k", V, Sr, V) + 1e-12), dtype=torch.float32, device=dev))
    loud_buf = {"on": None}

    def loud_hook(j):
        def hook(mod, a, o):
            if loud_buf["on"] is not None:
                t = a[0].reshape(-1, 768).float()
                loud_buf["on"].append((j, (t @ Mon[j]).pow(2).mean(0)))
        return hook
    for j, gr in enumerate(groups):
        gr[0].register_forward_hook(loud_hook(j))  # the original Linear still sees the module input inside its wrapper

    def loudness(X):
        """per module: mean over the monitor directions of log(synthetic-or-real energy / real fit energy)"""
        loud_buf["on"] = []
        pooled(X)
        acc = {}
        for j, E in loud_buf["on"]:
            acc.setdefault(j, []).append(E)
        loud_buf["on"] = None
        return np.array([(torch.stack(acc[j]).mean(0).log() - logE_real[j]).mean().item() for j in range(len(groups))])

    params = [p for p in beats.parameters() if p.requires_grad]
    log(f"{n_ad} adapters (fc1 {'included' if len(groups) == 36 else 'skipped: not nn.Linear'}), "
        f"{sum(p.numel() for p in params) / 1e6:.2f}M trainable (B only), bases in {time.time() - t0:.0f} s")

    adapters = [m for m in beats.modules() if isinstance(m, FrozenALoRA)]

    def set_scale(s):
        for m in adapters:
            m.s = s

    g = torch.Generator().manual_seed(args.seed)

    def crops(X, n, gain):
        idx = torch.randint(len(X), (n,), generator=g)
        sh = torch.randint(0, X.shape[1], (n,), generator=g)
        out = torch.stack([torch.roll(X[i], -int(s))[: args.crop_len] for i, s in zip(idx.tolist(), sh.tolist())])
        if gain:
            out = out * 10 ** ((torch.rand(n, 1, generator=g) * 4 - 2) / 20)
        return out

    # ---- fixed real target bank and probe references (adapter is zero here, so this is plain BEATs)
    bank_full = pooled(real_fit)
    bank = pooled(torch.cat([crops(real_fit, len(real_fit), False), crops(real_fit, len(real_fit), False)])) if args.crop else bank_full
    mu, sd = bank.mean(0), bank.std(0) + 1e-4
    bank_n = ((bank - mu) / sd).to(dev)
    with torch.no_grad():
        d2 = torch.cdist(bank_n[:400], bank_n[:400]).pow(2)
        med = d2[d2 > 0].median()
    bws = med * torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0], device=dev)
    P0r, P0s, T0 = pooled(real_probe), pooled(syn_probe), pooled(test)
    bank_full_n = ((bank_full - mu) / sd).to(dev)  # paired targets: plain BEATs of real_fit[i]
    c0 = P0r.mean(0)

    def mmd2(X, Y):
        Kxx, Kyy, Kxy = [sum(torch.exp(-torch.cdist(a, b).pow(2) / bw) for bw in bws) for a, b in ((X, X), (Y, Y), (X, Y))]
        n, m = len(X), len(Y)
        return ((Kxx.sum() - Kxx.diagonal().sum()) / (n * (n - 1)) + (Kyy.sum() - Kyy.diagonal().sum()) / (m * (m - 1))
                - 2 * Kxy.mean())

    def evaluate(step):
        beats.eval()
        Pr, Ps, T = pooled(real_probe), pooled(syn_probe), pooled(test)
        Sf = pooled(syn_fit[:100])
        ho, fit = probes(Pr.numpy(), Ps.numpy()), probes(bank_full[:400].numpy(), Sf.numpy())
        drift = ((Pr - P0r).norm() / (P0r - c0).norm()).item()
        moved = ((Ps - P0s).norm() / (P0s - P0s.mean(0)).norm()).item()
        Tn = T.numpy()
        na = probes(Tn[lab == 0], Tn[lab == 1])["linear"]
        Pt = pooled(rt_probe)
        cod, con = probes(Pr.numpy(), Pt.numpy()), probes(Pt.numpy(), Ps.numpy())
        perr = (((Pt - P0r) / sd) ** 2).mean().item()
        with torch.no_grad():
            ho_mmd = mmd2(((Ps - mu) / sd).to(dev), ((Pr - mu) / sd).to(dev)).item()
        log(f"step {step}: HELD-OUT real vs synthetic normal  linear {ho['linear']:.3f}  RBF {ho['RBF']:.3f}  "
            f"MLP {ho['MLP']:.3f}  kNN {ho['kNN']:.3f}  MMD2 {ho_mmd:.4f} | fit-set RBF {fit['RBF']:.3f} kNN {fit['kNN']:.3f} | "
            f"real drift {drift:.4f}, synthetic moved {moved:.3f} | real test n vs a {na:.3f}")
        log(f"step {step}: HELD-OUT real vs round-tripped real (codec only)  " + "  ".join(f"{k} {v:.3f}" for k, v in cod.items())
            + " | round-tripped vs synthetic (content)  " + "  ".join(f"{k} {v:.3f}" for k, v in con.items())
            + f" | pair error {perr:.4f}")
        Ls, Lr = loudness(syn_probe), loudness(real_probe)
        per_layer = np.exp(Ls.reshape(12, -1).mean(1))
        log(f"step {step}: loudness (synthetic / real energy in the {args.monitor} watched directions, 1 = real-like): "
            f"held-out synthetic median {np.exp(np.median(Ls)):.1f}x, held-out real {np.exp(np.median(Lr)):.2f}x | "
            f"per layer " + " ".join(f"{v:.1f}" for v in per_layer))

    evaluate(0)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    t0, losses = time.time(), []
    for s in range(1, args.steps + 1):
        xs = crops(syn_fit, 16, True) if args.crop else syn_fit[torch.randint(len(syn_fit), (16,))]
        loud_buf["on"] = [] if args.loud else None
        hs = (pooled(xs, bs=16, grad=True) - mu.to(dev)) / sd.to(dev)
        loss_l = (torch.stack([((E + 1e-12).log() - logE_real[j]).pow(2).mean() for j, E in loud_buf["on"]]).mean()
                  if args.loud else torch.zeros((), device=dev))
        loud_buf["on"] = None
        loss_m = mmd2(hs, bank_n[torch.randint(len(bank_n), (128,), device=dev)])
        loss_k = torch.zeros((), device=dev)
        if args.keep:
            xr = crops(real_fit, 8, False)
            set_scale(0.0); h0 = pooled(xr, bs=8); set_scale(args.scale)
            hr = pooled(xr, bs=8, grad=True)
            loss_k = (((hr - h0.to(dev)) / sd.to(dev)) ** 2).mean()
        loss_p = torch.zeros((), device=dev)
        if args.pair:
            ip = torch.randint(len(rt_fit), (8,))
            ht = (pooled(rt_fit[ip], bs=8, grad=True) - mu.to(dev)) / sd.to(dev)
            loss_p = (ht - bank_full_n[ip.to(dev)]).pow(2).mean()
        loss = args.mmd * loss_m + args.keep * loss_k + args.loud * loss_l + args.pair * loss_p
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append((loss_m.item(), loss_k.item(), loss_l.item(), loss_p.item()))
        if s % 25 == 0:
            lm, lk, ll, lp = np.mean(losses[-25:], 0)
            log(f"step {s}: MMD2 {lm:.4f} (weight {args.mmd:g}) keep {lk:.5f} loud {ll:.3f} pair {lp:.4f}, "
                f"{(time.time() - t0) / s:.2f} s/step")
        if s % args.eval == 0:
            evaluate(s)
    torch.save({n: t.detach().cpu() for n, t in beats.state_dict().items() if n.endswith(".A") or n.endswith(".B")},
               ROOT / "outputs" / f"clora_align_{tag.replace(' ', '_')}.pt")  # A (frozen bases) and B (trained)
    log("done")


if __name__ == "__main__":
    main()
