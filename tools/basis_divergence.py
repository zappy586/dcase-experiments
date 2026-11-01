"""How much of the real-vs-synthetic difference can the CLoRA-Align adapter see?

Per adapted BEATs module (q/k/v input, out_proj, fc1 in each of 12 layers), on plain-BEATs input tokens of 200 real
ToyCar training normals and the 480 synthetic training normals (synth_v1 training split + synth_v1_extra):
  total divergence = Gaussian KL(synthetic || real) over all 768 directions
                   = sum_i 1/2 (lam_i - 1 - log lam_i)  +  1/2 delta' Sigma_real^-1 delta
    (lam_i = generalized eigenvalues of the centred covariances, delta = mean gap; covariance part + mean part)
  share captured by a k-dim subspace U = KL of the two Gaussians projected onto U, divided by the total
    current : the CLoRA basis used now (top-k of uncentred S_syn a = lam S_real a)
    best    : the k generalized eigendirections with the largest per-direction KL (the "difference basis")
    random  : a random k-dim subspace (control)
Usage (main env): uv run python tools/basis_divergence.py --k 16
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from BEATs import BEATs, BEATsConfig  # noqa: E402

SYN = ROOT / "outputs" / "audiogen" / "synth_v1"


def kl_sub(U, mr, Sr, ms, Ss):
    """KL(N(ms,Ss) || N(mr,Sr)) after projecting onto the columns of U"""
    Mr, Ms, d = U.T @ Sr @ U, U.T @ Ss @ U, U.T @ (ms - mr)
    k = len(Mr)
    X = np.linalg.solve(Mr, Ms)
    sign, ld = np.linalg.slogdet(X)
    return 0.5 * (np.trace(X) - k - ld + d @ np.linalg.solve(Mr, d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    dev = torch.device("cuda")
    log = lambda m: A.log(f"basis_divergence [k{args.k}]: {m}")
    rng = np.random.default_rng(0)  # same draws as clora_align
    D = A.Data("ToyCar", dev)
    real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)])
    gen = torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in sorted((SYN / "normal").glob("*.wav"))]))
    perm = {c: rng.permutation(250) for c in ("normal", "faulty")}
    pid = rng.choice(1000, 200, replace=False)
    real_fit = real[np.setdiff1d(np.arange(1000), pid)][:200]
    ex = sorted((ROOT / "outputs" / "audiogen" / "synth_v1_extra" / "normal").glob("*.wav"))
    syn_fit = torch.cat([gen[perm["normal"][50:]], torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in ex]))])

    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    beats = BEATs(BEATsConfig(ck["cfg"])); beats.load_state_dict(ck["model"]); beats.eval().to(dev)
    mods = []
    for i, layer in enumerate(beats.encoder.layers):
        mods += [(f"{i + 1}.qkv", layer.self_attn.q_proj), (f"{i + 1}.out", layer.self_attn.out_proj), (f"{i + 1}.fc1", layer.fc1)]

    def stats(X):
        acc = [[torch.zeros(768, dtype=torch.float64, device=dev), torch.zeros(768, 768, dtype=torch.float64, device=dev), 0] for _ in mods]
        def grab(j):
            def hook(mod, a, o):
                t = a[0].reshape(-1, 768).double(); acc[j][0] += t.sum(0); acc[j][1] += t.T @ t; acc[j][2] += len(t)
            return hook
        hs = [m.register_forward_hook(grab(j)) for j, (_, m) in enumerate(mods)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(X), 16):
                x = X[i : i + 16].to(dev)
                beats.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))
        for h in hs:
            h.remove()
        out = []
        for s, S, n in acc:
            m = (s / n).cpu().numpy(); S2 = (S / n).cpu().numpy()
            out.append((m, S2, S2 - np.outer(m, m)))  # mean, uncentred second moment, covariance
        return out

    R, S = stats(real_fit), stats(syn_fit)
    k = args.k
    rows = []
    for j, (name, _) in enumerate(mods):
        mr, S2r, Cr = R[j]; ms, S2s, Cs = S[j]
        reg = lambda C: C + 1e-3 * np.trace(C) / 768 * np.eye(768)
        Cr_, Cs_ = reg(Cr), reg(Cs)
        lam, V = scipy.linalg.eigh(Cs_, Cr_)              # V' Cr_ V = I
        delta = ms - mr
        per = 0.5 * (lam - 1 - np.log(lam)) + 0.5 * (V.T @ delta) ** 2
        total = per.sum()
        cov_part = (0.5 * (lam - 1 - np.log(lam))).sum()
        best = V[:, np.argsort(per)[-k:]]
        S2r_ = reg(S2r)
        _, Vc = scipy.linalg.eigh(S2s, S2r_, subset_by_index=[768 - k, 767])   # current CLoRA basis
        cur = np.linalg.qr(Vc)[0]
        rnd = np.linalg.qr(rng.standard_normal((768, k)))[0]
        sh = {n: kl_sub(U, mr, Cr_, ms, Cs_) / total for n, U in (("current", cur), ("best", best), ("random", rnd))}
        # how much real-clip variance lives in each subspace (real-silent = small)
        rv = {n: np.trace(U.T @ Cr @ U) / np.trace(Cr) for n, U in (("current", cur), ("best", np.linalg.qr(best)[0]))}
        rows.append((name, total, cov_part / total, sh, rv))
        log(f"{name:7s} total KL {total:9.1f} (covariance {cov_part / total * 100:4.0f}%, mean {100 - cov_part / total * 100:4.0f}%) | "
            f"share in current {sh['current'] * 100:5.1f}%  best {sh['best'] * 100:5.1f}%  random {sh['random'] * 100:5.2f}% | "
            f"real variance in current {rv['current'] * 100:.2f}%  best {rv['best'] * 100:.2f}%")
    med = lambda f: np.median([f(r) for r in rows])
    log(f"MEDIAN over {len(rows)} modules: share of divergence in current {med(lambda r: r[3]['current']) * 100:.1f}%  "
        f"best {med(lambda r: r[3]['best']) * 100:.1f}%  random {med(lambda r: r[3]['random']) * 100:.2f}% | covariance part "
        f"{med(lambda r: r[2]) * 100:.0f}% | real variance in current {med(lambda r: r[4]['current']) * 100:.2f}%  "
        f"best {med(lambda r: r[4]['best']) * 100:.2f}%")


if __name__ == "__main__":
    main()
