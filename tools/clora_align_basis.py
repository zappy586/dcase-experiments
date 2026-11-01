"""Feasibility check for CLoRA-style alignment inside BEATs: does each adapted module have input directions that
synthetic normals excite but real clips leave (almost) silent?

CLoRA (clora_ft.py): the LoRA down-projection A is frozen to the top solutions of S_task a = lambda (S_keep + eps I) a,
S = E[x x^T] of that module's input tokens. Here
    S_keep = real ToyCar normal clips (what the adapter must not touch)
    S_task = synthetic normal clips (what the adapter must move onto the real distribution)
If such directions exist, a LoRA whose A is that basis has x @ A^T ~ 0 on real inputs, so real features (normals AND
anomalies) stay as in BEATs by construction, while synthetic features can be moved.

Fit: 200 real training normals + 200 synthetic normals. Held-out check (same split as grl_v2): 200 other real training
normals, 50 synthetic normals, 100 real test anomalies. Per module it logs lambda (synthetic/real energy along A on the
fit set) and, on held-out clips, the share of each set's input energy that falls in A (random rank-r subspace ~ r/dim).
Usage (main env): uv run python tools/clora_align_basis.py --rank 16
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
TARGETS = ("q_proj", "out_proj", "fc1")  # q/k/v share one input; fc2 (3072-d input) skipped as in clora_ft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()
    dev = torch.device("cuda")
    log = lambda m: A.log(f"clora_align_basis [r{args.rank}]: {m}")
    rng = np.random.default_rng(0)

    D = A.Data("ToyCar", dev)
    real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)])
    test = torch.cat([D.audio("test", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 200, 25)])
    lab = np.array(D.meta["test"]["label"])
    gen = {c: torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in sorted((SYN / c).glob("*.wav"))]))
           for c in ("normal", "faulty")}
    perm = {c: rng.permutation(250) for c in gen}
    pid = rng.choice(1000, 200, replace=False)
    fit_ids = np.setdiff1d(np.arange(1000), pid)
    sets = {"fit real": real[fit_ids[:200]], "fit synthetic": gen["normal"][perm["normal"][50:]],
            "held-out real normal": real[pid], "held-out synthetic": gen["normal"][perm["normal"][:50]],
            "real test anomaly": test[lab == 1]}

    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    beats = BEATs(BEATsConfig(ck["cfg"])); beats.load_state_dict(ck["model"]); beats.eval().to(dev)
    mods = {}
    for i, layer in enumerate(beats.encoder.layers):
        mods[f"{i + 1}.q_proj"] = layer.self_attn.q_proj
        mods[f"{i + 1}.out_proj"] = layer.self_attn.out_proj
        mods[f"{i + 1}.fc1"] = layer.fc1

    def moments(X):
        acc = {k: [torch.zeros(768, 768, dtype=torch.float64, device=dev), 0] for k in mods}
        def grab(k):
            def hook(mod, a, o):  # returns None, so the module output is untouched
                t = a[0].reshape(-1, 768).double()
                acc[k][0] += t.T @ t
                acc[k][1] += len(t)
            return hook
        hooks = [m.register_forward_hook(grab(k)) for k, m in mods.items()]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(X), 16):
                x = X[i : i + 16].to(dev)
                beats.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))
        for h in hooks:
            h.remove()
        return {k: (v[0] / v[1]).cpu().numpy() for k, v in acc.items()}

    M = {name: moments(X) for name, X in sets.items()}
    r = args.rank
    rows = []
    for k in mods:
        Sr, Ss = M["fit real"][k], M["fit synthetic"][k]
        Srr = Sr + 1e-3 * np.trace(Sr) / 768 * np.eye(768)
        lam, V = scipy.linalg.eigh(Ss, Srr, subset_by_index=[768 - r, 767])
        Q = np.linalg.qr(V)[0]  # 768 x r orthonormal
        share = {n: np.trace(Q.T @ M[n][k] @ Q) / np.trace(M[n][k]) for n in sets}
        rows.append((k, lam.mean(), lam.min(), share))
        log(f"{k:12s} lambda mean {lam.mean():8.1f} min {lam.min():7.1f} | energy share in A: "
            f"held-out real normal {share['held-out real normal'] * 100:.3f}%  real test anomaly "
            f"{share['real test anomaly'] * 100:.3f}%  held-out synthetic {share['held-out synthetic'] * 100:.2f}%  "
            f"(random {r / 768 * 100:.2f}%)")
    med = lambda n: np.median([s[n] for _, _, _, s in rows]) * 100
    log(f"median over modules: real normal {med('held-out real normal'):.3f}%  real anomaly {med('real test anomaly'):.3f}%  "
        f"synthetic {med('held-out synthetic'):.2f}%  lambda {np.median([l for _, l, _, _ in rows]):.1f}")


if __name__ == "__main__":
    main()
