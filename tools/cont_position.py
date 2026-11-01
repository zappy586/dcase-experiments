"""Where do continuation-generated synthetic normals land relative to real ones (plain BEATs)?
Compares outputs/audiogen/cont_test (continuations of real 3 s openings) with synth_v1 (from scratch).
Features: mean-pooled plain BEATs on (a) the generated part only, 3-10 s, with real / v1 clips cut the same way, and
(b) the full 10 s clip. Space: global LEACE (real vs v1) + standardised, as in tools/heldout_clip_check.py.
Per clip: real share of its 10 nearest neighbours among 1000 real training normals + 250 v1 synthetic normals, and the
distance to the nearest real clip as a percentile of real-to-real nearest-neighbour distances (50 = like a real clip).
Also whether the nearest real clip is the clip the opening came from (= near copy).
Usage (main env): uv run python tools/cont_position.py
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from BEATs import BEATs, BEATsConfig  # noqa: E402
from cluster_leace import cov_bal, leace  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/audiogen/cont_test")
    args = ap.parse_args()
    dev = torch.device("cuda")
    log = lambda m: A.log(f"cont_position: {m}")
    D = A.Data("ToyCar", dev)
    real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)])
    v1 = torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000]
                                   for f in sorted((ROOT / "outputs/audiogen/synth_v1/normal").glob("*.wav"))]))
    files = sorted((ROOT / args.dir / "normal").glob("*.wav"))
    ct = torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in files]))
    meta = json.loads((ROOT / args.dir / "meta.json").read_text())
    src = np.array([meta[f"normal/{f.name}"]["source_train_clip"] for f in files])

    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    beats = BEATs(BEATsConfig(ck["cfg"])); beats.load_state_dict(ck["model"]); beats.eval().to(dev)

    def pooled(X, bs=16):
        out = []
        for i in range(0, len(X), bs):
            x = X[i : i + bs].to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(beats.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0].float().mean(1).cpu())
        return torch.cat(out).numpy()

    for name, sl in (("generated part 3-10 s", slice(48000, 160000)), ("full 10 s", slice(0, 160000))):
        R, S, C = pooled(real[:, sl]), pooled(v1[:, sl]), pooled(ct[:, sl])
        g = leace(R, S, cov_bal(R, S))
        sc = StandardScaler().fit(np.r_[g(R), g(S)])
        Gr, Gs, Gc = sc.transform(g(R)), sc.transform(g(S)), sc.transform(g(C))
        allp, isreal = np.r_[Gr, Gs], np.r_[np.ones(len(Gr)), np.zeros(len(Gs))]
        drr = np.linalg.norm(Gr[:, None] - Gr[None], axis=-1); np.fill_diagonal(drr, np.inf)
        nn_rr = drr.min(1)

        def position(Q, own=None):
            sh, pc, cp = [], [], []
            for i, q in enumerate(Q):
                d = np.linalg.norm(allp - q, axis=1)
                if own is not None:
                    d[len(Gr) + own[i]] = np.inf
                sh.append(isreal[np.argsort(d)[:10]].mean())
                dr = d[: len(Gr)]
                pc.append((nn_rr < dr.min()).mean() * 100)
                cp.append(int(np.argmin(dr)))
            return np.array(sh), np.array(pc), np.array(cp)

        sv, pv, _ = position(Gs, own=np.arange(len(Gs)))
        sc_, pc_, cp = position(Gc)
        log(f"[{name}] from scratch (v1, 250): real-neighbour share median {np.median(sv):.2f} mean {sv.mean():.2f}, "
            f">= 0.7 for {(sv >= 0.7).mean() * 100:.0f}% | nearest-real distance percentile median {np.median(pv):.0f}")
        log(f"[{name}] continuation ({len(C)}):  real-neighbour share median {np.median(sc_):.2f} mean {sc_.mean():.2f}, "
            f">= 0.7 for {(sc_ >= 0.7).mean() * 100:.0f}% | nearest-real distance percentile median {np.median(pc_):.0f} | "
            f"nearest real = its own source clip for {(cp == src).sum()}/{len(C)}")
        log(f"[{name}] per continuation clip (share, percentile): " + "  ".join(f"({a:.1f}, {b:.0f})" for a, b in zip(sc_, pc_)))


if __name__ == "__main__":
    main()
