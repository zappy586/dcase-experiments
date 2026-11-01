"""Where do codec round-tripped REAL clips land (plain BEATs, full 10 s)? Tests whether AudioGen's EnCodec alone moves a
clip into the synthetic region.
Sets: 1000 real ToyCar training normals, the same 1000 sent through EnCodec and back (cache/toycar_rt_train.npy, from
tools/roundtrip_real.py), 250 synth_v1 synthetic normals (from scratch).
Same space and measures as tools/cont_position.py (global LEACE real vs v1 + standardised; real share of 10 nearest
neighbours among real + v1, own source clip excluded; nearest-real distance percentile). Also: share of a round-tripped
clip's 10 nearest neighbours that are synthetic when round-tripped clips are added to the pool, and RBF probes
real vs round-tripped, round-tripped vs synthetic (5-fold CV AUC, 200 vs 200).
Usage (main env): uv run python tools/rt_position.py
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
from BEATs import BEATs, BEATsConfig  # noqa: E402
from clora_align import probes  # noqa: E402
from cluster_leace import cov_bal, leace  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def main():
    dev = torch.device("cuda")
    log = lambda m: A.log(f"rt_position: {m}")
    D = A.Data("ToyCar", dev)
    real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)])
    rt = torch.from_numpy(np.load(ROOT / "cache" / "toycar_rt_train.npy").astype(np.float32))
    v1 = torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000]
                                   for f in sorted((ROOT / "outputs/audiogen/synth_v1/normal").glob("*.wav"))]))
    log(f"real {tuple(real.shape)}  round-tripped {tuple(rt.shape)}  v1 {tuple(v1.shape)} | rms dB real "
        f"{20 * np.log10(real.pow(2).mean(1).sqrt().mean()):.1f} rt {20 * np.log10(rt.pow(2).mean(1).sqrt().mean()):.1f}")
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

    R, T, S = pooled(real), pooled(rt), pooled(v1)
    g = leace(R, S, cov_bal(R, S))
    sc = StandardScaler().fit(np.r_[g(R), g(S)])
    Gr, Gt, Gs = sc.transform(g(R)), sc.transform(g(T)), sc.transform(g(S))
    drr = np.linalg.norm(Gr[:, None] - Gr[None], axis=-1); np.fill_diagonal(drr, np.inf)
    nn_rr = drr.min(1)
    sel = np.random.default_rng(0).choice(1000, 200, replace=False)
    sh, pc, own, syn_sh = [], [], [], []
    pool3 = np.r_[Gr, Gs, Gt]; kind = np.r_[np.zeros(1000), np.ones(250), np.full(1000, 2)]  # 0 real, 1 synthetic, 2 rt
    for i in sel:
        q = Gt[i]
        d = np.linalg.norm(np.r_[Gr, Gs] - q, axis=1); d[i] = np.inf          # exclude its own original
        sh.append((np.argsort(d)[:10] < 1000).mean())
        dr = d[:1000]; pc.append((nn_rr < dr.min()).mean() * 100)
        own.append(np.linalg.norm(Gr[i] - q))
        d3 = np.linalg.norm(pool3 - q, axis=1); d3[i] = np.inf; d3[1250 + i] = np.inf
        nb = kind[np.argsort(d3)[:10]]
        syn_sh.append([(nb == 0).mean(), (nb == 1).mean(), (nb == 2).mean()])
    sh, pc, syn_sh = np.array(sh), np.array(pc), np.array(syn_sh)
    log(f"round-tripped real (200): real-neighbour share median {np.median(sh):.2f} mean {sh.mean():.2f} | nearest-real "
        f"distance percentile median {np.median(pc):.0f} | distance to its own original = {np.median(own) / np.median(nn_rr):.2f}x "
        f"the typical real-to-real nearest-neighbour distance")
    log(f"  with round-tripped clips in the pool, their 10 neighbours are real {syn_sh[:, 0].mean() * 100:.0f}% / "
        f"synthetic {syn_sh[:, 1].mean() * 100:.0f}% / other round-tripped {syn_sh[:, 2].mean() * 100:.0f}%")
    a, b = probes(R[sel], T[sel]), probes(T[sel], S[:200])
    log(f"  RBF probe real vs round-tripped {a['RBF']:.3f} (linear {a['linear']:.3f}) | round-tripped vs synthetic "
        f"{b['RBF']:.3f} (linear {b['linear']:.3f})  [plain BEATs, 200 vs 200]")


if __name__ == "__main__":
    main()
