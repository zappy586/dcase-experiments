"""What is left of the real-vs-synthetic gap after CLoRA-Align: codec fingerprint or generated content?
Encoder: plain BEATs vs BEATs + saved CLoRA-Align adapters (default: best run, rank 16, 480 clips).
Sets (full 10 s, mean-pooled last layer): 200 held-out real training normals (the clora_align probe ids), the same 200
clips sent through EnCodec and back (cache/toycar_rt_train.npy), 50 held-out synthetic normals, 200 synthetic normals
used in training.  Probes: 5-fold CV AUC (linear / RBF / MLP / kNN, clora_align.probes).
  real vs round-tripped real = codec fingerprint alone (same content)
  round-tripped real vs synthetic = content alone (both decoded by EnCodec)
Usage (main env): uv run python tools/clora_gap_split.py
"""
import argparse
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
from clora_align import FrozenALoRA, probes  # noqa: E402

SYN = ROOT / "outputs" / "audiogen" / "synth_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/clora_align_r16_s1_lr0.0005_extra.pt")
    args = ap.parse_args()
    dev = torch.device("cuda")
    log = lambda m: A.log(f"clora_gap_split: {m}")
    rng = np.random.default_rng(0)
    perm = {c: rng.permutation(250) for c in ("normal", "faulty")}
    pid = rng.choice(1000, 200, replace=False)
    D = A.Data("ToyCar", dev)
    real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)])
    rt = torch.from_numpy(np.load(ROOT / "cache" / "toycar_rt_train.npy").astype(np.float32))[:, :160000]
    gen = torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in sorted((SYN / "normal").glob("*.wav"))]))
    sets = {"real": real[pid], "rt": rt[pid], "syn_ho": gen[perm["normal"][:50]], "syn_fit": gen[perm["normal"][50:]]}

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

    F = {"plain": {k: pooled(v) for k, v in sets.items()}}
    saved = torch.load(ROOT / args.ckpt)
    for name in sorted({n.rsplit(".", 1)[0] for n in saved}):
        parent_name, attr = name.rsplit(".", 1)
        parent = beats.get_submodule(parent_name)
        setattr(parent, attr, FrozenALoRA(getattr(parent, attr), saved[f"{name}.A"].to(dev), 1.0))
    sd = beats.state_dict(); sd.update({n: t.to(dev) for n, t in saved.items()}); beats.load_state_dict(sd)
    F["aligned"] = {k: pooled(v) for k, v in sets.items()}

    fmt = lambda p: "  ".join(f"{k} {v:.3f}" for k, v in p.items())
    for enc, f in F.items():
        log(f"[{enc}] real vs synthetic held-out (total gap)   {fmt(probes(f['real'], f['syn_ho']))}")
        log(f"[{enc}] real vs round-tripped real (codec only)  {fmt(probes(f['real'], f['rt']))}")
        log(f"[{enc}] round-tripped vs synth held-out (content) {fmt(probes(f['rt'], f['syn_ho']))}")
        log(f"[{enc}] round-tripped vs synth fit (content, seen) {fmt(probes(f['rt'], f['syn_fit']))}")


if __name__ == "__main__":
    main()
