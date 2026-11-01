"""Test: synthetic ToyCar normals by AudioGen CONTINUATION of real 3 s openings (instead of from scratch as synth_v1).
Same model and settings as tools/gen_synth_v1.py (AudioGen-medium + LoRA r16 'full10' + [toycar] token, cfg 3,
top-k 250, 10 s total, prompt 'the sound of <toycar>'). Openings: first 3 s of real training normals that are NOT in the
200-clip probe set used by clora_align (same draws). The returned clip = codec-decoded opening + 7 s generated.
Output: outputs/audiogen/cont_test/normal/NNN.wav + meta.json (source training clip per file).
Runs in envs/audiocraft:  envs/audiocraft/Scripts/python -u tools/gen_cont_test.py --n 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A  # noqa: E402

PROMPT = A.ANOMALY_PROMPTS["normal: '<toycar>'"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="outputs/audiogen/cont_test")
    ap.add_argument("--opening", type=float, default=3.0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = ROOT / args.out / "normal"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    _ = {c: rng.permutation(250) for c in ("normal", "faulty")}
    pid = rng.choice(1000, 200, replace=False)
    fit = np.setdiff1d(np.arange(1000), pid)
    src = np.random.default_rng(1).choice(fit, args.n, replace=False)
    A.LORA.update(rank=16, name="full10")
    m, cond, t5, tid = A.load_model(dev)
    m.set_generation_params(duration=10.0, cfg_coef=3.0, top_k=250)
    D = A.Data("ToyCar", dev)
    meta, L = {}, int(args.opening * 16000)
    for b in range(0, args.n, 4):
        ids = src[b : b + 4].tolist()
        x = D.audio("train", ids)
        torch.manual_seed(40000 + b)
        with torch.no_grad():
            w = m.generate_continuation(x[..., :L], 16000, [PROMPT] * len(ids)).cpu().numpy()[:, 0, :160000]
        for j, (i, y) in enumerate(zip(ids, w)):
            k = b + j
            sf.write(out / f"{k:03d}.wav", y, 16000)
            meta[f"normal/{k:03d}.wav"] = {"source_train_clip": int(i), "opening_s": args.opening, "prompt": PROMPT, "seed": 40000 + b}
        A.log(f"cont_test: {b + len(ids)}/{args.n} generated")
    (ROOT / args.out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
