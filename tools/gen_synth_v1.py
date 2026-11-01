"""Synthetic ToyCar set v1 for domain-adversarial training (real vs synthetic branch with gradient reversal).

AudioGen-medium + LoRA r16 'full10' + [toycar] token (trained on 1000 normal ToyCar near-minus-far clips, full 10 s),
cfg 3, top-k 250, 10 s per clip, generated from scratch (no real opening, so the fault prompts take effect).
  normal : 250 clips, 'the sound of <toycar>'
  faulty : 250 clips over the three fault prompts of audiogen_ti.ANOMALY_PROMPTS (84 / 83 / 83)
Output: outputs/audiogen/synth_v1/{normal,faulty}/NNN.wav (16 kHz mono) + meta.json (prompt, seed per clip).
Resumable: existing files are skipped. Each batch of 10 has its own seed (base + batch index); clips within a batch
differ because every token is sampled.
Runs in envs/audiocraft:  envs/audiocraft/Scripts/python -u tools/gen_synth_v1.py
"""
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A  # noqa: E402

OUT = ROOT / "outputs" / "audiogen" / "synth_v1"
BATCH = 4  # 10 x 10 s with guidance (= 20 sequences) overflows 16 GB into shared memory


def plan():
    names = list(A.ANOMALY_PROMPTS)
    jobs = [("normal", i, A.ANOMALY_PROMPTS[names[0]]) for i in range(250)]
    counts = [84, 83, 83]
    k = 0
    for name, n in zip(names[1:], counts):
        jobs += [("faulty", k + j, A.ANOMALY_PROMPTS[name]) for j in range(n)]
        k += n
    return jobs


def main():
    dev = torch.device("cuda")
    A.LORA.update(rank=16, name="full10")
    m, cond, t5, tid = A.load_model(dev)
    m.set_generation_params(duration=10.0, cfg_coef=3.0, top_k=250)
    for d in ("normal", "faulty"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    mpath = OUT / "meta.json"
    meta = json.loads(mpath.read_text()) if mpath.exists() else {}
    jobs = [j for j in plan() if not (OUT / j[0] / f"{j[1]:03d}.wav").exists()]
    A.log(f"synth_v1: {len(jobs)} clips to generate (of 500)")
    t0 = time.time()
    for b in range(0, len(jobs), BATCH):
        chunk = jobs[b : b + BATCH]
        seed = 20000 + (0 if chunk[0][0] == "normal" else 1000) + chunk[0][1]  # deterministic, reproducible
        torch.manual_seed(seed)
        with torch.no_grad():
            w = m.generate([p for _, _, p in chunk]).cpu().numpy()[:, 0, :160000]
        for (cls, idx, prompt), x in zip(chunk, w):
            sf.write(OUT / cls / f"{idx:03d}.wav", x, 16000)
            meta[f"{cls}/{idx:03d}.wav"] = {"prompt": prompt.replace("<toycar>", "[toycar]"), "batch_seed": seed}
        mpath.write_text(json.dumps(meta, indent=1))
        done = b + len(chunk)
        A.log(f"synth_v1: {done}/{len(jobs)} generated, {(time.time() - t0) / done:.1f} s per clip")
    A.log("synth_v1: done")


if __name__ == "__main__":
    main()
