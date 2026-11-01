"""750 extra synthetic ToyCar normals, same recipe as tools/gen_synth_v1.py (AudioGen-medium + LoRA r16 'full10' +
[toycar] token, cfg 3, top-k 250, 10 s, from scratch, prompt 'the sound of <toycar>'), for CLoRA-Align training.
Kept in a separate folder so synth_v1 (and its fixed 50-clip held-out split) stays unchanged.
Output: outputs/audiogen/synth_v1_extra/normal/NNN.wav (000-749) + meta.json. Seeds 30000 + first index of each batch
(synth_v1 used 20000+), so no clip repeats a v1 batch. Resumable.
Runs in envs/audiocraft:  envs/audiocraft/Scripts/python -u tools/gen_synth_extra.py
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

OUT = ROOT / "outputs" / "audiogen" / "synth_v1_extra"
N, BATCH = 750, 4
PROMPT = A.ANOMALY_PROMPTS["normal: '<toycar>'"]


def main():
    dev = torch.device("cuda")
    A.LORA.update(rank=16, name="full10")
    m, cond, t5, tid = A.load_model(dev)
    m.set_generation_params(duration=10.0, cfg_coef=3.0, top_k=250)
    (OUT / "normal").mkdir(parents=True, exist_ok=True)
    mpath = OUT / "meta.json"
    meta = json.loads(mpath.read_text()) if mpath.exists() else {}
    todo = [i for i in range(N) if not (OUT / "normal" / f"{i:03d}.wav").exists()]
    A.log(f"synth_extra: {len(todo)} clips to generate (of {N})")
    t0 = time.time()
    for b in range(0, len(todo), BATCH):
        chunk = todo[b : b + BATCH]
        seed = 30000 + chunk[0]
        torch.manual_seed(seed)
        with torch.no_grad():
            w = m.generate([PROMPT] * len(chunk)).cpu().numpy()[:, 0, :160000]
        for idx, x in zip(chunk, w):
            sf.write(OUT / "normal" / f"{idx:03d}.wav", x, 16000)
            meta[f"normal/{idx:03d}.wav"] = {"prompt": PROMPT.replace("<toycar>", "[toycar]"), "batch_seed": seed}
        mpath.write_text(json.dumps(meta, indent=1))
        done = b + len(chunk)
        if done % 20 == 0 or done == len(todo):
            el = time.time() - t0
            A.log(f"synth_extra: {done}/{len(todo)}, {el / done:.1f} s per clip, ~{el / done * (len(todo) - done) / 60:.0f} min left")
    A.log("synth_extra: done")


if __name__ == "__main__":
    main()
