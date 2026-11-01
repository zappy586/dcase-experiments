"""Send the real ToyCar clips (near-minus-far, machine gain, first 10 s) through AudioGen's EnCodec and back
(encode -> tokens -> decode), so they carry the same codec artefacts as the generated clips.
Output: cache/toycar_rt_{train,test}.npy, float16, (N, 160000), same order as cache/wav_ToyCar_{split}.json.
Runs in envs/audiocraft:  envs/audiocraft/Scripts/python -u tools/roundtrip_real.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A  # noqa: E402


@torch.no_grad()
def main():
    from audiocraft.models import AudioGen
    dev = torch.device("cuda")
    codec = AudioGen.get_pretrained("facebook/audiogen-medium", device="cuda").compression_model.eval()
    D = A.Data("ToyCar", dev)
    for split in ("train", "test"):
        n = D.wav[split].shape[0]
        out = np.zeros((n, 160000), np.float16)
        for i in range(0, n, 10):
            x = D.audio(split, list(range(i, min(i + 10, n))))[:, :, :160000]
            out[i : i + len(x)] = codec.decode(*codec.encode(x))[:, 0, :160000].clamp(-1, 1).cpu().numpy()
        np.save(ROOT / "cache" / f"toycar_rt_{split}.npy", out)
        A.log(f"roundtrip_real: {split} {n} clips done")


if __name__ == "__main__":
    main()
