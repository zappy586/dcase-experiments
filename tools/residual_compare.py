"""Add a 'near minus far' version of every real clip already on the comparison page (outputs/sam_compare/).

near minus far: per frequency band, the far-mic sound is matched in volume and timing to the near mic (one setting
per band per clip) and subtracted, so background both mics hear cancels and mostly the machine's own sound remains
(same processing as the Dasheng residual input, tools/simple_baselines.py near_only_audio).
Volume: one fixed boost per machine (median over its normal clips, to the near mic's loudness), the same for
normal and anomalous clips, so loudness differences between them are kept.
Usage: uv run python tools/residual_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from filter_compare import OUT, SR, mel_db, save_png  # noqa: E402
from simple_baselines import near_only_audio  # noqa: E402

VARIANT = "near minus far"


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpath = ROOT / OUT / "manifest.json"
    manifest = json.loads(mpath.read_text())
    items = manifest["items"]
    done = {(i["clip"], i["variant"]) for i in items}
    real = {(i["clip"], i["machine"], i["label"], i["file"]) for i in items
            if i["mic"] == "near" and i["variant"] == "original" and "synthetic" not in i["machine"]}
    by_machine = {}
    for c in sorted(real):
        by_machine.setdefault(c[1], []).append(c)
    for m, clips in by_machine.items():
        res = {}
        for clip, _, label, f in clips:
            w, sr = sf.read(ROOT / "data" / m / "test" / f, dtype="float32", always_2d=True)
            assert sr == SR
            r = near_only_audio(torch.from_numpy(w.T.copy())[None], dev)[0].cpu().numpy()
            res[clip] = (w[:, 0], r, label)
        ratios = [np.sqrt(np.mean(n**2)) / (np.sqrt(np.mean(r**2)) + 1e-12) for n, r, lab in res.values() if lab == "normal"]
        gain = float(np.median(ratios)) if ratios else 1.0
        for clip, _, label, f in clips:
            if (clip, VARIANT) in done:
                continue
            near, r, _ = res[clip]
            out = np.clip(r * gain, -1, 1).astype(np.float32)
            stem = f"{clip}_near_near-minus-far"
            sf.write(ROOT / OUT / "audio" / f"{stem}.wav", out, SR)
            vmax = float(mel_db(near).max())
            save_png(mel_db(out), ROOT / OUT / "img" / f"{stem}.png", len(out) / SR, vmax - 70, vmax)
            items.append({"clip": clip, "machine": m, "label": label, "file": f, "mic": "near", "variant": VARIANT,
                          "audio": f"audio/{stem}.wav", "img": f"img/{stem}.png",
                          "note": f"far mic matched per band and subtracted; boosted {20 * np.log10(gain):+.1f} dB (same for all {m} clips)"})
        print(f"{m}: {len(clips)} clips, boost {20 * np.log10(gain):+.1f} dB", flush=True)
    mpath.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
