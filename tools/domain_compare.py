"""Add target-domain test clips to the comparison page (outputs/sam_compare/), next to the source-domain ones.

Per machine: 2 normal + 2 anomalous target-domain test clips (fixed random pick), each with near mic, far mic and
'near minus far' (same processing and the same per-machine volume boost as tools/residual_compare.py).
The page's Domain filter (source / target) is read from the file name.
Usage: uv run python tools/domain_compare.py
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from filter_compare import OUT, SR, mel_db, save_png  # noqa: E402
from simple_baselines import MACHINES, near_only_audio  # noqa: E402


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpath = ROOT / OUT / "manifest.json"
    manifest = json.loads(mpath.read_text())
    items = manifest["items"]
    done = {(i["clip"], i["mic"], i["variant"]) for i in items}
    for m in MACHINES:
        # same volume boost as the source clips' 'near minus far' on the page
        boosts = [float(re.search(r"boosted ([+-][\d.]+) dB", i["note"]).group(1)) for i in items
                  if i["machine"] == m and i["variant"] == "near minus far" and "boosted" in i.get("note", "")]
        gain = 10 ** ((boosts[0] if boosts else 0.0) / 20)
        files = sorted((ROOT / "data" / m / "test").glob("*_target_test_*.wav"))
        rng = random.Random(0)
        pick = rng.sample([f for f in files if "_normal_" in f.name], 2) + rng.sample([f for f in files if "_anomaly_" in f.name], 2)
        for f in pick:
            label = "normal" if "_normal_" in f.name else "anomaly"
            clip = f"{m}_target_{label}_{f.stem.split('_')[5]}"
            w, sr = sf.read(f, dtype="float32", always_2d=True)
            assert sr == SR
            r = near_only_audio(torch.from_numpy(w.T.copy())[None], dev)[0].cpu().numpy()
            vmax = {"near": float(mel_db(w[:, 0]).max()), "far": float(mel_db(w[:, 1]).max())}
            for mic, variant, wav, note in (("near", "original", w[:, 0], ""), ("far", "original", w[:, 1], ""),
                                            ("near", "near minus far", np.clip(r * gain, -1, 1),
                                             f"far mic matched per band and subtracted; boosted {20 * np.log10(gain):+.1f} dB (same for all {m} clips)")):
                if (clip, mic, variant) in done:
                    continue
                stem = f"{clip}_{mic}_{variant.replace(' ', '-')}"
                sf.write(ROOT / OUT / "audio" / f"{stem}.wav", np.asarray(wav, dtype=np.float32), SR)
                save_png(mel_db(np.asarray(wav, dtype=np.float32)), ROOT / OUT / "img" / f"{stem}.png", len(wav) / SR,
                         vmax[mic] - 70, vmax[mic])
                items.append({"clip": clip, "machine": m, "label": label, "domain": "target", "file": f.name, "mic": mic,
                              "variant": variant, "audio": f"audio/{stem}.wav", "img": f"img/{stem}.png", "note": note})
        print(f"{m}: added {len(pick)} target clips", flush=True)
    for i in items:  # domain for every item, from the file name
        i.setdefault("domain", "target" if "_target_" in i["file"] else "source" if "_source_" in i["file"] else "")
    mpath.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
