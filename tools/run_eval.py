"""Evaluation-set pipeline for the noise-aware autoencoder (runs unattended; log -> outputs/live.log).

1. wait until the 10 evaluation zips in dataset/eval have their full size (sizes from the Zenodo records)
2. extract them and place the wav files in data/<machine>/{train,test}
3. build the waveform caches (test labels / domains from the official ground truth, see dcase_ae/efficientad.py)
4. microphone order: the near mic = the louder channel averaged over the machine's training clips (on the development
   set channel 0 is the near mic and ~15 dB louder); channels are swapped in the cache if needed, and logged
5. train + score each machine with exactly the development-set settings (ae_noiseaware.py, v4_all7 settings),
   exporting anomaly scores in the official format to third_party/dcase2026_task2_evaluator/teams/ours/<score>/
6. run the official evaluator (python 3.11 throwaway environment) and print the result tables
"""
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_ZIPS = ROOT / "dataset" / "eval"
EVALUATOR = ROOT / "third_party" / "dcase2026_task2_evaluator"
MACHINES = ["BlowerDustCollector", "Sander", "SewingMachine", "ToothBrush", "ToyDrone"]
RECORDS = {"train": "20151556", "test": "20437238"}
LOG = ROOT / "outputs" / "live.log"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')}  [eval-pipeline] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def expected_sizes():
    out = {}
    for split, rec in RECORDS.items():
        d = json.loads(urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}", timeout=60).read())
        out.update({f["key"]: f["size"] for f in d["files"]})
    return out


def run(cmd):
    log(f"$ {cmd}")
    p = subprocess.Popen(cmd, shell=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    with open(LOG, "a", encoding="utf-8") as f:
        for line in p.stdout:
            if not any(k in line.lower() for k in ("warn", "triton", "redirects")):
                f.write(line)
                f.flush()
    p.wait()
    if p.returncode:
        log(f"command failed with exit code {p.returncode}")
        sys.exit(p.returncode)


def main():
    sizes = expected_sizes()
    log(f"waiting for {len(sizes)} zips ({sum(sizes.values()) / 1e9:.2f} GB)")
    while True:
        have = {k: (EVAL_ZIPS / k).stat().st_size if (EVAL_ZIPS / k).exists() else 0 for k in sizes}
        if all(have[k] == sizes[k] for k in sizes):
            break
        stalled = [k for k in sizes if have[k] < sizes[k]]
        log(f"downloaded {sum(have.values()) / 1e9:.2f} of {sum(sizes.values()) / 1e9:.2f} GB ({len(stalled)} files incomplete)")
        time.sleep(60)
    log("all zips complete")

    # 2. extract and place
    tmp = ROOT / "data" / "_eval_extract"
    for m in MACHINES:
        if (ROOT / "data" / m / "train").is_dir() and (ROOT / "data" / m / "test").is_dir():
            continue
        for split in ("train", "test"):
            z = EVAL_ZIPS / f"eval_data_{m}_{split}.zip"
            with zipfile.ZipFile(z) as zf:
                zf.extractall(tmp)
        for split in ("train", "test"):
            cands = [p for p in tmp.rglob(split) if p.is_dir() and m in str(p) and any(p.glob("*.wav"))]
            assert len(cands) == 1, (m, split, cands)
            dst = ROOT / "data" / m / split
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(cands[0]), dst)
            log(f"{m} {split}: {len(list(dst.glob('*.wav')))} wav files -> {dst}")
    shutil.rmtree(tmp, ignore_errors=True)

    # 3 + 4. caches and microphone order
    sys.path.insert(0, str(ROOT))
    from dcase_ae.efficientad import build_cache
    for m in MACHINES:
        for split in ("train", "test"):
            build_cache(ROOT / "data", ROOT / "cache", m, split)
        tr = np.load(ROOT / "cache" / f"wav_{m}_train.npy", mmap_mode="r")
        rms = np.sqrt((tr[:200].astype(np.float64) ** 2).mean(axis=(0, 2)))
        db = 20 * np.log10(rms[0] / rms[1])
        flag = ROOT / "cache" / f"wav_{m}_micorder.json"
        if not flag.exists():
            swap = bool(rms[1] > rms[0])
            if swap:
                for split in ("train", "test"):
                    a = np.load(ROOT / "cache" / f"wav_{m}_{split}.npy", mmap_mode="r+")
                    a[:] = a[:, ::-1].copy()
                    a.flush()
            flag.write_text(json.dumps({"channel0_minus_channel1_db": float(db), "swapped": swap}))
        info = json.loads(flag.read_text())
        meta = json.loads((ROOT / "cache" / f"wav_{m}_test.json").read_text())
        log(f"{m}: channel 0 is {info['channel0_minus_channel1_db']:+.1f} dB vs channel 1 -> "
            f"{'SWAPPED so channel 0 = near' if info['swapped'] else 'channel 0 = near (as on the development set)'}; "
            f"test clips {len(meta['label'])} (anomalous {sum(meta['label'])}, target domain {sum(meta['domain'])})")

    # 5. train + export
    team = EVALUATOR / "teams" / "ours"
    for m in MACHINES:
        run(f"uv run python -u ae_noiseaware.py --name eval_v4 --machine {m} --holdout 0 --target_frac 0 "
            f"--domain_norm 0 --export_dir \"{team}\"")

    # 6. official evaluator
    run(f"cd \"{EVALUATOR}\" && uv run --no-project --python 3.11 --with pandas --with scikit-learn --with scipy "
        f"--with matplotlib --with seaborn --with librosa --with fasteners python dcase2026_task2_evaluator.py "
        f"--teams_root_dir ./teams --result_dir ./teams_result --additional_result_dir ./teams_additional_result "
        f"--dir_depth 2 --out_all False")
    for f in sorted((EVALUATOR / "teams_result").glob("*.csv")):
        log(f"--- {f.name}")
        with open(LOG, "a", encoding="utf-8") as out:
            out.write(f.read_text() + "\n")
    log("done")


if __name__ == "__main__":
    main()
