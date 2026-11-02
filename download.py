"""Download the DCASE 2026 Challenge Task 2 data from Zenodo and unpack it into data/<machine>/{train,test}.

  uv run python download.py dev                  # development set: 7 machines, train + test (~4.8 GB)
  uv run python download.py eval                 # evaluation set: 5 machines, train + test (~3.4 GB)
  uv run python download.py all
  uv run python download.py dev --machines ToyCar fan   # only some machines
  uv run python download.py eval --list          # show the files and sizes, download nothing

File names, sizes and MD5 checksums come from the Zenodo records below. Archives go to dataset/ (dev) and
dataset/eval/ (eval), the downloads resume after an interruption (.part files), every file is checked against its MD5,
and archives already complete are not downloaded again. Machines already unpacked in data/ are skipped.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RECORDS = {  # Zenodo record id -> where its archives are stored
    "dev": [("19336329", ROOT / "dataset")],                       # DCASE 2026 Task 2 Development Dataset
    "eval": [("20151556", ROOT / "dataset" / "eval"),              # Additional Training Dataset (eval machines' train)
             ("20437238", ROOT / "dataset" / "eval")],             # Evaluation Dataset (eval machines' test)
}
CHUNK = 1 << 20


def record_files(rec):
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}", timeout=60) as r:
        d = json.loads(r.read())
    return [{"key": f["key"], "size": f["size"], "md5": f["checksum"].split(":", 1)[1], "url": f["links"]["self"]}
            for f in d["files"]]


def machine_of(zip_path):
    """Top-level folder inside the archive = machine name (e.g. dev_ToyCar_r2.zip -> ToyCar)."""
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()[0].split("/")[0]


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def download(f, dest, pos):
    out, part = dest / f["key"], dest / (f["key"] + ".part")
    if out.exists() and out.stat().st_size == f["size"]:
        return out
    have = part.stat().st_size if part.exists() else 0
    if have > f["size"]:
        part.unlink(); have = 0
    if have < f["size"]:
        req = urllib.request.Request(f["url"], headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have and r.status == 206 else "wb") as fo, \
                tqdm(total=f["size"], initial=have if r.status == 206 else 0, unit="B", unit_scale=True,
                     unit_divisor=1024, desc=f["key"][:32], position=pos, leave=False) as bar:
            for b in iter(lambda: r.read(CHUNK), b""):
                fo.write(b); bar.update(len(b))
    if part.stat().st_size != f["size"]:
        raise IOError(f"{f['key']}: got {part.stat().st_size} bytes, expected {f['size']}")
    if md5sum(part) != f["md5"]:
        part.unlink()
        raise IOError(f"{f['key']}: MD5 mismatch (partial file removed, run again)")
    part.replace(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", choices=["dev", "eval", "all"])
    ap.add_argument("--machines", nargs="+", help="only these machines (names as in data/, e.g. ToyCar fan)")
    ap.add_argument("--list", action="store_true", help="list the files and exit")
    ap.add_argument("--no-extract", action="store_true", help="download only")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = []
    for s in (["dev", "eval"] if args.set == "all" else [args.set]):
        for rec, dest in RECORDS[s]:
            for f in record_files(rec):
                jobs.append((f, dest))
    if args.machines:
        want = {m.lower() for m in args.machines}
        # dev_ToyCar_r2.zip / eval_data_ToyDrone_train.zip -> machine part of the name
        name = lambda k: k.removesuffix(".zip").removeprefix("dev_").removeprefix("eval_data_") \
            .removesuffix("_train").removesuffix("_test").removesuffix("_r2")
        jobs = [(f, d) for f, d in jobs if name(f["key"]).lower() in want]
        if not jobs:
            sys.exit(f"no archives match {args.machines}")
    total = sum(f["size"] for f, _ in jobs)
    print(f"{len(jobs)} archives, {total / 1e9:.2f} GB")
    if args.list:
        for f, d in jobs:
            print(f"  {f['key']:42s} {f['size'] / 1e6:8.1f} MB -> {d.relative_to(ROOT)}")
        return

    for _, d in jobs:
        d.mkdir(parents=True, exist_ok=True)
    done, failed = [], []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download, f, d, i % args.workers + 1): f["key"] for i, (f, d) in enumerate(jobs)}
        with tqdm(total=len(futs), desc="archives", unit="file", position=0) as bar:
            for fu in as_completed(futs):
                try:
                    done.append(fu.result())
                except Exception as e:  # keep going; report at the end
                    failed.append(f"{futs[fu]}: {e}")
                bar.update(1)
    for msg in failed:
        print("FAILED", msg)

    if not args.no_extract:
        for z in sorted(done):
            m = machine_of(z)
            split = "test" if z.stem.endswith("_test") else "train" if z.stem.endswith("_train") else None
            target = [DATA / m / split] if split else [DATA / m / "train", DATA / m / "test"]
            if all(t.is_dir() and any(t.glob("*.wav")) for t in target):
                print(f"{z.name}: already in {DATA.relative_to(ROOT)}/{m}, skipped")
                continue
            with zipfile.ZipFile(z) as zf:
                zf.extractall(DATA)
            print(f"{z.name} -> {DATA.relative_to(ROOT)}/{m} ({', '.join(f'{t.name} {len(list(t.glob(chr(42) + '.wav')))} wav' for t in target)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
