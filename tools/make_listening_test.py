"""Build a blind listening test: per machine, 5 normal reference clips + 10 test clips (5 normal, 5 anomaly).

All clips are from the SOURCE domain, near mic only (channel 1), original level, 16 kHz.
File names are random ids, so the label can't be read from the URL.

Usage: uv run python tools/make_listening_test.py   -> writes listening_test/ (open with a local web server)
"""
import base64
import json
import random
import secrets
import shutil
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "listening_test"
N_REF, N_TEST_EACH = 5, 5
rng = random.Random(1234)


def near_mic_copy(src: Path, dst: Path):
    w, sr = sf.read(src, dtype="int16", always_2d=True)
    sf.write(dst, w[:, 0], sr, subtype="PCM_16")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "audio").mkdir(parents=True)
    machines = sorted(p.name for p in (ROOT / "data").iterdir() if (p / "test").is_dir())
    data = []
    for m in machines:
        refs = rng.sample(sorted((ROOT / "data" / m / "train").glob("*source_train_normal*.wav")), N_REF)
        test_n = rng.sample(sorted((ROOT / "data" / m / "test").glob("*source_test_normal*.wav")), N_TEST_EACH)
        test_a = rng.sample(sorted((ROOT / "data" / m / "test").glob("*source_test_anomaly*.wav")), N_TEST_EACH)
        tests = [(f, 0) for f in test_n] + [(f, 1) for f in test_a]
        rng.shuffle(tests)
        entry = {"machine": m, "refs": [], "tests": [], "source_files": []}
        for f in refs:
            name = secrets.token_hex(6) + ".wav"
            near_mic_copy(f, OUT / "audio" / name)
            entry["refs"].append(name)
        for f, lab in tests:
            name = secrets.token_hex(6) + ".wav"
            near_mic_copy(f, OUT / "audio" / name)
            entry["tests"].append(name)
            entry["source_files"].append((f.name, lab))
        data.append(entry)

    answers = {d["machine"]: [lab for _, lab in d["source_files"]] for d in data}
    (ROOT / "outputs" / "listening_test_answer_key.json").write_text(json.dumps({d["machine"]: d["source_files"] for d in data}, indent=1))
    page = {"machines": [{k: d[k] for k in ("machine", "refs", "tests")} for d in data]}
    key = base64.b64encode(json.dumps(answers).encode()).decode()  # only to avoid reading answers at a glance
    html = (ROOT / "tools" / "listening_test.html").read_text(encoding="utf-8")
    html = html.replace("__DATA__", json.dumps(page)).replace("__KEY__", key)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote {OUT} ({sum(len(d['refs']) + len(d['tests']) for d in data)} clips, {len(data)} machines)")


if __name__ == "__main__":
    main()
