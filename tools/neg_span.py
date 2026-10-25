"""SAM-Audio negative-span test: "separate what is in this recording but NOT in the normal examples".

Input per test clip (near mic, 48 kHz): 4 s from the middle of each of its K nearest normal training clips
(tools/nearest_normals.py), then the 10 s test clip. Spans "-" (target absent) over the normal parts, test part unmarked.
Text describes the target (= the anomaly). Score = share of the TEST part's energy that goes to the target, minus the
same share over the normal parts. Higher = more sound that the normal examples don't have.

Runs in the SAM-Audio environment (PYTHONPATH=tools):
  envs/samaudio/Scripts/python tools/neg_span.py --bench        # memory / speed / output difference per precision
  envs/samaudio/Scripts/python tools/neg_span.py [--k 3] [--precision bf16]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import roc_auc_score

from sam_compare import OUT, mel_db, save_png

SEG = 4.0
MODEL_ID = "facebook/sam-audio-small-tv"
TEXTS = {"neg span, no text": "", "neg span + 'mechanical fault noise'": "mechanical fault noise"}


def load_model(precision):
    from sam_audio import SAMAudio, SAMAudioProcessor

    model = SAMAudio.from_pretrained(MODEL_ID, visual_ranker=None, text_ranker=None, span_predictor=None).eval()
    model.vision_encoder.model.to("cpu")  # only its output size is used when there is no video
    model.to("cuda") if precision == "fp32-all-gpu" else None
    for name, mod in model.named_children():  # everything except the unused video encoder goes to the GPU
        if name != "vision_encoder":
            mod.to("cuda")
    if precision == "bf16-weights":
        model.transformer.to(torch.bfloat16)
    return model, SAMAudioProcessor.from_pretrained(MODEL_ID)


def separate(model, proc, x, text, anchors, precision, seed=0):
    batch = proc(audios=[x[None]], descriptions=[text], anchors=anchors).to("cuda")
    torch.manual_seed(seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision != "fp32" and precision != "fp32-all-gpu"):
        res = model.separate(batch, predict_spans=False, reranking_candidates=1)
    n = len(x)
    return res.target[0].float().cpu().reshape(-1)[:n], res.residual[0].float().cpu().reshape(-1)[:n]


def build_input(m, entry, k, sr):
    parts = []
    for f in entry["neighbours"][:k]:
        w, in_sr = sf.read(Path("data") / m / "train" / f, dtype="float32", always_2d=True)
        mid = len(w) // 2
        parts.append(torch.from_numpy(w[mid - int(SEG / 2 * in_sr): mid + int(SEG / 2 * in_sr), 0]))
    w, in_sr = sf.read(Path("data") / m / "test" / entry["test"], dtype="float32", always_2d=True)
    parts.append(torch.from_numpy(w[: 10 * in_sr, 0]))
    x = torchaudio.functional.resample(torch.cat(parts), in_sr, sr)
    return x, int(k * SEG * sr), [["-", i * SEG, (i + 1) * SEG] for i in range(k)]


def bench(k):
    nn = json.loads(Path("outputs/span_nn/neighbours.json").read_text())
    entry = nn["fan"][4]
    ref = None
    for precision in ("fp32-all-gpu", "fp32", "bf16-autocast", "bf16-weights"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, proc = load_model(precision)
        loaded = torch.cuda.memory_allocated() / 1e9
        x, _, anchors = build_input("fan", entry, k, proc.audio_sampling_rate)
        separate(model, proc, x, "", [anchors], precision)  # warm-up
        torch.cuda.synchronize()
        t0 = time.time()
        tgt, _ = separate(model, proc, x, "", [anchors], precision)
        torch.cuda.synchronize()
        dt = time.time() - t0
        diff = "reference" if ref is None else f"output differs from fp32 by {10 * np.log10(((tgt - ref) ** 2).sum() / (ref**2).sum() + 1e-12):.1f} dB"
        ref = tgt if ref is None else ref
        print(f"{precision:14s} weights on GPU {loaded:5.2f} GB | peak {torch.cuda.max_memory_allocated() / 1e9:5.2f} GB | "
              f"{dt:5.1f} s per {len(x) / proc.audio_sampling_rate:.0f} s of audio | {diff}", flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--precision", default="bf16-autocast")
    ap.add_argument("--machine", default="fan")
    args = ap.parse_args()
    if args.bench:
        return bench(args.k)

    nn = json.loads(Path("outputs/span_nn/neighbours.json").read_text())
    model, proc = load_model(args.precision)
    sr = proc.audio_sampling_rate
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text())
    done = {(i["clip"], i["mic"], i["variant"]) for i in manifest["items"]}
    results = []
    for e in nn[args.machine]:
        x, split, anchors = build_input(args.machine, e, args.k, sr)
        clip = f"{args.machine}_{e['label']}_{e['test'].split('_')[5]}"
        row = {"label": e["label"], "test": e["test"]}
        for name, text in TEXTS.items():
            t0 = time.time()
            tgt, rsd = separate(model, proc, x, text, [anchors], args.precision)
            share = lambda a, b: float((tgt[a:b] ** 2).sum() / ((tgt[a:b] ** 2).sum() + (rsd[a:b] ** 2).sum() + 1e-12))
            s_test, s_norm = share(split, len(x)), share(0, split)
            row[name] = s_test - s_norm
            row[name + " parts"] = (round(s_test, 3), round(s_norm, 3), round(time.time() - t0, 1))
            for variant, wav in (("neg-span input", x.numpy()), (f"neg-span target ({name})", tgt.numpy())):
                if (clip, "near", variant) in done:
                    continue
                stem = f"{clip}_near_{''.join(c if c.isalnum() else '-' for c in variant)}"
                sf.write(OUT / "audio" / f"{stem}.wav", wav, sr)
                save_png(mel_db(wav, sr), float(mel_db(x.numpy(), sr).max()), len(wav) / sr, OUT / "img" / f"{stem}.png")
                note = f"0-{args.k * SEG:.0f} s: {args.k} nearest normal clips marked '-'; then the test clip"
                if "target" in variant:
                    note += f" | target share: test {s_test:.0%}, normal parts {s_norm:.0%}"
                manifest["items"].append({"clip": clip, "machine": args.machine, "label": e["label"], "file": e["test"],
                                          "mic": "near", "variant": variant, "audio": f"audio/{stem}.wav",
                                          "img": f"img/{stem}.png", "note": note})
                done.add((clip, "near", variant))
        results.append(row)
        mpath.write_text(json.dumps(manifest, indent=1))
        print(f"{e['label']:7s} " + "  ".join(f"{n}: {row[n]:+.3f} (test, normal, sec) {row[n + ' parts']}" for n in TEXTS), flush=True)
    lab = [r["label"] == "anomaly" for r in results]
    print("AUC " + "  ".join(f"{n}: {roc_auc_score(lab, [r[n] for r in results]):.2f}" for n in TEXTS))


if __name__ == "__main__":
    main()
