"""Few-shot anomaly check with Audio Flamingo Next (8B), language model quantised to 4-bit NF4 on load.

Same clips as tools/alm_fewshot.py (same seed and draw order). The processor takes one audio per prompt, so the
few-shot prompt is ONE recording: N_REF normal references and the test clip joined with 1 s of silence, and the text
says which part is which.
Score = log p(yes) - log p(no) for "does recording N contain an anomaly?" (both capitalisations pooled).

Usage: uv run python tools/alm_fewshot_afnext.py [--machines fan,valveEmu] [--n_test 20] [--n_ref 4]
"""
import argparse
import csv
import glob
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoProcessor, BitsAndBytesConfig, MusicFlamingoForConditionalGeneration

MODEL = "pretrained/audio-flamingo-next-hf"
SR = 16000
ANOMALY = "an anomaly (a fault, damage, an unusual noise or a change in the machine's sound)"


def near_mic(path):
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR
    return w[:, 0]


def prompt(machine, refs, test):
    """(text question, one audio array)."""
    if not refs:
        return (f"This is a 10-second recording of a {machine}, made with a microphone next to it. "
                f"Does it contain {ANOMALY}? Answer with one word: yes or no."), near_mic(test)
    gap = np.zeros(SR, dtype=np.float32)
    parts = []
    for r in refs:
        parts += [near_mic(r), gap]
    audio = np.concatenate(parts + [near_mic(test)])
    n = len(refs) + 1
    return (f"This audio contains {n} recordings of a {machine}, each about 10 seconds long and separated by 1 second "
            f"of silence, made with a microphone next to the machine. Recordings 1 to {n - 1} were made while the machine "
            f"was working normally. Compared with recordings 1 to {n - 1}, does recording {n} contain {ANOMALY}? "
            f"Answer with one word: yes or no."), audio


def inputs(proc, question, audio, device):
    conv = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], audio=[audio], sampling_rate=SR, return_tensors="pt").to(device)
    return {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in inp.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", default="fan,valveEmu")
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--n_ref", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(MODEL)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               llm_int8_skip_modules=["audio_tower", "multi_modal_projector", "lm_head"])
    model = MusicFlamingoForConditionalGeneration.from_pretrained(MODEL, quantization_config=quant, dtype=torch.bfloat16,
                                                                  device_map="cuda").eval()
    print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    tok = proc.tokenizer
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]

    out = Path("outputs/alm_fewshot_afnext")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for m in args.machines.split(","):
        refs = rng.sample(sorted(glob.glob(f"data/{m}/train/*source_train_normal*.wav")), args.n_ref)
        tests = (rng.sample(sorted(glob.glob(f"data/{m}/test/*source_test_normal*.wav")), args.n_test // 2)
                 + rng.sample(sorted(glob.glob(f"data/{m}/test/*source_test_anomaly*.wav")), args.n_test // 2))
        labels = np.array([0] * (args.n_test // 2) + [1] * (args.n_test // 2))
        rows = []
        for cond, r in (("zero-shot", []), ("few-shot", refs)):
            scores = []
            for t in tests:
                with torch.no_grad():
                    logp = torch.log_softmax(model(**inputs(proc, *prompt(m, r, t), model.device)).logits[0, -1].float(), -1)
                scores.append((torch.logsumexp(logp[yes], 0) - torch.logsumexp(logp[no], 0)).item())
            scores = np.array(scores)
            rows.append((cond, scores))
            print(f"{m:10s} {cond:9s} AUC {roc_auc_score(labels, scores):.3f}  accuracy {((scores > 0) == labels).mean():.2f}  "
                  f"says 'yes, anomaly' for {(scores > 0).sum()}/{len(scores)}", flush=True)
        ref = {row["file"]: float(row["ensemble"]) for row in csv.DictReader(open(f"outputs/near_only/scores_{m}.csv"))}
        ens = np.array([ref[Path(t).name] for t in tests])
        print(f"{m:10s} near-only ensemble, same clips  AUC {roc_auc_score(labels, ens):.3f}")
        with open(out / f"{m}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "label", *(c for c, _ in rows), "near_only_ensemble"])
            for i, t in enumerate(tests):
                w.writerow([Path(t).name, labels[i], *(f"{s[i]:.4f}" for _, s in rows), f"{ens[i]:.4f}"])

        for cond, r in (("zero-shot", []), ("few-shot", refs)):  # free-text answer on one anomalous clip
            q, a = prompt(m, r, tests[-1])
            q = q.rsplit(" Answer", 1)[0].replace("does recording", "describe what you hear in recording").replace(
                f"Does it contain {ANOMALY}?", "Describe what you hear and whether it sounds faulty.")
            inp = inputs(proc, q, a, model.device)
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=100, do_sample=False, repetition_penalty=1.2)
            print(f"  [{cond}, anomalous clip] {tok.decode(g[0, inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()}")


if __name__ == "__main__":
    main()
