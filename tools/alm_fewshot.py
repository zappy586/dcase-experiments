"""Few-shot anomaly check with an audio-language model (Qwen2.5-Omni-3B, thinker only, frozen).

Per machine: N_TEST source-domain test clips (half normal, half anomaly) and N_REF normal source-domain training
clips as references; near mic only. Conditions:
  zero-shot : test clip + question
  few-shot  : N_REF normal references, then the test clip + question
Score = log p(first token of "anomaly") - log p(first token of "normal") at the answer position.
Reports AUC and accuracy (score > 0) and, for comparison, the near-only ensemble's AUC on the same clips.

Usage: uv run python tools/alm_fewshot.py [--machines fan,valveEmu] [--n_test 20] [--n_ref 4]
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
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

MODEL = "pretrained/Qwen2.5-Omni-3B"
SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory "
          "and visual inputs, as well as generating text and speech.")
QUESTION = ("Decide whether the test recording sounds like the machine working normally or whether it contains an anomaly "
            "(a fault, damage, an unusual noise or a change in the machine's sound). Answer with one word: normal or anomaly.")


def near_mic(path):
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 16000
    return w[:, 0]


def build(machine, refs, test):
    """Conversation content and the list of audio arrays in the same order."""
    content, audios = [], []
    if refs:
        content.append({"type": "text", "text": f"These are recordings of a {machine} made with a microphone next to it. "
                                                f"The first {len(refs)} recordings were made while the machine was working normally."})
        for i, r in enumerate(refs):
            content += [{"type": "text", "text": f"Normal reference {i + 1}:"}, {"type": "audio", "audio": r}]
            audios.append(near_mic(r))
    else:
        content.append({"type": "text", "text": f"This is a recording of a {machine} made with a microphone next to it."})
    content += [{"type": "text", "text": "Test recording:"}, {"type": "audio", "audio": test}, {"type": "text", "text": QUESTION}]
    audios.append(near_mic(test))
    conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]
    return conv, audios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", default="fan,valveEmu")
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--n_ref", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    proc = Qwen2_5OmniProcessor.from_pretrained(MODEL)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    tok = proc.tokenizer
    ids = {w: tok.encode(w, add_special_tokens=False)[0] for w in ("anomaly", "normal")}
    print("answer tokens:", {w: tok.decode([i]) for w, i in ids.items()})

    out = Path("outputs/alm_fewshot")
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
                conv, audios = build(m, r, t)
                text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                inp = proc(text=text, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True).to(model.device)
                inp = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in inp.items()}
                with torch.no_grad():
                    logp = torch.log_softmax(model(**inp).logits[0, -1].float(), -1)
                scores.append((logp[ids["anomaly"]] - logp[ids["normal"]]).item())
            scores = np.array(scores)
            rows.append((cond, scores))
            print(f"{m:10s} {cond:9s} AUC {roc_auc_score(labels, scores):.3f}  accuracy {((scores > 0) == labels).mean():.2f}  "
                  f"says 'anomaly' for {(scores > 0).sum()}/{len(scores)}", flush=True)
        ref = {row["file"]: float(row["ensemble"]) for row in csv.DictReader(open(f"outputs/near_only/scores_{m}.csv"))}
        ens = np.array([ref[Path(t).name] for t in tests])
        print(f"{m:10s} {'near-only ensemble, same clips':s}  AUC {roc_auc_score(labels, ens):.3f}")
        with open(out / f"{m}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "label", *(c for c, _ in rows), "near_only_ensemble"])
            for i, t in enumerate(tests):
                w.writerow([Path(t).name, labels[i], *(f"{s[i]:.4f}" for _, s in rows), f"{ens[i]:.4f}"])

        # one free-text answer per condition, to see what the model "hears"
        for cond, r in (("zero-shot", []), ("few-shot", refs)):
            conv, audios = build(m, r, tests[-1])
            conv[-1]["content"][-1]["text"] = "Describe what you hear in the test recording and whether it sounds faulty."
            text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            inp = proc(text=text, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True).to(model.device)
            inp = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in inp.items()}
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=80, do_sample=False)
            print(f"  [{cond}, anomalous clip] {tok.decode(g[0, inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()}")


if __name__ == "__main__":
    main()
