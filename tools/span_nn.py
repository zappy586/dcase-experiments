"""SAM-Audio with nearest normal neighbours as the span prompt (neighbours from tools/nearest_normals.py).

Input per test clip (near mic, 48 kHz, 30 s): 4 s from the middle of each of its 5 nearest normal training clips,
then the full 10 s test clip. Conditions:
  text only   : description "<machine> running", no spans
  span + text : same description, spans "+" over the 5 normal parts
  span only   : empty description, spans "+" over the 5 normal parts
Score = share of energy in the residual over the TEST part minus the same share over the 5 NORMAL parts
(a swap of target/residual moves both, so it cancels). Higher = less like the normal neighbours.

Runs in the SAM-Audio environment:  envs/samaudio/Scripts/python tools/span_nn.py   (PYTHONPATH=tools)
Adds input / residual spectrograms to the comparison page (outputs/sam_compare/).
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import roc_auc_score

from sam_audio_demo import machine_phrase
from sam_compare import OUT, mel_db, save_png

SEG, K = 4.0, 5
CONDITIONS = ("text only", "span + text", "span only")


def near(path):
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(w[:, 0]), sr


def main():
    from sam_audio import SAMAudio, SAMAudioProcessor

    nn = json.loads(Path("outputs/span_nn/neighbours.json").read_text())
    model_id = "facebook/sam-audio-small-tv"
    model = SAMAudio.from_pretrained(model_id, visual_ranker=None, text_ranker=None, span_predictor=None).cuda().eval()
    proc = SAMAudioProcessor.from_pretrained(model_id)
    sr = proc.audio_sampling_rate
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text())
    done = {(i["clip"], i["mic"], i["variant"]) for i in manifest["items"]}
    results = []

    for m, entries in nn.items():
        for e in entries:
            parts = []
            for f in e["neighbours"][:K]:
                x, in_sr = near(Path("data") / m / "train" / f)
                mid = len(x) // 2
                parts.append(x[mid - int(SEG / 2 * in_sr): mid + int(SEG / 2 * in_sr)])
            t, in_sr = near(Path("data") / m / "test" / e["test"])
            parts.append(t[: 10 * in_sr])
            x = torchaudio.functional.resample(torch.cat(parts), in_sr, sr)
            n_split = int(K * SEG * sr)  # where the test part starts
            spans = [["+", i * SEG, (i + 1) * SEG] for i in range(K)]
            clip = f"{m}_{e['label']}_{e['test'].split('_')[5]}"
            row = {"machine": m, "label": e["label"], "test": e["test"]}
            for cond in CONDITIONS:
                text = "" if cond == "span only" else f"{machine_phrase(m)} running"
                batch = proc(audios=[x[None]], descriptions=[text], anchors=None if cond == "text only" else [spans]).to("cuda")
                torch.manual_seed(0)
                with torch.inference_mode():
                    res = model.separate(batch, predict_spans=False, reranking_candidates=1)
                tgt = res.target[0].float().cpu().reshape(-1)[: len(x)]
                rsd = res.residual[0].float().cpu().reshape(-1)[: len(x)]
                share = lambda a, b: float((rsd[a:b] ** 2).sum() / ((rsd[a:b] ** 2).sum() + (tgt[a:b] ** 2).sum() + 1e-12))
                r_nb, r_test = share(0, n_split), share(n_split, len(x))
                row[cond] = r_test - r_nb
                row[cond + " (test, normal parts)"] = (round(r_test, 3), round(r_nb, 3))
                # page: whole 30 s input and residual, so the normal parts and the test part can be compared by eye
                for variant, wav in ((f"span-NN input", x.numpy()), (f"span-NN residual ({cond})", rsd.numpy())):
                    if (clip, "near", variant) in done:
                        continue
                    stem = f"{clip}_near_{variant.replace(' ', '-').replace('(', '').replace(')', '').replace('+', 'plus')}"
                    sf.write(OUT / "audio" / f"{stem}.wav", wav, sr)
                    db = mel_db(wav, sr)
                    vmax = float(mel_db(x.numpy(), sr).max())
                    save_png(db, vmax, len(wav) / sr, OUT / "img" / f"{stem}.png")
                    manifest["items"].append({"clip": clip, "machine": m, "label": e["label"], "file": e["test"], "mic": "near",
                                              "variant": variant, "audio": f"audio/{stem}.wav", "img": f"img/{stem}.png",
                                              "note": "0-20 s: 5 nearest normal clips (4 s each); 20-30 s: test clip"
                                              + ("" if variant.endswith("input") else f" | residual share test {r_test:.0%}, normal parts {r_nb:.0%}")})
                    done.add((clip, "near", variant))
            results.append(row)
            print(f"{m:9s} {e['label']:7s} " + "  ".join(f"{c}: {row[c]:+.3f} {row[c + ' (test, normal parts)']}" for c in CONDITIONS), flush=True)
        mpath.write_text(json.dumps(manifest, indent=1))

    print("\nAUC (anomaly vs normal, 4 vs 4 per machine; higher score = more anomalous)")
    for m in nn:
        rows = [r for r in results if r["machine"] == m]
        lab = [r["label"] == "anomaly" for r in rows]
        print(f"  {m:9s} " + "  ".join(f"{c}: {roc_auc_score(lab, [r[c] for r in rows]):.2f}" for c in CONDITIONS))
    Path("outputs/span_nn/results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
