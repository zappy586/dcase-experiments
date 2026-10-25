"""Show what the (fine-tuned) SAM-Audio normality separator does, on the comparison page (outputs/sam_compare/).

For 4 synthetic examples (one per fake type, subtle) and 2 normal + 2 anomalous real test clips:
  input [5 s normal reference | 5 s test], separated target, separated residual  (pretrained and fine-tuned)
  plus the true fake on its own for the synthetic examples.
Also prints, per example, the residual's share of the test part's energy and (synthetic) the fake's true share.

Runs in the SAM-Audio environment (PYTHONPATH=tools):
  envs/samaudio/Scripts/python tools/sam_normality_viz.py --machine fan
"""
import argparse
import json
import random
from pathlib import Path

import soundfile as sf
import torch
from torchdiffeq import odeint

from sam_audio_demo import machine_phrase
from sam_compare import OUT, mel_db, save_png
from sam_normality_ft import EVAL_ODE, MODEL_ID, SEG, SR_IN, Runner, add_lora, crop, fake_anomaly, load_near

KINDS = ["gain", "tone", "wobble", "burst"]


@torch.inference_mode()
def separate(run, ref, test, seed=0):
    b = run.batch([torch.cat([ref, test])])
    args = run.model._get_forward_args(b)
    g = torch.Generator(device=run.dev).manual_seed(seed)
    noise = torch.randn(args["audio_features"].shape, device=run.dev, generator=g)
    field = lambda t, x: run.model(noisy_audio=x, time=t.expand(x.size(0)), **args)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gen = odeint(field, noise, torch.tensor([0.0, 1.0], device=run.dev), **EVAL_ODE)[-1].float()
    _, T, C2 = gen.shape
    lat = gen.transpose(1, 2).reshape(2, C2 // 2, T)
    wav = torch.cat([run.model.audio_codec.decode(lat[i : i + 1]).float() for i in range(2)]).reshape(2, -1)
    return wav[0].cpu(), wav[1].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="fan")
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()
    from sam_audio import SAMAudio, SAMAudioProcessor

    dev = torch.device("cuda")
    model = SAMAudio.from_pretrained(MODEL_ID, visual_ranker=None, text_ranker=None, span_predictor=None).eval()
    model.vision_encoder.model.to("cpu")
    for name, mod in model.named_children():
        if name != "vision_encoder":
            mod.to(dev)
    model.transformer.to(torch.bfloat16)
    add_lora(model.transformer, args.rank)
    proc = SAMAudioProcessor.from_pretrained(MODEL_ID)
    run = Runner(model, proc, f"{machine_phrase(args.machine)} running", dev)
    sr = run.sr
    adapter = torch.load(Path(f"outputs/sam_normality/{args.machine}/adapter.pt"), map_location=dev)
    base_anchor = {k: v.clone() for k, v in model.embed_anchors.state_dict().items()}

    g = random.Random(1)
    train = sorted(Path(f"data/{args.machine}/train").glob("*.wav"))
    examples = []  # (clip id, label, ref, test, true fake or None)
    for k, kind in enumerate(KINDS):
        ref, x = crop(load_near(g.choice(train)).to(dev), g), crop(load_near(g.choice(train)).to(dev), g)
        n = fake_anomaly(x, g, 0.0, kind=k)
        examples.append((f"samft-synthetic-{kind}", "anomaly", ref, x + n, n))
    for lab in ("normal", "anomaly"):
        for f in g.sample(sorted(Path(f"data/{args.machine}/test").glob(f"*source_test_{lab}*.wav")), 2):
            ref = crop(load_near(g.choice(train)).to(dev), g)
            examples.append((f"samft-real-{lab}-{f.stem.split('_')[5]}", lab, ref, load_near(f).to(dev)[: int(SEG * SR_IN)], None))

    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["items"] = [i for i in manifest["items"] if not i["clip"].startswith("samft-")]
    up = lambda w: run.up(w).cpu()
    split = int(SEG * sr)

    def add(clip, lab, variant, wav, note, vmax):
        stem = f"{clip}_{variant.replace(' ', '-')}"
        sf.write(OUT / "audio" / f"{stem}.wav", wav.numpy(), sr)
        save_png(mel_db(wav.numpy(), sr), vmax, len(wav) / sr, OUT / "img" / f"{stem}.png")
        manifest["items"].append({"clip": clip, "machine": f"{args.machine} SAM fine-tune", "label": lab, "file": clip,
                                  "mic": "near", "variant": variant, "audio": f"audio/{stem}.wav",
                                  "img": f"img/{stem}.png", "note": note})

    results = {}
    for state in ("pretrained", "fine-tuned"):
        for mod in model.transformer.modules():  # LoRA off (b = 0) for pretrained, loaded weights for fine-tuned
            if hasattr(mod, "b") and isinstance(getattr(mod, "b"), torch.nn.Parameter):
                mod.b.data.zero_()
        model.embed_anchors.load_state_dict(base_anchor)
        if state == "fine-tuned":
            missing = model.load_state_dict(adapter, strict=False)
            print("loaded adapter tensors:", len(adapter), "| unexpected:", len(missing.unexpected_keys))
        for clip, lab, ref, test, n in examples:
            mix = up(torch.cat([ref, test]))
            vmax = float(mel_db(mix.numpy(), sr).max())
            tg, rs = separate(run, ref, test)
            share = float((rs[split:] ** 2).sum() / ((rs[split:] ** 2).sum() + (tg[split:] ** 2).sum() + 1e-12))
            results.setdefault(clip, {})[state] = share
            if state == "pretrained":
                add(clip, lab, "input (ref | test)", mix, "0-5 s: normal reference ('+' span); 5-10 s: test", vmax)
                if n is not None:
                    true = float((n**2).sum() / ((test**2).sum() + 1e-12))
                    results[clip]["true fake share"] = true
                    add(clip, lab, "true fake (alone)", up(torch.cat([torch.zeros_like(ref), n])),
                        f"the added fake on its own; {true:.1%} of the test part's energy", vmax)
            add(clip, lab, f"{state} target", tg, f"{state}", vmax)
            add(clip, lab, f"{state} residual", rs, f"{state}: residual share of test part {share:.1%}", vmax)
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"\n{'example':34s}{'true fake share':>16s}{'pretrained':>12s}{'fine-tuned':>12s}   (residual share of the test part)")
    for clip, r in results.items():
        t = f"{r['true fake share']:.2%}" if "true fake share" in r else "-"
        print(f"{clip:34s}{t:>16s}{r['pretrained']:>12.1%}{r['fine-tuned']:>12.1%}")


if __name__ == "__main__":
    main()
