"""Generate synthetic ToyCar normals with AudioGen-medium + our LoRA r16 'full10' + <toycar> token (for Colab / any GPU).
Same recipe as tools/gen_synth_v1.py / gen_synth_extra.py: prompt 'the sound of <toycar>', cfg 3, top-k 250, 10 s at
16 kHz, from scratch. Seeds --seed0 + first index of each batch (v1 used 20000+, extra 30000+; default 50000+).
Adapter files (lora_r16_full10.pt, toycar_token_lora_r16_full10.pt) come from the Hugging Face repo --hf_repo, or from
--weights_dir if given. Output: <out>/normal/0000.wav ... (--n clips) and <out>/faulty/0000.wav ... (--faulty clips,
cycling through the three fault prompts of tools/audiogen_ti.py, as in gen_synth_v1.py) + <out>/meta.json.
Resumable: existing clips are skipped. Larger --batch is much faster per clip (generation is step-by-step).
Needs audiocraft 1.3.0 installed WITHOUT its pinned torch (see the Colab cell in README.md).
Usage: python tools/colab_gen_synth.py --out /content/drive/MyDrive/dcase_gens --n 1000 --faulty 1000 --batch 64
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/content/drive/MyDrive/dcase_gens")
ap.add_argument("--n", type=int, default=1000, help="normal clips")
ap.add_argument("--faulty", type=int, default=0, help="faulty clips (fault prompts used in turn)")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--seed0", type=int, default=50000)
ap.add_argument("--hf_repo", default="ZappY-AI/audiogen-toycar-lora")
ap.add_argument("--weights_dir", default=None, help="folder with the two .pt files (instead of Hugging Face)")
args = ap.parse_args()
PROMPT = "the sound of <toycar>"
FAULT_PROMPTS = ["a broken <toycar> with subtle deviations from a working <toycar>",  # = audiogen_ti.ANOMALY_PROMPTS
                 "a <toycar> with a faint intermittent rattle",
                 "a <toycar> with a slightly worn, grinding gear"]
FILES = ("lora_r16_full10.pt", "toycar_token_lora_r16_full10.pt")

if args.weights_dir:
    W = {f: Path(args.weights_dir) / f for f in FILES}
else:
    from huggingface_hub import hf_hub_download
    W = {f: Path(hf_hub_download(args.hf_repo, f)) for f in FILES}
for f, p in W.items():
    assert p.exists(), f"missing {p}"

# audiocraft imports xformers at module load but generates with torch's own attention by default; a stub avoids
# installing an xformers build that would replace the runtime's torch
if "xformers" not in sys.modules:
    try:
        import xformers.ops  # noqa: F401
    except Exception:
        # with audiocraft's default 'torch' attention backend, LowerTriangularMask() is only used as a "causal" flag
        # and attention runs in torch.nn.functional.scaled_dot_product_attention; the real xformers kernel is never called
        xf = types.ModuleType("xformers"); xf.ops = types.ModuleType("xformers.ops")

        class LowerTriangularMask:
            pass

        def memory_efficient_attention(*a, **k):
            raise RuntimeError("xformers stub: set audiocraft's attention backend to 'torch'")
        import torch as _torch
        xf.ops.LowerTriangularMask, xf.ops.memory_efficient_attention = LowerTriangularMask, memory_efficient_attention
        xf.ops.unbind = _torch.unbind                         # xformers.ops.unbind == torch.unbind (used on packed qkv)
        sys.modules["xformers"], sys.modules["xformers.ops"] = xf, xf.ops

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from audiocraft.models import AudioGen  # noqa: E402
from peft import LoraConfig, inject_adapter_in_model  # noqa: E402

dev = "cuda"
m = AudioGen.get_pretrained("facebook/audiogen-medium", device=dev)
cond = m.lm.condition_provider.conditioners["description"]
tok, t5 = cond.t5_tokenizer, cond.__dict__["t5"]
if "<toycar>" not in tok.get_vocab():
    tok.add_tokens(["<toycar>"]); t5.resize_token_embeddings(len(tok))
with torch.no_grad():
    t5.get_input_embeddings().weight[tok.convert_tokens_to_ids("<toycar>")] = torch.load(
        W["toycar_token_lora_r16_full10.pt"], map_location=dev).to(dev)
names = [n for n, mod in m.lm.transformer.named_modules()
         if isinstance(mod, torch.nn.Linear) and n.split(".")[-1] in ("out_proj", "linear1", "linear2")]
inject_adapter_in_model(LoraConfig(r=16, lora_alpha=16, target_modules=names), m.lm.transformer)
m.lm.to(dev)
for n, p in m.lm.transformer.named_parameters():
    if "lora_" in n:
        p.data = p.data.float()
sd = torch.load(W["lora_r16_full10.pt"], map_location=dev)
res = m.lm.transformer.load_state_dict(sd, strict=False)
lora_missing = [k for k in res.missing_keys if "lora_" in k]
assert not res.unexpected_keys and not lora_missing, (res.unexpected_keys[:3], lora_missing[:3])
print(f"loaded <toycar> token + {len(sd)} LoRA tensors")
m.set_generation_params(duration=10.0, cfg_coef=3.0, top_k=250)

ROOT_OUT = Path(args.out)
mpath = ROOT_OUT / "meta.json"
meta = json.loads(mpath.read_text()) if mpath.exists() else {}
jobs = [("normal", i, PROMPT) for i in range(args.n)] +        [("faulty", i, FAULT_PROMPTS[i % 3]) for i in range(args.faulty)]
for c in ("normal", "faulty"):
    (ROOT_OUT / c).mkdir(parents=True, exist_ok=True)
todo = [j for j in jobs if not (ROOT_OUT / j[0] / f"{j[1]:04d}.wav").exists()]
print(f"{len(todo)} clips to generate (of {args.n} normal + {args.faulty} faulty) -> {ROOT_OUT}")
t0 = time.time()
for b in range(0, len(todo), args.batch):
    chunk = todo[b : b + args.batch]
    seed = args.seed0 + (100000 if chunk[0][0] == "faulty" else 0) + chunk[0][1]
    torch.manual_seed(seed)
    with torch.no_grad():
        w = m.generate([p for _, _, p in chunk]).cpu().numpy()[:, 0, :160000]
    for (c, idx, p), x in zip(chunk, w):
        sf.write(ROOT_OUT / c / f"{idx:04d}.wav", x, 16000)
        meta[f"{c}/{idx:04d}.wav"] = {"prompt": p.replace("<toycar>", "[toycar]"), "batch_seed": seed}
    mpath.write_text(json.dumps(meta, indent=1))
    done = b + len(chunk); el = time.time() - t0
    print(f"{done}/{len(todo)}  {el / done:.1f} s per clip, ~{el / done * (len(todo) - done) / 60:.0f} min left", flush=True)
print("done:", ROOT_OUT)
