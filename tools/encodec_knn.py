"""kNN on EnCodec embeddings of the near-minus-far (near-only residual) audio, one machine.

Audio: residual r = near - H(f) * far (least-squares H per frequency, STFT 1024/512, as in ae_noiseaware.Spec), 16 -> 24 kHz.
Embedding: EnCodec 24 kHz encoder output (continuous latent before quantisation, 128 x 75 frames/s) -> mean and std
over time -> 256 numbers per clip, standardised with the 1000 training clips.
Score: distance to the nearest training clips (all 1000 normal training clips), k = 1 and k = 5, cosine and Euclidean.
All four settings shown; nothing picked on test labels.
Usage: uv run python tools/encodec_knn.py --machine ToyCar
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dcase_ae.evaluation import compute_metrics  # noqa: E402


def residual(w):
    win = torch.hann_window(1024, device=w.device)
    S = torch.stft(w.reshape(-1, w.shape[-1]), 1024, 512, window=win, return_complex=True).reshape(len(w), 2, 513, -1)
    N, F = S[:, 0], S[:, 1]
    H = (N * F.conj()).mean(-1, keepdim=True) / (F.abs().pow(2).mean(-1, keepdim=True) + 1e-12)
    return torch.istft(N - H * F, 1024, 512, window=win, length=w.shape[-1])


@torch.no_grad()
def embed(model, machine, split, dev):
    f = ROOT / "outputs" / "probe_cache" / f"encodec_resid_{machine}_{split}.npy"
    if f.exists():
        return np.load(f)
    wav = np.load(ROOT / "cache" / f"wav_{machine}_{split}.npy", mmap_mode="r")
    E = []
    for i in range(0, len(wav), 16):
        w = torch.from_numpy(wav[i : i + 16].astype(np.float32) / 32768).to(dev)
        r = torchaudio.functional.resample(residual(w), 16000, 24000)
        r = r / (r.pow(2).mean(-1, keepdim=True).sqrt().mean() + 1e-9) * 0.1  # one gain for the batch, keeps level differences
        z = model.encoder(r[:, None])  # (B, 128, frames)
        E.append(torch.cat([z.mean(-1), z.std(-1)], 1).cpu())
    E = torch.cat(E).numpy()
    f.parent.mkdir(exist_ok=True)
    np.save(f, E)
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="ToyCar")
    args = ap.parse_args()
    from encodec import EncodecModel
    dev = torch.device("cuda")
    model = EncodecModel.encodec_model_24khz().to(dev).eval()
    Etr, Ete = embed(model, args.machine, "train", dev), embed(model, args.machine, "test", dev)
    mu, sd = Etr.mean(0), Etr.std(0) + 1e-6
    Etr, Ete = (Etr - mu) / sd, (Ete - mu) / sd
    meta = json.loads((ROOT / "cache" / f"wav_{args.machine}_test.json").read_text())
    lab, dom = np.array(meta["label"]), np.array(meta["domain"])
    Tr, Te = torch.from_numpy(Etr), torch.from_numpy(Ete)
    D = {"euclid": torch.cdist(Te, Tr),
         "cosine": 1 - torch.nn.functional.normalize(Te, dim=1) @ torch.nn.functional.normalize(Tr, dim=1).T}
    print(f"{args.machine}: EnCodec near-minus-far embeddings, train {Etr.shape}, test {Ete.shape}")
    for name, d in D.items():
        for k in (1, 5):
            s = d.topk(k, dim=1, largest=False).values.mean(1).numpy()
            r = compute_metrics(s, lab, dom)
            print(f"  kNN k={k} {name:6s}: AUC {r['auc_all']:.3f} src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} "
                  f"pAUC {r['pauc']:.3f} official {r['official']:.3f}")


if __name__ == "__main__":
    main()
