"""Variety check for synthetic set v1 (outputs/audiogen/synth_v1): are the 250 normal / 250 faulty generated clips as
varied as 250 real ToyCar training normals (near-minus-far, machine gain, full 10 s)? Two spaces:
  BEATs iter3 (final layer, average over tokens, 768 numbers per clip) and the time-averaged 128-band log-mel spectrum.
Per set: mean pairwise cosine distance (spread), nearest-neighbour cosine distance within the set (median, min),
near-duplicates (NN distance below the 1st percentile of real NN distances), participation ratio of the PCA
eigenvalues (how many dimensions the set really uses), and generated->nearest-real distance.
Usage (main env): uv run python tools/synth_variety.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from BEATs import BEATs, BEATsConfig  # noqa: E402

SYN = ROOT / "outputs" / "audiogen" / "synth_v1"


@torch.no_grad()
def beats_embed(model, wavs, dev):
    out = []
    for i in range(0, len(wavs), 8):
        x = torch.as_tensor(np.stack(wavs[i : i + 8]), dtype=torch.float32, device=dev)
        f = model.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0]
        out.append(f.mean(1).cpu())
    return torch.cat(out).numpy()


def mel_embed(wavs):
    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=160, n_mels=128, f_max=8000)
    return np.stack([np.log(mel(torch.from_numpy(w)).mean(-1).numpy() + 1e-8) for w in wavs])


def cos_dist(X, Y=None):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Yn = Xn if Y is None else Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return 1 - Xn @ Yn.T


def stats(X, real_nn_p1, R):
    D = cos_dist(X)
    n = len(X)
    iu = np.triu_indices(n, 1)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    ev = np.linalg.eigvalsh(np.cov((X - X.mean(0)).T))
    pr = ev.sum() ** 2 / (ev ** 2).sum()
    to_real = cos_dist(X, R).min(1) if R is not None else None
    return dict(spread=D[iu].mean(), nn_med=np.median(nn), nn_min=nn.min(), dups=int((nn < real_nn_p1).sum()),
                pr=pr, to_real=None if to_real is None else np.median(to_real))


def main():
    dev = torch.device("cuda")
    gen = {c: [sf.read(f, dtype="float32")[0][:160000] for f in sorted((SYN / c).glob("*.wav"))] for c in ("normal", "faulty")}
    D = A.Data("ToyCar", dev)
    ids = np.random.default_rng(0).choice(len(D.meta["train"]["label"]) if "label" in D.meta["train"] else D.wav["train"].shape[0],
                                          250, replace=False)
    real = []
    for i in range(0, 250, 25):
        real += list(D.audio("train", [int(j) for j in ids[i : i + 25]])[:, 0, :160000].cpu().numpy())

    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    model = BEATs(BEATsConfig(ck["cfg"]))
    model.load_state_dict(ck["model"])
    model = model.to(dev).eval()

    for space, fn in (("BEATs", lambda w: beats_embed(model, w, dev)), ("log-mel", mel_embed)):
        R = fn(real)
        Dr = cos_dist(R)
        np.fill_diagonal(Dr, np.inf)
        p1 = np.percentile(Dr.min(1), 1)
        print(f"\n{space}   (near-duplicate threshold = 1st pct of real NN distance = {p1:.2e})")
        print(f"{'set':14s} {'spread':>9s} {'NN med':>9s} {'NN min':>9s} {'dups':>5s} {'PR':>6s} {'->real':>9s}")
        for name, W in (("real normal", None), ("gen normal", gen["normal"]), ("gen faulty", gen["faulty"])):
            X = R if W is None else fn(W)
            s = stats(X, p1, None if W is None else R)
            tr = "" if s["to_real"] is None else f"{s['to_real']:9.2e}"
            print(f"{name:14s} {s['spread']:9.2e} {s['nn_med']:9.2e} {s['nn_min']:9.2e} {s['dups']:5d} {s['pr']:6.1f} {tr}")


if __name__ == "__main__":
    main()
