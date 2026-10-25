"""EfficientAD adapted to machine sound (student-teacher distillation + autoencoder branch).

Input (per clip, on GPU): 4 x 256 x T  =  [near log-power, far log-power, cos IPD, sin IPD]
  1024-pt STFT at 16 kHz, hop 256 (16 ms), DC dropped, adjacent bin pairs averaged -> 256 bins of 31 Hz.
  Log-power channels are standardised with the machine's own training statistics.

Networks (same for every machine):
  PDN      narrow in frequency (receptive field 9 bins), long in time (~1 s), time downsampled x8, frequency kept.
  Teacher  PDN pretrained to classify machine|domain|attribute on the OTHER machines' normal clips, then frozen.
  Student  PDN with 2D outputs: [imitate teacher | imitate autoencoder].
  AE       global bottleneck: 64 channels x 8 frequency regions, averaged over time; decoded to the teacher map.

Loss (normal clips of the machine; penalty clips from the other machines):
  L = hard-quantile( (T - S_st)^2 ) + mean(S_st(x_other)^2) + mean((T - A)^2) + mean((A - S_ae)^2)

Score: maps M_st = mean_c (T - S_st)^2, M_ae = mean_c (A - S_ae)^2, each rescaled with quantiles (q90, q99.5) of
held-out normal clips, averaged; steady = max_f mean_t M, transient = max of 3x3-smoothed M; both z-normalised on the
held-out normal clips and summed.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

SR, CLIP = 16000, 160000  # 10 s
NFFT, HOP = 1024, 256
D = 128  # teacher feature channels


# ---- waveform cache -----------------------------------------------------------------------------------------
def _ground_truth(machine):
    """(label, domain) dicts from the official evaluator repo, for evaluation-set test files; None if not available"""
    root = Path(__file__).resolve().parent.parent / "third_party" / "dcase2026_task2_evaluator"
    fl, fd = root / "ground_truth_data" / f"ground_truth_{machine}_section_00_test.csv", root / "ground_truth_domain" / f"ground_truth_{machine}_section_00_test.csv"
    if not fl.exists():
        return None
    rd = lambda f: {a: int(b) for a, b in (l.strip().split(",") for l in f.read_text().splitlines() if l.strip())}
    return rd(fl), rd(fd)


def build_cache(data_root, cache_dir, machine, split):
    """int16 (N, 2, 160000) memmap + meta json, first 10 s of every clip."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arr_path, meta_path = cache_dir / f"wav_{machine}_{split}.npy", cache_dir / f"wav_{machine}_{split}.json"
    if arr_path.exists() and meta_path.exists():
        return
    files = sorted((Path(data_root) / machine / split).glob("*.wav"))
    arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.int16, shape=(len(files), 2, CLIP))
    meta = {"names": [], "label": [], "domain": [], "attr": []}
    gt = _ground_truth(machine) if split == "test" else None
    for i, f in enumerate(files):
        w, _ = sf.read(f, dtype="int16", always_2d=True)
        w = w[:CLIP].T
        arr[i, :, : w.shape[1]] = w
        p = f.stem.split("_")
        meta["names"].append(f.name)
        if len(p) < 5:  # evaluation set: 'section_00_0000.wav', labels from the official ground truth
            meta["label"].append(gt[0][f.name])
            meta["domain"].append(gt[1][f.name])
            meta["attr"].append("")
            continue
        meta["label"].append(0 if p[4] == "normal" else 1)
        meta["domain"].append(0 if p[2] == "source" else 1)
        meta["attr"].append("_".join(p[6:]))
    arr.flush()
    meta_path.write_text(json.dumps(meta))


class WavStore:
    def __init__(self, cache_dir, machine, split):
        self.wav = np.load(Path(cache_dir) / f"wav_{machine}_{split}.npy", mmap_mode="r")
        self.meta = json.loads((Path(cache_dir) / f"wav_{machine}_{split}.json").read_text())
        self.machine = machine

    def __len__(self):
        return len(self.wav)

    def get(self, idx, start=0, length=CLIP):
        return torch.from_numpy(np.stack([self.wav[i, :, start : start + length] for i in idx]).astype(np.float32) / 32768)


# ---- features -----------------------------------------------------------------------------------------------
_WIN = {}


def features(w, stats=None):
    """w (B, 2, L) float on GPU -> (B, 4, 256, T). stats = (mean, std) of the two log-power channels,
    each of shape (2,) for one machine or (B, 2) for a batch mixing machines."""
    dev = w.device
    if dev not in _WIN:
        _WIN[dev] = torch.hann_window(NFFT, device=dev)
    B = w.shape[0]
    S = torch.stft(w.reshape(2 * B, -1), NFFT, HOP, window=_WIN[dev], return_complex=True)[:, 1:]  # drop DC
    S = S.reshape(B, 2, 256, 2, -1)  # pair adjacent bins
    pw = (S.abs() ** 2).mean(3)  # (B, 2, 256, T)
    X = (S[:, 0] * S[:, 1].conj()).mean(2)  # (B, 256, T) cross-spectrum per pooled band
    lp = 10 * torch.log10(pw + 1e-10)
    if stats is not None:
        lp = (lp - stats[0][..., :, None, None]) / stats[1][..., :, None, None]
    ang = torch.angle(X)
    return torch.cat([lp, torch.cos(ang)[:, None], torch.sin(ang)[:, None]], 1)


@torch.no_grad()
def logpower_stats(store, device, n=100, seed=0):
    idx = random.Random(seed).sample(range(len(store)), min(n, len(store)))
    lp = features(store.get(idx).to(device))[:, :2]
    return lp.mean(dim=(0, 2, 3)), lp.std(dim=(0, 2, 3)) + 1e-6


# ---- networks -----------------------------------------------------------------------------------------------
class PDN(nn.Module):
    """Patch description network for spectrograms: receptive field 9 bins x ~67 frames (~1.1 s), time / 8."""

    def __init__(self, out_ch=D, in_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, (3, 7), padding=(1, 3)), nn.ReLU(), nn.AvgPool2d((1, 2)),
            nn.Conv2d(64, 128, (3, 7), padding=(1, 3)), nn.ReLU(), nn.AvgPool2d((1, 2)),
            nn.Conv2d(128, 128, (3, 5), padding=(1, 2)), nn.ReLU(), nn.AvgPool2d((1, 2)),
            nn.Conv2d(128, out_ch, (3, 5), padding=(1, 2)),
        )

    def forward(self, x):
        return self.net(x)


class Teacher(nn.Module):
    """PDN + classification head (head only used for pretraining)."""

    def __init__(self, n_classes):
        super().__init__()
        self.pdn = PDN(D)
        self.head = nn.Sequential(nn.Linear(2 * D, 256), nn.ReLU(), nn.Linear(256, n_classes))

    def forward(self, x):
        f = F.relu(self.pdn(x)).mean(3)  # (B, D, F)
        return self.head(torch.cat([f.mean(2), f.amax(2)], 1))


class AE(nn.Module):
    """Global bottleneck: 64 channels x 8 frequency regions, averaged over time -> teacher-shaped map."""

    def __init__(self, in_ch=4, bott=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(), nn.Conv2d(64, bott, 4, 2, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(bott, 64, (3, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(64, D, (3, 1), padding=(1, 0)),
        )

    def forward(self, x, out_hw):
        z = self.enc(x).mean(3, keepdim=True)  # average over time
        z = F.adaptive_avg_pool2d(z, (8, 1))  # 8 frequency regions
        h = F.interpolate(z, size=(out_hw[0], 1), mode="bilinear", align_corners=False)
        return self.dec(h).expand(-1, -1, -1, out_hw[1])


# ---- training -----------------------------------------------------------------------------------------------
def random_crops(store, n, length, rng, weights=None):
    idx = rng.choices(range(len(store)), weights=weights, k=n)
    out = []
    for i in idx:
        s = rng.randrange(0, CLIP - length + 1)
        out.append(store.get([i], s, length)[0])
    return torch.stack(out)


def train_teacher(stores, stats, device, steps, log, seed=0, batch=32, crop_s=4.0):
    """stores: list of WavStore (training split of the OTHER machines); classes = machine|domain|attribute."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    labels, items = {}, []
    for k, st in enumerate(stores):
        for i in range(len(st)):
            key = f"{st.machine}|{st.meta['domain'][i]}|{st.meta['attr'][i]}"
            labels.setdefault(key, len(labels))
            items.append((k, i, labels[key]))
    model = Teacher(len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    length = int(crop_s * SR)
    log(f"  teacher: {len(labels)} classes from {len(stores)} machines, {len(items)} clips")
    for step in range(1, steps + 1):
        picks = rng.sample(items, batch)
        w, y, mk = [], [], []
        for k, i, lab in picks:
            s = rng.randrange(0, CLIP - length + 1)
            w.append(stores[k].get([i], s, length)[0])
            y.append(lab)
            mk.append(k)
        w = torch.stack(w).to(device)
        x = features(w, (torch.stack([stats[k][0] for k in mk]), torch.stack([stats[k][1] for k in mk])))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), torch.tensor(y, device=device), label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps:
            acc = (logits.argmax(1).cpu() == torch.tensor(y)).float().mean().item()
            log(f"  teacher step {step:5d}/{steps}  loss {loss.item():.3f}  batch acc {acc:.2f}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.pdn


class Detector:
    def __init__(self, teacher, device):
        self.t = teacher
        self.s = PDN(2 * D).to(device)
        self.ae = AE().to(device)
        self.dev = device

    @torch.no_grad()
    def set_teacher_norm(self, x):
        f = self.t(x)
        self.t_mu, self.t_sd = f.mean(dim=(0, 2, 3), keepdim=True), f.std(dim=(0, 2, 3), keepdim=True) + 1e-6

    def teach(self, x):
        with torch.no_grad():
            return (self.t(x) - self.t_mu) / self.t_sd

    def losses(self, x, x_pen, q=0.99):
        t = self.teach(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = self.s(x).float()
            a = self.ae(x, t.shape[2:]).float()
            s_pen = self.s(x_pen)[:, :D].float()
        s_st, s_ae = s[:, :D], s[:, D:]
        d = (t - s_st) ** 2
        flat = d.flatten()
        sample = flat[torch.randint(len(flat), (min(len(flat), 1_000_000),), device=flat.device)]
        thr = torch.quantile(sample, q)
        l_hard = d[d >= thr].mean()
        l_pen = (s_pen**2).mean()
        l_ae = ((t - a) ** 2).mean()
        l_stae = ((a.detach() - s_ae) ** 2).mean()
        return l_hard, l_pen, l_ae, l_stae

    @torch.no_grad()
    def maps(self, x):
        t = self.teach(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = self.s(x).float()
            a = self.ae(x, t.shape[2:]).float()
        return ((t - s[:, :D]) ** 2).mean(1), ((a - s[:, D:]) ** 2).mean(1)  # (B, F, T') each


def clip_scores(m_st, m_ae, norm):
    """Combined normalised map -> steady / transient clip scores (raw, before z-normalisation)."""
    ms = 0.1 * (m_st - norm["st"][0]) / (norm["st"][1] - norm["st"][0])
    ma = 0.1 * (m_ae - norm["ae"][0]) / (norm["ae"][1] - norm["ae"][0])
    m = 0.5 * ms + 0.5 * ma
    steady = m.mean(2).amax(1)
    transient = F.avg_pool2d(m[:, None], 3, 1, 1)[:, 0].flatten(1).amax(1)
    parts = {"st steady": ms.mean(2).amax(1), "ae steady": ma.mean(2).amax(1)}
    return steady, transient, parts


def quantiles(v, lo=0.9, hi=0.995):
    flat = v.flatten()
    sample = flat[torch.randperm(len(flat), device=flat.device)[: min(len(flat), 2_000_000)]]
    return torch.quantile(sample, lo).item(), torch.quantile(sample, hi).item()
