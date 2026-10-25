"""Qualitative plots: input / reconstruction / anomaly heatmap per sample."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABEL_NAMES = ("normal", "anomaly")
DOMAIN_NAMES = ("source", "target")


def select_indices(labels: np.ndarray, domains: np.ndarray, n: int, seed: int = 0) -> list[int]:
    """Fixed, stratified pick over (label, domain) groups, round-robin so every group is covered."""
    rng = np.random.RandomState(seed)
    groups = [rng.permutation(np.where((labels == l) & (domains == d))[0]).tolist() for l in (0, 1) for d in (0, 1)]
    picked = []
    while len(picked) < n and any(groups):
        for g in groups:
            if g and len(picked) < n:
                picked.append(g.pop())
    return sorted(picked, key=lambda i: (labels[i], domains[i], i))


def _draw(axes, x, r, e, title, vmax_err, hop_s, n_mels):
    kw = dict(origin="lower", aspect="auto", extent=[0, x.shape[-1] * hop_s, 0, n_mels])
    axes[0].imshow(x, cmap="magma", vmin=-3, vmax=3, **kw)
    axes[1].imshow(r, cmap="magma", vmin=-3, vmax=3, **kw)
    im = axes[2].imshow(e, cmap="inferno", vmin=0, vmax=vmax_err, **kw)
    axes[0].set_ylabel("mel bin")
    axes[0].set_title(title, fontsize=9, loc="left")
    axes[1].set_title("reconstruction", fontsize=9)
    axes[2].set_title("error heatmap", fontsize=9)
    for a in axes:
        a.set_xlabel("time (s)")
    return im


def save_step_plots(out_dir, step, idxs, ds, recon, err, scores, audio_cfg, err_vmax=None):
    """Writes <out_dir>/step_XXXXXX/{grid.png, sample_k_<label>_<domain>.png}."""
    step_dir = Path(out_dir) / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    hop_s = audio_cfg.hop_length / audio_cfg.sample_rate
    xs = ds.specs[idxs].mean(1).numpy()
    rs = recon[idxs].mean(1).numpy()
    es = err[idxs].mean(1).numpy()
    vmax_err = err_vmax or float(np.percentile(es, 99.5))

    def title(k):
        i = idxs[k]
        return f"#{i} {LABEL_NAMES[int(ds.labels[i])]} / {DOMAIN_NAMES[int(ds.domains[i])]}  score={scores[i]:.3f}"

    n = len(idxs)
    fig, axs = plt.subplots(n, 3, figsize=(15, 2.3 * n), squeeze=False, constrained_layout=True)
    for k in range(n):
        im = _draw(axs[k], xs[k], rs[k], es[k], title(k), vmax_err, hop_s, audio_cfg.n_mels)
    fig.colorbar(im, ax=axs[:, 2], shrink=0.6)
    fig.suptitle(f"step {step}", fontsize=12)
    fig.savefig(step_dir / "grid.png", dpi=80)
    plt.close(fig)

    for k in range(n):
        i = idxs[k]
        fig, ax = plt.subplots(1, 3, figsize=(15, 3), constrained_layout=True)
        im = _draw(ax, xs[k], rs[k], es[k], title(k), vmax_err, hop_s, audio_cfg.n_mels)
        fig.colorbar(im, ax=ax[2])
        fig.savefig(step_dir / f"sample_{k}_{LABEL_NAMES[int(ds.labels[i])]}_{DOMAIN_NAMES[int(ds.domains[i])]}.png", dpi=80)
        plt.close(fig)
    return step_dir
