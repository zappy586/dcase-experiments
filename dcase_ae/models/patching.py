"""Patchify helpers and 2D sin-cos positional embeddings."""
import numpy as np
import torch


def patchify(x: torch.Tensor, patch: tuple[int, int]) -> torch.Tensor:
    """(B, C, F, T) -> (B, Fp*Tp, C*pf*pt); tokens ordered freq-major, time-minor."""
    B, C, F, T = x.shape
    pf, pt = patch
    x = x.reshape(B, C, F // pf, pf, T // pt, pt).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (F // pf) * (T // pt), C * pf * pt)


def unpatchify(p: torch.Tensor, patch: tuple[int, int], grid: tuple[int, int], channels: int) -> torch.Tensor:
    """Inverse of `patchify`."""
    B = p.shape[0]
    pf, pt = patch
    Fp, Tp = grid
    p = p.reshape(B, Fp, Tp, channels, pf, pt).permute(0, 3, 1, 4, 2, 5)
    return p.reshape(B, channels, Fp * pf, Tp * pt)


def _sincos_1d(dim: int, pos: np.ndarray) -> np.ndarray:
    omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_2d(dim: int, grid: tuple[int, int]) -> torch.Tensor:
    """(Fp*Tp, dim) fixed embedding, freq-major to match `patchify`."""
    Fp, Tp = grid
    f, t = np.meshgrid(np.arange(Fp), np.arange(Tp), indexing="ij")
    emb = np.concatenate([_sincos_1d(dim // 2, f), _sincos_1d(dim // 2, t)], axis=1)
    return torch.from_numpy(emb).float()
