"""Frequency-conditioned normalizing flow (CFLOW-style) on teacher features: a learned density of normal sound.

Each feature vector v (D dims) at frequency row f is mapped invertibly to z with K affine-coupling blocks whose
sub-networks see a sinusoidal encoding of f. Under a standard-normal z:
    -log p(v | f) = 0.5 * ||z||^2 - log|det dz/dv|  (+ const)
Trained on normal clips only (maximum likelihood); the anomaly score is this negative log-likelihood, averaged over
time per frequency row, then aggregated over rows (max / top-10% / mean).
"""
import math

import torch
import torch.nn as nn


def freq_encoding(f_idx, n_freq, dim=32):
    """Sinusoidal encoding of frequency-row indices (N,) -> (N, dim)."""
    k = torch.arange(dim // 2, device=f_idx.device).float()
    ang = f_idx.float()[:, None] / (n_freq ** (2 * k / dim))[None]  # transformer-style, wavelengths 2*pi .. ~n_freq
    return torch.cat([torch.sin(ang), torch.cos(ang)], 1)


class Coupling(nn.Module):
    def __init__(self, dim, cond_dim, hidden=256, clamp=1.9, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("perm", torch.randperm(dim, generator=g))
        self.d1 = dim // 2
        d2 = dim - self.d1
        self.net = nn.Sequential(nn.Linear(self.d1 + cond_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * d2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.clamp = clamp

    def forward(self, x, c):
        x = x[:, self.perm]
        x1, x2 = x[:, : self.d1], x[:, self.d1 :]
        s, t = self.net(torch.cat([x1, c], 1)).chunk(2, 1)
        s = self.clamp * torch.tanh(s / self.clamp)  # soft clamp keeps training stable
        return torch.cat([x1, x2 * torch.exp(s) + t], 1), s.sum(1)


class CondFlow(nn.Module):
    def __init__(self, dim, n_freq, blocks=8, cond_dim=32, hidden=256):
        super().__init__()
        self.n_freq, self.cond_dim = n_freq, cond_dim
        self.blocks = nn.ModuleList([Coupling(dim, cond_dim, hidden, seed=i) for i in range(blocks)])

    def nll(self, v, f_idx):
        """v (N, D), f_idx (N,) -> per-vector negative log-likelihood divided by D (nats per dimension)."""
        c = freq_encoding(f_idx, self.n_freq, self.cond_dim)
        logdet = torch.zeros(len(v), device=v.device)
        for b in self.blocks:
            v, ld = b(v, c)
            logdet = logdet + ld
        return (0.5 * (v**2).sum(1) - logdet) / v.shape[1]
