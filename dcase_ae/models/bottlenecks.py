"""Bottlenecks: (token grid (B, Fp, Tp, D), visible mask or None) -> memory tokens (B, M, D')."""
import torch
from torch import nn

from ..registry import BOTTLENECKS


@BOTTLENECKS.register("mean_time")
class MeanTimeBottleneck(nn.Module):
    """Mean-pool over the time axis -> one vector per frequency patch row (M = Fp).
    With a visible mask, only the encoded (unmasked) tokens of each row are averaged."""

    def __init__(self, in_dim, dim, grid):
        super().__init__()
        self.num_tokens = grid[0]
        self.out_dim = dim
        self.proj = nn.Linear(in_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, visible=None) -> torch.Tensor:
        if visible is None:
            pooled = tokens.mean(dim=2)
        else:
            v = visible[..., None].to(tokens.dtype)
            pooled = (tokens * v).sum(dim=2) / v.sum(dim=2).clamp_min(1.0)
        return self.norm(self.proj(pooled))


@BOTTLENECKS.register("none")
class IdentityBottleneck(nn.Module):
    """Keeps every token (M = Fp*Tp). Sanity baseline: should reconstruct almost perfectly."""

    def __init__(self, in_dim, dim, grid):
        super().__init__()
        self.num_tokens = grid[0] * grid[1]
        self.out_dim = dim
        self.proj = nn.Linear(in_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, visible=None) -> torch.Tensor:
        return self.norm(self.proj(tokens.flatten(1, 2)))
