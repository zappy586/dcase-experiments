"""Decoders: memory tokens (B, M, D') -> patch pixels (B, Fp*Tp, C*pf*pt)."""
import torch
from torch import nn

from ..registry import DECODERS
from .patching import sincos_2d


@DECODERS.register("transformer")
class TransformerDecoder(nn.Module):
    """One learned query per output patch; queries see the memory only via cross-attention,
    so everything the decoder knows about the input is the bottleneck tokens."""

    def __init__(self, mem_dim, num_memory, grid, out_dim, dim=256, depth=4, heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.mem_proj = nn.Linear(mem_dim, dim) if mem_dim != dim else nn.Identity()
        self.mem_pos = nn.Parameter(torch.randn(1, num_memory, dim) * 0.02)
        self.query = nn.Parameter(sincos_2d(dim, grid)[None])
        layer = nn.TransformerDecoderLayer(
            dim, heads, int(dim * mlp_ratio), dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerDecoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        memory = self.mem_proj(memory) + self.mem_pos
        q = self.query.expand(memory.shape[0], -1, -1)
        return self.head(self.norm(self.blocks(q, memory)))
