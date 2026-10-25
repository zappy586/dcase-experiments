"""Encoders: spectrogram (B, C, F, T) -> (token grid (B, Fp, Tp, D), visible mask or None).

`visible` is a bool grid (B, Fp, Tp) marking tokens that were actually encoded; it is only returned
when patches were masked (training with mask_ratio > 0). Any new backbone (e.g. BEATs) only needs to
return the same pair and register itself with @ENCODERS.register("name").
"""
import torch
from torch import nn

from ..registry import ENCODERS
from .patching import sincos_2d


@ENCODERS.register("vit")
class ViTEncoder(nn.Module):
    def __init__(
        self, in_chans, input_size, patch_size, dim=256, depth=6, heads=8, mlp_ratio=4, dropout=0.0, mask_ratio=0.0
    ):
        super().__init__()
        pf, pt = patch_size
        self.grid = (input_size[0] // pf, input_size[1] // pt)
        self.out_dim = dim
        self.mask_ratio = mask_ratio
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(sincos_2d(dim, self.grid)[None])
        layer = nn.TransformerEncoderLayer(
            dim, heads, int(dim * mlp_ratio), dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        tok = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos  # (B, N, D), freq-major
        N, D = tok.shape[1], tok.shape[2]
        if self.training and self.mask_ratio > 0:
            # random per-clip subset of patches; the encoder only ever sees the kept ones
            keep = max(1, int(round(N * (1 - self.mask_ratio))))
            idx = torch.rand(B, N, device=x.device).argsort(1)[:, :keep]
            enc = self.norm(self.blocks(tok.gather(1, idx[..., None].expand(-1, -1, D))))
            full = enc.new_zeros(B, N, D).scatter(1, idx[..., None].expand(-1, -1, D), enc)
            visible = torch.zeros(B, N, dtype=torch.bool, device=x.device).scatter(1, idx, True)
            return full.reshape(B, *self.grid, D), visible.reshape(B, *self.grid)
        return self.norm(self.blocks(tok)).reshape(B, *self.grid, D), None
