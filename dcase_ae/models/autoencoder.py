"""Encoder -> bottleneck -> decoder composed from the config."""
import torch
import torch.nn.functional as F
from torch import nn

from ..registry import BOTTLENECKS, DECODERS, ENCODERS
from . import bottlenecks, decoders, encoders  # noqa: F401  (populate registries)
from .patching import unpatchify

LOGVAR_RANGE = (-6.0, 4.0)


class SpectrogramAE(nn.Module):
    """The model reconstructs `target(x)`: x itself, or x average-pooled by `target_pool` (rf, rt).
    With target_pool=(1, 1) each 16x16 input patch is decoded to 16x16 values; otherwise to a
    (16/rf) x (16/rt) block.

    With predict_var=True the decoder also predicts a per-value log-variance and is trained with the
    Gaussian negative log-likelihood, so values that are intrinsically noisy are down-weighted.
    """

    def __init__(self, encoder, bottleneck, decoder, patch_size, grid, in_chans, target_pool=(1, 1), predict_var=False):
        super().__init__()
        self.encoder, self.bottleneck, self.decoder = encoder, bottleneck, decoder
        self.patch_size, self.grid, self.in_chans = tuple(patch_size), tuple(grid), in_chans
        self.target_pool = tuple(target_pool)
        self.predict_var = predict_var
        self.out_patch = (patch_size[0] // target_pool[0], patch_size[1] // target_pool[1])

    def target(self, x):
        """What the decoder is trained to output (B, C, F/rf, T/rt)."""
        return x if self.target_pool == (1, 1) else F.avg_pool2d(x, self.target_pool)

    def forward(self, x):
        """Returns (reconstruction of target(x), log-variance or None, bottleneck tokens (B,M,D'))."""
        tokens, visible = self.encoder(x)
        memory = self.bottleneck(tokens, visible)
        out = self.decoder(memory)
        logvar = None
        if self.predict_var:
            out, logvar = out.chunk(2, dim=-1)
            logvar = unpatchify(logvar.clamp(*LOGVAR_RANGE), self.out_patch, self.grid, self.in_chans)
        return unpatchify(out, self.out_patch, self.grid, self.in_chans), logvar, memory

    @staticmethod
    def nll_map(recon, logvar, target):
        """Per-value Gaussian negative log-likelihood (up to a constant)."""
        return 0.5 * ((recon - target) ** 2 * torch.exp(-logvar) + logvar)

    def loss(self, recon, logvar, target):
        if logvar is None:
            return F.mse_loss(recon, target)
        return self.nll_map(recon, logvar, target).mean()


def _kwargs(section) -> dict:
    return {k: v for k, v in section.items() if k != "name"}


def build_model(cfg, in_chans: int, input_size: tuple[int, int]) -> SpectrogramAE:
    patch = tuple(cfg.model.patch_size)
    pool = tuple(cfg.model.get("target_pool", (1, 1)))
    if patch[0] % pool[0] or patch[1] % pool[1]:
        raise ValueError(f"model.target_pool {pool} must divide patch_size {patch}")
    out_patch = (patch[0] // pool[0], patch[1] // pool[1])
    predict_var = bool(cfg.model.get("predict_var", False))
    enc = ENCODERS.get(cfg.model.encoder.name)(in_chans, input_size, patch, **_kwargs(cfg.model.encoder))
    bott = BOTTLENECKS.get(cfg.model.bottleneck.name)(enc.out_dim, cfg.model.bottleneck.dim, enc.grid)
    out_dim = in_chans * out_patch[0] * out_patch[1] * (2 if predict_var else 1)
    dec = DECODERS.get(cfg.model.decoder.name)(bott.out_dim, bott.num_tokens, enc.grid, out_dim, **_kwargs(cfg.model.decoder))
    return SpectrogramAE(enc, bott, dec, patch, enc.grid, in_chans, pool, predict_var)
