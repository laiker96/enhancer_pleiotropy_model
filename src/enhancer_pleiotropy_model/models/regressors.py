"""ATAC/H3K27ac profile regressors (the pretraining task).

Every regressor reads 2,048 bp of one-hot DNA and predicts, for the eight
contexts, ATAC in 32 x 16-bp bins over the central 512 bp and H3K27ac in
24 x 64-bp bins over the central 1,536 bp, in normalized signal units.

Architectures (``ARCHITECTURES``):

* ``cnn`` - convolutional stem + six dilated residual convolutions (manuscript model)
* ``attention`` - convolutional stem + four relative-position transformer blocks
* ``dense`` - convolutional stem + flatten + two dense layers
* ``enhancernet_cnn`` / ``enhancernet_attention`` - see :mod:`.enhancernet`

Subclasses build the parent's layers before replacing them. That consumes the
same random numbers as the original implementation, so a given seed yields the
initialisation used in the manuscript.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..constants import ATAC_BINS, CONTEXTS, H3K27AC_BINS, H3K27AC_OUTPUT_POOL_SIZE, INPUT_BP
from .layers import (ConvolutionalStem, DilatedBlock, RelativeTransformerBlock, downsample_mask,
                     inverse_soft_clip, profile_head, soft_clip)

FILTERS = (96, 128, 160, 192)
KERNELS = (15, 5, 5, 5)
DILATIONS = (1, 2, 4, 8, 16, 32)
TRANSFORMER_LAYERS, TRANSFORMER_HEADS, TRANSFORMER_FEEDFORWARD = 4, 8, 768
ATAC_SLICE = slice(48, 80)         # stem positions (16 bp each) covering the central 512 bp
H3K27AC_SLICE = slice(16, 112)     # stem positions covering the central 1,536 bp


class AttentionRegressor(nn.Module):
    """Convolutional stem, relative-position transformer and two profile heads."""

    def __init__(self, context_count: int = len(CONTEXTS), dropout: float = 0.1, head_dropout: float = 0.1,
                 h3k27ac_output_pool_size: int = H3K27AC_OUTPUT_POOL_SIZE,
                 output_scaling: dict[str, list[float]] | None = None) -> None:
        super().__init__()
        self.convolution_filters, self.convolution_kernels = FILTERS, KERNELS
        self.context_count = context_count
        self.h3k27ac_output_pool_size = h3k27ac_output_pool_size
        self.output_scaling = None
        if output_scaling is not None:
            # Predictions are made in scaled units, soft_clip(signal / training nonzero mean).
            self.output_scaling = {}
            for assay, values in output_scaling.items():
                means = torch.as_tensor(values, dtype=torch.float32)
                self.register_buffer(f"{assay}_output_means", means)
                self.output_scaling[assay] = means.tolist()
        dimension = FILTERS[-1]
        self.convolutional_body = ConvolutionalStem(dropout, FILTERS, KERNELS)
        self.transformer = nn.ModuleList(
            RelativeTransformerBlock(dimension, TRANSFORMER_HEADS, TRANSFORMER_FEEDFORWARD, dropout)
            for _ in range(TRANSFORMER_LAYERS))
        self.atac_head = profile_head(dimension, context_count, head_dropout)
        self.h3k27ac_decoder = nn.ModuleList()  # kept empty for checkpoint compatibility
        self.h3k27ac_head = profile_head(dimension, context_count, head_dropout)

    @staticmethod
    def _initialize_head_means(head: nn.Sequential, means) -> None:
        """Set the final bias so that softplus(bias) equals the training mean."""
        means = torch.as_tensor(means, dtype=head[-1].bias.dtype).clamp_min(1e-4)
        with torch.no_grad():
            head[-1].bias.copy_(torch.where(means > 20, means, torch.log(torch.expm1(means))))

    def initialize_output_means(self, atac_means: np.ndarray, h3k27ac_means: np.ndarray) -> None:
        if self.output_scaling is not None:
            atac_means, h3k27ac_means = (
                soft_clip(torch.as_tensor(values) / torch.tensor(self.output_scaling[assay])).numpy()
                for assay, values in (("atac", atac_means), ("h3k27ac", h3k27ac_means)))
        self._initialize_head_means(self.atac_head, atac_means)
        self._initialize_head_means(self.h3k27ac_head, h3k27ac_means)

    def _unscale(self, outputs):
        if self.output_scaling is None:
            return outputs
        return tuple(inverse_soft_clip(values) * getattr(self, f"{assay}_output_means")
                     for assay, values in zip(("atac", "h3k27ac"), outputs, strict=True))

    def encode(self, one_hot: torch.Tensor, attention_mask: torch.Tensor, atac_target_mask: torch.Tensor):
        """Stem + sequence mixer -> [batch, 128, 192] latent positions."""
        hidden, mask, pooled_atac = self.convolutional_body(one_hot, attention_mask, atac_target_mask)
        hidden = hidden.transpose(1, 2)
        for block in self.transformer:
            hidden = block(hidden, mask)
        return hidden, mask, pooled_atac

    def forward(self, one_hot, attention_mask, atac_target_mask, h3k27ac_target_mask):
        hidden, mask, pooled_atac = self.encode(one_hot, attention_mask, atac_target_mask)
        pooled_h3 = downsample_mask(h3k27ac_target_mask, len(FILTERS)) & mask
        atac = hidden[pooled_atac].reshape(len(hidden), -1, hidden.shape[-1])
        h3 = hidden[pooled_h3].reshape(len(hidden), -1, hidden.shape[-1])
        pool = self.h3k27ac_output_pool_size
        h3 = h3.reshape(len(h3), h3.shape[1] // pool, pool, h3.shape[2]).mean(dim=2)
        return self._unscale((F.softplus(self.atac_head(atac)), F.softplus(self.h3k27ac_head(h3))))


class CNNRegressor(AttentionRegressor):
    """Same stem and heads; six dilated convolutions (d = 1..32) replace attention."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transformer = nn.ModuleList(DilatedBlock(d, dropout=kwargs.get("dropout", .1)) for d in DILATIONS)


class DenseRegressor(AttentionRegressor):
    """Same stem; all 128 x 192 latent values are flattened into two dense layers."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        dropout = kwargs.get("dropout", .1)
        self.transformer = nn.ModuleList()
        self.dense = nn.Sequential(
            nn.Linear(128 * 192, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout))
        self.atac_head = nn.Sequential(nn.Linear(256, ATAC_BINS * self.context_count))
        self.h3k27ac_head = nn.Sequential(nn.Linear(256, H3K27AC_BINS * self.context_count))

    def initialize_output_means(self, atac_means, h3k27ac_means) -> None:
        for assay, means, bins in (("atac", atac_means, ATAC_BINS), ("h3k27ac", h3k27ac_means, H3K27AC_BINS)):
            values = torch.as_tensor(means)
            if self.output_scaling is not None:
                values = soft_clip(values / torch.tensor(self.output_scaling[assay]))
            self._initialize_head_means(getattr(self, f"{assay}_head"), values.repeat(bins))

    def forward(self, one_hot, attention_mask, atac_target_mask, h3k27ac_target_mask):
        if one_hot.shape[1:] != (4, INPUT_BP):
            raise ValueError("The dense model requires unpadded 2,048-bp inputs")
        hidden, mask, _ = self.convolutional_body(one_hot, attention_mask, atac_target_mask)
        representation = self.dense((hidden.transpose(1, 2) * mask.unsqueeze(-1)).flatten(1))
        outputs = tuple(F.softplus(getattr(self, f"{assay}_head")(representation)).reshape(
            len(one_hot), bins, self.context_count)
            for assay, bins in (("atac", ATAC_BINS), ("h3k27ac", H3K27AC_BINS)))
        return self._unscale(outputs)
