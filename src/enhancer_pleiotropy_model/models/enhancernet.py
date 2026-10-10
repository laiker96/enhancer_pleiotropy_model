"""enhancerNet baselines: a DeepSTARR-style CNN, optionally followed by a transformer.

Architecture from https://github.com/laiker96/enhancerNet (commit f64e338),
adapted to 2,048-bp inputs and the eight-context profile/classification heads.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..constants import ATAC_BINS, H3K27AC_BINS
from .layers import soft_clip
from .regressors import AttentionRegressor


class EnhancerNetEncoder(nn.Module):
    """Four valid Conv-BN-ReLU-Dropout-MaxPool blocks (256/60/60/120 filters), optional
    two-layer transformer, global max + mean pooling and two dense layers (256)."""

    def __init__(self, with_attention: bool) -> None:
        super().__init__()
        blocks, channels = [], 4
        for filters, kernel in zip((256, 60, 60, 120), (7, 3, 5, 3)):
            blocks.append(nn.Sequential(nn.Conv1d(channels, filters, kernel), nn.BatchNorm1d(filters),
                                        nn.ReLU(), nn.Dropout1d(.1), nn.MaxPool1d(2, 2)))
            channels = filters
        self.convs = nn.Sequential(*blocks)
        if with_attention:
            position = torch.arange(5000).unsqueeze(1)
            frequency = torch.exp(torch.arange(0, 120, 2) * (-math.log(10000.) / 120))
            encoding = torch.zeros(1, 5000, 120)
            encoding[0, :, 0::2] = torch.sin(position * frequency)
            encoding[0, :, 1::2] = torch.cos(position * frequency)
            self.register_buffer("positional_encoding", encoding)
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=120, nhead=8, dim_feedforward=256, dropout=.2,
                                       activation="gelu", batch_first=True, norm_first=False)
            for _ in range(2 if with_attention else 0)])
        self.dense = nn.Sequential(nn.Linear(240, 256), nn.Dropout(.4), nn.ReLU(),
                                   nn.Linear(256, 256), nn.Dropout(.4), nn.ReLU())

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.convs(sequence).transpose(1, 2)
        if self.transformer:
            hidden = hidden + self.positional_encoding[:, :hidden.shape[1]]
            for layer in self.transformer:
                hidden = layer(hidden)
        # Max before mean matches the upstream concatenation order.
        return self.dense(torch.cat((hidden.max(dim=1).values, hidden.mean(dim=1)), dim=1))


class EnhancerNetRegressor(AttentionRegressor):
    def __init__(self, *, with_attention: bool, **kwargs) -> None:
        super().__init__(**kwargs)  # parent layers are created, then replaced (seed compatibility)
        del self.convolutional_body, self.transformer
        self.encoder = EnhancerNetEncoder(with_attention)
        self.atac_head = nn.Sequential(nn.Linear(256, ATAC_BINS * self.context_count))
        self.h3k27ac_head = nn.Sequential(nn.Linear(256, H3K27AC_BINS * self.context_count))

    def initialize_output_means(self, atac_means, h3k27ac_means) -> None:
        for assay, means, bins in (("atac", atac_means, ATAC_BINS), ("h3k27ac", h3k27ac_means, H3K27AC_BINS)):
            values = torch.as_tensor(means)
            if self.output_scaling is not None:
                values = soft_clip(values / torch.tensor(self.output_scaling[assay]))
            self._initialize_head_means(getattr(self, f"{assay}_head"), values.repeat(bins))

    def forward(self, one_hot, attention_mask, atac_target_mask, h3k27ac_target_mask):
        representation = self.encoder(one_hot)
        outputs = tuple(F.softplus(getattr(self, f"{assay}_head")(representation)).reshape(
            len(one_hot), bins, self.context_count)
            for assay, bins in (("atac", ATAC_BINS), ("h3k27ac", H3K27AC_BINS)))
        return self._unscale(outputs)


class EnhancerNetCNNRegressor(EnhancerNetRegressor):
    def __init__(self, **kwargs) -> None:
        super().__init__(with_attention=False, **kwargs)


class EnhancerNetAttentionRegressor(EnhancerNetRegressor):
    def __init__(self, **kwargs) -> None:
        super().__init__(with_attention=True, **kwargs)
