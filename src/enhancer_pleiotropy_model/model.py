"""Checkpoint-compatible production joint ATAC/H3K27ac model."""

from __future__ import annotations

from typing import Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .constants import CONTEXTS, H3K27AC_OUTPUT_POOL_SIZE


POOL_SIZE: Final = 2
RELATIVE_POSITION_MAX_DISTANCE: Final = 128
CONVOLUTION_KERNELS: Final = (15, 5, 5, 5)
MODEL_PRESETS = {
    "base": {
        "convolution_filters": (64, 80, 100, 120),
        "transformer_layers": 2,
        "transformer_heads": 8,
        "transformer_feedforward_dimension": 480,
    },
    "4x": {
        "convolution_filters": (96, 128, 160, 192),
        "transformer_layers": 4,
        "transformer_heads": 8,
        "transformer_feedforward_dimension": 768,
    },
}


class MaskedBatchNorm1d(nn.Module):
    """Batch normalization over valid batch/length positions only."""

    def __init__(
        self, features: int, eps: float = 1e-5, momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.ones(features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or attention_mask.shape != (
            hidden.shape[0],
            hidden.shape[2],
        ):
            raise ValueError("MaskedBatchNorm1d received incompatible shapes")
        mask = attention_mask.bool().unsqueeze(1)
        if self.training:
            statistics = hidden.float()
            float_mask = mask.to(dtype=statistics.dtype)
            count = float_mask.sum()
            if count.item() < 1:
                raise ValueError("MaskedBatchNorm1d requires a valid position")
            mean = (statistics * float_mask).sum(dim=(0, 2)) / count
            centered = statistics - mean.view(1, -1, 1)
            variance = (centered.square() * float_mask).sum(dim=(0, 2)) / count
            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                correction = count / (count - 1.0) if count.item() > 1 else 1.0
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_((variance * correction).detach(), self.momentum)
        else:
            mean = self.running_mean
            variance = self.running_var
        output = (hidden.float() - mean.view(1, -1, 1)) * torch.rsqrt(
            variance.view(1, -1, 1) + self.eps
        )
        output = output * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return output.to(dtype=hidden.dtype) * mask


class SoftmaxPooling1d(nn.Module):
    """Learned per-channel softmax pooling over adjacent positions."""

    def __init__(self, channels: int, pool_size: int = POOL_SIZE) -> None:
        super().__init__()
        if pool_size < 1:
            raise ValueError("pool_size must be positive")
        self.pool_size = pool_size
        self.logit_scale = nn.Parameter(torch.full((channels,), 2.0))
        self.logit_bias = nn.Parameter(torch.zeros(channels))

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 3 or attention_mask.shape != (
            hidden.shape[0],
            hidden.shape[2],
        ):
            raise ValueError("Softmax pooling received incompatible shapes")
        remainder = hidden.shape[-1] % self.pool_size
        if remainder:
            padding = self.pool_size - remainder
            hidden = F.pad(hidden, (0, padding))
            attention_mask = F.pad(
                attention_mask.bool(), (0, padding), value=False
            )
        batch, channels, length = hidden.shape
        groups = length // self.pool_size
        grouped_hidden = hidden.reshape(batch, channels, groups, self.pool_size)
        grouped_mask = attention_mask.reshape(batch, groups, self.pool_size)
        logits = (
            grouped_hidden * self.logit_scale.view(1, channels, 1, 1)
            + self.logit_bias.view(1, channels, 1, 1)
        )
        logits = logits.masked_fill(~grouped_mask.unsqueeze(1), -1e4)
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=hidden.dtype)
        pooled_mask = grouped_mask.any(dim=-1)
        pooled = (weights * grouped_hidden).sum(dim=-1)
        return pooled * pooled_mask.unsqueeze(1), pooled_mask


class EnformerLikeConvolutionalBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, kernel_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.normalization = MaskedBatchNorm1d(output_channels)
        self.residual_normalization = MaskedBatchNorm1d(output_channels)
        self.pointwise = nn.Conv1d(output_channels, output_channels, kernel_size=1)
        self.dropout = nn.Dropout1d(dropout)
        self.pooling = SoftmaxPooling1d(output_channels)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = attention_mask.bool()
        hidden = F.gelu(self.normalization(self.convolution(hidden), mask))
        residual = self.pointwise(
            F.gelu(self.residual_normalization(hidden, mask))
        )
        hidden = self.dropout(hidden + residual) * mask.unsqueeze(1)
        return self.pooling(hidden, mask)


class EnformerLikeConvolutionalBody(nn.Module):
    def __init__(
        self,
        dropout: float,
        convolution_filters: tuple[int, ...],
        convolution_kernels: tuple[int, ...] = CONVOLUTION_KERNELS,
    ) -> None:
        super().__init__()
        if not convolution_filters or len(convolution_filters) != len(
            convolution_kernels
        ):
            raise ValueError("Convolution filters and kernels must align")
        channels = (4, *convolution_filters)
        self.blocks = nn.ModuleList(
            EnformerLikeConvolutionalBlock(
                channels[index],
                channels[index + 1],
                convolution_kernels[index],
                dropout,
            )
            for index in range(len(convolution_filters))
        )

    def forward(
        self,
        one_hot: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if target_mask.shape != attention_mask.shape:
            raise ValueError("Target and attention masks must have the same shape")
        if torch.any(target_mask.bool() & ~attention_mask.bool()):
            raise ValueError("Target mask cannot include padding")
        hidden = one_hot * attention_mask.bool().unsqueeze(1)
        mask = attention_mask.bool()
        pooled_target_mask = target_mask.bool()
        for block in self.blocks:
            hidden, mask = block(hidden, mask)
            remainder = pooled_target_mask.shape[-1] % POOL_SIZE
            if remainder:
                pooled_target_mask = F.pad(
                    pooled_target_mask,
                    (0, POOL_SIZE - remainder),
                    value=False,
                )
            pooled_target_mask = pooled_target_mask.reshape(
                pooled_target_mask.shape[0], -1, POOL_SIZE
            ).any(dim=-1)
        return hidden, mask, pooled_target_mask & mask


class RelativePositionBias(nn.Module):
    def __init__(self, heads: int, maximum_distance: int) -> None:
        super().__init__()
        self.maximum_distance = maximum_distance
        self.embedding = nn.Embedding(2 * maximum_distance + 1, heads)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, length: int, reference: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(length, device=reference.device)
        relative = positions[:, None] - positions[None, :]
        relative = relative.clamp(
            -self.maximum_distance, self.maximum_distance
        ) + self.maximum_distance
        return self.embedding(relative).permute(2, 0, 1).to(dtype=reference.dtype)


class RelativeTransformerBlock(nn.Module):
    def __init__(
        self, dimension: int, heads: int, feedforward: int, dropout: float
    ) -> None:
        super().__init__()
        self.heads = heads
        self.attention_normalization = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.relative_bias = RelativePositionBias(
            heads, RELATIVE_POSITION_MAX_DISTANCE
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_normalization = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, dimension),
            nn.Dropout(dropout),
        )

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, length, _dimension = hidden.shape
        normalized = self.attention_normalization(hidden)
        attention_bias = self.relative_bias(length, hidden)
        attention_bias = attention_bias.unsqueeze(0).expand(batch, -1, -1, -1)
        attention_bias = attention_bias.masked_fill(
            ~attention_mask[:, None, None, :].bool(), -1e4
        ).reshape(batch * self.heads, length, length)
        attended, _weights = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_bias,
            need_weights=False,
        )
        hidden = hidden + self.attention_dropout(attended)
        hidden = hidden + self.feedforward(self.feedforward_normalization(hidden))
        return hidden * attention_mask.unsqueeze(-1)


def downsample_boolean_mask(mask: torch.Tensor, levels: int) -> torch.Tensor:
    pooled = mask.bool()
    for _ in range(levels):
        remainder = pooled.shape[-1] % POOL_SIZE
        if remainder:
            pooled = F.pad(pooled, (0, POOL_SIZE - remainder), value=False)
        pooled = pooled.reshape(pooled.shape[0], -1, POOL_SIZE).any(dim=-1)
    return pooled


def profile_head(dimension: int, contexts: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(dimension),
        nn.Linear(dimension, 2 * dimension),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(2 * dimension, contexts),
    )


class EnformerLikeJointProfileRegressor(nn.Module):
    """Shared sequence encoder with separate dense ATAC and H3K27ac heads."""

    def __init__(
        self,
        context_count: int = len(CONTEXTS),
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        model_size: str = "4x",
        h3k27ac_output_pool_size: int = H3K27AC_OUTPUT_POOL_SIZE,
    ) -> None:
        super().__init__()
        if model_size not in MODEL_PRESETS:
            raise ValueError(f"Unknown model preset: {model_size}")
        if h3k27ac_output_pool_size < 1:
            raise ValueError("H3K27ac output pool size must be positive")
        configuration = MODEL_PRESETS[model_size]
        self.convolution_filters = tuple(configuration["convolution_filters"])
        self.convolution_kernels = CONVOLUTION_KERNELS
        self.transformer_layers = int(configuration["transformer_layers"])
        self.transformer_heads = int(configuration["transformer_heads"])
        self.transformer_feedforward_dimension = int(
            configuration["transformer_feedforward_dimension"]
        )
        dimension = self.convolution_filters[-1]
        self.context_count = context_count
        self.h3k27ac_output_pool_size = h3k27ac_output_pool_size
        self.convolutional_body = EnformerLikeConvolutionalBody(
            dropout,
            convolution_filters=self.convolution_filters,
            convolution_kernels=self.convolution_kernels,
        )
        self.transformer = nn.ModuleList(
            RelativeTransformerBlock(
                dimension,
                self.transformer_heads,
                self.transformer_feedforward_dimension,
                dropout,
            )
            for _ in range(self.transformer_layers)
        )
        self.atac_head = profile_head(dimension, context_count, head_dropout)
        self.h3k27ac_decoder = nn.ModuleList()
        self.h3k27ac_head = profile_head(dimension, context_count, head_dropout)

    @staticmethod
    def _initialize_head_means(head: nn.Sequential, means: np.ndarray) -> None:
        means_tensor = torch.as_tensor(means, dtype=head[-1].bias.dtype).clamp_min(
            1e-4
        )
        inverse_softplus = torch.where(
            means_tensor > 20,
            means_tensor,
            torch.log(torch.expm1(means_tensor)),
        )
        with torch.no_grad():
            head[-1].bias.copy_(inverse_softplus)

    def initialize_output_means(
        self, atac_means: np.ndarray, h3k27ac_means: np.ndarray
    ) -> None:
        expected = (self.context_count,)
        if atac_means.shape != expected or h3k27ac_means.shape != expected:
            raise ValueError("Output means do not match contexts")
        self._initialize_head_means(self.atac_head, atac_means)
        self._initialize_head_means(self.h3k27ac_head, h3k27ac_means)

    @staticmethod
    def _select_target(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        counts = mask.sum(dim=1)
        if torch.any(counts != counts[0]) or counts[0] < 1:
            raise ValueError("Every batch item must retain equal non-empty target bins")
        return hidden[mask].reshape(
            hidden.shape[0], int(counts[0].item()), hidden.shape[-1]
        )

    def forward(
        self,
        one_hot: torch.Tensor,
        attention_mask: torch.Tensor,
        atac_target_mask: torch.Tensor,
        h3k27ac_target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, mask, pooled_atac_mask = self.convolutional_body(
            one_hot, attention_mask, atac_target_mask
        )
        pooled_h3k27ac_mask = downsample_boolean_mask(
            h3k27ac_target_mask, len(self.convolution_filters)
        )
        pooled_h3k27ac_mask &= mask
        hidden = hidden.transpose(1, 2)
        for block in self.transformer:
            hidden = block(hidden, mask)
        atac_hidden = self._select_target(hidden, pooled_atac_mask)
        h3k27ac_hidden = self._select_target(hidden, pooled_h3k27ac_mask)
        if h3k27ac_hidden.shape[1] % self.h3k27ac_output_pool_size:
            raise ValueError("H3K27ac target bins are not divisible by output pooling")
        if self.h3k27ac_output_pool_size > 1:
            h3k27ac_hidden = h3k27ac_hidden.reshape(
                h3k27ac_hidden.shape[0],
                h3k27ac_hidden.shape[1] // self.h3k27ac_output_pool_size,
                self.h3k27ac_output_pool_size,
                h3k27ac_hidden.shape[2],
            ).mean(dim=2)
        return (
            F.softplus(self.atac_head(atac_hidden)),
            F.softplus(self.h3k27ac_head(h3k27ac_hidden)),
        )


def infer_model_preset(architecture: dict[str, object]) -> str:
    filters = tuple(int(value) for value in architecture["convolution_filters"])
    matches = [
        name
        for name, values in MODEL_PRESETS.items()
        if tuple(values["convolution_filters"]) == filters
    ]
    if len(matches) != 1:
        raise ValueError(f"Unsupported convolution filter configuration: {filters}")
    return matches[0]
