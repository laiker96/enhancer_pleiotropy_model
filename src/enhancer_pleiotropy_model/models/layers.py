"""Building blocks shared by the regressors and classifiers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

POOL_SIZE = 2
SOFT_CLIP = 10.0


def soft_clip(values: torch.Tensor) -> torch.Tensor:
    """Identity below 10, square-root growth above (AlphaGenome target scaling)."""
    values = values.float()
    return torch.where(values > SOFT_CLIP,
                       2 * torch.sqrt(values.clamp_min(SOFT_CLIP) * SOFT_CLIP) - SOFT_CLIP, values)


def inverse_soft_clip(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return torch.where(values > SOFT_CLIP, (values + SOFT_CLIP).square() / (4 * SOFT_CLIP), values)


class MaskedBatchNorm1d(nn.Module):
    """Batch normalization over valid (unmasked) positions only."""

    def __init__(self, features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.ones(features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.bool().unsqueeze(1)
        if self.training:
            statistics = hidden.float()
            float_mask = mask.to(dtype=statistics.dtype)
            count = float_mask.sum()
            mean = (statistics * float_mask).sum(dim=(0, 2)) / count
            variance = ((statistics - mean.view(1, -1, 1)).square() * float_mask).sum(dim=(0, 2)) / count
            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                correction = count / (count - 1.0) if count.item() > 1 else 1.0
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_((variance * correction).detach(), self.momentum)
        else:
            mean, variance = self.running_mean, self.running_var
        output = (hidden.float() - mean.view(1, -1, 1)) * torch.rsqrt(variance.view(1, -1, 1) + self.eps)
        output = output * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return output.to(dtype=hidden.dtype) * mask


class SoftmaxPooling1d(nn.Module):
    """Learned per-channel softmax pooling over pairs of adjacent positions."""

    def __init__(self, channels: int, pool_size: int = POOL_SIZE) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.logit_scale = nn.Parameter(torch.full((channels,), 2.0))
        self.logit_bias = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor):
        remainder = hidden.shape[-1] % self.pool_size
        if remainder:
            padding = self.pool_size - remainder
            hidden = F.pad(hidden, (0, padding))
            attention_mask = F.pad(attention_mask.bool(), (0, padding), value=False)
        batch, channels, length = hidden.shape
        groups = length // self.pool_size
        grouped_hidden = hidden.reshape(batch, channels, groups, self.pool_size)
        grouped_mask = attention_mask.reshape(batch, groups, self.pool_size)
        logits = grouped_hidden * self.logit_scale.view(1, channels, 1, 1) + self.logit_bias.view(1, channels, 1, 1)
        logits = logits.masked_fill(~grouped_mask.unsqueeze(1), -1e4)
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=hidden.dtype)
        pooled_mask = grouped_mask.any(dim=-1)
        return (weights * grouped_hidden).sum(dim=-1) * pooled_mask.unsqueeze(1), pooled_mask


class ConvolutionalBlock(nn.Module):
    """Conv -> BN -> GELU, a pointwise residual branch, dropout and softmax pooling (x2)."""

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(input_channels, output_channels, kernel_size, padding=kernel_size // 2)
        self.normalization = MaskedBatchNorm1d(output_channels)
        self.residual_normalization = MaskedBatchNorm1d(output_channels)
        self.pointwise = nn.Conv1d(output_channels, output_channels, kernel_size=1)
        self.dropout = nn.Dropout1d(dropout)
        self.pooling = SoftmaxPooling1d(output_channels)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor):
        mask = attention_mask.bool()
        hidden = F.gelu(self.normalization(self.convolution(hidden), mask))
        residual = self.pointwise(F.gelu(self.residual_normalization(hidden, mask)))
        hidden = self.dropout(hidden + residual) * mask.unsqueeze(1)
        return self.pooling(hidden, mask)


class ConvolutionalStem(nn.Module):
    """Four residual convolution blocks: 2,048 bp x 4 -> 128 positions x 192 channels."""

    def __init__(self, dropout: float, filters: tuple[int, ...], kernels: tuple[int, ...]) -> None:
        super().__init__()
        channels = (4, *filters)
        self.blocks = nn.ModuleList(ConvolutionalBlock(channels[i], channels[i + 1], kernels[i], dropout)
                                    for i in range(len(filters)))

    def forward(self, one_hot: torch.Tensor, attention_mask: torch.Tensor, target_mask: torch.Tensor):
        hidden = one_hot * attention_mask.bool().unsqueeze(1)
        mask = attention_mask.bool()
        pooled_target = target_mask.bool()
        for block in self.blocks:
            hidden, mask = block(hidden, mask)
            pooled_target = downsample_mask(pooled_target, 1)
        return hidden, mask, pooled_target & mask


def downsample_mask(mask: torch.Tensor, levels: int) -> torch.Tensor:
    pooled = mask.bool()
    for _ in range(levels):
        remainder = pooled.shape[-1] % POOL_SIZE
        if remainder:
            pooled = F.pad(pooled, (0, POOL_SIZE - remainder), value=False)
        pooled = pooled.reshape(pooled.shape[0], -1, POOL_SIZE).any(dim=-1)
    return pooled


class RelativePositionBias(nn.Module):
    def __init__(self, heads: int, maximum_distance: int) -> None:
        super().__init__()
        self.maximum_distance = maximum_distance
        self.embedding = nn.Embedding(2 * maximum_distance + 1, heads)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, length: int, reference: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(length, device=reference.device)
        relative = (positions[:, None] - positions[None, :]).clamp(
            -self.maximum_distance, self.maximum_distance) + self.maximum_distance
        return self.embedding(relative).permute(2, 0, 1).to(dtype=reference.dtype)


class RelativeTransformerBlock(nn.Module):
    """Pre-norm multi-head self-attention with a learned relative-position bias."""

    def __init__(self, dimension: int, heads: int, feedforward: int, dropout: float,
                 maximum_distance: int = 128) -> None:
        super().__init__()
        self.heads = heads
        self.attention_normalization = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.relative_bias = RelativePositionBias(heads, maximum_distance)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_normalization = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(nn.Linear(dimension, feedforward), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(feedforward, dimension), nn.Dropout(dropout))

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        normalized = self.attention_normalization(hidden)
        bias = self.relative_bias(length, hidden).unsqueeze(0).expand(batch, -1, -1, -1)
        bias = bias.masked_fill(~attention_mask[:, None, None, :].bool(), -1e4).reshape(
            batch * self.heads, length, length)
        attended, _ = self.attention(normalized, normalized, normalized, attn_mask=bias, need_weights=False)
        hidden = hidden + self.attention_dropout(attended)
        hidden = hidden + self.feedforward(self.feedforward_normalization(hidden))
        return hidden * attention_mask.unsqueeze(-1)


def interleaved_conv1d(values: torch.Tensor, convolution: nn.Conv1d) -> torch.Tensor:
    """The same dilated convolution computed without dilation.

    Positions are split into ``dilation`` interleaved residue classes that become
    separate batch items, so neighbours in each item are ``dilation`` apart.
    Mathematically identical to ``convolution(values)``; much faster than native
    dilation when deterministic CUDA kernels are required for gradients.
    """
    dilation = convolution.dilation[0]
    if dilation == 1:
        return convolution(values)
    batch, channels, length = values.shape
    padded = F.pad(values, (0, (-length) % dilation))
    grouped = padded.reshape(batch, channels, -1, dilation).permute(0, 3, 1, 2).reshape(batch * dilation, channels, -1)
    outputs = F.conv1d(grouped, convolution.weight, convolution.bias, padding=1)
    outputs = outputs.reshape(batch, dilation, convolution.out_channels, -1)
    return outputs.permute(0, 2, 3, 1).reshape(batch, convolution.out_channels, -1)[..., :length]


class DilatedBlock(nn.Module):
    """Pre-norm residual dilated convolution: x + Dropout(GELU(Conv_k3,d(LN(x))))."""

    def __init__(self, dilation: int, dimension: int = 192, dropout: float = .1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.conv = nn.Conv1d(dimension, dimension, 3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.interleaved = False  # inference switch, see interleaved_conv1d; training uses native dilation

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = (self.norm(hidden) * mask.unsqueeze(-1)).transpose(1, 2)
        values = (interleaved_conv1d(values, self.conv) if self.interleaved else self.conv(values)).transpose(1, 2)
        return (hidden + self.dropout(F.gelu(values))) * mask.unsqueeze(-1)


def profile_head(dimension: int, contexts: int, dropout: float) -> nn.Sequential:
    """LN -> Linear(2d) -> GELU -> Dropout -> Linear(contexts); softplus is applied by the model."""
    return nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 2 * dimension), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(2 * dimension, contexts))
