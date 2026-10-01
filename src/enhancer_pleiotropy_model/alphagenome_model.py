# Copyright 2026 Google LLC.
# PyTorch adaptation and small-target modifications, 2026-09-10.
# Licensed under Apache-2.0; see LICENSES/AlphaGenome-Apache-2.0.txt.
# Distributed WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

"""Small AlphaGenome-style trunk for the existing 2-kb ATAC/H3 target contract.

PyTorch adaptation of google-deepmind/alphagenome_research at
0db53bd4352c66d1e00a049a81da373a066e6670 (Apache-2.0): convolutions.py,
layers.py, attention.py, embeddings.py and heads.py. This is not the released
AlphaGenome model: widths are reduced, decoding stops at 16 bp, and there is
no pairwise/contact-map tower, organism conditioning or additional modalities.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .alphagenome_loss import UPSTREAM_COMMIT, inverse_soft_clip, soft_clip
from .constants import CONTEXTS, INPUT_BP


PRESET = "alphagenome_small"
ARCHITECTURE_NAME = "alphagenome_small_joint_profiles_v1"
ARCHITECTURE_16BP_NAME = "alphagenome_small_joint_profiles_16bp_v1"
WIDTHS = (96, 112, 128, 144, 160, 176, 192)


def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


class RMSBatchNorm(nn.Module):
    """Uncentered, stop-gradient batch/position second moment, as upstream."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_second_moment", torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                moment = x.float().square().mean(dim=tuple(range(x.ndim - 1)))
                self.running_second_moment.lerp_(moment, .1)
        else:
            moment = self.running_second_moment
        # FP32 statistics/arithmetic avoid squaring FP16 activations in FP16.
        return (x.float() * torch.rsqrt(moment + 1e-5) * self.weight + self.bias).to(x.dtype)


class StandardizedConv1d(nn.Conv1d):
    def __init__(self, inputs: int, outputs: int, width: int):
        super().__init__(inputs, outputs, width, padding=width // 2)
        self.gain = nn.Parameter(torch.ones(outputs, 1, 1))
        nn.init.normal_(self.weight, std=(inputs * width) ** -.5)
        nn.init.trunc_normal_(self.bias, std=1e-4, a=-2e-4, b=2e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float()
        centered = weight - weight.mean(dim=(1, 2), keepdim=True)
        fan_in = weight.shape[1] * weight.shape[2]
        weight = centered * self.gain * torch.rsqrt(
            (fan_in * centered.square().mean(dim=(1, 2), keepdim=True)).clamp_min(1e-4)
        )
        return F.conv1d(x.transpose(1, 2), weight, self.bias, padding=self.padding).transpose(1, 2)


class ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, width: int):
        super().__init__()
        self.norm = RMSBatchNorm(inputs)
        self.conv = nn.Linear(inputs, outputs) if width == 1 else StandardizedConv1d(inputs, outputs, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(fast_gelu(self.norm(x)))


class DownResBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.added_channels = outputs - inputs
        self.first = ConvBlock(inputs, outputs, 5)
        self.second = ConvBlock(outputs, outputs, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x) + F.pad(x, (0, self.added_channels))
        return x + self.second(x)


class UpResBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, upsample_factor: int = 2):
        super().__init__()
        self.outputs = outputs
        if upsample_factor not in (1, 2):
            raise ValueError("Decoder supports refinement or factor-two upsampling")
        self.upsample_factor = upsample_factor
        self.first = ConvBlock(inputs, outputs, 5)
        self.skip = ConvBlock(outputs, outputs, 1)
        self.second = ConvBlock(outputs, outputs, 5)
        self.residual_scale = nn.Parameter(torch.tensor(.1))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = (self.first(x) + x[..., :self.outputs]).repeat_interleave(self.upsample_factor, dim=1)
        x = x * self.residual_scale + self.skip(skip)
        return x + self.second(x)


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Upstream interleaved rotary convention and 8192-position frequency grid."""
    frequencies = x.shape[-1] // 2
    inverse = 1 / (torch.arange(frequencies, device=x.device, dtype=torch.float32)
                   + torch.exp(torch.linspace(0, math.log(8192 - frequencies + 1), frequencies, device=x.device)))
    theta = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[:, None] * inverse
    theta = theta.repeat_interleave(2, -1)[None, :, None, :]
    rotated = torch.stack((-x[..., 1::2], x[..., ::2]), -1).flatten(-2)
    return (x.float() * theta.cos() + rotated.float() * theta.sin()).to(x.dtype)


class MultiQueryBlock(nn.Module):
    def __init__(self, channels: int = 192):
        super().__init__()
        self.norm = RMSBatchNorm(channels)
        self.query = nn.Linear(channels, 8 * 16, bias=False)
        self.key = nn.Linear(channels, 16, bias=False)
        self.value = nn.Linear(channels, 24, bias=False)
        self.query_norm = nn.LayerNorm(16, eps=1e-5)
        self.key_norm = nn.LayerNorm(16, eps=1e-5)
        self.value_norm = nn.LayerNorm(24, eps=1e-5)
        self.output = nn.Linear(8 * 24, channels)
        nn.init.trunc_normal_(self.output.weight, std=1e-6, a=-2e-6, b=2e-6)
        nn.init.zeros_(self.output.bias)
        self.output_norm = RMSBatchNorm(channels)
        self.mlp_norm = RMSBatchNorm(channels)
        self.mlp_in = nn.Linear(channels, 2 * channels)
        self.mlp_out = nn.Linear(2 * channels, channels)
        nn.init.trunc_normal_(self.mlp_out.weight, std=1e-6, a=-2e-6, b=2e-6)
        nn.init.zeros_(self.mlp_out.bias)
        self.mlp_output_norm = RMSBatchNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        batch, length, _ = h.shape
        q = apply_rope(self.query_norm(self.query(h).reshape(batch, length, 8, 16)))
        k = apply_rope(self.key_norm(self.key(h).reshape(batch, length, 1, 16)))
        v = self.value_norm(self.value(h).reshape(batch, length, 1, 24))
        # Explicit FP32 logits/softmax, with upstream soft-cap 5.
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = q.float().transpose(1, 2) @ k.float().permute(0, 2, 3, 1) / 4
            weights = (5 * torch.tanh(logits / 5)).softmax(-1)
            attended = weights @ v.float().transpose(1, 2)
        attended = attended.transpose(1, 2).reshape(batch, length, 8 * 24).to(h.dtype)
        x = x + self.output_norm(self.output(attended))
        h = self.mlp_out(F.relu(self.mlp_in(self.mlp_norm(x))))
        return x + self.mlp_output_norm(h)


class OutputEmbedder(nn.Module):
    def __init__(self, channels: int, coarse_channels: int | None = None):
        super().__init__()
        self.linear = nn.Linear(channels, 2 * channels)
        self.skip = nn.Linear(coarse_channels, 2 * channels, bias=False) if coarse_channels else None
        self.norm = RMSBatchNorm(2 * channels)

    def forward(self, x: torch.Tensor, coarse: torch.Tensor | None = None) -> torch.Tensor:
        x = self.linear(x)
        if self.skip is not None:
            x = x + self.skip(coarse).repeat_interleave(x.shape[1] // coarse.shape[1], dim=1)
        return fast_gelu(self.norm(x))


class TrackHead(nn.Module):
    def __init__(self, channels: int, contexts: int):
        super().__init__()
        self.linear = nn.Linear(channels, contexts)
        self.learned_scale = nn.Parameter(torch.ones(contexts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.linear(x).float()) * F.softplus(self.learned_scale)


class AlphaGenomeSmall(nn.Module):
    def __init__(self, context_count=len(CONTEXTS), dropout=0., head_dropout=0.,
                 model_size=PRESET, h3k27ac_output_pool_size=4, output_scaling=None,
                 transformer_bin_bp=128):
        super().__init__()
        if (model_size != PRESET or dropout != 0 or head_dropout != 0
                or h3k27ac_output_pool_size != 4 or context_count != len(CONTEXTS)):
            raise ValueError("AlphaGenome-small requires its fixed no-dropout 8-context geometry")
        if type(transformer_bin_bp) is not int or transformer_bin_bp not in (16, 128):
            raise ValueError("Transformer spacing must be 16 or 128 bp")
        self.transformer_bin_bp = transformer_bin_bp
        if output_scaling is None or set(output_scaling) != {"atac", "h3k27ac"}:
            raise ValueError("AlphaGenome-small requires both training-only track means")
        self.output_scaling = {}
        for assay, values in output_scaling.items():
            means = torch.as_tensor(values, dtype=torch.float32)
            if means.shape != (context_count,) or not torch.isfinite(means).all() or (means <= 0).any():
                raise ValueError("Invalid output scaling means")
            self.register_buffer(f"{assay}_output_means", means)
            self.output_scaling[assay] = means.tolist()
        self.h3k27ac_output_pool_size = 4
        self.stem = nn.Conv1d(4, WIDTHS[0], 15, padding=7)
        self.stem_residual = ConvBlock(WIDTHS[0], WIDTHS[0], 5)
        self.encoder = nn.ModuleList(DownResBlock(a, b) for a, b in zip(WIDTHS[:-1], WIDTHS[1:]))
        self.transformer = nn.Sequential(*(MultiQueryBlock() for _ in range(9)))
        factor = 1 if transformer_bin_bp == 16 else 2
        self.decoder = nn.ModuleList(UpResBlock(a, b, factor) for a, b in ((192, 192), (192, 176), (176, 160)))
        self.coarse_embed = OutputEmbedder(192)
        self.atac_embed = OutputEmbedder(160, 384)
        self.h3_embed = OutputEmbedder(192, 384)
        self.atac_head = TrackHead(320, context_count)
        self.h3k27ac_head = TrackHead(384, context_count)

    def initialize_output_means(self, atac_means: np.ndarray, h3k27ac_means: np.ndarray):
        for assay, means in (("atac", atac_means), ("h3k27ac", h3k27ac_means)):
            head = getattr(self, f"{assay}_head")
            target = torch.as_tensor(means, dtype=torch.float32, device=head.linear.bias.device)
            target = soft_clip(target / getattr(self, f"{assay}_output_means"))
            target = (target / F.softplus(head.learned_scale)).clamp_min(1e-4)
            with torch.no_grad():
                head.linear.bias.copy_(target + torch.log(-torch.expm1(-target)))

    @staticmethod
    def select_target(x, mask, bin_bp, target_bp):
        grouped = mask.bool().reshape(len(x), -1, bin_bp)
        selected = grouped.all(-1)
        if ((grouped.any(-1) != selected).any()
                or (selected.sum(-1) != target_bp // bin_bp).any()):
            raise ValueError("Targets must align exactly to the assay output bins")
        return x[selected].reshape(len(x), target_bp // bin_bp, x.shape[-1])

    def forward(self, one_hot, attention_mask, atac_target_mask, h3k27ac_target_mask):
        if (one_hot.shape[1:] != (4, INPUT_BP) or attention_mask.shape != (len(one_hot), INPUT_BP)
                or not attention_mask.bool().all()
                or atac_target_mask.shape != attention_mask.shape
                or h3k27ac_target_mask.shape != attention_mask.shape):
            raise ValueError("AlphaGenome-small requires complete unpadded 2048-bp inputs and aligned masks")
        x = self.stem(one_hot).transpose(1, 2)
        x = x + self.stem_residual(x)
        skips = {1: x}
        x = x.reshape(len(x), -1, 2, x.shape[-1]).amax(2)
        for bin_bp, block in zip((2, 4, 8, 16, 32, 64), self.encoder):
            x = block(x)
            # Keys retain encoder stage identity, even when several stages now
            # refine the same 16-bp grid. This preserves all widths and weights.
            skips[bin_bp] = x
            if bin_bp < self.transformer_bin_bp:
                x = x.reshape(len(x), -1, 2, x.shape[-1]).amax(2)
        x = self.transformer(x)
        coarse = self.coarse_embed(x)
        for bin_bp, block in zip((64, 32, 16), self.decoder):
            x = block(x, skips[bin_bp])
            if bin_bp == 64:
                h3 = self.h3_embed(x, coarse)
                if self.transformer_bin_bp == 16:
                    # Keep the existing 64-bp H3 target contract. Pool embeddings,
                    # not nonlinear raw predictions, before applying the head.
                    h3 = h3.reshape(len(h3), -1, 4, h3.shape[-1]).mean(2)
        atac = self.atac_embed(x, coarse)
        scaled = (self.atac_head(self.select_target(atac, atac_target_mask, 16, 512)),
                  self.h3k27ac_head(self.select_target(h3, h3k27ac_target_mask, 64, 1536)))
        return tuple(inverse_soft_clip(value) * getattr(self, f"{assay}_output_means")
                     for assay, value in zip(("atac", "h3k27ac"), scaled))

    def architecture_metadata(self):
        metadata = {"name": ARCHITECTURE_NAME, "preset": PRESET, "upstream_commit": UPSTREAM_COMMIT,
                "input_bp": INPUT_BP, "convolution_filters": list(WIDTHS),
                "encoder_bin_bp": [1, 2, 4, 8, 16, 32, 64], "trunk_bin_bp": 128,
                "decoder_bin_bp": [64, 32, 16], "pooling": "max, factor 2",
                "normalization": "RMSBatchNorm, detached FP32 second moment, decay 0.9",
                "weight_standardization": True, "transformer_layers": 9,
                "transformer_heads": 8, "query_key_channels": 16, "value_channels": 24,
                "attention": "multi-query, normalized Q/K/V, RoPE, logits soft-cap 5",
                "pairwise_tower": False, "learned_track_scale": True,
                "h3k27ac_output_pool_size": 4, "h3k27ac_output_bin_size_bp": 64,
                "atac_output_bin_size_bp": 16, "output_scaling": self.output_scaling,
                "parameter_count": sum(p.numel() for p in self.parameters())}
        if self.transformer_bin_bp == 16:
            metadata.update(name=ARCHITECTURE_16BP_NAME, trunk_bin_bp=16, transformer_positions=128,
                            encoder_bin_bp=[1, 2, 4, 8, 16, 16, 16], decoder_bin_bp=[16, 16, 16],
                            pooling="max factor 2 until 16 bp; upper encoder/decoder refine at 16 bp",
                            h3k27ac_feature_pooling="mean of four 16-bp embeddings before track head")
        return metadata
