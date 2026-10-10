"""AlphaGenome-style profile loss used to pretrain the regressors.

Adapted from google-deepmind/alphagenome_research (model/losses.py and
model/heads.py at commit 0db53bd4352c66d1e00a049a81da373a066e6670), Apache
License 2.0; see LICENSES/AlphaGenome-Apache-2.0.txt.

Targets and predictions are scaled per context by the training-set mean of the
non-zero bins and soft-clipped above 10. The loss has a profile term (Poisson
deviance of the window total + multinomial over bins) and an auxiliary
cross-context term (Poisson deviance of the summed window means + multinomial
over the eight contexts). Our targets are bin means rather than counts, so the
bin-width factor in AlphaGenome's count scaling cancels.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..models.layers import soft_clip

EPSILON = 1e-7


def fit_nonzero_means(profiles: np.ndarray, pool_size: int = 1, chunk_size: int = 4096) -> np.ndarray:
    """Per-context mean of the non-zero (pooled) bins of the training split."""
    totals = np.zeros(profiles.shape[-1], np.float64)
    counts = np.zeros(profiles.shape[-1], np.int64)
    for start in range(0, len(profiles), chunk_size):
        chunk = np.asarray(profiles[start:start + chunk_size], np.float32)
        if not np.isfinite(chunk).all() or np.any(chunk < 0):
            raise ValueError("Training profiles must be finite and non-negative")
        if pool_size > 1:
            chunk = chunk.reshape(len(chunk), -1, pool_size, chunk.shape[-1]).mean(axis=2)
        totals += chunk.sum(axis=(0, 1), dtype=np.float64)
        counts += (chunk > 0).sum(axis=(0, 1))
    if np.any(counts == 0):
        raise ValueError("A context has no signal in the training split")
    return (totals / counts).astype(np.float32)


def poisson_deviance_half(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    predicted, observed = predicted.float(), observed.float()
    return (predicted - observed + observed * (
        torch.log(observed + EPSILON) - torch.log(predicted + EPSILON))).mean()


class AlphaGenomeProfileLoss(nn.Module):
    """Loss for one assay; inputs are ``[batch, bins, contexts]`` in signal units."""

    def __init__(self, track_means: np.ndarray, *, bins: int, positional_weight: float = 5.0,
                 cross_context_weight: float = 5.0, auxiliary_weight: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("track_means", torch.from_numpy(np.asarray(track_means, np.float32).copy()))
        self.bins = bins
        self.positional_weight = positional_weight
        self.cross_context_weight = cross_context_weight
        self.auxiliary_weight = auxiliary_weight

    def components(self, predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        predictions, labels = predictions.float(), labels.float()
        predicted = soft_clip(predictions / self.track_means)
        observed = soft_clip(labels / self.track_means)
        predicted_total = predicted.sum(dim=1, keepdim=True)
        observed_total = observed.sum(dim=1, keepdim=True)
        count = poisson_deviance_half(predicted_total, observed_total) / self.bins
        positional = -(observed * torch.log(predicted / (predicted_total + EPSILON) + EPSILON)).mean()
        predicted_context, observed_context = predicted.mean(dim=1), observed.mean(dim=1)
        predicted_across = predicted_context.sum(dim=-1, keepdim=True)
        observed_across = observed_context.sum(dim=-1, keepdim=True)
        cross_count = poisson_deviance_half(predicted_across, observed_across) / len(self.track_means)
        cross_distribution = -(observed_context * torch.log(
            predicted_context / (predicted_across + EPSILON) + EPSILON)).mean()
        profile = count + self.positional_weight * positional
        auxiliary = cross_count + self.cross_context_weight * cross_distribution
        return {"profile": profile, "auxiliary": auxiliary, "total": profile + self.auxiliary_weight * auxiliary}

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.components(predictions, labels)["total"]
