"""AlphaGenome track loss with its RNA-style cross-track term adapted to windows.

Equations follow google-deepmind/alphagenome_research model/losses.py and
model/heads.py at commit 0db53bd4352c66d1e00a049a81da373a066e6670
(inspected 2026-09-10). This is not an RNA prediction head.
Our dense targets are bin means, not summed counts: the bin-width factor in
AlphaGenome's count scaling cancels when converting both counts and means.
"""

from __future__ import annotations

import os
import socket

import numpy as np
import torch
from torch import nn

from .execution import execution_backend, require_runpod


LOSS_NAME = "alphagenome_profile_cross_context"
UPSTREAM_COMMIT = "0db53bd4352c66d1e00a049a81da373a066e6670"
EPSILON = 1e-7
SOFT_CLIP = 10.0


def require_training_node(device: str) -> None:
    if execution_backend() == "runpod":
        require_runpod()
        return
    host = socket.gethostname().split(".")[0]
    if host in {"neocranex", "nodo3", "nodo5", "nodo9", "nodo10", "nodo12"}:
        raise RuntimeError("AlphaGenome experiment refuses login/excluded nodes")
    if device != "cpu" and (
        not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_NODELIST")
    ):
        raise RuntimeError("AlphaGenome CUDA experiment requires a Slurm compute allocation")


def soft_clip(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    # Clamp only the inactive square-root branch, avoiding a 0 * infinity
    # gradient at zero without changing the transform itself.
    return torch.where(
        values > SOFT_CLIP,
        2 * torch.sqrt(values.clamp_min(SOFT_CLIP) * SOFT_CLIP) - SOFT_CLIP,
        values,
    )


def inverse_soft_clip(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return torch.where(
        values > SOFT_CLIP,
        (values + SOFT_CLIP).square() / (4 * SOFT_CLIP),
        values,
    )


def fit_nonzero_means(profiles: np.ndarray, pool_size: int = 1, chunk_size: int = 4096) -> np.ndarray:
    """Fit once on TRAIN bins, after the same mean pooling as the target loader."""
    if (profiles.ndim != 3 or not len(profiles) or pool_size < 1
            or profiles.shape[1] % pool_size or chunk_size < 1):
        raise ValueError("Nonzero means require nonempty, poolable profile arrays")
    totals = np.zeros(profiles.shape[-1], np.float64)
    counts = np.zeros(profiles.shape[-1], np.int64)
    for start in range(0, len(profiles), chunk_size):
        chunk = np.asarray(profiles[start:start + chunk_size], np.float32)
        if not np.isfinite(chunk).all() or np.any(chunk < 0):
            raise ValueError("Training profiles must be finite and nonnegative")
        if pool_size > 1:
            chunk = chunk.reshape(len(chunk), -1, pool_size, chunk.shape[-1]).mean(axis=2)
        totals += chunk.sum(axis=(0, 1), dtype=np.float64)
        counts += (chunk > 0).sum(axis=(0, 1))
    if np.any(counts == 0):
        raise ValueError("Cannot fit a nonzero mean to an all-zero track")
    means = (totals / counts).astype(np.float32)
    if not np.isfinite(means).all() or np.any(means <= 0):
        raise ValueError("Invalid training nonzero means")
    return means


def poisson_deviance_half(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Poisson NLL minus its target-dependent minimum, as in AlphaGenome."""
    predicted, observed = predicted.float(), observed.float()
    return (predicted - observed + observed * (
        torch.log(observed + EPSILON) - torch.log(predicted + EPSILON)
    )).mean()


class AlphaGenomeProfileLoss(nn.Module):
    """Single target-window segment, with all eight unstranded tracks present.

Inputs/outputs of the public model stay in experimental signal units. The
model internally predicts scaled values and applies inverse_soft_clip; this
loss maps back into that space. RC ensembling therefore remains in raw units.
"""

    def __init__(self, track_means: np.ndarray, *, bins: int,
                 positional_weight: float = 5.0, cross_context_weight: float = 5.0,
                 auxiliary_weight: float = 0.1) -> None:
        super().__init__()
        means = np.asarray(track_means, np.float32)
        if means.ndim != 1 or not np.isfinite(means).all() or np.any(means <= 0):
            raise ValueError("Track means must be finite positive vectors")
        if bins < 1 or any(not np.isfinite(w) or w < 0 for w in (
            positional_weight, cross_context_weight, auxiliary_weight
        )):
            raise ValueError("Invalid loss weights or number of bins")
        self.register_buffer("track_means", torch.from_numpy(means.copy()))
        self.bins = bins
        self.positional_weight = positional_weight
        self.cross_context_weight = cross_context_weight
        self.auxiliary_weight = auxiliary_weight

    def components(self, predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        if (predictions.shape != labels.shape or predictions.ndim != 3
                or predictions.shape[1:] != (self.bins, len(self.track_means))):
            raise ValueError("AlphaGenome loss expects aligned [batch, bins, contexts]")
        predictions, labels = predictions.float(), labels.float()
        predicted = soft_clip(predictions / self.track_means)
        observed = soft_clip(labels / self.track_means)
        predicted_total = predicted.sum(dim=1, keepdim=True)
        observed_total = observed.sum(dim=1, keepdim=True)
        count = poisson_deviance_half(predicted_total, observed_total) / self.bins
        positional = -(observed * torch.log(
            predicted / (predicted_total + EPSILON) + EPSILON
        )).mean()

        # RNA's length-normalized gene aggregation becomes an assay-window
        # mean. No strand filtering is needed for these unstranded tracks.
        predicted_context = predicted.mean(dim=1)
        observed_context = observed.mean(dim=1)
        predicted_across = predicted_context.sum(dim=-1, keepdim=True)
        observed_across = observed_context.sum(dim=-1, keepdim=True)
        cross_count = poisson_deviance_half(predicted_across, observed_across) / len(self.track_means)
        cross_distribution = -(observed_context * torch.log(
            predicted_context / (predicted_across + EPSILON) + EPSILON
        )).mean()
        profile = count + self.positional_weight * positional
        auxiliary = cross_count + self.cross_context_weight * cross_distribution
        return {
            "profile_count": count, "profile_positional": positional,
            "cross_context_count": cross_count,
            "cross_context_distribution": cross_distribution,
            "profile": profile, "auxiliary": auxiliary,
            "mse": (torch.log1p(predictions) - torch.log1p(labels)).square().mean(),
            "total": profile + self.auxiliary_weight * auxiliary,
        }

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.components(predictions, labels)["total"]
