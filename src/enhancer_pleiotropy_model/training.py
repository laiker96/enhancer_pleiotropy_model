"""Train the production joint ATAC/H3K27ac profile regressor."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
import yaml

from .alphagenome_loss import (
    LOSS_NAME as ALPHAGENOME_LOSS_NAME,
    UPSTREAM_COMMIT,
    AlphaGenomeProfileLoss,
    fit_nonzero_means,
    require_training_node,
    soft_clip,
)
from .alphagenome_model import AlphaGenomeSmall, PRESET as ALPHAGENOME_PRESET
from .alphagenome_schedule import AlphaGenomeSchedule, SCHEDULE_NAME
from .continuation_schedule import AlphaGenomeContinuationSchedule, CONTINUATION_SCHEDULE_NAME
from .two_stage_schedule import AlphaGenomeTwoStageSchedule, SCHEDULE_NAME as TWO_STAGE_SCHEDULE_NAME
from .constants import (
    ASSAYS,
    ATAC_TARGET_BP,
    CONTEXTS,
    H3K27AC_OUTPUT_POOL_SIZE,
    H3K27AC_TARGET_BP,
    INPUT_BP,
    SOURCE_BIN_BP,
)
from .data import (
    JointProfileDataset,
    WindowRecord,
    load_profiles,
    make_loader,
    read_windows,
    streamed_h3_log_statistics,
    streamed_profile_means,
)
from .inference import resolve_device
from .io import atomic_write_json, sha256_file
from .metrics import (
    assay_validation_metrics,
    h3k27ac_segment_metrics,
    regulatory_overcorrelation_summary,
    scientific_composite,
)
from .model import EnformerLikeJointProfileRegressor, MODEL_PRESETS
from .preprocessing.windows import (
    H3K27AC_PEAK_SOURCE,
    JOINT_PEAK_SOURCE,
    PEAK_SOURCE,
)


ALPHAGENOME_SMOKE_MAX_BATCHES = 32
CRESTED_TARGET_SCALING = "alphagenome_nonzero_mean_softclip"


def crested_target_scaling_enabled(training: dict[str, Any]) -> bool:
    scaling = training.get("target_scaling")
    if scaling is None:
        return False
    if (scaling != CRESTED_TARGET_SCALING
            or training.get("loss", {}).get("name") != "crested_cosine_mse_log_both"):
        raise ValueError("Explicit target_scaling requires CREsted loss and alphagenome_nonzero_mean_softclip")
    return True


class StandardizedLog1pHuberLoss(nn.Module):
    def __init__(self, means: np.ndarray, standard_deviations: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float32))
        self.register_buffer(
            "standard_deviations",
            torch.as_tensor(standard_deviations, dtype=torch.float32),
        )

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        predicted = (torch.log1p(predictions.float().clamp_min(0)) - self.means) / self.standard_deviations
        target = (torch.log1p(labels.float().clamp_min(0)) - self.means) / self.standard_deviations
        return torch.nn.functional.smooth_l1_loss(predicted, target, beta=1.0)


class CrestedCosineMSELogLoss(nn.Module):
    """PyTorch implementation of CREsted's CosineMSELogLoss.

    Dense profiles are shaped ``[batch, bins, contexts]``. The logarithmic MSE
    is reduced over all elements, while cosine similarity is calculated across
    the context axis for every genomic bin and then averaged.
    """

    def __init__(
        self,
        *,
        max_weight: float = 100.0,
        multiplier: float = 1.0,
        minimum_target_norm: float = 0.0,
        track_means: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if max_weight < 1:
            raise ValueError("CREsted max_weight must be at least 1")
        if multiplier <= 0:
            raise ValueError("CREsted multiplier must be positive")
        if minimum_target_norm < 0:
            raise ValueError("CREsted minimum_target_norm cannot be negative")
        self.max_weight = float(max_weight)
        self.multiplier = float(multiplier)
        self.minimum_target_norm = float(minimum_target_norm)
        if track_means is not None:
            means = np.asarray(track_means, dtype=np.float32)
            if means.ndim != 1 or not len(means) or not np.isfinite(means).all() or np.any(means <= 0):
                raise ValueError("CREsted track means must be finite positive vectors")
            self.register_buffer("track_means", torch.from_numpy(means.copy()))
        else:
            self.track_means = None

    def components(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if predictions.shape != labels.shape or predictions.ndim != 3:
            raise ValueError("CREsted loss expects aligned [batch, bins, contexts]")
        predictions = predictions.float()
        labels = labels.float()
        if self.track_means is not None:
            if predictions.shape[-1] != len(self.track_means):
                raise ValueError("CREsted context count differs from track means")
            predictions = soft_clip(predictions / self.track_means)
            labels = soft_clip(labels / self.track_means)
        transformed_predictions = torch.sign(predictions) * torch.log1p(
            self.multiplier * predictions.abs()
        )
        transformed_labels = torch.log1p(self.multiplier * labels)
        mse = torch.mean(torch.square(transformed_predictions - transformed_labels))
        cosine_weight = mse.abs().clamp(1.0, self.max_weight)
        normalized_predictions = torch.nn.functional.normalize(
            predictions, dim=-1
        )
        normalized_labels = torch.nn.functional.normalize(labels, dim=-1)
        cosine_similarity = torch.sum(
            normalized_predictions * normalized_labels, dim=-1
        )
        if self.minimum_target_norm > 0:
            eligible = (
                torch.linalg.vector_norm(labels, dim=-1)
                > self.minimum_target_norm
            )
            mean_cosine = (
                cosine_similarity[eligible].mean()
                if eligible.any()
                else cosine_similarity.new_zeros(())
            )
        else:
            mean_cosine = cosine_similarity.mean()
        total = mse - cosine_weight * mean_cosine
        return {
            "mse": mse,
            "cosine_similarity": mean_cosine,
            "cosine_weight": cosine_weight,
            "total": total,
        }

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.components(predictions, labels)["total"]


def build_loss_criteria(
    training: dict[str, Any],
    h3_means: np.ndarray,
    h3_standard_deviations: np.ndarray,
    device: torch.device,
    *,
    track_means: dict[str, np.ndarray] | None = None,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    loss_config = dict(training.get("loss", {}))
    scaled_crested = crested_target_scaling_enabled(training)
    if scaled_crested and (track_means is None or set(track_means) != set(ASSAYS)):
        raise ValueError("Scaled CREsted loss requires training-only nonzero means")
    name = loss_config.get(
        "name", "poisson_atac_standardized_log1p_huber_h3k27ac"
    )
    if name == ALPHAGENOME_LOSS_NAME:
        if track_means is None or set(track_means) != set(ASSAYS):
            raise ValueError("AlphaGenome loss requires training-only nonzero means")
        weights = {key: float(loss_config[key]) for key in (
            "positional_weight", "cross_context_weight", "auxiliary_weight"
        )}
        criteria = [AlphaGenomeProfileLoss(track_means[assay], bins=bins, **weights).to(device)
                    for assay, bins in (("atac", 32), ("h3k27ac", 24))]
        return (*criteria, {
            "name": ALPHAGENOME_LOSS_NAME, "upstream_commit": UPSTREAM_COMMIT, **weights,
            "fit_split": "train", "nonzero_means": {a: m.tolist() for a, m in track_means.items()},
            "scaling": "bin-mean / training nonzero bin-mean; soft clip above 10; no RNA power transform",
            "segments": {"atac": "one 512-bp segment", "h3k27ac": "one 1536-bp segment"},
            "auxiliary": "RNA-style total + distribution across contexts of scaled window means; no RNA labels",
            "assay_reduction": "unweighted sum", "output_units": "original background-TMM bin means",
        })
    if name == "poisson_atac_standardized_log1p_huber_h3k27ac":
        return (
            nn.PoissonNLLLoss(log_input=False, full=False, eps=1e-8),
            StandardizedLog1pHuberLoss(
                h3_means, h3_standard_deviations
            ).to(device),
            {
                "name": "ATAC raw Poisson NLL plus H3K27ac train-standardized log1p SmoothL1",
                "main_loss": "poisson",
                "h3k27ac_main_loss": "standardized_log1p_huber",
                "h3k27ac_target_standardization": {
                    "transform": "log1p then per-context mean/std standardization",
                    "fit_split": "train",
                    "means": h3_means.tolist(),
                    "standard_deviations": h3_standard_deviations.tolist(),
                },
            },
        )
    if name != "crested_cosine_mse_log_both":
        raise ValueError(f"Unsupported training loss: {name}")
    max_weight = float(loss_config.get("max_weight", 100.0))
    minimum_target_norm = float(loss_config.get("minimum_target_norm", 0.0))
    multipliers = dict(loss_config.get("multipliers", {}))
    expected = set(ASSAYS)
    if set(multipliers) != expected:
        raise ValueError(f"CREsted multipliers must be provided for {sorted(expected)}")
    criteria = {
        assay: CrestedCosineMSELogLoss(
            max_weight=max_weight,
            multiplier=float(multipliers[assay]),
            minimum_target_norm=minimum_target_norm,
            track_means=track_means[assay] if scaled_crested else None,
        ).to(device)
        for assay in ASSAYS
    }
    metadata = {
        "name": "CREsted CosineMSELogLoss applied independently to ATAC and H3K27ac",
        "main_loss": "crested_cosine_mse_log",
        "h3k27ac_main_loss": "crested_cosine_mse_log",
        "implementation": "PyTorch port of aertslab/CREsted CosineMSELogLoss",
        "assay_reduction": "unweighted sum of independently reduced assay losses",
        "context_axis": -1,
        "max_weight": max_weight,
        "minimum_target_norm": minimum_target_norm,
        "multipliers": {assay: float(multipliers[assay]) for assay in ASSAYS},
        "mse": "global mean squared error after signed log1p(multiplier * signal)",
        "cosine": "negative mean raw-signal cosine similarity across contexts per genomic bin",
        "dynamic_weight": "clamp(absolute log-MSE, 1, max_weight)",
    }
    if scaled_crested:
        metadata.update(
            target_scaling=CRESTED_TARGET_SCALING,
            fit_split="train",
            nonzero_means={a: m.tolist() for a, m in track_means.items()},
            scaling="bin-mean / training nonzero bin-mean; soft clip above 10; no RNA power transform",
            mse="global log1p MSE of scaled predictions and targets",
            cosine="negative mean scaled-signal cosine similarity across contexts per genomic bin",
            output_units="original background-TMM bin means",
        )
    return criteria["atac"], criteria["h3k27ac"], metadata


def context_gini(activity: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Calculate the Gini index across contexts for each genomic window."""
    values = np.asarray(activity, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Gini activity must be [windows, contexts]")
    if np.any(values < 0) or np.any(~np.isfinite(values)):
        raise ValueError("Gini activity must be finite and nonnegative")
    ordered = np.sort(values, axis=1)
    context_count = ordered.shape[1]
    coefficients = 2 * np.arange(1, context_count + 1) - context_count - 1
    totals = ordered.sum(axis=1)
    numerator = (ordered * coefficients).sum(axis=1)
    return np.divide(
        numerator,
        context_count * totals,
        out=np.zeros_like(totals),
        where=totals > epsilon,
    )


def peak_specificity_scores(
    records: list[WindowRecord],
    profiles: np.ndarray,
    eligible_sources: frozenset[str],
    *,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-eligible indices and Gini scores of window-mean profiles."""
    indices = np.asarray(
        [index for index, record in enumerate(records) if record.source in eligible_sources],
        dtype=np.int64,
    )
    scores = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), chunk_size):
        chunk_indices = indices[start : start + chunk_size]
        chunk = np.asarray(profiles[chunk_indices], dtype=np.float32)
        scores[start : start + len(chunk_indices)] = context_gini(
            chunk.mean(axis=1, dtype=np.float64)
        )
    return indices, scores


def fit_specificity_thresholds(
    records: list[WindowRecord],
    atac_profiles: np.ndarray,
    h3_profiles: np.ndarray,
    standard_deviation_multiplier: float,
) -> dict[str, dict[str, float]]:
    """Fit CREsted-style Gini thresholds using training peaks only."""
    if standard_deviation_multiplier < 0:
        raise ValueError("Gini standard-deviation multiplier cannot be negative")
    assay_inputs = {
        "atac": (
            atac_profiles,
            frozenset((PEAK_SOURCE, JOINT_PEAK_SOURCE)),
        ),
        "h3k27ac": (
            h3_profiles,
            frozenset((H3K27AC_PEAK_SOURCE, JOINT_PEAK_SOURCE)),
        ),
    }
    thresholds: dict[str, dict[str, float]] = {}
    for assay, (profiles, eligible_sources) in assay_inputs.items():
        _, scores = peak_specificity_scores(records, profiles, eligible_sources)
        if len(scores) < 2:
            raise ValueError(f"Not enough {assay} peak windows to fit specificity")
        mean = float(scores.mean())
        standard_deviation = float(scores.std())
        thresholds[assay] = {
            "mean": mean,
            "standard_deviation": standard_deviation,
            "standard_deviation_multiplier": standard_deviation_multiplier,
            "threshold": mean + standard_deviation_multiplier * standard_deviation,
            "eligible_windows": int(len(scores)),
        }
    return thresholds


def select_specific_peak_indices(
    records: list[WindowRecord],
    atac_profiles: np.ndarray,
    h3_profiles: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Select peak windows specific in ATAC or H3K27ac using fixed thresholds."""
    selected_indices, _dominant_contexts, counts = (
        select_specific_peak_indices_and_contexts(
            records, atac_profiles, h3_profiles, thresholds
        )
    )
    return selected_indices, counts


def select_specific_peak_indices_and_contexts(
    records: list[WindowRecord],
    atac_profiles: np.ndarray,
    h3_profiles: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    *,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Select specific peaks and assign each to one dominant target context.

    A peak passing both assay thresholds is assigned using the assay with the
    larger threshold-normalized Gini excess. Its dominant context is the
    largest window-mean target in that assay.
    """
    assay_inputs = {
        "atac": (
            atac_profiles,
            frozenset((PEAK_SOURCE, JOINT_PEAK_SOURCE)),
        ),
        "h3k27ac": (
            h3_profiles,
            frozenset((H3K27AC_PEAK_SOURCE, JOINT_PEAK_SOURCE)),
        ),
    }
    best_margin = np.full(len(records), -np.inf, dtype=np.float64)
    dominant_context = np.full(len(records), -1, dtype=np.int64)
    counts: dict[str, int] = {}
    for assay, (profiles, eligible_sources) in assay_inputs.items():
        indices, scores = peak_specificity_scores(records, profiles, eligible_sources)
        threshold = float(thresholds[assay]["threshold"])
        selected_mask = scores > threshold
        assay_selected = indices[selected_mask]
        assay_scores = scores[selected_mask]
        counts[f"{assay}_specific"] = int(len(assay_selected))
        scale = max(float(thresholds[assay]["standard_deviation"]), 1e-12)
        margins = (assay_scores - threshold) / scale
        for start in range(0, len(assay_selected), chunk_size):
            end = min(start + chunk_size, len(assay_selected))
            chunk_indices = assay_selected[start:end]
            chunk_margins = margins[start:end]
            window_means = np.asarray(
                profiles[chunk_indices], dtype=np.float32
            ).mean(axis=1, dtype=np.float64)
            chunk_contexts = window_means.argmax(axis=1)
            replace = chunk_margins > best_margin[chunk_indices]
            replaced_indices = chunk_indices[replace]
            best_margin[replaced_indices] = chunk_margins[replace]
            dominant_context[replaced_indices] = chunk_contexts[replace]
    selected_indices = np.flatnonzero(np.isfinite(best_margin))
    if not len(selected_indices):
        raise ValueError("No peak windows passed the specificity thresholds")
    counts["union_specific"] = int(len(selected_indices))
    assignments = dominant_context[selected_indices]
    if np.any(assignments < 0):
        raise RuntimeError("A selected specificity peak lacks a dominant context")
    return selected_indices, assignments, counts


def balance_specific_peak_contexts(
    indices: np.ndarray,
    dominant_contexts: np.ndarray,
    context_count: int,
    seed: int,
    maximum_oversampling_factor: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return deterministic equal-sized dominant-context groups.

    The group size is the smaller of the median group count and the rarest
    group count times ``maximum_oversampling_factor``. Common groups are
    sampled without replacement; rare groups retain every unique example and
    add deterministic replacement draws only when needed.
    """
    indices = np.asarray(indices, dtype=np.int64)
    dominant_contexts = np.asarray(dominant_contexts, dtype=np.int64)
    if indices.ndim != 1 or dominant_contexts.shape != indices.shape:
        raise ValueError("Specificity indices and dominant contexts must align")
    if context_count < 2 or maximum_oversampling_factor < 1:
        raise ValueError("Context count and oversampling factor are invalid")
    if np.any((dominant_contexts < 0) | (dominant_contexts >= context_count)):
        raise ValueError("Dominant context index is outside the configured contexts")
    counts = np.bincount(dominant_contexts, minlength=context_count)
    if np.any(counts == 0):
        raise ValueError("Cannot balance specificity data with an empty context")
    median_count = int(np.median(counts))
    oversampling_cap = max(
        1, int(np.floor(counts.min() * maximum_oversampling_factor))
    )
    target_count = min(median_count, oversampling_cap)
    rng = np.random.default_rng(seed)
    balanced_groups: list[np.ndarray] = []
    unique_retained: list[int] = []
    for context_index in range(context_count):
        group = indices[dominant_contexts == context_index]
        if len(group) >= target_count:
            chosen = rng.choice(group, size=target_count, replace=False)
            unique_retained.append(target_count)
        else:
            additional = rng.choice(
                group, size=target_count - len(group), replace=True
            )
            chosen = np.concatenate((group, additional))
            unique_retained.append(len(group))
        balanced_groups.append(chosen.astype(np.int64, copy=False))
    balanced = np.concatenate(balanced_groups)
    rng.shuffle(balanced)
    return balanced, {
        "method": "dominant_context_equal_resampling",
        "target_examples_per_context": target_count,
        "maximum_oversampling_factor": maximum_oversampling_factor,
        "before_counts": counts.astype(int).tolist(),
        "after_counts": [target_count] * context_count,
        "unique_retained_counts": unique_retained,
        "examples": int(len(balanced)),
        "unique_examples": int(len(np.unique(balanced))),
    }


class WarmupPlateauScheduler:
    """Linear warmup, cosine transition, then validation-driven reductions."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        maximum_learning_rate: float,
        post_warmup_learning_rate: float,
        warmup_steps: int,
        decay_steps: int,
        plateau_factor: float,
        plateau_patience: int,
        plateau_threshold: float,
        minimum_learning_rate: float,
    ) -> None:
        self.optimizer = optimizer
        self.maximum_learning_rate = maximum_learning_rate
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.scheduled_steps = warmup_steps + decay_steps
        self.optimizer_steps = 0
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            threshold_mode="rel",
            min_lr=minimum_learning_rate,
        )
        self._set_learning_rate(self._learning_rate_for_step(1))

    def _set_learning_rate(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def _learning_rate_for_step(self, step: int) -> float:
        if step <= self.warmup_steps:
            return self.maximum_learning_rate * step / self.warmup_steps
        if self.decay_steps and step <= self.scheduled_steps:
            progress = (step - self.warmup_steps) / self.decay_steps
            return self.post_warmup_learning_rate + 0.5 * (
                self.maximum_learning_rate - self.post_warmup_learning_rate
            ) * (1.0 + math.cos(math.pi * progress))
        return self.post_warmup_learning_rate

    def step(self) -> None:
        self.optimizer_steps += 1
        next_step = self.optimizer_steps + 1
        if next_step <= self.scheduled_steps:
            self._set_learning_rate(self._learning_rate_for_step(next_step))
        elif self.optimizer_steps == self.scheduled_steps:
            self._set_learning_rate(self.post_warmup_learning_rate)

    def step_validation(self, score: float) -> dict[str, Any]:
        before = float(self.optimizer.param_groups[0]["lr"])
        eligible = self.optimizer_steps >= self.scheduled_steps
        if eligible:
            self.plateau.step(score)
        after = float(self.optimizer.param_groups[0]["lr"])
        return {
            "score": score,
            "eligible_after_scheduled_decay": eligible,
            "learning_rate_before": before,
            "learning_rate_after": after,
            "reduced": after < before,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "maximum_learning_rate": self.maximum_learning_rate,
            "post_warmup_learning_rate": self.post_warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "optimizer_steps": self.optimizer_steps,
            "plateau": self.plateau.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (
            self.maximum_learning_rate,
            self.post_warmup_learning_rate,
            self.warmup_steps,
            self.decay_steps,
        )
        observed = (
            float(state["maximum_learning_rate"]),
            float(state["post_warmup_learning_rate"]),
            int(state["warmup_steps"]),
            int(state["decay_steps"]),
        )
        if observed != expected:
            raise ValueError("Learning-rate scheduler configuration changed")
        self.optimizer_steps = int(state["optimizer_steps"])
        self.plateau.load_state_dict(state["plateau"])


def seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_tensors_to_cpu(value: Any) -> Any:
    """Copy tensors in a nested checkpoint value to CPU before serialization."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {
            key: checkpoint_tensors_to_cpu(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [checkpoint_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_tensors_to_cpu(item) for item in value)
    return value


def autocast_context(device: torch.device, mixed_precision: str):
    if mixed_precision == "no":
        return nullcontext()
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, ...]:
    names = (
        "one_hot",
        "attention_mask",
        "atac_target_mask",
        "h3k27ac_target_mask",
        "atac_labels",
        "h3k27ac_labels",
    )
    return tuple(batch[name].to(device, non_blocking=True) for name in names)


def reverse_complement_batch(
    one_hot: torch.Tensor,
    attention_mask: torch.Tensor,
    atac_mask: torch.Tensor,
    h3_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        one_hot.flip((1, 2)),
        attention_mask.flip(1),
        atac_mask.flip(1),
        h3_mask.flip(1),
    )


def calculate_losses(
    predictions: tuple[torch.Tensor, torch.Tensor],
    labels: tuple[torch.Tensor, torch.Tensor],
    atac_criterion: nn.Module,
    h3_criterion: nn.Module,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    assay_totals = []
    for assay, prediction, target, criterion in zip(
        ASSAYS,
        predictions,
        labels,
        (atac_criterion, h3_criterion),
        strict=True,
    ):
        if isinstance(criterion, CrestedCosineMSELogLoss):
            components = criterion.components(prediction.float(), target.float())
            assay_total = components["total"]
            for name in ("mse", "cosine_similarity", "cosine_weight"):
                losses[f"{assay}_{name}"] = components[name]
        elif isinstance(criterion, AlphaGenomeProfileLoss):
            components = criterion.components(prediction, target)
            assay_total = components["total"]
            losses.update({f"{assay}_{name}": value for name, value in components.items() if name != "total"})
        else:
            assay_total = criterion(prediction.float(), target.float())
        losses[assay] = assay_total
        assay_totals.append(assay_total)
    losses["total"] = sum(assay_totals)
    return losses


def alphagenome_optimizer_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupPlateauScheduler,
    scaler: torch.amp.GradScaler,
    gradient_clip_norm: float,
) -> dict[str, Any]:
    """Allow AMP's overflow backoff, but never advance LR on a skipped update."""
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite AlphaGenome loss; stopping before backward")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    try:
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip_norm, error_if_nonfinite=True
        )
    except RuntimeError:
        # Do not confuse overflow of a norm of finite gradients with AMP's
        # recorded non-finite gradients. error_if_nonfinite leaves grads intact.
        bad_gradient = any(p.grad is not None and not torch.isfinite(p.grad).all()
                           for p in model.parameters())
        if not bad_gradient:
            raise FloatingPointError("Non-finite AlphaGenome gradient norm or clipping failure")
        if not scaler.is_enabled():
            raise FloatingPointError("Non-finite AlphaGenome gradients without AMP scaling")
        norm = None
    before = scaler.get_scale()
    # unscale_ already recorded non-finite gradients. In that case GradScaler
    # skips optimizer.step and update halves its scale (standard AMP behavior).
    scaler.step(optimizer)
    scaler.update()
    after = scaler.get_scale()
    skipped = after < before
    if (not math.isfinite(after) or after <= 0 or (norm is None and not skipped)):
        raise FloatingPointError("AlphaGenome AMP did not safely back off its loss scale")
    if not skipped:
        scheduler.step()
    return {"gradient_norm": float(norm) if norm is not None else None,
            "optimizer_step_skipped": skipped,
            "loss_scale_before": before, "loss_scale_after": after,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer_steps": scheduler.optimizer_steps}


@torch.no_grad()
def evaluate(
    model: EnformerLikeJointProfileRegressor,
    loader,
    atac_criterion: nn.Module,
    h3_criterion: nn.Module,
    device: torch.device,
    mixed_precision: str,
    rc_ensemble: bool,
    maximum_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray]]:
    model.eval()
    loss_sums: dict[str, float] = {}
    count = 0
    labels_all = {assay: [] for assay in ASSAYS}
    predictions_all = {assay: [] for assay in ASSAYS}
    for batch_index, batch in enumerate(loader, start=1):
        one_hot, attention_mask, atac_mask, h3_mask, atac_labels, h3_labels = move_batch(
            batch, device
        )
        with autocast_context(device, mixed_precision):
            predictions = model(one_hot, attention_mask, atac_mask, h3_mask)
            if rc_ensemble:
                rc_inputs = reverse_complement_batch(
                    one_hot, attention_mask, atac_mask, h3_mask
                )
                rc_predictions = model(*rc_inputs)
                predictions = tuple(
                    0.5 * (forward + reverse.flip(1))
                    for forward, reverse in zip(
                        predictions, rc_predictions, strict=True
                    )
                )
            losses = calculate_losses(
                predictions,
                (atac_labels, h3_labels),
                atac_criterion,
                h3_criterion,
            )
        batch_count = len(atac_labels)
        count += batch_count
        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + (
                float(value.item()) * batch_count
            )
        for assay, target, prediction in zip(
            ASSAYS,
            (atac_labels, h3_labels),
            predictions,
            strict=True,
        ):
            labels_all[assay].append(target.cpu().numpy())
            predictions_all[assay].append(prediction.float().cpu().numpy())
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
    if count == 0:
        raise ValueError("Validation loader was empty")
    return (
        {name: value / count for name, value in loss_sums.items()},
        {assay: np.concatenate(values) for assay, values in labels_all.items()},
        {assay: np.concatenate(values) for assay, values in predictions_all.items()},
    )


def validation_metric_sets(
    labels: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    regulatory_mask: np.ndarray,
    contexts: tuple[str, ...],
    specificity_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Calculate full and optional specificity-subset validation metrics."""
    full_metrics = {
        assay: assay_validation_metrics(
            labels[assay],
            predictions[assay],
            regulatory_mask[: len(labels[assay])],
            contexts,
        )
        for assay in ASSAYS
    }
    full_metrics["h3k27ac"]["segments"] = h3k27ac_segment_metrics(
        labels["h3k27ac"], predictions["h3k27ac"], contexts
    )
    if specificity_mask is None:
        return full_metrics, None

    selected_mask = specificity_mask[: len(labels["atac"])]
    if not selected_mask.any():
        raise ValueError("No specific peaks were evaluated in validation")
    specificity_metrics = {
        assay: assay_validation_metrics(
            labels[assay][selected_mask],
            predictions[assay][selected_mask],
            np.ones(int(selected_mask.sum()), dtype=np.bool_),
            contexts,
        )
        for assay in ASSAYS
    }
    specificity_metrics["h3k27ac"]["segments"] = h3k27ac_segment_metrics(
        labels["h3k27ac"][selected_mask],
        predictions["h3k27ac"][selected_mask],
        contexts,
    )
    return full_metrics, specificity_metrics


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    if tuple(config["contexts"]) != CONTEXTS:
        raise ValueError(f"Production context order must be {CONTEXTS}")
    if config["model"]["preset"] not in {*MODEL_PRESETS, ALPHAGENOME_PRESET}:
        raise ValueError("Unsupported model preset")
    return config


def architecture_metadata(
    model: EnformerLikeJointProfileRegressor,
    contexts: tuple[str, ...],
) -> dict[str, Any]:
    if isinstance(model, AlphaGenomeSmall):
        return model.architecture_metadata()
    return {
        "name": "enformer_like_dense_atac_h3k27ac_profile_regressor_v1",
        "input_encoding": "one_hot_ACGT",
        "convolution_filters": list(model.convolution_filters),
        "convolution_kernels": list(model.convolution_kernels),
        "convolution_blocks": len(model.convolution_filters),
        "pooling": "learned per-channel softmax pooling, size 2 after every block",
        "downsampling_factor": 2 ** len(model.convolution_filters),
        "transformer_layers": model.transformer_layers,
        "transformer_dimension": model.convolution_filters[-1],
        "transformer_heads": model.transformer_heads,
        "transformer_feedforward_dimension": model.transformer_feedforward_dimension,
        "relative_position_max_distance_bins": 128,
        "heads": {
            assay: (
                f"LayerNorm-Linear({2 * model.convolution_filters[-1]})-GELU-"
                f"Dropout-Linear({len(contexts)})-Softplus"
            )
            for assay in ASSAYS
        },
        "h3k27ac_decoder": {
            "layers": 0,
            "dilations_bins": [],
            "kernel_size_bins": 3,
            "receptive_field_bins": 1,
            "identity_initialized": True,
        },
        "h3k27ac_output_pool_size": model.h3k27ac_output_pool_size,
        "h3k27ac_output_bin_size_bp": 16 * model.h3k27ac_output_pool_size,
        "h3k27ac_atac_cross_attention": {
            "enabled": False,
            "direction": "H3K27ac queries; ATAC keys and values",
            "latent_positions": "full encoder output before target cropping",
            "heads": 0,
            "residual_gate": "not present",
            "assay_projections": "not present",
        },
        "parameter_count": sum(value.numel() for value in model.parameters()),
        **({"output_scaling": model.output_scaling} if model.output_scaling is not None else {}),
    }


def save_training_state(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupPlateauScheduler,
    scaler: torch.amp.GradScaler,
    run_signature: dict[str, Any],
    progress: dict[str, Any],
    device: torch.device,
) -> None:
    checkpoint = {
        "kind": "enhancer_pleiotropy_training_state",
        "version": 1,
        "run_signature": run_signature,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(device),
        "progress": progress,
    }
    atomic_torch_save(checkpoint_tensors_to_cpu(checkpoint), path)


def load_training_state(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupPlateauScheduler,
    scaler: torch.amp.GradScaler,
    run_signature: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "enhancer_pleiotropy_training_state":
        raise ValueError(f"{path}: not a training-state checkpoint")
    if checkpoint.get("run_signature") != run_signature:
        raise ValueError("Resume configuration or input hashes changed")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint["rng_state"], device)
    return dict(checkpoint["progress"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after-epoch", type=int, help="Pause after this complete epoch without shortening the LR schedule")
    parser.add_argument("--pause-after-step", type=int, help="Validate and pause at this absolute successful-update count; keep the LR schedule and epoch history unchanged")
    parser.add_argument("--stop-file", type=Path, help="Checkpoint and pause at the next batch/epoch boundary when this file exists")
    parser.add_argument(
        "--stage",
        choices=("base", "specificity"),
        default="base",
        help="Train the broad base model or fine-tune it on specific peaks.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--evaluate-initialization-only",
        action="store_true",
        help=(
            "Evaluate the configured specificity initialization checkpoint on "
            "the full and balanced specificity validation sets without training."
        ),
    )
    parser.add_argument(
        "--evaluation-checkpoint",
        type=Path,
        help="Override the specificity initialization checkpoint for evaluation only.",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        help="Write initialization-only evaluation JSON to this path.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = dict(config["training"])
    stage = args.stage
    if args.evaluate_initialization_only and stage != "specificity":
        raise ValueError("Initialization-only evaluation requires --stage specificity")
    if (args.evaluation_checkpoint or args.evaluation_output) and not args.evaluate_initialization_only:
        raise ValueError("Evaluation checkpoint/output overrides require --evaluate-initialization-only")
    specificity_config = dict(config.get("specificity_finetuning", {}))
    if stage == "specificity":
        if not specificity_config.get("enabled", False):
            raise ValueError("Specificity fine-tuning is not enabled in the config")
        training.update(dict(specificity_config["training"]))
    if args.device is not None:
        training["device"] = args.device
    if args.mixed_precision is not None:
        training["mixed_precision"] = args.mixed_precision
    alphagenome_experiment = training.get("loss", {}).get("name") == ALPHAGENOME_LOSS_NAME
    scaled_crested = crested_target_scaling_enabled(training)
    continuation = training.get("schedule", {}).get("name") == CONTINUATION_SCHEDULE_NAME
    epoch_schedule = training.get("schedule", {}).get("name") == TWO_STAGE_SCHEDULE_NAME
    ag_4x_legacy_schedule = (alphagenome_experiment and config["model"]["preset"] == "4x"
                             and not training.get("schedule"))
    if args.pause_after_epoch is not None or args.pause_after_step is not None or args.stop_file is not None:
        if not (epoch_schedule or scaled_crested or ag_4x_legacy_schedule) or args.smoke_test or stage != "base":
            raise ValueError("Operational pausing requires non-smoke two-stage or scaled 4x base training")
        if args.pause_after_step is not None and not epoch_schedule:
            raise ValueError("Step-based pausing requires the two-stage schedule")
        if args.pause_after_epoch is not None and not 1 <= args.pause_after_epoch < int(training["epochs"]):
            raise ValueError("Pause epoch must precede the final scheduled epoch")
        if args.pause_after_step is not None and (args.pause_after_step < 1 or args.pause_after_epoch is not None):
            raise ValueError("Pause step must be positive and cannot be combined with pause-after-epoch")
        if args.stop_file is not None and args.stop_file.exists():
            raise FileExistsError("Stop file already exists; review it before starting/resuming")
    fixed_schedule = training.get("schedule", {}).get("name") in {
        SCHEDULE_NAME, CONTINUATION_SCHEDULE_NAME, TWO_STAGE_SCHEDULE_NAME}
    if epoch_schedule and (config["model"]["preset"] != ALPHAGENOME_PRESET or config.get("continuation")):
        raise ValueError("Two-stage training requires AlphaGenome-small without continuation initialization")
    if continuation and (config["model"]["preset"] != ALPHAGENOME_PRESET or not args.resume
                         or args.smoke_test or not config.get("continuation")):
        raise ValueError("Continuation requires AlphaGenome-small, provenance and --resume; use its separate CUDA gate")
    if config["model"]["preset"] == ALPHAGENOME_PRESET and (not alphagenome_experiment or stage != "base"):
        raise ValueError("AlphaGenome-small requires the from-scratch AlphaGenome loss experiment")
    if training.get("schedule") and not fixed_schedule:
        raise ValueError("Unknown explicitly configured learning-rate schedule")
    if fixed_schedule and not alphagenome_experiment:
        raise ValueError("Fixed AlphaGenome schedule requires the AMP-safe AlphaGenome training path")
    if alphagenome_experiment or scaled_crested:
        require_training_node(training["device"])
        if stage != "base":
            raise ValueError("Scaled-target experiments require a from-scratch base run")
    device = resolve_device(training["device"])
    if training["mixed_precision"] == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training requires CUDA")
    seed = int(config["seed"])
    seed_everything(seed, device)

    root = Path(config["output_directory"])
    data_directory = root / "data"
    model_directory = root / (
        str(specificity_config["model_subdirectory"])
        if stage == "specificity"
        else "model"
    )
    if continuation and not (model_directory / "last_checkpoint.pt").is_file():
        raise FileNotFoundError("Continuation requires its explicitly migrated restart checkpoint")
    if (epoch_schedule or scaled_crested or ag_4x_legacy_schedule) and args.resume and not (model_directory / "last_checkpoint.pt").is_file():
        raise FileNotFoundError("Scaled-target resume requires its existing restart checkpoint")
    dataset_path = data_directory / "windows.tsv.gz"
    records = read_windows(dataset_path)
    sequence_lengths = {
        len(record.sequence) for split_records in records.values() for record in split_records
    }
    if sequence_lengths != {INPUT_BP}:
        raise ValueError(f"Production model requires {INPUT_BP}-bp inputs")
    counts = {split: len(values) for split, values in records.items()}
    atac_profiles, atac_metadata = load_profiles(
        data_directory / "profiles" / "atac", dataset_path, counts
    )
    h3_profiles, h3_metadata = load_profiles(
        data_directory / "profiles" / "h3k27ac", dataset_path, counts
    )
    contexts = tuple(config["contexts"])
    if tuple(atac_metadata["contexts"]) != contexts or tuple(h3_metadata["contexts"]) != contexts:
        raise ValueError("Profile context order differs from configuration")
    profiles_config = config["profiles"]
    expected_geometry = {
        "source_bin_bp": SOURCE_BIN_BP,
        "atac_target_bp": ATAC_TARGET_BP,
        "h3k27ac_target_bp": H3K27AC_TARGET_BP,
        "h3k27ac_output_pool_size": H3K27AC_OUTPUT_POOL_SIZE,
    }
    observed_geometry = {
        name: int(profiles_config[name]) for name in expected_geometry
    }
    if observed_geometry != expected_geometry:
        raise ValueError(
            f"Production profile geometry must be {expected_geometry}, found {observed_geometry}"
        )
    if (
        int(atac_metadata["bin_size_bp"]) != int(profiles_config["source_bin_bp"])
        or int(h3_metadata["bin_size_bp"]) != int(profiles_config["source_bin_bp"])
        or int(atac_metadata["target_window_size_bp"]) != int(profiles_config["atac_target_bp"])
        or int(h3_metadata["target_window_size_bp"]) != int(profiles_config["h3k27ac_target_bp"])
    ):
        raise ValueError("Profile target geometry differs from configuration")
    h3_pool_size = int(profiles_config["h3k27ac_output_pool_size"])

    specificity_metadata: dict[str, Any] | None = None
    train_indices: np.ndarray | None = None
    validation_specific_mask: np.ndarray | None = None
    if stage == "specificity":
        gini_multiplier = float(specificity_config["gini_standard_deviations"])
        thresholds = fit_specificity_thresholds(
            records["train"],
            atac_profiles["train"],
            h3_profiles["train"],
            gini_multiplier,
        )
        train_indices, train_dominant_contexts, train_specific_counts = (
            select_specific_peak_indices_and_contexts(
                records["train"],
                atac_profiles["train"],
                h3_profiles["train"],
                thresholds,
            )
        )
        validation_indices, validation_dominant_contexts, validation_specific_counts = (
            select_specific_peak_indices_and_contexts(
                records["validation"],
                atac_profiles["validation"],
                h3_profiles["validation"],
                thresholds,
            )
        )
        balancing_config = dict(specificity_config.get("context_balancing", {}))
        balancing_metadata: dict[str, object] | None = None
        if balancing_config.get("enabled", False):
            maximum_oversampling_factor = float(
                balancing_config["maximum_training_oversampling_factor"]
            )
            train_indices, train_balance = balance_specific_peak_contexts(
                train_indices,
                train_dominant_contexts,
                len(contexts),
                seed,
                maximum_oversampling_factor,
            )
            validation_indices, validation_balance = balance_specific_peak_contexts(
                validation_indices,
                validation_dominant_contexts,
                len(contexts),
                seed + 1,
                1.0,
            )
            balancing_metadata = {
                "assignment": (
                    "assay with largest threshold-standardized Gini excess; "
                    "argmax window-mean signal within that assay"
                ),
                "train": train_balance,
                "validation": validation_balance,
                "validation_policy": "downsample every context to the rarest context",
            }
        validation_specific_mask = np.zeros(len(records["validation"]), dtype=np.bool_)
        validation_specific_mask[validation_indices] = True
        specificity_metadata = {
            "definition": "ATAC-specific OR H3K27ac-specific peak window",
            "activity_summary": "mean signal across the assay target bins",
            "score": "Gini index across the eight contexts",
            "threshold_fit_split": "train",
            "threshold_rule": "training peak mean + multiplier * training peak standard deviation",
            "thresholds": thresholds,
            "train_counts": train_specific_counts,
            "validation_counts": validation_specific_counts,
            "context_order": list(contexts),
            "context_balancing": balancing_metadata,
        }

    train_dataset = JointProfileDataset(
        records["train"],
        atac_profiles["train"],
        h3_profiles["train"],
        h3_pool_size,
        training=True,
        rc_probability=float(training["stochastic_rc_probability"]),
        seed=seed,
        indices=train_indices,
    )
    validation_dataset = JointProfileDataset(
        records["validation"],
        atac_profiles["validation"],
        h3_profiles["validation"],
        h3_pool_size,
        training=False,
        rc_probability=0,
        seed=seed,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=int(training["evaluation_batch_size"]),
        workers=int(training["num_workers"]),
        epoch=0,
        seed=seed,
        training=False,
        pin_memory=device.type == "cuda",
    )
    regulatory_mask = np.asarray(
        [record.source != "genomic_background" for record in records["validation"]],
        dtype=np.bool_,
    )

    h3_means, h3_standard_deviations = streamed_h3_log_statistics(
        h3_profiles["train"], h3_pool_size
    )
    target_means = {
        "atac": streamed_profile_means(atac_profiles["train"]),
        "h3k27ac": streamed_profile_means(h3_profiles["train"], h3_pool_size),
    }
    track_means = None
    breadth_monitor = None
    if alphagenome_experiment or scaled_crested:
        track_means = {
            "atac": fit_nonzero_means(atac_profiles["train"]),
            "h3k27ac": fit_nonzero_means(h3_profiles["train"], h3_pool_size),
        }
        from .training_breadth import TrainingBreadthMonitor

        breadth_monitor = TrainingBreadthMonitor(
            records["train"], atac_profiles["train"], h3_profiles["train"]
        )
        print(json.dumps({"event": "scaled_crested_training_statistics" if scaled_crested else "alphagenome_training_statistics",
                          "track_means": {a: m.tolist() for a, m in track_means.items()},
                          "breadth_reference": breadth_monitor.metadata}), flush=True)
    model_class = AlphaGenomeSmall if config["model"]["preset"] == ALPHAGENOME_PRESET else EnformerLikeJointProfileRegressor
    model = model_class(
        context_count=len(contexts),
        dropout=float(config["model"]["dropout"]),
        head_dropout=float(config["model"]["head_dropout"]),
        model_size=str(config["model"]["preset"]),
        h3k27ac_output_pool_size=h3_pool_size,
        output_scaling={a: m.tolist() for a, m in track_means.items()} if track_means is not None else None,
        **({"transformer_bin_bp": config["model"].get("transformer_bin_bp", 128)}
           if model_class is AlphaGenomeSmall else {}),
    ).to(device)
    model.initialize_output_means(target_means["atac"], target_means["h3k27ac"])
    initialization_metadata: dict[str, Any] | None = dict(config["continuation"]) if continuation else None
    if stage == "specificity":
        initialization_setting = (
            args.evaluation_checkpoint
            if args.evaluate_initialization_only and args.evaluation_checkpoint
            else Path(str(specificity_config["initialization_checkpoint"]))
        )
        initialization_path = (
            initialization_setting
            if initialization_setting.is_absolute()
            else root / initialization_setting
        )
        if not initialization_path.is_file():
            raise FileNotFoundError(
                f"Specificity fine-tuning requires {initialization_path}"
            )
        initial_checkpoint = torch.load(
            initialization_path, map_location="cpu", weights_only=False
        )
        if initial_checkpoint.get("kind") != "enformer_like_dense_atac_h3k27ac_profile_regressor":
            raise ValueError("Specificity initialization checkpoint has the wrong kind")
        if tuple(initial_checkpoint.get("contexts", ())) != contexts:
            raise ValueError("Specificity initialization context order differs")
        dataset_hash = sha256_file(dataset_path)
        if initial_checkpoint.get("dataset_sha256") != dataset_hash:
            raise ValueError("Specificity initialization dataset differs")
        model.load_state_dict(initial_checkpoint["state_dict"], strict=True)
        initialization_metadata = {
            "path": str(initialization_path),
            "sha256": sha256_file(initialization_path),
            "base_epoch": int(initial_checkpoint["epoch"]),
            "base_score": float(initial_checkpoint["checkpoint_selection"]["score"]),
        }
    atac_criterion, h3_criterion, loss_metadata = build_loss_criteria(
        training, h3_means, h3_standard_deviations, device, track_means=track_means
    )
    if args.evaluate_initialization_only:
        validation_losses, labels, predictions = evaluate(
            model,
            validation_loader,
            atac_criterion,
            h3_criterion,
            device,
            training["mixed_precision"],
            bool(training["validation_rc_ensemble"]),
        )
        full_metrics, specificity_metrics = validation_metric_sets(
            labels,
            predictions,
            regulatory_mask,
            contexts,
            validation_specific_mask,
        )
        if specificity_metrics is None:
            raise RuntimeError("Specificity metrics were not calculated")
        report = {
            "event": "specificity_initialization_evaluation",
            "training_stage": stage,
            "initialization": initialization_metadata,
            "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
            "split_counts": counts,
            "specificity": specificity_metadata,
            "validation_reverse_complement_ensemble": training[
                "validation_rc_ensemble"
            ],
            "validation_losses": validation_losses,
            "scientific_composite": scientific_composite(specificity_metrics),
            "regulatory_overcorrelation": regulatory_overcorrelation_summary(
                specificity_metrics
            ),
            "validation": full_metrics,
            "specificity_validation": specificity_metrics,
        }
        if args.evaluation_output is None:
            output_path = model_directory / "initialization_evaluation.json"
        else:
            output_path = args.evaluation_output
            if not output_path.is_absolute():
                output_path = root / output_path
        atomic_write_json(output_path, report)
        print(
            json.dumps(
                {
                    "event": report["event"],
                    "checkpoint": initialization_metadata,
                    "scientific_composite": report["scientific_composite"],
                    "regulatory_overcorrelation": report[
                        "regulatory_overcorrelation"
                    ],
                    "output": str(output_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["max_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    batches_per_epoch = math.ceil(len(train_dataset) / batch_size)
    epochs = int(training["epochs"])
    maximum_steps = None
    if epoch_schedule:
        scheduler = AlphaGenomeTwoStageSchedule.from_config(optimizer, training, batches_per_epoch)
        warmup_steps = scheduler.warmup_steps
        decay_steps = scheduler.scheduled_steps - warmup_steps
    elif fixed_schedule:
        warmup_steps = int(training["schedule"]["warmup_steps"])
        maximum_steps = int(training["schedule"]["total_steps"])
        decay_steps = maximum_steps - warmup_steps
        if continuation:
            scheduler = AlphaGenomeContinuationSchedule.from_config(optimizer, training)
            decay_steps = maximum_steps - scheduler.start_step - warmup_steps - scheduler.hold_steps
        else:
            scheduler = AlphaGenomeSchedule(optimizer, float(training["max_learning_rate"]),
                                           warmup_steps, maximum_steps)
    else:
        warmup_steps = max(
            1, round(epochs * batches_per_epoch * float(training["warmup_fraction"]))
        )
        decay_steps = max(
            0, round(batches_per_epoch * float(training.get("post_warmup_decay_epochs", 0.0)))
        )
        scheduler = WarmupPlateauScheduler(
            optimizer,
            maximum_learning_rate=float(training["max_learning_rate"]),
            post_warmup_learning_rate=float(training["post_warmup_learning_rate"]),
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            plateau_factor=float(training["plateau_factor"]),
            plateau_patience=int(training["plateau_patience"]),
            plateau_threshold=float(training["plateau_threshold"]),
            minimum_learning_rate=float(training["minimum_learning_rate"]),
        )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and training["mixed_precision"] == "fp16"
    )
    architecture = architecture_metadata(model, contexts)
    run_signature = {
        "config": config,
        "effective_training": training,
        "dataset_sha256": sha256_file(dataset_path),
        "atac_profiles_sha256": atac_metadata["outputs"]["train"]["sha256"],
        "h3k27ac_profiles_sha256": h3_metadata["outputs"]["train"]["sha256"],
        "architecture": architecture,
        "training_stage": stage,
        "specificity": specificity_metadata,
        "initialization": initialization_metadata,
    }
    best_path = model_directory / "best_model.pt"
    best_low_overcorrelation_path = (
        model_directory / "best_low_overcorrelation_model.pt"
    )
    last_path = model_directory / "last_checkpoint.pt"
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    best_low_overcorrelation_score = math.inf
    best_low_overcorrelation_epoch = 0
    best_low_overcorrelation_metrics: dict[str, float | int] | None = None
    epochs_without_improvement = 0
    start_epoch = 0
    start_batch = 0
    running_sums = {"atac": 0.0, "h3k27ac": 0.0, "total": 0.0}
    running_examples = 0

    def save_compact_checkpoint(
        path: Path,
        checkpoint_epoch: int,
        selection: dict[str, Any],
    ) -> None:
        atomic_torch_save(
            {
                "kind": ("alphagenome_small_joint_profile_regressor" if isinstance(model, AlphaGenomeSmall)
                         else "enformer_like_dense_atac_h3k27ac_profile_regressor"),
                "version": 1,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "architecture": architecture,
                "contexts": contexts,
                "profile_metadata": {
                    "atac": atac_metadata,
                    "h3k27ac": h3_metadata,
                },
                "dataset_sha256": run_signature["dataset_sha256"],
                "epoch": checkpoint_epoch,
                "checkpoint_selection": selection,
                "training_stage": stage,
                "specificity": specificity_metadata,
                "initialization": initialization_metadata,
                "loss": loss_metadata,
                "training_target_means": {
                    assay: values.tolist() for assay, values in target_means.items()
                },
                "learning_rate_schedule": scheduler.state_dict(),
                **({"optimizer_steps": scheduler.optimizer_steps} if fixed_schedule else {}),
                **({"continuation_optimizer_steps": scheduler.continuation_steps} if continuation else {}),
                "reverse_complement_augmentation": {
                    "strategy": "stochastic",
                    "stochastic_probability": training[
                        "stochastic_rc_probability"
                    ],
                },
                "validation_reverse_complement_ensemble": training[
                    "validation_rc_ensemble"
                ],
            },
            path,
        )

    if args.resume and last_path.is_file():
        progress = load_training_state(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            run_signature,
            device,
        )
        start_epoch = int(progress["next_epoch"])
        start_batch = int(progress["next_batch"])
        history = list(progress["history"])
        best_score = float(progress["best_score"])
        best_epoch = int(progress["best_epoch"])
        if "best_low_overcorrelation_score" in progress:
            best_low_overcorrelation_score = float(
                progress["best_low_overcorrelation_score"]
            )
            best_low_overcorrelation_epoch = int(
                progress["best_low_overcorrelation_epoch"]
            )
            stored_overcorrelation_metrics = progress[
                "best_low_overcorrelation_metrics"
            ]
            best_low_overcorrelation_metrics = (
                dict(stored_overcorrelation_metrics)
                if stored_overcorrelation_metrics is not None
                else None
            )
        else:
            for previous in history:
                previous_metrics = previous.get("specificity_validation") or previous[
                    "validation"
                ]
                summary = regulatory_overcorrelation_summary(previous_metrics)
                if summary["mean_positive_excess"] < best_low_overcorrelation_score:
                    best_low_overcorrelation_score = float(
                        summary["mean_positive_excess"]
                    )
                    best_low_overcorrelation_epoch = int(previous["epoch"])
                    best_low_overcorrelation_metrics = summary
        epochs_without_improvement = int(progress["epochs_without_improvement"])
        running_sums = dict(progress["running_sums"])
        running_examples = int(progress["running_examples"])
        print(json.dumps({"event": "training_resumed", "epoch": start_epoch + 1, "batch": start_batch}), flush=True)
        if (
            best_low_overcorrelation_epoch
            and not best_low_overcorrelation_path.is_file()
        ):
            selection = {
                "metric": "regulatory_mean_positive_pairwise_overcorrelation",
                "mode": "min",
                "score": best_low_overcorrelation_score,
                "details": best_low_overcorrelation_metrics,
            }
            seeded = False
            compact_checkpoint: dict[str, Any] | None = None
            if best_path.is_file():
                compact_checkpoint = torch.load(
                    best_path, map_location="cpu", weights_only=False
                )
                if (
                    int(compact_checkpoint.get("epoch", 0))
                    == best_low_overcorrelation_epoch
                ):
                    compact_checkpoint["checkpoint_selection"] = selection
                    atomic_torch_save(
                        compact_checkpoint, best_low_overcorrelation_path
                    )
                    seeded = True
            if (
                not seeded
                and start_batch == 0
                and start_epoch == best_low_overcorrelation_epoch
            ):
                save_compact_checkpoint(
                    best_low_overcorrelation_path,
                    best_low_overcorrelation_epoch,
                    selection,
                )
                seeded = True
            if not seeded:
                unavailable_epoch = best_low_overcorrelation_epoch
                unavailable_score = best_low_overcorrelation_score
                fallback_epoch = (
                    int(compact_checkpoint.get("epoch", 0))
                    if compact_checkpoint is not None
                    else 0
                )
                fallback_result = next(
                    (
                        result
                        for result in history
                        if int(result["epoch"]) == fallback_epoch
                    ),
                    None,
                )
                if compact_checkpoint is not None and fallback_result is not None:
                    fallback_metrics = (
                        fallback_result.get("specificity_validation")
                        or fallback_result["validation"]
                    )
                    fallback_summary = regulatory_overcorrelation_summary(
                        fallback_metrics
                    )
                    best_low_overcorrelation_score = float(
                        fallback_summary["mean_positive_excess"]
                    )
                    best_low_overcorrelation_epoch = fallback_epoch
                    best_low_overcorrelation_metrics = fallback_summary
                    compact_checkpoint["checkpoint_selection"] = {
                        "metric": "regulatory_mean_positive_pairwise_overcorrelation",
                        "mode": "min",
                        "score": best_low_overcorrelation_score,
                        "details": best_low_overcorrelation_metrics,
                    }
                    atomic_torch_save(
                        compact_checkpoint, best_low_overcorrelation_path
                    )
                    seeded = True
                    action = "seed_best_available_checkpoint"
                else:
                    best_low_overcorrelation_score = math.inf
                    best_low_overcorrelation_epoch = 0
                    best_low_overcorrelation_metrics = None
                    action = "start_checkpoint_selection_at_next_validation"
                print(
                    json.dumps(
                        {
                            "event": "historical_low_overcorrelation_checkpoint_unavailable",
                            "epoch": unavailable_epoch,
                            "score": unavailable_score,
                            "action": action,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if seeded:
                print(
                    json.dumps(
                        {
                            "event": "low_overcorrelation_checkpoint_seeded",
                            "epoch": best_low_overcorrelation_epoch,
                            "score": best_low_overcorrelation_score,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    elif args.resume:
        print(json.dumps({"event": "resume_checkpoint_absent", "action": "new_run"}), flush=True)

    if maximum_steps is not None and history and history[-1].get("optimizer_steps", 0) >= maximum_steps:
        print(json.dumps({"event": "training_already_complete", "optimizer_steps": scheduler.optimizer_steps,
                          "learning_rate": optimizer.param_groups[0]["lr"]}), flush=True)
        # Skip optimization but finish/recover the final JSON report if an
        # interruption occurred after the terminal checkpoint was saved.
        start_epoch = epochs
    if epoch_schedule and history and history[-1]["epoch"] == epochs and history[-1]["training_epoch_complete"]:
        print(json.dumps({"event": "training_already_complete", "optimizer_steps": scheduler.optimizer_steps,
                          "completed_epochs": epochs, "learning_rate": optimizer.param_groups[0]["lr"]}), flush=True)
        start_epoch = epochs

    print(
        json.dumps(
            {
                "event": "training_start",
                "training_stage": stage,
                "device": str(device),
                "architecture": architecture,
                "split_counts": counts,
                "training_examples": len(train_dataset),
                "batches_per_epoch": batches_per_epoch,
                "warmup_steps": warmup_steps,
                "decay_steps": decay_steps,
                **({"maximum_optimizer_steps": maximum_steps, "schedule": scheduler.state_dict()}
                   if maximum_steps is not None else {}),
                **({"planned_optimizer_steps": scheduler.scheduled_steps, "maximum_epochs": epochs,
                    "schedule": scheduler.state_dict(), "stop_policy": "completed_epochs"}
                   if epoch_schedule else {}),
                "loss": loss_metadata,
                "specificity": specificity_metadata,
                "initialization": initialization_metadata,
                "target_shapes": {
                    "atac": [int(atac_metadata["bins_per_target"]), len(contexts)],
                    "h3k27ac": [int(h3_metadata["bins_per_target"]) // h3_pool_size, len(contexts)],
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.smoke_test:
        epochs = 1

    def evaluate_step_pause(epoch_index, absolute_batch):
        # This extra validation is diagnostic only. The pre-validation restart
        # preserves data position, optimizer/RNG state and epoch-based selection.
        count = scheduler.optimizer_steps
        report_path = model_directory / f"step_{count:09d}_metrics.json"
        compact_path = model_directory / f"step_{count:09d}_model.pt"
        restart_sha = sha256_file(last_path)
        if report_path.exists():
            previous = json.loads(report_path.read_text())
            if (previous["optimizer_steps"] != count
                    or previous["restart_checkpoint_sha256"] != restart_sha
                    or previous["checkpoint_sha256"] != sha256_file(compact_path)):
                raise ValueError("Existing step evaluation belongs to a different checkpoint")
        else:
            validation_losses, labels, predictions = evaluate(
                model, validation_loader, atac_criterion, h3_criterion, device,
                training["mixed_precision"], bool(training["validation_rc_ensemble"]),
            )
            validation, _ = validation_metric_sets(labels, predictions, regulatory_mask, contexts, None)
            epoch = epoch_index + int(absolute_batch > 0)
            save_compact_checkpoint(compact_path, epoch,
                                    {"metric": "diagnostic_optimizer_step", "mode": "fixed",
                                     "optimizer_steps": count})
            atomic_write_json(report_path, {
                "event": "step_validation_complete", "diagnostic_only": True,
                "epoch": epoch, "training_batches_completed": absolute_batch,
                "training_epoch_complete": absolute_batch == batches_per_epoch,
                "completed_epochs": sum(row.get("training_epoch_complete", False) for row in history),
                "optimizer_steps": count, "learning_rate": optimizer.param_groups[0]["lr"],
                "learning_rate_phase": scheduler.phase, "loss_scale": scaler.get_scale(),
                "validation_examples": len(labels["atac"]),
                "validation_reverse_complement_ensemble": bool(training["validation_rc_ensemble"]),
                "validation_losses": validation_losses, "validation": validation,
                "scientific_composite": scientific_composite(validation),
                "regulatory_overcorrelation": regulatory_overcorrelation_summary(validation),
                "continuous_breadth": breadth_monitor.evaluate(labels, predictions, regulatory_mask),
                "checkpoint_sha256": sha256_file(compact_path),
                "restart_checkpoint_sha256": restart_sha,
            })
        print(json.dumps({"event": "step_validation_complete", "optimizer_steps": count,
                          "learning_rate": optimizer.param_groups[0]["lr"],
                          "output": str(report_path), "diagnostic_only": True}), flush=True)
        report_pause("optimizer_step_boundary")

    def report_pause(reason):
        print(json.dumps({"event": "training_paused", "reason": reason,
                          "checkpoint": str(last_path), "completed_epochs": len(history),
                          "optimizer_steps": scheduler.optimizer_steps,
                          "learning_rate": optimizer.param_groups[0]["lr"]}), flush=True)

    if args.pause_after_step is not None:
        if args.pause_after_step > scheduler.scheduled_steps or scheduler.optimizer_steps > args.pause_after_step:
            raise ValueError("Pause step is outside the remaining scheduled update horizon")
        if scheduler.optimizer_steps == args.pause_after_step:
            evaluate_step_pause(start_epoch, start_batch)
            return
    if args.pause_after_epoch is not None and len(history) >= args.pause_after_epoch:
        report_pause("requested_epoch_already_completed")
        return

    model_directory.mkdir(parents=True, exist_ok=True)
    for epoch_index in range(start_epoch, epochs):
        resume_batch = start_batch if epoch_index == start_epoch else 0
        if not resume_batch:
            running_sums = {"atac": 0.0, "h3k27ac": 0.0, "total": 0.0}
            running_examples = 0
        train_loader = make_loader(
            train_dataset,
            batch_size=batch_size,
            workers=int(training["num_workers"]),
            epoch=epoch_index,
            seed=seed,
            training=True,
            start_batch=resume_batch,
            pin_memory=device.type == "cuda",
        )
        model.train()
        absolute_batch = resume_batch
        # Rebuilding a partially consumed iterator draws a new worker base seed.
        # Data order/RC are index-and-epoch deterministic; preserve the restored
        # model RNG instead of consuming this extra draw on scaled-target resume.
        loader_rng_state = torch.get_rng_state() if (epoch_schedule or scaled_crested or ag_4x_legacy_schedule) and resume_batch else None
        train_iterator = iter(train_loader)
        if loader_rng_state is not None:
            torch.set_rng_state(loader_rng_state)
        for relative_batch, batch in enumerate(train_iterator, start=1):
            if maximum_steps is not None and scheduler.optimizer_steps >= maximum_steps:
                break
            absolute_batch = resume_batch + relative_batch
            one_hot, attention_mask, atac_mask, h3_mask, atac_labels, h3_labels = move_batch(
                batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, training["mixed_precision"]):
                predictions = model(one_hot, attention_mask, atac_mask, h3_mask)
                losses = calculate_losses(
                    predictions,
                    (atac_labels, h3_labels),
                    atac_criterion,
                    h3_criterion,
                )
            if alphagenome_experiment:
                step_report = alphagenome_optimizer_step(
                    losses["total"], model, optimizer, scheduler, scaler,
                    float(training["gradient_clip_norm"]),
                )
                if step_report["optimizer_step_skipped"] or relative_batch <= 2:
                    print(json.dumps({"event": "alphagenome_optimizer_step", "epoch": epoch_index + 1,
                                      "batch": absolute_batch, **step_report}), flush=True)
            else:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            example_count = len(atac_labels)
            running_examples += example_count
            for name, value in losses.items():
                running_sums[name] = running_sums.get(name, 0.0) + (
                    float(value.item()) * example_count
                )
            step_budget_reached = maximum_steps is not None and scheduler.optimizer_steps >= maximum_steps
            step_pause_reached = args.pause_after_step is not None and scheduler.optimizer_steps >= args.pause_after_step
            if absolute_batch % 100 == 0 or absolute_batch == batches_per_epoch or step_budget_reached or step_pause_reached:
                print(
                    json.dumps(
                        {
                            "event": "training_progress",
                            "epoch": epoch_index + 1,
                            "batch": absolute_batch,
                            "batches": batches_per_epoch,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            **({"loss_scale": scaler.get_scale(),
                                "optimizer_steps": scheduler.optimizer_steps}
                               if alphagenome_experiment else {}),
                            **({"continuation_optimizer_steps": scheduler.continuation_steps} if continuation else {}),
                            **({"learning_rate_phase": scheduler.phase} if epoch_schedule else {}),
                            "running_losses": {
                                name: value / running_examples
                                for name, value in running_sums.items()
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            checkpoint_interval = int(training["checkpoint_every_batches"])
            stop_requested = args.stop_file is not None and args.stop_file.exists()
            if ((checkpoint_interval and absolute_batch % checkpoint_interval == 0) or step_budget_reached or step_pause_reached
                    or (epoch_schedule and absolute_batch == batches_per_epoch) or stop_requested):
                save_training_state(
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    run_signature,
                    {
                        "next_epoch": epoch_index,
                        "next_batch": absolute_batch,
                        "history": history,
                        "best_score": best_score,
                        "best_epoch": best_epoch,
                        "best_low_overcorrelation_score": best_low_overcorrelation_score,
                        "best_low_overcorrelation_epoch": best_low_overcorrelation_epoch,
                        "best_low_overcorrelation_metrics": best_low_overcorrelation_metrics,
                        "epochs_without_improvement": epochs_without_improvement,
                        "running_sums": running_sums,
                        "running_examples": running_examples,
                    },
                    device,
                )
            if stop_requested:
                report_pause("stop_file")
                return
            if step_pause_reached:
                evaluate_step_pause(epoch_index, absolute_batch)
                return
            if args.smoke_test:
                if fixed_schedule or ag_4x_legacy_schedule:
                    if scheduler.optimizer_steps >= 2:
                        break
                    if absolute_batch >= ALPHAGENOME_SMOKE_MAX_BATCHES:
                        raise RuntimeError(
                            f"AlphaGenome smoke failed to reach two successful optimizer updates "
                            f"within {ALPHAGENOME_SMOKE_MAX_BATCHES} attempted batches"
                        )
                elif absolute_batch >= 2:
                    break
            if step_budget_reached:
                break

        if args.smoke_test and (fixed_schedule or ag_4x_legacy_schedule) and scheduler.optimizer_steps < 2:
            raise RuntimeError("AlphaGenome smoke exhausted training data before two successful optimizer updates")

        validation_losses, labels, predictions = evaluate(
            model,
            validation_loader,
            atac_criterion,
            h3_criterion,
            device,
            training["mixed_precision"],
            bool(training["validation_rc_ensemble"]),
            maximum_batches=1 if args.smoke_test else None,
        )
        epoch_metrics, specificity_metrics = validation_metric_sets(
            labels,
            predictions,
            regulatory_mask,
            contexts,
            validation_specific_mask if stage == "specificity" else None,
        )
        score_metrics = specificity_metrics or epoch_metrics
        score = scientific_composite(score_metrics)
        overcorrelation_metrics = regulatory_overcorrelation_summary(score_metrics)
        plateau = scheduler.step_validation(score)
        epoch_result = {
            "epoch": epoch_index + 1,
            "training_losses": {
                name: value / running_examples for name, value in running_sums.items()
            },
            "validation_losses": validation_losses,
            "scientific_composite": score,
            "regulatory_overcorrelation": overcorrelation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "plateau": plateau,
            "validation": epoch_metrics,
        }
        if breadth_monitor is not None:
            epoch_result["loss_scale"] = scaler.get_scale()
            epoch_result["optimizer_steps"] = scheduler.optimizer_steps
            epoch_result["continuous_breadth"] = breadth_monitor.evaluate(
                labels, predictions, regulatory_mask[:len(labels["atac"])]
            )
        if fixed_schedule or ag_4x_legacy_schedule:
            epoch_result["training_epoch_complete"] = absolute_batch == batches_per_epoch
            epoch_result["training_batches_completed"] = absolute_batch
        if maximum_steps is not None:
            epoch_result["maximum_optimizer_steps"] = maximum_steps
        if epoch_schedule:
            epoch_result.update(planned_optimizer_steps=scheduler.scheduled_steps,
                                maximum_epochs=scheduler.epochs, learning_rate_phase=scheduler.phase)
        if continuation:
            epoch_result["continuation_optimizer_steps"] = scheduler.continuation_steps
        if specificity_metrics is not None:
            epoch_result["specificity_validation"] = specificity_metrics
        history.append(epoch_result)
        if alphagenome_experiment or scaled_crested:
            atomic_write_json(model_directory / "history.json", history)
        print(json.dumps({"event": "epoch_complete", **epoch_result}, sort_keys=True), flush=True)

        if score > best_score:
            best_score = score
            best_epoch = epoch_index + 1
            epochs_without_improvement = 0
            save_compact_checkpoint(
                best_path,
                best_epoch,
                {
                    "metric": (
                        "specificity_scientific_composite"
                        if stage == "specificity"
                        else "scientific_composite"
                    ),
                    "mode": "max",
                    "score": best_score,
                },
            )
        else:
            epochs_without_improvement += 1
        overcorrelation_score = float(overcorrelation_metrics["mean_positive_excess"])
        if overcorrelation_score < best_low_overcorrelation_score:
            best_low_overcorrelation_score = overcorrelation_score
            best_low_overcorrelation_epoch = epoch_index + 1
            best_low_overcorrelation_metrics = overcorrelation_metrics
            save_compact_checkpoint(
                best_low_overcorrelation_path,
                best_low_overcorrelation_epoch,
                {
                    "metric": "regulatory_mean_positive_pairwise_overcorrelation",
                    "mode": "min",
                    "score": best_low_overcorrelation_score,
                    "details": best_low_overcorrelation_metrics,
                },
            )
        if maximum_steps is not None and scheduler.optimizer_steps >= maximum_steps:
            save_compact_checkpoint(model_directory / "final_model.pt", epoch_index + 1,
                                    {"metric": "fixed_optimizer_step_budget", "mode": "fixed",
                                     "optimizer_steps": scheduler.optimizer_steps, "score": score})
        if epoch_schedule and not args.smoke_test and epoch_index + 1 == epochs:
            save_compact_checkpoint(model_directory / "final_model.pt", epoch_index + 1,
                                    {"metric": "fixed_full_epoch_budget", "mode": "fixed",
                                     "epochs": epochs, "optimizer_steps": scheduler.optimizer_steps, "score": score})
        save_training_state(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            run_signature,
            {
                "next_epoch": epoch_index if maximum_steps is not None and absolute_batch < batches_per_epoch else epoch_index + 1,
                "next_batch": absolute_batch if maximum_steps is not None and absolute_batch < batches_per_epoch else 0,
                "history": history,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_low_overcorrelation_score": best_low_overcorrelation_score,
                "best_low_overcorrelation_epoch": best_low_overcorrelation_epoch,
                "best_low_overcorrelation_metrics": best_low_overcorrelation_metrics,
                "epochs_without_improvement": epochs_without_improvement,
                "running_sums": running_sums if maximum_steps is not None and absolute_batch < batches_per_epoch else {"atac": 0.0, "h3k27ac": 0.0, "total": 0.0},
                "running_examples": running_examples if maximum_steps is not None and absolute_batch < batches_per_epoch else 0,
            },
            device,
        )
        start_batch = 0
        if epoch_index + 1 < epochs and (
                args.pause_after_epoch == epoch_index + 1
                or (args.stop_file is not None and args.stop_file.exists())):
            report_pause("epoch_boundary")
            return
        if args.smoke_test or (maximum_steps is not None and scheduler.optimizer_steps >= maximum_steps):
            break
        if not fixed_schedule and epochs_without_improvement >= int(training["patience"]):
            break

    if maximum_steps is not None and not args.smoke_test and scheduler.optimizer_steps < maximum_steps:
        raise RuntimeError("Epoch safety limit reached before the fixed optimizer-step budget")

    metrics = {
        "method": "joint_atac_h3k27ac_profile_training_v1",
        "training_stage": stage,
        "contexts": list(contexts),
        "dataset": {"path": str(dataset_path), "sha256": run_signature["dataset_sha256"]},
        "split_counts": counts,
        "model": architecture,
        "best_epoch": best_epoch,
        "best_scientific_composite": best_score,
        "best_low_overcorrelation_epoch": best_low_overcorrelation_epoch,
        "best_low_overcorrelation": best_low_overcorrelation_metrics,
        "checkpoint_selection_metric": (
            "specificity_scientific_composite"
            if stage == "specificity"
            else "scientific_composite"
        ),
        "specificity": specificity_metadata,
        "initialization": initialization_metadata,
        "history": history,
        "configuration": config,
    }
    atomic_write_json(model_directory / "metrics.json", metrics)
    print(
        json.dumps(
            {
                "event": "training_complete" if not args.smoke_test else "smoke_test_complete",
                "best_epoch": best_epoch,
                "best_scientific_composite": best_score,
                "best_low_overcorrelation_epoch": best_low_overcorrelation_epoch,
                "best_low_overcorrelation": best_low_overcorrelation_metrics,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
