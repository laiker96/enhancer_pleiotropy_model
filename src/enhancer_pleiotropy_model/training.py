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
    scientific_composite,
)
from .model import EnformerLikeJointProfileRegressor, MODEL_PRESETS
from .preprocessing.windows import (
    H3K27AC_PEAK_SOURCE,
    JOINT_PEAK_SOURCE,
    PEAK_SOURCE,
)


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

    def components(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if predictions.shape != labels.shape or predictions.ndim != 3:
            raise ValueError("CREsted loss expects aligned [batch, bins, contexts]")
        predictions = predictions.float()
        labels = labels.float()
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
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    loss_config = dict(training.get("loss", {}))
    name = loss_config.get(
        "name", "poisson_atac_standardized_log1p_huber_h3k27ac"
    )
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
    selected = np.zeros(len(records), dtype=np.bool_)
    counts: dict[str, int] = {}
    for assay, (profiles, eligible_sources) in assay_inputs.items():
        indices, scores = peak_specificity_scores(records, profiles, eligible_sources)
        assay_selected = indices[scores > float(thresholds[assay]["threshold"])]
        selected[assay_selected] = True
        counts[f"{assay}_specific"] = int(len(assay_selected))
    selected_indices = np.flatnonzero(selected)
    if not len(selected_indices):
        raise ValueError("No peak windows passed the specificity thresholds")
    counts["union_specific"] = int(len(selected_indices))
    return selected_indices, counts


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
        else:
            assay_total = criterion(prediction.float(), target.float())
        losses[assay] = assay_total
        assay_totals.append(assay_total)
    losses["total"] = sum(assay_totals)
    return losses


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


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    if tuple(config["contexts"]) != CONTEXTS:
        raise ValueError(f"Production context order must be {CONTEXTS}")
    if config["model"]["preset"] not in MODEL_PRESETS:
        raise ValueError("Unsupported model preset")
    return config


def architecture_metadata(
    model: EnformerLikeJointProfileRegressor,
    contexts: tuple[str, ...],
) -> dict[str, Any]:
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
    atomic_torch_save(
        {
            "kind": "enhancer_pleiotropy_training_state",
            "version": 1,
            "run_signature": run_signature,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": capture_rng_state(device),
            "progress": progress,
        },
        path,
    )


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
    parser.add_argument(
        "--stage",
        choices=("base", "specificity"),
        default="base",
        help="Train the broad base model or fine-tune it on specific peaks.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = dict(config["training"])
    stage = args.stage
    specificity_config = dict(config.get("specificity_finetuning", {}))
    if stage == "specificity":
        if not specificity_config.get("enabled", False):
            raise ValueError("Specificity fine-tuning is not enabled in the config")
        training.update(dict(specificity_config["training"]))
    if args.device is not None:
        training["device"] = args.device
    if args.mixed_precision is not None:
        training["mixed_precision"] = args.mixed_precision
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
        train_indices, train_specific_counts = select_specific_peak_indices(
            records["train"],
            atac_profiles["train"],
            h3_profiles["train"],
            thresholds,
        )
        validation_indices, validation_specific_counts = select_specific_peak_indices(
            records["validation"],
            atac_profiles["validation"],
            h3_profiles["validation"],
            thresholds,
        )
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
    model = EnformerLikeJointProfileRegressor(
        context_count=len(contexts),
        dropout=float(config["model"]["dropout"]),
        head_dropout=float(config["model"]["head_dropout"]),
        model_size=str(config["model"]["preset"]),
        h3k27ac_output_pool_size=h3_pool_size,
    ).to(device)
    model.initialize_output_means(target_means["atac"], target_means["h3k27ac"])
    initialization_metadata: dict[str, Any] | None = None
    if stage == "specificity":
        initialization_path = root / str(specificity_config["initialization_checkpoint"])
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
        training, h3_means, h3_standard_deviations, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["max_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    batches_per_epoch = math.ceil(len(train_dataset) / batch_size)
    epochs = int(training["epochs"])
    warmup_steps = max(
        1, round(epochs * batches_per_epoch * float(training["warmup_fraction"]))
    )
    decay_steps = max(
        0,
        round(
            batches_per_epoch
            * float(training.get("post_warmup_decay_epochs", 0.0))
        ),
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
    last_path = model_directory / "last_checkpoint.pt"
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 0
    start_batch = 0
    running_sums = {"atac": 0.0, "h3k27ac": 0.0, "total": 0.0}
    running_examples = 0
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
        epochs_without_improvement = int(progress["epochs_without_improvement"])
        running_sums = dict(progress["running_sums"])
        running_examples = int(progress["running_examples"])
        print(json.dumps({"event": "training_resumed", "epoch": start_epoch + 1, "batch": start_batch}), flush=True)
    elif args.resume:
        print(json.dumps({"event": "resume_checkpoint_absent", "action": "new_run"}), flush=True)

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
        for relative_batch, batch in enumerate(train_loader, start=1):
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
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            example_count = len(atac_labels)
            running_examples += example_count
            for name, value in losses.items():
                running_sums[name] = running_sums.get(name, 0.0) + (
                    float(value.item()) * example_count
                )
            if absolute_batch % 100 == 0 or absolute_batch == batches_per_epoch:
                print(
                    json.dumps(
                        {
                            "event": "training_progress",
                            "epoch": epoch_index + 1,
                            "batch": absolute_batch,
                            "batches": batches_per_epoch,
                            "learning_rate": optimizer.param_groups[0]["lr"],
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
            if checkpoint_interval and absolute_batch % checkpoint_interval == 0:
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
                        "epochs_without_improvement": epochs_without_improvement,
                        "running_sums": running_sums,
                        "running_examples": running_examples,
                    },
                    device,
                )
            if args.smoke_test and absolute_batch >= 2:
                break

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
        epoch_metrics = {
            assay: assay_validation_metrics(
                labels[assay], predictions[assay],
                regulatory_mask[: len(labels[assay])], contexts
            )
            for assay in ASSAYS
        }
        epoch_metrics["h3k27ac"]["segments"] = h3k27ac_segment_metrics(
            labels["h3k27ac"], predictions["h3k27ac"], contexts
        )
        specificity_metrics: dict[str, Any] | None = None
        if stage == "specificity":
            if validation_specific_mask is None:
                raise RuntimeError("Specificity validation mask was not initialized")
            selected_mask = validation_specific_mask[: len(labels["atac"])]
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
        score_metrics = specificity_metrics or epoch_metrics
        score = scientific_composite(score_metrics)
        plateau = scheduler.step_validation(score)
        epoch_result = {
            "epoch": epoch_index + 1,
            "training_losses": {
                name: value / running_examples for name, value in running_sums.items()
            },
            "validation_losses": validation_losses,
            "scientific_composite": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "plateau": plateau,
            "validation": epoch_metrics,
        }
        if specificity_metrics is not None:
            epoch_result["specificity_validation"] = specificity_metrics
        history.append(epoch_result)
        print(json.dumps({"event": "epoch_complete", **epoch_result}, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch_index + 1
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "kind": "enformer_like_dense_atac_h3k27ac_profile_regressor",
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
                    "epoch": best_epoch,
                    "checkpoint_selection": {
                        "metric": (
                            "specificity_scientific_composite"
                            if stage == "specificity"
                            else "scientific_composite"
                        ),
                        "score": best_score,
                    },
                    "training_stage": stage,
                    "specificity": specificity_metadata,
                    "initialization": initialization_metadata,
                    "loss": loss_metadata,
                    "training_target_means": {
                        assay: values.tolist() for assay, values in target_means.items()
                    },
                    "learning_rate_schedule": scheduler.state_dict(),
                    "reverse_complement_augmentation": {
                        "strategy": "stochastic",
                        "stochastic_probability": training["stochastic_rc_probability"],
                    },
                    "validation_reverse_complement_ensemble": training["validation_rc_ensemble"],
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        save_training_state(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            run_signature,
            {
                "next_epoch": epoch_index + 1,
                "next_batch": 0,
                "history": history,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "running_sums": {"atac": 0.0, "h3k27ac": 0.0, "total": 0.0},
                "running_examples": 0,
            },
            device,
        )
        start_batch = 0
        if args.smoke_test or epochs_without_improvement >= int(training["patience"]):
            break

    metrics = {
        "method": "joint_atac_h3k27ac_profile_training_v1",
        "training_stage": stage,
        "contexts": list(contexts),
        "dataset": {"path": str(dataset_path), "sha256": run_signature["dataset_sha256"]},
        "split_counts": counts,
        "model": architecture,
        "best_epoch": best_epoch,
        "best_scientific_composite": best_score,
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
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
