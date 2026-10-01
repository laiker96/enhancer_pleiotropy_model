"""Evaluate a base/specificity residual ensemble on validation data only."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .constants import ASSAYS
from .data import JointProfileDataset, load_profiles, make_loader, read_windows
from .inference import load_model, resolve_device
from .io import atomic_write_json, atomic_write_text, sha256_file
from .metrics import (
    correlation_structure,
    regression_metrics,
    regulatory_overcorrelation_summary,
    scientific_composite,
    tissue_pattern_metrics,
)
from .preprocessing.windows import BACKGROUND_SOURCE
from .training import (
    balance_specific_peak_contexts,
    evaluate,
    fit_specificity_thresholds,
    load_config,
    select_specific_peak_indices_and_contexts,
    validation_metric_sets,
)


def residual_ensemble_predictions(
    base: np.ndarray,
    specific: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Blend context residuals in log1p space while retaining the base mean."""
    if base.shape != specific.shape or base.ndim != 3:
        raise ValueError("Predictions must align as [examples, bins, contexts]")
    if not 0 <= alpha <= 1:
        raise ValueError("Residual ensemble alpha must be between zero and one")
    if np.any(base < 0) or np.any(specific < 0):
        raise ValueError("Residual ensemble predictions must be nonnegative")
    if alpha == 0:
        return np.asarray(base, dtype=np.float32).copy(), 0.0

    base_log = np.log1p(base.astype(np.float64))
    specific_log = np.log1p(specific.astype(np.float64))
    base_mean = base_log.mean(axis=-1, keepdims=True)
    base_residual = base_log - base_mean
    specific_residual = specific_log - specific_log.mean(axis=-1, keepdims=True)
    combined_log = (
        base_mean + (1 - alpha) * base_residual + alpha * specific_residual
    )
    clipped = combined_log < 0
    combined = np.expm1(np.maximum(combined_log, 0)).astype(np.float32)
    return combined, float(clipped.mean())


def compact_assay_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    regulatory_mask: np.ndarray,
    contexts: tuple[str, ...],
) -> dict[str, Any]:
    """Calculate only metrics needed during the alpha sweep."""
    mean_labels = np.log1p(labels.mean(axis=1))
    mean_predictions = np.log1p(np.maximum(predictions, 0).mean(axis=1))
    return {
        "window_mean": regression_metrics(mean_labels, mean_predictions, contexts),
        "tissue_pattern": tissue_pattern_metrics(
            mean_labels, mean_predictions, regulatory_mask
        ),
        "regulatory_windows": {
            "correlation_structure": correlation_structure(
                mean_labels[regulatory_mask],
                mean_predictions[regulatory_mask],
                contexts,
            )
        },
    }


def combination_summary(
    sweep: dict[str, dict[float, dict[str, Any]]],
    alpha_atac: float,
    alpha_h3k27ac: float,
) -> dict[str, Any]:
    alpha_by_assay = {"atac": alpha_atac, "h3k27ac": alpha_h3k27ac}
    full = {
        assay: sweep[assay][alpha_by_assay[assay]]["validation"]
        for assay in ASSAYS
    }
    specific = {
        assay: sweep[assay][alpha_by_assay[assay]]["specificity_validation"]
        for assay in ASSAYS
    }
    return {
        "alpha_atac": alpha_atac,
        "alpha_h3k27ac": alpha_h3k27ac,
        "validation_composite": scientific_composite(full),
        "specificity_composite": scientific_composite(specific),
        "validation_overcorrelation": regulatory_overcorrelation_summary(full),
        "specificity_overcorrelation": regulatory_overcorrelation_summary(specific),
    }


def parse_alphas(value: str) -> tuple[float, ...]:
    alphas = tuple(float(item) for item in value.split(","))
    if not alphas or any(not 0 <= alpha <= 1 for alpha in alphas):
        raise argparse.ArgumentTypeError("Alphas must be comma-separated values in [0,1]")
    if len(set(alphas)) != len(alphas):
        raise argparse.ArgumentTypeError("Alphas must be unique")
    return alphas


def resolve_under_root(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def checkpoint_dataset_hash(path: Path) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    value = checkpoint.get("dataset_sha256")
    if not value:
        raise ValueError(f"{path}: checkpoint lacks dataset_sha256")
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--specific-checkpoint", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument(
        "--alphas",
        type=parse_alphas,
        default=parse_alphas("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"),
    )
    parser.add_argument(
        "--maximum-general-composite-drop-fraction", default=0.01, type=float
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.maximum_general_composite_drop_fraction < 1:
        raise ValueError("Maximum general composite drop fraction must be in [0,1)")
    config = load_config(args.config)
    root = Path(config["output_directory"])
    data_directory = root / "data"
    dataset_path = data_directory / "windows.tsv.gz"
    base_checkpoint = resolve_under_root(args.base_checkpoint, root)
    specific_checkpoint = resolve_under_root(args.specific_checkpoint, root)
    dataset_hash = sha256_file(dataset_path)
    for checkpoint in (base_checkpoint, specific_checkpoint):
        if checkpoint_dataset_hash(checkpoint) != dataset_hash:
            raise ValueError(f"{checkpoint}: checkpoint and validation dataset differ")

    records = read_windows(dataset_path)
    counts = {split: len(values) for split, values in records.items()}
    atac_profiles, atac_metadata = load_profiles(
        data_directory / "profiles" / "atac", dataset_path, counts
    )
    h3_profiles, h3_metadata = load_profiles(
        data_directory / "profiles" / "h3k27ac", dataset_path, counts
    )
    contexts = tuple(config["contexts"])
    if tuple(atac_metadata["contexts"]) != contexts or tuple(
        h3_metadata["contexts"]
    ) != contexts:
        raise ValueError("Profile context order differs from configuration")

    specificity_config = config["specificity_finetuning"]
    thresholds = fit_specificity_thresholds(
        records["train"],
        atac_profiles["train"],
        h3_profiles["train"],
        float(specificity_config["gini_standard_deviations"]),
    )
    validation_indices, dominant_contexts, specificity_counts = (
        select_specific_peak_indices_and_contexts(
            records["validation"],
            atac_profiles["validation"],
            h3_profiles["validation"],
            thresholds,
        )
    )
    validation_indices, balance_metadata = balance_specific_peak_contexts(
        validation_indices,
        dominant_contexts,
        len(contexts),
        int(config["seed"]) + 1,
        1.0,
    )
    specificity_mask = np.zeros(counts["validation"], dtype=np.bool_)
    specificity_mask[validation_indices] = True
    regulatory_mask = np.asarray(
        [record.source != BACKGROUND_SOURCE for record in records["validation"]],
        dtype=np.bool_,
    )

    validation_dataset = JointProfileDataset(
        records["validation"],
        atac_profiles["validation"],
        h3_profiles["validation"],
        int(config["profiles"]["h3k27ac_output_pool_size"]),
        training=False,
        rc_probability=0,
        seed=int(config["seed"]),
    )
    device = resolve_device(args.device)
    loader = make_loader(
        validation_dataset,
        batch_size=int(config["training"]["evaluation_batch_size"]),
        workers=int(config["training"]["num_workers"]),
        epoch=0,
        seed=int(config["seed"]),
        training=False,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.MSELoss()
    predictions: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, np.ndarray] | None = None
    model_metadata = {}
    for name, checkpoint in (
        ("base", base_checkpoint),
        ("specific", specific_checkpoint),
    ):
        model, metadata = load_model(checkpoint, device)
        _, current_labels, current_predictions = evaluate(
            model,
            loader,
            criterion,
            criterion,
            device,
            args.mixed_precision,
            True,
        )
        if labels is None:
            labels = current_labels
        elif any(
            not np.array_equal(labels[assay], current_labels[assay])
            for assay in ASSAYS
        ):
            raise RuntimeError("Validation labels changed between model evaluations")
        predictions[name] = current_predictions
        model_metadata[name] = {
            "path": str(checkpoint),
            "sha256": metadata.checkpoint_sha256,
            "epoch": metadata.epoch,
            "architecture": metadata.architecture,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert labels is not None
    if model_metadata["base"]["architecture"] != model_metadata["specific"]["architecture"]:
        raise ValueError("Residual ensemble checkpoints have different architectures")

    sweep: dict[str, dict[float, dict[str, Any]]] = {assay: {} for assay in ASSAYS}
    for assay in ASSAYS:
        for alpha in args.alphas:
            combined, clipped_fraction = residual_ensemble_predictions(
                predictions["base"][assay], predictions["specific"][assay], alpha
            )
            full_metrics = compact_assay_metrics(
                labels[assay], combined, regulatory_mask, contexts
            )
            specific_metrics = compact_assay_metrics(
                labels[assay][specificity_mask],
                combined[specificity_mask],
                np.ones(int(specificity_mask.sum()), dtype=np.bool_),
                contexts,
            )
            sweep[assay][alpha] = {
                "clipped_log_value_fraction": clipped_fraction,
                "validation": full_metrics,
                "specificity_validation": specific_metrics,
            }

    combinations = [
        combination_summary(sweep, alpha_atac, alpha_h3k27ac)
        for alpha_atac in args.alphas
        for alpha_h3k27ac in args.alphas
    ]
    baseline = next(
        row for row in combinations if row["alpha_atac"] == row["alpha_h3k27ac"] == 0
    )
    minimum_general_composite = baseline["validation_composite"] * (
        1 - args.maximum_general_composite_drop_fraction
    )
    eligible = [
        row
        for row in combinations
        if row["validation_composite"] >= minimum_general_composite
    ]
    selected = max(
        eligible,
        key=lambda row: (
            row["specificity_composite"],
            -row["specificity_overcorrelation"]["mean_positive_excess"],
            row["validation_composite"],
        ),
    )

    selected_predictions = {}
    for assay in ASSAYS:
        alpha = selected[f"alpha_{assay}"]
        selected_predictions[assay], _ = residual_ensemble_predictions(
            predictions["base"][assay], predictions["specific"][assay], alpha
        )
    detailed = {}
    for name, candidate_predictions in (
        ("base", predictions["base"]),
        ("specific", predictions["specific"]),
        ("selected_ensemble", selected_predictions),
    ):
        full_metrics, specific_metrics = validation_metric_sets(
            labels,
            candidate_predictions,
            regulatory_mask,
            contexts,
            specificity_mask,
        )
        assert specific_metrics is not None
        detailed[name] = {
            "validation": full_metrics,
            "specificity_validation": specific_metrics,
            "validation_composite": scientific_composite(full_metrics),
            "specificity_composite": scientific_composite(specific_metrics),
            "validation_overcorrelation": regulatory_overcorrelation_summary(
                full_metrics
            ),
            "specificity_overcorrelation": regulatory_overcorrelation_summary(
                specific_metrics
            ),
        }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    table_buffer = io.StringIO()
    writer = csv.DictWriter(
        table_buffer,
        fieldnames=(
            "alpha_atac",
            "alpha_h3k27ac",
            "validation_composite",
            "specificity_composite",
            "validation_overcorrelation",
            "specificity_overcorrelation",
            "eligible",
        ),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in combinations:
        writer.writerow(
            {
                "alpha_atac": row["alpha_atac"],
                "alpha_h3k27ac": row["alpha_h3k27ac"],
                "validation_composite": row["validation_composite"],
                "specificity_composite": row["specificity_composite"],
                "validation_overcorrelation": row["validation_overcorrelation"][
                    "mean_positive_excess"
                ],
                "specificity_overcorrelation": row[
                    "specificity_overcorrelation"
                ]["mean_positive_excess"],
                "eligible": row["validation_composite"]
                >= minimum_general_composite,
            }
        )
    atomic_write_text(args.output_directory / "alpha_grid.tsv", table_buffer.getvalue())
    report = {
        "method": "log1p_common_mean_context_residual_ensemble_v1",
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "dataset": {"path": str(dataset_path), "sha256": dataset_hash},
        "contexts": list(contexts),
        "checkpoints": model_metadata,
        "reverse_complement_ensemble": True,
        "alpha_grid": list(args.alphas),
        "selection": {
            "criterion": "maximum specificity composite subject to general constraint",
            "maximum_general_composite_drop_fraction": args.maximum_general_composite_drop_fraction,
            "minimum_general_composite": minimum_general_composite,
            "eligible_combinations": len(eligible),
            "selected": selected,
        },
        "specificity_subset": {
            "definition": "training-thresholded high-Gini peak windows, context-balanced",
            "thresholds": thresholds,
            "natural_counts": specificity_counts,
            "balance": balance_metadata,
        },
        "detailed_metrics": detailed,
    }
    output_path = args.output_directory / "metrics.json"
    atomic_write_json(output_path, report)
    print(
        json.dumps(
            {
                "event": "residual_ensemble_evaluation_complete",
                "selected": selected,
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
