"""Calibrate enhancer activity from frozen ATAC/H3K27ac regressor outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .constants import CONTEXTS
from .enhancer_catalog_evaluation import (
    binary_curve,
    choose_f1_threshold,
    empirical_percentiles,
)
from .io import atomic_write_json, atomic_write_text, sha256_file


MODEL_NAMES = (
    "geometric_mean_percentiles",
    "fuzzy_and_percentiles",
    "component_probability_product",
    "cross_context_component_probability_product",
    "context_local_logistic",
    "atac_only_8_feature_logistic",
    "h3k27ac_only_8_feature_logistic",
    "joint_16_feature_logistic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--activity-threshold", default=0.6, type=float)
    parser.add_argument("--buffer-bp", default=10_000, type=int)
    parser.add_argument("--l2", default=1e-2, type=float)
    parser.add_argument("--reliability-bins", default=10, type=int)
    return parser.parse_args()


def genomic_calibration_split(
    validation: np.ndarray,
    chromosomes: np.ndarray,
    positions: np.ndarray,
    buffer_bp: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    """Split one validation chromosome at its coordinate midpoint with a gap."""
    if buffer_bp < 0:
        raise ValueError("buffer_bp must be non-negative")
    validation_chromosomes = np.unique(chromosomes[validation])
    if len(validation_chromosomes) != 1:
        raise ValueError("Calibration requires exactly one validation chromosome")
    chromosome = str(validation_chromosomes[0])
    coordinates = positions[validation]
    midpoint = (int(coordinates.min()) + int(coordinates.max())) // 2
    fit = validation & (positions <= midpoint - buffer_bp)
    threshold = validation & (positions >= midpoint + buffer_bp)
    if not fit.any() or not threshold.any():
        raise ValueError("Buffered calibration split produced an empty interval")
    return fit, threshold, {
        "chromosome": chromosome,
        "midpoint": midpoint,
        "buffer_bp": buffer_bp,
        "excluded_n": int(np.sum(validation & ~(fit | threshold))),
        "fit_n": int(fit.sum()),
        "threshold_selection_n": int(threshold.sum()),
    }


def _standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (values - mean) / scale, mean, scale


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Fit one deterministic L2-regularized logistic head with L-BFGS."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError("Expected features [N,F] and labels [N]")
    if not np.isin(labels, (0, 1)).all() or not 0 < labels.sum() < len(labels):
        raise ValueError("Logistic labels must contain both classes")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    design = np.column_stack((np.ones(len(features)), features))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ parameters
        loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)
        loss += 0.5 * l2 * float(parameters[1:] @ parameters[1:])
        gradient = design.T @ (expit(logits) - labels) / len(labels)
        gradient[1:] += l2 * parameters[1:]
        return float(loss), gradient

    initial = np.zeros(design.shape[1], dtype=np.float64)
    prevalence = np.clip(labels.mean(), 1e-6, 1 - 1e-6)
    initial[0] = np.log(prevalence / (1 - prevalence))
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Logistic optimization failed: {result.message}")
    return result.x, {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "message": str(result.message),
    }


def predict_logistic(features: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if features.ndim != 2 or parameters.shape != (features.shape[1] + 1,):
        raise ValueError("Logistic parameter shape does not match features")
    return expit(parameters[0] + features @ parameters[1:])


def probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    curve = binary_curve(labels, probabilities)
    calls = probabilities >= threshold
    true_positive = int(np.sum(calls & (labels == 1)))
    false_positive = int(np.sum(calls & (labels == 0)))
    false_negative = int(np.sum(~calls & (labels == 1)))
    precision = true_positive / (true_positive + false_positive) if calls.any() else 0.0
    recall = true_positive / (true_positive + false_negative)
    denominator = precision + recall
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "n": int(len(labels)),
        "positive_n": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / denominator) if denominator else 0.0,
        "average_precision": float(curve["average_precision"]),
        "auprc_trapezoidal": float(curve["auprc_trapezoidal"]),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
        ),
    }


def reliability_rows(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> tuple[list[dict[str, float | int]], float]:
    if bins < 2:
        raise ValueError("At least two reliability bins are required")
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    assignments = np.minimum((probabilities * bins).astype(int), bins - 1)
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        confidence = float(probabilities[selected].mean()) if count else float("nan")
        observed = float(labels[selected].mean()) if count else float("nan")
        if count:
            ece += count / len(labels) * abs(confidence - observed)
        rows.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "n": count,
                "mean_probability": confidence,
                "observed_fraction": observed,
            }
        )
    return rows, float(ece)


def _load_catalog(
    path: Path,
    identifiers: np.ndarray,
    stored_chromosomes: np.ndarray,
    stored_labels: np.ndarray,
    activity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = ["master_dhs_id", "chrom", "summit"]
    for context in CONTEXTS:
        columns.extend(
            (
                f"{context}__context_membership",
                f"{context}__h3k27ac_max_500_background_percentile",
            )
        )
    frame = pd.read_csv(path, sep="\t", compression="infer", usecols=columns)
    if frame["master_dhs_id"].duplicated().any():
        raise ValueError("Catalog contains duplicate enhancer IDs")
    aligned = frame.set_index("master_dhs_id").reindex(identifiers)
    if aligned.isna().any().any():
        raise ValueError("Catalog metadata are missing for cached predictions")
    chromosomes = aligned["chrom"].astype(str).to_numpy(dtype=np.str_)
    if not np.array_equal(chromosomes, stored_chromosomes):
        raise ValueError("Catalog chromosome order disagrees with predictions")
    membership = aligned[
        [f"{context}__context_membership" for context in CONTEXTS]
    ].to_numpy(float) > 0.5
    h3_active = aligned[
        [f"{context}__h3k27ac_max_500_background_percentile" for context in CONTEXTS]
    ].to_numpy(float) > activity_threshold
    if not np.array_equal(membership & h3_active, stored_labels):
        raise ValueError("Catalog activity labels disagree with cached predictions")
    return aligned["summit"].to_numpy(np.int64), membership, h3_active


def _macro(by_context: dict[str, dict[str, float | int]]) -> dict[str, float]:
    fields = (
        "precision",
        "recall",
        "f1",
        "average_precision",
        "auprc_trapezoidal",
        "brier",
        "log_loss",
        "ece",
    )
    return {
        field: float(np.mean([float(metrics[field]) for metrics in by_context.values()]))
        for field in fields
    }


def main() -> None:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    cached = np.load(args.predictions, allow_pickle=False)
    required = {"ids", "chrom", "split", "labels", "features", "contexts", "feature_names"}
    missing = required - set(cached.files)
    if missing:
        raise ValueError(f"Predictions lack arrays: {sorted(missing)}")
    if tuple(cached["contexts"].tolist()) != CONTEXTS:
        raise ValueError("Cached context ordering is not the production ordering")
    identifiers = cached["ids"]
    chromosomes = cached["chrom"]
    splits = cached["split"]
    labels = cached["labels"].astype(np.int8)
    features = cached["features"].astype(np.float64)
    if features.shape != (len(labels), len(CONTEXTS), 6):
        raise ValueError(f"Unexpected summarized feature shape: {features.shape}")
    positions, membership, h3_active = _load_catalog(
        args.catalog,
        identifiers,
        chromosomes,
        labels,
        args.activity_threshold,
    )

    validation = splits == "validation"
    calibration_fit, threshold_selection, partition = genomic_calibration_split(
        validation, chromosomes, positions, args.buffer_bp
    )
    test = splits == "test"
    if not test.any():
        raise ValueError("Predictions contain no held-out test examples")

    raw_joint = np.concatenate((features[:, :, 0], features[:, :, 5]), axis=1)
    transformed_joint = np.log1p(np.maximum(raw_joint, 0))
    standardized_fit, mean, scale = _standardize_fit(transformed_joint[calibration_fit])
    standardized = (transformed_joint - mean) / scale

    probabilities: dict[str, np.ndarray] = {
        name: np.empty(labels.shape, dtype=np.float64) for name in MODEL_NAMES
    }
    train = splits == "train"
    for context_index in range(len(CONTEXTS)):
        atac_percentile = empirical_percentiles(
            raw_joint[train, context_index], raw_joint[:, context_index]
        )
        h3_percentile = empirical_percentiles(
            raw_joint[train, len(CONTEXTS) + context_index],
            raw_joint[:, len(CONTEXTS) + context_index],
        )
        probabilities["geometric_mean_percentiles"][:, context_index] = np.sqrt(
            atac_percentile * h3_percentile
        )
        probabilities["fuzzy_and_percentiles"][:, context_index] = np.minimum(
            atac_percentile, h3_percentile
        )

    parameter_records: dict[str, object] = {}
    component_atac_probabilities = np.empty(labels.shape, dtype=np.float64)
    component_h3_probabilities = np.empty(labels.shape, dtype=np.float64)
    cross_context_atac_probabilities = np.empty(labels.shape, dtype=np.float64)
    cross_context_h3_probabilities = np.empty(labels.shape, dtype=np.float64)
    component_atac_parameters = []
    component_h3_parameters = []
    component_atac_optimization = {}
    component_h3_optimization = {}
    cross_context_atac_parameters = []
    cross_context_h3_parameters = []
    cross_context_atac_optimization = {}
    cross_context_h3_optimization = {}
    local_parameters = []
    local_optimization = {}
    atac_parameters = []
    atac_optimization = {}
    h3_parameters = []
    h3_optimization = {}
    joint_parameters = []
    joint_optimization = {}
    for context_index, context in enumerate(CONTEXTS):
        local_columns = (context_index, len(CONTEXTS) + context_index)

        parameters, optimization = fit_logistic(
            standardized_fit[:, [context_index]],
            membership[calibration_fit, context_index],
            args.l2,
        )
        component_atac_probabilities[:, context_index] = predict_logistic(
            standardized[:, [context_index]], parameters
        )
        component_atac_parameters.append(parameters)
        component_atac_optimization[context] = optimization

        h3_column = len(CONTEXTS) + context_index
        parameters, optimization = fit_logistic(
            standardized_fit[:, [h3_column]],
            h3_active[calibration_fit, context_index],
            args.l2,
        )
        component_h3_probabilities[:, context_index] = predict_logistic(
            standardized[:, [h3_column]], parameters
        )
        component_h3_parameters.append(parameters)
        component_h3_optimization[context] = optimization

        parameters, optimization = fit_logistic(
            standardized_fit[:, : len(CONTEXTS)],
            membership[calibration_fit, context_index],
            args.l2,
        )
        cross_context_atac_probabilities[:, context_index] = predict_logistic(
            standardized[:, : len(CONTEXTS)], parameters
        )
        cross_context_atac_parameters.append(parameters)
        cross_context_atac_optimization[context] = optimization

        parameters, optimization = fit_logistic(
            standardized_fit[:, len(CONTEXTS) :],
            h3_active[calibration_fit, context_index],
            args.l2,
        )
        cross_context_h3_probabilities[:, context_index] = predict_logistic(
            standardized[:, len(CONTEXTS) :], parameters
        )
        cross_context_h3_parameters.append(parameters)
        cross_context_h3_optimization[context] = optimization

        parameters, optimization = fit_logistic(
            standardized_fit[:, local_columns],
            labels[calibration_fit, context_index],
            args.l2,
        )
        probabilities["context_local_logistic"][:, context_index] = predict_logistic(
            standardized[:, local_columns], parameters
        )
        local_parameters.append(parameters)
        local_optimization[context] = optimization

        parameters, optimization = fit_logistic(
            standardized_fit[:, : len(CONTEXTS)],
            labels[calibration_fit, context_index],
            args.l2,
        )
        probabilities["atac_only_8_feature_logistic"][:, context_index] = (
            predict_logistic(standardized[:, : len(CONTEXTS)], parameters)
        )
        atac_parameters.append(parameters)
        atac_optimization[context] = optimization

        parameters, optimization = fit_logistic(
            standardized_fit[:, len(CONTEXTS) :],
            labels[calibration_fit, context_index],
            args.l2,
        )
        probabilities["h3k27ac_only_8_feature_logistic"][:, context_index] = (
            predict_logistic(standardized[:, len(CONTEXTS) :], parameters)
        )
        h3_parameters.append(parameters)
        h3_optimization[context] = optimization

        parameters, optimization = fit_logistic(
            standardized_fit, labels[calibration_fit, context_index], args.l2
        )
        probabilities["joint_16_feature_logistic"][:, context_index] = predict_logistic(
            standardized, parameters
        )
        joint_parameters.append(parameters)
        joint_optimization[context] = optimization
    probabilities["component_probability_product"] = (
        component_atac_probabilities * component_h3_probabilities
    )
    probabilities["cross_context_component_probability_product"] = (
        cross_context_atac_probabilities * cross_context_h3_probabilities
    )
    parameter_records["component_atac_membership_logistic"] = (
        component_atac_optimization
    )
    parameter_records["component_h3k27ac_activity_logistic"] = (
        component_h3_optimization
    )
    parameter_records["cross_context_atac_membership_logistic"] = (
        cross_context_atac_optimization
    )
    parameter_records["cross_context_h3k27ac_activity_logistic"] = (
        cross_context_h3_optimization
    )
    parameter_records["context_local_logistic"] = local_optimization
    parameter_records["atac_only_8_feature_logistic"] = atac_optimization
    parameter_records["h3k27ac_only_8_feature_logistic"] = h3_optimization
    parameter_records["joint_16_feature_logistic"] = joint_optimization

    result_models: dict[str, object] = {}
    reliability: list[dict[str, object]] = []
    selected_thresholds: dict[str, np.ndarray] = {}
    for model_name, model_probabilities in probabilities.items():
        thresholds = np.empty(len(CONTEXTS), dtype=np.float64)
        threshold_f1 = {}
        for context_index, context in enumerate(CONTEXTS):
            thresholds[context_index], threshold_f1[context] = choose_f1_threshold(
                labels[threshold_selection, context_index],
                model_probabilities[threshold_selection, context_index],
            )
        selected_thresholds[model_name] = thresholds
        split_metrics = {}
        for split_name, mask in (
            ("calibration_fit", calibration_fit),
            ("threshold_selection", threshold_selection),
            ("test", test),
        ):
            by_context = {}
            for context_index, context in enumerate(CONTEXTS):
                metrics = probability_metrics(
                    labels[mask, context_index],
                    model_probabilities[mask, context_index],
                    thresholds[context_index],
                )
                rows, ece = reliability_rows(
                    labels[mask, context_index],
                    model_probabilities[mask, context_index],
                    args.reliability_bins,
                )
                metrics["ece"] = ece
                by_context[context] = metrics
                for row in rows:
                    reliability.append(
                        {
                            "model": model_name,
                            "split": split_name,
                            "context": context,
                            **row,
                        }
                    )
            split_metrics[split_name] = {
                "by_context": by_context,
                "macro": _macro(by_context),
            }
        result_models[model_name] = {
            "threshold_selection": {
                "split": "right buffered validation interval",
                "criterion": "maximum F1 independently per context",
                "thresholds": dict(zip(CONTEXTS, thresholds.tolist(), strict=True)),
                "f1": threshold_f1,
            },
            **split_metrics,
        }

    np.savez_compressed(
        args.output_directory / "calibrated_predictions.npz",
        ids=identifiers,
        split=splits,
        labels=labels.astype(np.uint8),
        atac_membership_labels=membership.astype(np.uint8),
        h3k27ac_activity_labels=h3_active.astype(np.uint8),
        contexts=np.asarray(CONTEXTS),
        model_names=np.asarray(MODEL_NAMES),
        probabilities=np.stack([probabilities[name] for name in MODEL_NAMES]),
        thresholds=np.stack([selected_thresholds[name] for name in MODEL_NAMES]),
        atac_activity_probabilities=component_atac_probabilities,
        h3k27ac_activity_probabilities=component_h3_probabilities,
        smooth_enhancer_scores=probabilities["component_probability_product"],
        cross_context_atac_activity_probabilities=cross_context_atac_probabilities,
        cross_context_h3k27ac_activity_probabilities=cross_context_h3_probabilities,
        cross_context_smooth_enhancer_scores=probabilities[
            "cross_context_component_probability_product"
        ],
    )
    np.savez_compressed(
        args.output_directory / "calibrator_parameters.npz",
        feature_names=np.asarray(
            [f"atac_mean_512__{context}" for context in CONTEXTS]
            + [f"h3k27ac_max_mean_512__{context}" for context in CONTEXTS]
        ),
        transform_mean=mean,
        transform_scale=scale,
        component_atac_parameters=np.stack(component_atac_parameters),
        component_h3k27ac_parameters=np.stack(component_h3_parameters),
        cross_context_atac_parameters=np.stack(cross_context_atac_parameters),
        cross_context_h3k27ac_parameters=np.stack(cross_context_h3_parameters),
        context_local_parameters=np.stack(local_parameters),
        atac_only_8_feature_parameters=np.stack(atac_parameters),
        h3k27ac_only_8_feature_parameters=np.stack(h3_parameters),
        joint_16_feature_parameters=np.stack(joint_parameters),
        contexts=np.asarray(CONTEXTS),
    )

    reliability_path = args.output_directory / "reliability.tsv"
    with reliability_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reliability[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(reliability)

    result = {
        "method": "frozen_regressor_enhancer_calibration_v1",
        "inputs": {
            "predictions": {
                "path": str(args.predictions),
                "sha256": sha256_file(args.predictions),
            },
            "catalog": {"path": str(args.catalog), "sha256": sha256_file(args.catalog)},
        },
        "target": f"ATAC context membership AND H3K27ac background percentile > {args.activity_threshold}",
        "features": {
            "atac": "predicted central-512-bp mean per context",
            "h3k27ac": "predicted maximum of left/center/right 512-bp means per context",
            "transform": "log1p, then mean/std fitted on calibration-fit interval only",
            "joint_order": "eight ATAC contexts followed by eight H3K27ac contexts",
            "smooth_enhancer_score": "separately calibrated ATAC-membership probability multiplied by separately calibrated H3K27ac>0.6 probability",
            "cross_context_smooth_enhancer_score": "same component product, but each assay-specific probability head may use all eight contexts from that assay",
        },
        "partition": {
            **partition,
            "fit_interval": f"summit <= {partition['midpoint'] - args.buffer_bp}",
            "threshold_selection_interval": f"summit >= {partition['midpoint'] + args.buffer_bp}",
            "test_split": "existing held-out test chromosome",
            "test_chromosomes": sorted(set(chromosomes[test].tolist())),
            "test_n": int(test.sum()),
            "regressor_train_examples_excluded_from_classifier_fitting": True,
        },
        "optimization": {
            "classifier": "independent L2-regularized logistic heads fitted with L-BFGS",
            "l2": args.l2,
            "class_weighting": "none (preserves probability calibration)",
            "details": parameter_records,
        },
        "models": result_models,
        "metric_notes": {
            "average_precision": "ranking metric; prevalence is its random baseline",
            "auprc_trapezoidal": "trapezoidal area under the precision-recall curve",
            "brier": "mean squared probability error; lower is better",
            "log_loss": "binary cross-entropy of probabilities; lower is better",
            "ece": f"expected calibration error over {args.reliability_bins} equal-width probability bins; lower is better",
            "baseline_calibration": "percentile baselines are unit-interval scores rather than fitted probabilities; their Brier/ECE values quantify this lack of calibration",
            "component_product_limit": "the product is an interpretable continuous score, not a guaranteed joint probability, because ATAC and H3K27ac activity are not conditionally independent",
            "negative_class": "inactive in the target context but active in at least one other retained context",
        },
    }
    atomic_write_json(args.output_directory / "metrics.json", result)

    test_rows = []
    for name in MODEL_NAMES:
        macro = result_models[name]["test"]["macro"]  # type: ignore[index]
        test_rows.append(
            "| " + name + " | " + " | ".join(
                f"{float(macro[field]):.4f}"
                for field in (
                    "average_precision",
                    "auprc_trapezoidal",
                    "precision",
                    "recall",
                    "f1",
                    "brier",
                    "ece",
                )
            ) + " |"
        )
    context_rows = []
    baseline_contexts = result_models["geometric_mean_percentiles"]["test"]["by_context"]  # type: ignore[index]
    atac_contexts = result_models["atac_only_8_feature_logistic"]["test"]["by_context"]  # type: ignore[index]
    h3_contexts = result_models["h3k27ac_only_8_feature_logistic"]["test"]["by_context"]  # type: ignore[index]
    joint_contexts = result_models["joint_16_feature_logistic"]["test"]["by_context"]  # type: ignore[index]
    for context in CONTEXTS:
        baseline = baseline_contexts[context]
        atac = atac_contexts[context]
        h3 = h3_contexts[context]
        joint = joint_contexts[context]
        context_rows.append(
            f"| {context} | {float(baseline['average_precision']):.4f} | "
            f"{float(atac['average_precision']):.4f} | "
            f"{float(h3['average_precision']):.4f} | "
            f"{float(joint['average_precision']):.4f} | "
            f"{float(joint['average_precision']) - float(atac['average_precision']):+.4f} |"
        )
    summary = "\n".join(
        (
            "# Enhancer calibration summary",
            "",
            f"Frozen predictions: `{args.predictions}`",
            "",
            f"Calibration fit: {partition['chromosome']} left interval ({partition['fit_n']} enhancers); "
            f"threshold selection: right interval ({partition['threshold_selection_n']} enhancers); "
            f"buffer exclusion: {partition['excluded_n']}; test: "
            f"{', '.join(sorted(set(chromosomes[test].tolist())))} ({int(test.sum())} enhancers).",
            "",
            "| Model | AP | AUPRC | Precision | Recall | F1 | Brier | ECE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *test_rows,
            "",
            "## Assay ablation by context",
            "",
            "| Context | Baseline AP | ATAC-only AP | H3K27ac-only AP | Joint AP | Joint minus ATAC |",
            "|---|---:|---:|---:|---:|---:|",
            *context_rows,
            "",
            "Lower Brier score and ECE indicate better probability calibration. Test data were not used to fit weights, standardization, or thresholds.",
            "",
        )
    )
    atomic_write_text(args.output_directory / "summary.md", summary)
    print(
        json.dumps(
            {
                "event": "enhancer_calibration_complete",
                "metrics": str(args.output_directory / "metrics.json"),
                "summary": str(args.output_directory / "summary.md"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
