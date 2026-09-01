"""Validation metrics used by checkpoint selection and diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import rankdata

from .constants import ASSAYS, CONTEXTS


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    contexts: tuple[str, ...] = CONTEXTS,
) -> dict[str, Any]:
    if labels.shape != predictions.shape or labels.ndim != 2:
        raise ValueError("Regression arrays must align as [examples, contexts]")
    if labels.shape[1] != len(contexts):
        raise ValueError("Context names do not match prediction columns")
    by_context = {}
    for index, context in enumerate(contexts):
        truth = labels[:, index].astype(np.float64)
        prediction = predictions[:, index].astype(np.float64)
        residual_sum = np.square(truth - prediction).sum()
        total_sum = np.square(truth - truth.mean()).sum()
        by_context[context] = {
            "n": len(truth),
            "pearson": correlation(truth, prediction),
            "spearman": correlation(rankdata(truth), rankdata(prediction)),
            "mae": float(np.abs(truth - prediction).mean()),
            "r2": float(1 - residual_sum / total_sum)
            if total_sum > 0
            else float("nan"),
        }
    macro = {
        metric: float(np.nanmean([values[metric] for values in by_context.values()]))
        for metric in ("pearson", "spearman", "mae", "r2")
    }
    return {"by_context": by_context, "macro": macro}


def dense_profile_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    contexts: tuple[str, ...] = CONTEXTS,
) -> dict[str, Any]:
    if labels.shape != predictions.shape or labels.ndim != 3:
        raise ValueError("Dense arrays must align as [examples, bins, contexts]")
    transformed_labels = np.log1p(labels).reshape(-1, labels.shape[-1])
    transformed_predictions = np.log1p(np.maximum(predictions, 0)).reshape(
        -1, predictions.shape[-1]
    )
    return regression_metrics(transformed_labels, transformed_predictions, contexts)


def tissue_pattern_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    if labels.shape != predictions.shape or labels.ndim != 2:
        raise ValueError("Tissue-pattern arrays must align")
    if eligible.shape != (len(labels),):
        raise ValueError("Eligibility mask has the wrong shape")
    label_centered = labels - labels.mean(axis=1, keepdims=True)
    prediction_centered = predictions - predictions.mean(axis=1, keepdims=True)
    variable = eligible.astype(bool) & (np.linalg.norm(label_centered, axis=1) > 0)
    numerator = np.sum(label_centered[variable] * prediction_centered[variable], axis=1)
    denominator = np.linalg.norm(label_centered[variable], axis=1) * np.linalg.norm(
        prediction_centered[variable], axis=1
    )
    values = np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan),
        where=denominator > 0,
    )
    finite = values[np.isfinite(values)]
    return {
        "eligible_windows": int(eligible.sum()),
        "variable_windows": int(variable.sum()),
        "finite_windows": int(len(finite)),
        "mean_pearson": float(finite.mean()) if len(finite) else float("nan"),
        "median_pearson": float(np.median(finite)) if len(finite) else float("nan"),
        "q25_pearson": float(np.quantile(finite, 0.25)) if len(finite) else float("nan"),
        "q75_pearson": float(np.quantile(finite, 0.75)) if len(finite) else float("nan"),
    }


def correlation_structure(
    labels: np.ndarray,
    predictions: np.ndarray,
    contexts: tuple[str, ...] = CONTEXTS,
) -> dict[str, Any]:
    if labels.shape != predictions.shape or labels.ndim != 2 or len(labels) < 2:
        raise ValueError("Correlation matrices need aligned loci by context arrays")
    true_matrix = np.corrcoef(labels, rowvar=False)
    predicted_matrix = np.corrcoef(predictions, rowvar=False)
    rows, columns = np.triu_indices(len(contexts), 1)
    differences = predicted_matrix[rows, columns] - true_matrix[rows, columns]
    finite = np.isfinite(differences)
    return {
        "contexts": list(contexts),
        "true_matrix": true_matrix.tolist(),
        "predicted_matrix": predicted_matrix.tolist(),
        "mean_prediction_minus_true": float(differences[finite].mean())
        if finite.any()
        else float("nan"),
        "mean_absolute_error": float(np.abs(differences[finite]).mean())
        if finite.any()
        else float("nan"),
        "fraction_more_correlated_than_true": float((differences[finite] > 0).mean())
        if finite.any()
        else float("nan"),
    }


def target_pca_projection_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    contexts: tuple[str, ...] = CONTEXTS,
) -> dict[str, Any]:
    if labels.shape != predictions.shape or labels.ndim != 2 or len(labels) < 2:
        raise ValueError("PCA diagnostics need aligned loci by context arrays")
    target_mean = labels.astype(np.float64).mean(axis=0)
    centered_labels = labels.astype(np.float64) - target_mean
    centered_predictions = predictions.astype(np.float64) - target_mean
    covariance = centered_labels.T @ centered_labels / (len(labels) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    eigenvectors = eigenvectors[:, order]
    target_scores = centered_labels @ eigenvectors
    prediction_scores = centered_predictions @ eigenvectors
    total_variance = float(eigenvalues.sum())
    components = []
    for index in range(len(contexts)):
        truth = target_scores[:, index]
        prediction = prediction_scores[:, index]
        total_sum = np.square(truth - truth.mean()).sum()
        components.append(
            {
                "component": index + 1,
                "target_explained_variance_ratio": float(eigenvalues[index] / total_variance)
                if total_variance > 0
                else 0.0,
                "target_eigenvalue": float(eigenvalues[index]),
                "prediction_pearson": correlation(truth, prediction),
                "prediction_r2": float(1 - np.square(truth - prediction).sum() / total_sum)
                if total_sum > 0
                else float("nan"),
                "loadings": eigenvectors[:, index].tolist(),
            }
        )
    return {"contexts": list(contexts), "components": components}


def assay_validation_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    regulatory_mask: np.ndarray,
    contexts: tuple[str, ...] = CONTEXTS,
) -> dict[str, Any]:
    mean_labels = np.log1p(labels.mean(axis=1))
    mean_predictions = np.log1p(np.maximum(predictions, 0).mean(axis=1))
    result = {
        "window_mean": regression_metrics(mean_labels, mean_predictions, contexts),
        "dense_profile": dense_profile_metrics(labels, predictions, contexts),
        "tissue_pattern": tissue_pattern_metrics(
            mean_labels, mean_predictions, regulatory_mask
        ),
        "all_windows": {
            "correlation_structure": correlation_structure(
                mean_labels, mean_predictions, contexts
            ),
            "target_pca": target_pca_projection_metrics(
                mean_labels, mean_predictions, contexts
            ),
        },
    }
    if regulatory_mask.sum() >= 2:
        result["regulatory_windows"] = {
            "correlation_structure": correlation_structure(
                mean_labels[regulatory_mask],
                mean_predictions[regulatory_mask],
                contexts,
            ),
            "target_pca": target_pca_projection_metrics(
                mean_labels[regulatory_mask],
                mean_predictions[regulatory_mask],
                contexts,
            ),
        }
    return result


def h3k27ac_segment_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    contexts: tuple[str, ...] = CONTEXTS,
) -> dict[str, Any]:
    if labels.shape[1] % 3:
        raise ValueError("H3K27ac profile cannot be divided into three segments")
    bins = labels.shape[1] // 3
    truth = np.log1p(labels.reshape(len(labels), 3, bins, len(contexts)).mean(axis=2))
    predicted = np.log1p(
        np.maximum(predictions, 0).reshape(len(labels), 3, bins, len(contexts)).mean(axis=2)
    )
    return {
        "all_segments": regression_metrics(
            truth.reshape(-1, len(contexts)),
            predicted.reshape(-1, len(contexts)),
            contexts,
        ),
        "center": regression_metrics(truth[:, 1], predicted[:, 1], contexts),
        "maximum": regression_metrics(
            truth.max(axis=1), predicted.max(axis=1), contexts
        ),
    }


def scientific_composite(metrics: dict[str, dict[str, Any]]) -> float:
    values = [
        metrics[assay]["window_mean"]["macro"]["pearson"] for assay in ASSAYS
    ] + [metrics[assay]["tissue_pattern"]["mean_pearson"] for assay in ASSAYS]
    score = float(np.mean(values))
    if not math.isfinite(score):
        raise ValueError("Scientific composite is not finite")
    return score
