"""Validation metrics for the regressors.

The checkpoint-selection score (``scientific_composite``) is the mean of four
Pearson correlations on log1p window means: the macro (per-context) correlation
across windows for ATAC and H3K27ac, and the mean per-window correlation across
the eight contexts ("tissue pattern") on regulatory windows for each assay.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata

from ..constants import ASSAYS, CONTEXTS


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def per_context(labels: np.ndarray, predictions: np.ndarray) -> dict:
    by_context = {}
    for index, context in enumerate(CONTEXTS):
        truth, estimate = labels[:, index].astype(np.float64), predictions[:, index].astype(np.float64)
        total = np.square(truth - truth.mean()).sum()
        by_context[context] = {
            "pearson": correlation(truth, estimate),
            "spearman": correlation(rankdata(truth), rankdata(estimate)),
            "r2": float(1 - np.square(truth - estimate).sum() / total) if total > 0 else float("nan"),
        }
    macro = {k: float(np.nanmean([v[k] for v in by_context.values()])) for k in ("pearson", "spearman", "r2")}
    return {"by_context": by_context, "macro": macro}


def tissue_pattern(labels: np.ndarray, predictions: np.ndarray, eligible: np.ndarray) -> dict:
    """Per-window Pearson correlation of the eight-context pattern."""
    centred_labels = labels - labels.mean(axis=1, keepdims=True)
    centred_predictions = predictions - predictions.mean(axis=1, keepdims=True)
    variable = eligible.astype(bool) & (np.linalg.norm(centred_labels, axis=1) > 0)
    numerator = np.sum(centred_labels[variable] * centred_predictions[variable], axis=1)
    denominator = (np.linalg.norm(centred_labels[variable], axis=1)
                   * np.linalg.norm(centred_predictions[variable], axis=1))
    values = np.divide(numerator, denominator, out=np.full(len(numerator), np.nan), where=denominator > 0)
    finite = values[np.isfinite(values)]
    return {"windows": int(len(finite)), "mean_pearson": float(finite.mean()) if len(finite) else float("nan")}


def assay_metrics(labels: np.ndarray, predictions: np.ndarray, regulatory: np.ndarray) -> dict:
    mean_labels = np.log1p(labels.mean(axis=1))
    mean_predictions = np.log1p(np.maximum(predictions, 0).mean(axis=1))
    dense_labels = np.log1p(labels).reshape(-1, labels.shape[-1])
    dense_predictions = np.log1p(np.maximum(predictions, 0)).reshape(-1, predictions.shape[-1])
    return {"window_mean": per_context(mean_labels, mean_predictions),
            "dense_profile": per_context(dense_labels, dense_predictions),
            "tissue_pattern": tissue_pattern(mean_labels, mean_predictions, regulatory)}


def scientific_composite(metrics: dict) -> float:
    values = ([metrics[a]["window_mean"]["macro"]["pearson"] for a in ASSAYS]
              + [metrics[a]["tissue_pattern"]["mean_pearson"] for a in ASSAYS])
    score = float(np.mean(values))
    if not math.isfinite(score):
        raise ValueError("Validation score is not finite")
    return score
