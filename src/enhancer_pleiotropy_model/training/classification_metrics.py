"""Classifier evaluation: per-context AP/AUROC, predicted breadth and background separation."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ..constants import CONTEXTS


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Step-wise average precision (ties share one threshold)."""
    labels, scores = np.asarray(labels, np.int8), np.asarray(scores, float)
    positives = int(labels.sum())
    if not 0 < positives < len(labels):
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels, sorted_scores = labels[order], scores[order]
    distinct = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    true_positive = np.cumsum(sorted_labels)[distinct].astype(float)
    false_positive = np.cumsum(1 - sorted_labels)[distinct].astype(float)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return float("nan")
    return float((rankdata(scores)[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def summarize(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Metrics over enhancers; ``macro_average_precision`` selects checkpoints."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, np.float64)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    contexts = {
        context: {"average_precision": average_precision(labels[:, i], probabilities[:, i]),
                  "auroc": auroc(labels[:, i], probabilities[:, i]),
                  "prevalence": float(labels[:, i].mean())}
        for i, context in enumerate(CONTEXTS)}
    observed, predicted = labels.sum(1), probabilities.sum(1)
    valid_ap = [v["average_precision"] for v in contexts.values() if np.isfinite(v["average_precision"])]
    return {
        "macro_average_precision": float(np.mean(valid_ap)) if valid_ap else float("nan"),
        "macro_auroc": float(np.nanmean([v["auroc"] for v in contexts.values()])),
        "bce": float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean()),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "breadth": {"observed_mean": float(observed.mean()), "predicted_mean": float(predicted.mean()),
                    "pearson": float(np.corrcoef(observed, predicted)[0, 1]) if np.ptp(observed) else float("nan"),
                    "mae": float(np.abs(predicted - observed).mean())},
        "contexts": contexts,
    }


def background_metrics(labels: np.ndarray, enhancer_probabilities: np.ndarray,
                       background_probabilities: np.ndarray, paired_index: np.ndarray) -> dict:
    """How well enhancers are separated from their GC-matched genomic backgrounds.

    * ``active_vs_background`` - each context's active enhancers vs. their matched backgrounds.
    * ``enhancers_plus_background`` - all enhancers plus all backgrounds (backgrounds negative).
    * ``background_fpr_at_80_recall`` - background false-positive rate at the threshold
      that recovers 80% of the context's active enhancers.
    """
    y, p, bg, idx = map(np.asarray, (labels, enhancer_probabilities, background_probabilities, paired_index))
    result = {"active_vs_background": {}, "enhancers_plus_background": {}, "background_fpr_at_80_recall": {}}
    for j, context in enumerate(CONTEXTS):
        active = y[:, j].astype(bool)
        paired_active = y[idx, j].astype(bool)
        result["active_vs_background"][context] = average_precision(
            np.r_[np.ones(paired_active.sum()), np.zeros(paired_active.sum())],
            np.r_[p[idx[paired_active], j], bg[paired_active, j]])
        result["enhancers_plus_background"][context] = average_precision(
            np.r_[y[:, j], np.zeros(len(bg))], np.r_[p[:, j], bg[:, j]])
        threshold = np.sort(p[active, j])[-int(np.ceil(.8 * active.sum()))]
        result["background_fpr_at_80_recall"][context] = float((bg[:, j] >= threshold).mean())
    return {f"macro_{key}": float(np.mean(list(values.values()))) for key, values in result.items()} | result
