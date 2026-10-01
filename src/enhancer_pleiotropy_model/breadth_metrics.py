"""Evaluate additive expected activity counts without thresholding probabilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .constants import CONTEXTS
from .enhancer_catalog_evaluation import binary_curve


GROUPS = {
    "embryo": ("e5", "e13"),
    "imaginal_discs": ("ead", "hid", "wid"),
    "brain": ("ab", "lb"),
    "ovary": ("o",),
}
GROUP_INDICES = tuple(
    tuple(CONTEXTS.index(context) for context in members) for members in GROUPS.values()
)
GROUP_SIZES = np.asarray([len(indices) for indices in GROUP_INDICES])


def validate_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> None:
    if labels.shape != probabilities.shape or labels.ndim != 2 or labels.shape[1] != 8:
        raise ValueError("Labels and probabilities must align as [enhancers, 8]")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Activity labels must be binary")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Probabilities must be finite and in [0, 1]")


def group_counts(values: np.ndarray) -> np.ndarray:
    return np.column_stack([values[:, indices].sum(axis=1) for indices in GROUP_INDICES])


def correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    if rank:
        left, right = rankdata(left), rankdata(right)
    return float(np.corrcoef(left, right)[0, 1])


def count_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Score real-valued expected counts against observed counts."""
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("Count vectors must align")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Count vectors must be finite")
    if not len(observed):
        return {"n": 0}
    error = predicted - observed
    return {
        "n": len(observed),
        "observed_mean": float(observed.mean()),
        "predicted_mean": float(predicted.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
        "pearson": correlation(observed, predicted),
        "spearman": correlation(observed, predicted, rank=True),
    }


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    if not len(labels):
        return {"n": 0, "brier": float("nan"), "average_precision": float("nan")}
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    positives = int(labels.sum())
    return {
        "n": len(labels),
        "positive_n": positives,
        "prevalence": float(labels.mean()),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
        "average_precision": float(binary_curve(labels, probabilities)["average_precision"])
        if 0 < positives < len(labels) else float("nan"),
    }


def reliability_rows(
    observed: np.ndarray, predicted: np.ndarray, maximum: int, bins: int = 10
) -> list[dict]:
    """Bin on predictions; works for probabilities and expected context counts."""
    if bins < 1 or maximum < 1:
        raise ValueError("Positive reliability range and bin count required")
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("Reliability vectors must align")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Reliability values must be finite")
    if np.any((predicted < 0) | (predicted > maximum)):
        raise ValueError("Prediction outside reliability range")
    assignments = np.minimum((predicted / maximum * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        selected = assignments == index
        truth, estimate = observed[selected], predicted[selected]
        rows.append({
            "bin": index, "lower": maximum * index / bins,
            "upper": maximum * (index + 1) / bins, "n": int(selected.sum()),
            "predicted_mean": float(estimate.mean()) if len(estimate) else float("nan"),
            "observed_mean": float(truth.mean()) if len(truth) else float("nan"),
        })
    return rows


def pairwise_ranking(labels: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    """Evaluate discordant observed labels; ties receive half credit."""
    rows = []
    for name, indices in zip(GROUPS, GROUP_INDICES, strict=True):
        for position, first in enumerate(indices):
            for second in indices[position + 1 :]:
                selected = labels[:, first] != labels[:, second]
                difference = probabilities[selected, first] - probabilities[selected, second]
                truth = labels[selected, first].astype(float) - labels[selected, second]
                credit = (difference * truth > 0).astype(float) + 0.5 * (difference == 0)
                rows.append({
                    "group": name, "context_a": CONTEXTS[first], "context_b": CONTEXTS[second],
                    "n": len(credit),
                    "accuracy": float(credit.mean()) if len(credit) else float("nan"),
                    "tie_fraction": float(np.mean(difference == 0)) if len(credit) else float("nan"),
                })
    return rows


def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    validate_probabilities(labels, probabilities)
    observed, predicted = labels.sum(axis=1), probabilities.sum(axis=1)
    observed_groups, predicted_groups = group_counts(labels), group_counts(probabilities)
    eligible = (observed > 0) & (predicted > 0)
    allocation = 0.5 * np.abs(
        observed_groups[eligible] / observed[eligible, None]
        - predicted_groups[eligible] / predicted[eligible, None]
    ).sum(axis=1)
    by_context = {
        context: probability_metrics(labels[:, index], probabilities[:, index])
        for index, context in enumerate(CONTEXTS)
    }
    pairs = pairwise_ranking(labels, probabilities)
    valid_pairs = [row["accuracy"] for row in pairs if row["n"]]
    valid_ap = [r["average_precision"] for r in by_context.values() if np.isfinite(r["average_precision"])]
    return {
        "breadth": count_metrics(observed, predicted),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "macro_average_precision": float(np.mean(valid_ap)) if valid_ap else float("nan"),
        "group_normalized_mae": float(np.mean(np.abs(predicted_groups - observed_groups) / GROUP_SIZES)),
        "allocation_error": float(allocation.mean()) if len(allocation) else float("nan"),
        "allocation_n": int(eligible.sum()),
        "allocation_zero_observed_n": int((observed == 0).sum()),
        "allocation_zero_prediction_n": int((predicted == 0).sum()),
        "related_pair_macro_accuracy": float(np.mean(valid_pairs)) if valid_pairs else float("nan"),
        "contexts": by_context,
        "groups": {
            name: {**count_metrics(observed_groups[:, index], predicted_groups[:, index]),
                   "size": int(GROUP_SIZES[index]),
                   "normalized_mae": float(np.mean(np.abs(predicted_groups[:, index] - observed_groups[:, index])) / GROUP_SIZES[index])}
            for index, name in enumerate(GROUPS)
        },
        "related_pairs": pairs,
    }


def block_bootstrap(
    labels: np.ndarray, predictions: dict[str, np.ndarray], blocks: np.ndarray,
    *, replicates: int = 500, seed: int = 20260904, reference: str = "fine_tuned_joint",
) -> list[dict]:
    """Paired whole-block bootstrap, conditional on the fitted models (no refits)."""
    if replicates < 2:
        raise ValueError("At least two bootstrap replicates required")
    _, inverse = np.unique(blocks, return_inverse=True)
    block_n = int(inverse.max()) + 1
    if block_n < 2:
        return []
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(block_n, np.full(block_n, 1 / block_n), size=replicates)
    counts = np.bincount(inverse, minlength=block_n)
    denominator = weights @ counts
    samples, point = {}, {}
    for name, probabilities in predictions.items():
        error = probabilities.sum(axis=1) - labels.sum(axis=1)
        contributions = np.column_stack((
            error**2, np.abs(error), error, np.mean((probabilities - labels)**2, axis=1),
            np.mean(np.abs(group_counts(probabilities) - group_counts(labels)) / GROUP_SIZES, axis=1),
        ))
        totals = np.stack([np.bincount(inverse, weights=column, minlength=block_n) for column in contributions.T], axis=1)
        samples[name] = (weights @ totals) / denominator[:, None]
        samples[name][:, 0] = np.sqrt(samples[name][:, 0])
        point[name] = contributions.mean(axis=0)
        point[name][0] = np.sqrt(point[name][0])
    rows = []
    for name in predictions:
        for index, metric in enumerate(("breadth_rmse", "breadth_mae", "breadth_bias", "brier", "group_normalized_mae")):
            for kind in ("estimate", "difference_from_reference"):
                values = samples[name][:, index]
                estimate = point[name][index]
                if kind == "difference_from_reference":
                    values = values - samples[reference][:, index]
                    estimate -= point[reference][index]
                low, high = np.quantile(values, [0.025, 0.975])
                rows.append({"model": name, "metric": metric, "kind": kind,
                             "estimate": float(estimate), "lower_95": float(low), "upper_95": float(high),
                             "reference": reference, "blocks": block_n, "replicates": replicates})
    return rows
