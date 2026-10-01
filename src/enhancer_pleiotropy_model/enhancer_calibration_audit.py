"""Audit cross-context behavior of saved enhancer calibrator predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .constants import CONTEXTS
from .io import atomic_write_json, atomic_write_text, sha256_file


CONTEXT_GROUPS = {
    "brain": ("ab", "lb"),
    "embryo": ("e5", "e13"),
    "imaginal_discs": ("ead", "hid", "wid"),
}
AUDIT_MODELS = (
    "geometric_mean_percentiles",
    "component_probability_product",
    "cross_context_component_probability_product",
    "atac_only_8_feature_logistic",
    "h3k27ac_only_8_feature_logistic",
    "joint_16_feature_logistic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def close_context_metrics(
    labels: np.ndarray,
    calls: np.ndarray,
    contexts: tuple[str, ...],
    groups: dict[str, tuple[str, ...]],
) -> tuple[dict[str, dict[str, float | int | str]], dict[str, float | int]]:
    """Measure whether false positives are active in a related context."""
    labels = np.asarray(labels, dtype=bool)
    calls = np.asarray(calls, dtype=bool)
    if labels.shape != calls.shape or labels.shape[1] != len(contexts):
        raise ValueError("Labels, calls, and context ordering do not align")
    context_index = {context: index for index, context in enumerate(contexts)}
    by_context: dict[str, dict[str, float | int | str]] = {}
    totals = {
        "false_positive_n": 0,
        "close_false_positive_n": 0,
        "close_negative_n": 0,
        "unrelated_false_positive_n": 0,
        "unrelated_negative_n": 0,
    }
    for group_name, members in groups.items():
        for context in members:
            index = context_index[context]
            peers = [context_index[peer] for peer in members if peer != context]
            negative = ~labels[:, index]
            false_positive = calls[:, index] & negative
            close = labels[:, peers].any(axis=1)
            close_negative_n = int(np.sum(negative & close))
            unrelated_negative_n = int(np.sum(negative & ~close))
            close_false_positive_n = int(np.sum(false_positive & close))
            unrelated_false_positive_n = int(np.sum(false_positive & ~close))
            false_positive_n = int(false_positive.sum())
            close_fpr = _safe_ratio(close_false_positive_n, close_negative_n)
            unrelated_fpr = _safe_ratio(
                unrelated_false_positive_n, unrelated_negative_n
            )
            by_context[context] = {
                "group": group_name,
                "false_positive_n": false_positive_n,
                "close_false_positive_n": close_false_positive_n,
                "close_fraction_among_false_positives": _safe_ratio(
                    close_false_positive_n, false_positive_n
                ),
                "close_fraction_in_negative_pool": _safe_ratio(
                    close_negative_n, int(negative.sum())
                ),
                "close_false_positive_rate": close_fpr,
                "unrelated_false_positive_rate": unrelated_fpr,
                "close_to_unrelated_fpr_ratio": (
                    close_fpr / unrelated_fpr
                    if np.isfinite(close_fpr) and unrelated_fpr > 0
                    else float("nan")
                ),
            }
            totals["false_positive_n"] += false_positive_n
            totals["close_false_positive_n"] += close_false_positive_n
            totals["close_negative_n"] += close_negative_n
            totals["unrelated_false_positive_n"] += unrelated_false_positive_n
            totals["unrelated_negative_n"] += unrelated_negative_n
    aggregate_close_fpr = _safe_ratio(
        totals["close_false_positive_n"], totals["close_negative_n"]
    )
    aggregate_unrelated_fpr = _safe_ratio(
        totals["unrelated_false_positive_n"], totals["unrelated_negative_n"]
    )
    aggregate: dict[str, float | int] = {
        **totals,
        "close_fraction_among_false_positives": _safe_ratio(
            totals["close_false_positive_n"], totals["false_positive_n"]
        ),
        "close_fraction_in_negative_pool": _safe_ratio(
            totals["close_negative_n"],
            totals["close_negative_n"] + totals["unrelated_negative_n"],
        ),
        "close_false_positive_rate": aggregate_close_fpr,
        "unrelated_false_positive_rate": aggregate_unrelated_fpr,
        "close_to_unrelated_fpr_ratio": (
            aggregate_close_fpr / aggregate_unrelated_fpr
            if aggregate_unrelated_fpr > 0
            else float("nan")
        ),
    }
    return by_context, aggregate


def breadth_metrics(labels: np.ndarray, calls: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    calls = np.asarray(calls, dtype=bool)
    if labels.shape != calls.shape:
        raise ValueError("Labels and calls do not align")
    observed = labels.sum(axis=1)
    predicted = calls.sum(axis=1)
    difference = predicted - observed
    return {
        "n": int(len(labels)),
        "observed_mean": float(observed.mean()),
        "predicted_mean": float(predicted.mean()),
        "mean_error": float(difference.mean()),
        "mean_absolute_error": float(np.abs(difference).mean()),
        "exact_fraction": float(np.mean(difference == 0)),
        "overcalled_fraction": float(np.mean(difference > 0)),
        "undercalled_fraction": float(np.mean(difference < 0)),
        "pearson": float(pearsonr(observed, predicted).statistic),
        "spearman": float(spearmanr(observed, predicted).statistic),
    }


def breadth_rows(
    labels: np.ndarray, calls: np.ndarray
) -> list[dict[str, float | int]]:
    observed = np.asarray(labels, dtype=bool).sum(axis=1)
    predicted = np.asarray(calls, dtype=bool).sum(axis=1)
    rows = []
    for breadth in sorted(np.unique(observed).tolist()):
        selected = observed == breadth
        rows.append(
            {
                "observed_breadth": int(breadth),
                "n": int(selected.sum()),
                "predicted_mean": float(predicted[selected].mean()),
                "predicted_median": float(np.median(predicted[selected])),
                "exact_fraction": float(np.mean(predicted[selected] == breadth)),
                "overcalled_fraction": float(np.mean(predicted[selected] > breadth)),
                "undercalled_fraction": float(np.mean(predicted[selected] < breadth)),
            }
        )
    return rows


def correlation_comparison(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    true_correlation = np.corrcoef(np.asarray(labels, dtype=float), rowvar=False)
    predicted_correlation = np.corrcoef(np.asarray(scores, dtype=float), rowvar=False)
    upper = np.triu_indices(labels.shape[1], k=1)
    difference = predicted_correlation[upper] - true_correlation[upper]
    return true_correlation, predicted_correlation, {
        "true_mean_off_diagonal": float(true_correlation[upper].mean()),
        "predicted_mean_off_diagonal": float(predicted_correlation[upper].mean()),
        "mean_predicted_minus_true": float(difference.mean()),
        "mean_absolute_difference": float(np.abs(difference).mean()),
        "root_mean_squared_difference": float(np.sqrt(np.mean(difference**2))),
        "pairs": int(len(difference)),
    }


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix(path: Path, matrix: np.ndarray, contexts: tuple[str, ...]) -> None:
    rows = []
    for row_index, context in enumerate(contexts):
        rows.append(
            {"context": context, **dict(zip(contexts, matrix[row_index], strict=True))}
        )
    _write_tsv(path, rows)


def main() -> None:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    predictions = np.load(args.predictions, allow_pickle=False)
    parameters = np.load(args.parameters, allow_pickle=False)
    contexts = tuple(predictions["contexts"].tolist())
    if contexts != CONTEXTS or tuple(parameters["contexts"].tolist()) != CONTEXTS:
        raise ValueError("Unexpected context ordering")
    model_names = tuple(predictions["model_names"].tolist())
    missing = set(AUDIT_MODELS) - set(model_names)
    if missing:
        raise ValueError(f"Saved predictions lack audit models: {sorted(missing)}")
    probabilities = predictions["probabilities"]
    thresholds = predictions["thresholds"]
    labels = predictions["labels"].astype(bool)
    test = predictions["split"] == "test"

    coefficient_rows: list[dict[str, object]] = []
    feature_names = parameters["feature_names"].tolist()
    coefficient_matrix = parameters["joint_16_feature_parameters"][:, 1:]
    for target_index, target_context in enumerate(CONTEXTS):
        coefficients = coefficient_matrix[target_index]
        ranks = np.argsort(-np.abs(coefficients))
        rank_by_feature = np.empty(len(ranks), dtype=int)
        rank_by_feature[ranks] = np.arange(1, len(ranks) + 1)
        for feature_index, (feature_name, coefficient) in enumerate(
            zip(feature_names, coefficients, strict=True)
        ):
            assay, feature_context = feature_name.split("__", maxsplit=1)
            coefficient_rows.append(
                {
                    "target_context": target_context,
                    "feature": feature_name,
                    "assay_summary": assay,
                    "feature_context": feature_context,
                    "is_target_context": feature_context == target_context,
                    "standardized_coefficient": float(coefficient),
                    "absolute_coefficient": float(abs(coefficient)),
                    "absolute_rank": int(rank_by_feature[feature_index]),
                }
            )
    _write_tsv(args.output_directory / "joint_coefficients.tsv", coefficient_rows)

    audit_models: dict[str, object] = {}
    close_rows: list[dict[str, object]] = []
    all_breadth_rows: list[dict[str, object]] = []
    for model_name in AUDIT_MODELS:
        model_index = model_names.index(model_name)
        test_probabilities = probabilities[model_index, test]
        test_calls = test_probabilities >= thresholds[model_index]
        test_labels = labels[test]
        by_context, close_aggregate = close_context_metrics(
            test_labels, test_calls, CONTEXTS, CONTEXT_GROUPS
        )
        for context, metrics in by_context.items():
            close_rows.append({"model": model_name, "context": context, **metrics})
        model_breadth_rows = breadth_rows(test_labels, test_calls)
        all_breadth_rows.extend(
            {"model": model_name, **row} for row in model_breadth_rows
        )
        true_correlation, predicted_correlation, correlation_metrics = (
            correlation_comparison(test_labels, test_probabilities)
        )
        _write_matrix(
            args.output_directory / f"{model_name}.predicted_context_correlation.tsv",
            predicted_correlation,
            CONTEXTS,
        )
        if model_name == AUDIT_MODELS[0]:
            _write_matrix(
                args.output_directory / "true_context_correlation.tsv",
                true_correlation,
                CONTEXTS,
            )
        audit_models[model_name] = {
            "close_context_false_positives": {
                "by_context": by_context,
                "aggregate": close_aggregate,
            },
            "tissue_breadth": breadth_metrics(test_labels, test_calls),
            "tissue_breadth_by_observed_count": model_breadth_rows,
            "context_probability_correlation": correlation_metrics,
        }
    _write_tsv(args.output_directory / "close_context_errors.tsv", close_rows)
    _write_tsv(args.output_directory / "tissue_breadth.tsv", all_breadth_rows)

    top_features = {}
    for target_context in CONTEXTS:
        selected = [
            row for row in coefficient_rows if row["target_context"] == target_context
        ]
        top_features[target_context] = {
            "positive": [
                {"feature": row["feature"], "coefficient": row["standardized_coefficient"]}
                for row in sorted(
                    selected,
                    key=lambda row: float(row["standardized_coefficient"]),
                    reverse=True,
                )[:3]
            ],
            "negative": [
                {"feature": row["feature"], "coefficient": row["standardized_coefficient"]}
                for row in sorted(
                    selected, key=lambda row: float(row["standardized_coefficient"])
                )[:3]
            ],
        }

    result = {
        "method": "enhancer_calibration_cross_context_audit_v1",
        "inputs": {
            "predictions": {
                "path": str(args.predictions),
                "sha256": sha256_file(args.predictions),
            },
            "parameters": {
                "path": str(args.parameters),
                "sha256": sha256_file(args.parameters),
            },
        },
        "evaluation_split": "test",
        "test_n": int(test.sum()),
        "context_groups": CONTEXT_GROUPS,
        "models": audit_models,
        "joint_model_top_standardized_coefficients": top_features,
        "interpretation_limits": [
            "Coefficient magnitude is conditional on correlated standardized inputs and is not a causal feature importance measure.",
            "Close-context support means a false-positive enhancer is truly active in a predefined related context; it does not prove why it was misclassified.",
            "The evaluation set contains elements active in at least one retained context and does not include genomic-background negatives.",
        ],
    }
    atomic_write_json(args.output_directory / "metrics.json", result)

    summary_rows = []
    for model_name in AUDIT_MODELS:
        model = audit_models[model_name]
        close = model["close_context_false_positives"]["aggregate"]  # type: ignore[index]
        breadth = model["tissue_breadth"]  # type: ignore[index]
        correlation = model["context_probability_correlation"]  # type: ignore[index]
        summary_rows.append(
            f"| {model_name} | {int(close['false_positive_n'])} | "
            f"{float(close['close_fraction_among_false_positives']):.3f} | "
            f"{float(close['close_to_unrelated_fpr_ratio']):.2f} | "
            f"{float(breadth['predicted_mean']):.3f} | "
            f"{float(breadth['mean_absolute_error']):.3f} | "
            f"{float(breadth['spearman']):.3f} | "
            f"{float(correlation['mean_predicted_minus_true']):+.3f} |"
        )
    feature_rows = []
    for context in CONTEXTS:
        positive = ", ".join(
            f"{entry['feature']} ({float(entry['coefficient']):+.2f})"
            for entry in top_features[context]["positive"][:2]
        )
        negative = ", ".join(
            f"{entry['feature']} ({float(entry['coefficient']):+.2f})"
            for entry in top_features[context]["negative"][:2]
        )
        feature_rows.append(f"| {context} | {positive} | {negative} |")
    summary = "\n".join(
        (
            "# Enhancer calibrator cross-context audit",
            "",
            f"Evaluation: {int(test.sum())} held-out test enhancers; no refitting or threshold selection was performed.",
            "",
            "| Model | Grouped-context FP n | Close fraction of FPs | Close/unrelated FPR | Predicted breadth mean | Breadth MAE | Breadth Spearman | Mean correlation excess |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *summary_rows,
            "",
            "The close-context columns pool brain, embryo, and imaginal-disc targets; `o` has no predefined peer and is excluded from that calculation. Mean correlation excess is the average predicted-minus-observed correlation over 28 context pairs.",
            "",
            "## Largest standardized joint-model coefficients",
            "",
            "| Target | Positive | Negative |",
            "|---|---|---|",
            *feature_rows,
            "",
            "Coefficients are conditional associations among correlated inputs, not causal effects.",
            "",
        )
    )
    atomic_write_text(args.output_directory / "summary.md", summary)


if __name__ == "__main__":
    main()
