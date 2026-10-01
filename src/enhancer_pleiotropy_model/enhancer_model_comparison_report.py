"""Evaluate three regressor prediction sets on active enhancer elements."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .browser_report import finite_or_none, git_commit, html_table, write_tsv
from .constants import ASSAYS, CONTEXTS
from .continuous_activity import RELATED_GROUPS, contrast_metrics
from .continuous_activity import participation_ratio
from .enhancer_catalog_evaluation import binary_curve
from .io import atomic_write_json, sha256_file
from .master_element_calibration import calibrate
from .metrics import correlation, correlation_structure, target_pca_projection_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--background-reference", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def row_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Row-correlation arrays must be aligned matrices")
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.full(len(left), np.nan),
        where=denominator > 0,
    )


def matrix_profile_metrics(
    observed: np.ndarray, predicted: np.ndarray, *, log1p: bool
) -> dict[str, object]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise ValueError("Profile metric arrays must align by enhancer and context")
    if log1p:
        observed = np.log1p(np.maximum(observed, 0))
        predicted = np.log1p(np.maximum(predicted, 0))
    by_context = {}
    for index, context in enumerate(CONTEXTS):
        by_context[context] = {
            "pearson": correlation(observed[:, index], predicted[:, index]),
            "spearman": correlation(
                rankdata(observed[:, index]), rankdata(predicted[:, index])
            ),
        }
    pattern_pearson = row_correlations(observed, predicted)
    pattern_spearman = row_correlations(
        rankdata(observed, axis=1), rankdata(predicted, axis=1)
    )
    return {
        "by_context": by_context,
        "macro_pearson": float(np.nanmean([x["pearson"] for x in by_context.values()])),
        "macro_spearman": float(np.nanmean([x["spearman"] for x in by_context.values()])),
        "tissue_pattern_mean_pearson": float(np.nanmean(pattern_pearson)),
        "tissue_pattern_median_pearson": float(np.nanmedian(pattern_pearson)),
        "tissue_pattern_mean_spearman": float(np.nanmean(pattern_spearman)),
        "tissue_pattern_median_spearman": float(np.nanmedian(pattern_spearman)),
        "tissue_pattern_n": int(np.isfinite(pattern_pearson).sum()),
    }


def state_summary(observed: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    result = matrix_profile_metrics(observed, predicted, log1p=False)
    observed_pleiotropy = participation_ratio(observed)
    predicted_pleiotropy = participation_ratio(predicted)
    result["pleiotropy"] = {
        "pearson": correlation(observed_pleiotropy, predicted_pleiotropy),
        "spearman": correlation(
            rankdata(observed_pleiotropy), rankdata(predicted_pleiotropy)
        ),
        "mae": float(np.mean(np.abs(observed_pleiotropy - predicted_pleiotropy))),
        "observed_mean": float(observed_pleiotropy.mean()),
        "predicted_mean": float(predicted_pleiotropy.mean()),
    }
    return result


def hard_activity_ranking(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("Hard labels and scores must align by enhancer and context")
    eligible = labels.any(axis=1) & (~labels).any(axis=1)
    selected_labels = labels[eligible]
    selected_scores = scores[eligible]
    active_mean = np.divide(
        (selected_scores * selected_labels).sum(axis=1),
        selected_labels.sum(axis=1),
    )
    inactive_mean = np.divide(
        (selected_scores * ~selected_labels).sum(axis=1),
        (~selected_labels).sum(axis=1),
    )
    margin = active_mean - inactive_mean
    top = selected_scores.argmax(axis=1)
    top_is_active = selected_labels[np.arange(len(top)), top]
    curve = binary_curve(selected_labels.ravel(), selected_scores.ravel())
    return {
        "eligible_elements": int(eligible.sum()),
        "mean_active_minus_inactive": float(margin.mean()),
        "positive_margin_fraction": float(np.mean(margin > 0)),
        "top_context_is_active_fraction": float(top_is_active.mean()),
        "average_precision": float(curve["average_precision"]),
    }


def context_specific_top_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    contexts: tuple[str, ...] = CONTEXTS,
    groups: dict[str, tuple[str, ...]] = RELATED_GROUPS,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    selected = labels.sum(axis=1) == 1
    truth = labels[selected].argmax(axis=1)
    selected_scores = scores[selected]
    prediction = selected_scores.argmax(axis=1)
    no_positive_score = selected_scores.max(axis=1) <= 0
    exact = (prediction == truth) & ~no_positive_score
    group_for_context = {}
    for group, members in groups.items():
        for context in members:
            group_for_context[context] = group
    related_error = np.asarray(
        [
            not is_exact
            and not no_score
            and group_for_context.get(contexts[true_index])
            == group_for_context.get(contexts[predicted_index])
            for true_index, predicted_index, is_exact, no_score in zip(
                truth, prediction, exact, no_positive_score, strict=True
            )
        ],
        dtype=bool,
    )
    errors = ~exact
    return {
        "elements": int(selected.sum()),
        "exact_fraction": float(exact.mean()),
        "group_tolerant_fraction": float((exact | related_error).mean()),
        "no_positive_score_fraction": float(no_positive_score.mean()),
        "related_fraction_among_errors": float(related_error[errors].mean())
        if errors.any()
        else 0.0,
    }


def _mean_contrasts(rows: list[dict[str, object]], related: bool, field: str) -> float:
    values = [float(row[field]) for row in rows if bool(row["is_related"]) == related]
    return float(np.nanmean(values))


def render_report(
    path: Path,
    summary: list[dict[str, object]],
    context_rows: list[dict[str, object]],
    breadth_rows: list[dict[str, object]],
    provenance: dict[str, object],
) -> None:
    summary_columns = [
        "source",
        "split",
        "assay",
        "elements",
        "signal_macro_pearson",
        "signal_macro_spearman",
        "tissue_pattern_pearson",
        "tissue_pattern_spearman",
        "overcorrelation",
        "pleiotropy_pearson",
        "pleiotropy_spearman",
        "pleiotropy_mae",
        "active_context_average_precision",
        "top_context_is_active",
        "specific_exact_context",
        "specific_group_tolerant",
    ]
    context_columns = [
        "source",
        "assay",
        "context",
        "pearson",
        "spearman",
        "active_context_average_precision",
    ]
    breadth_columns = [
        "source",
        "assay",
        "hard_breadth",
        "elements",
        "tissue_pattern_pearson",
        "tissue_pattern_spearman",
        "pleiotropy_pearson",
        "pleiotropy_spearman",
        "pleiotropy_mae",
        "observed_pleiotropy_mean",
        "predicted_pleiotropy_mean",
    ]
    test_summary = [row for row in summary if row["split"] == "test"]
    all_summary = [row for row in summary if row["split"] == "all"]
    test_contexts = [row for row in context_rows if row["split"] == "test"]
    test_breadth = [row for row in breadth_rows if row["split"] == "test"]
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Enhancer model comparison</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#222}}table{{border-collapse:collapse;margin:1rem 0;font-size:.81rem}}th,td{{border:1px solid #ccc;padding:.3rem .45rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eee;position:sticky;top:0}}code{{background:#eee;padding:.1rem .25rem}}.warning{{border-left:5px solid #d95f02;padding:.7rem;background:#fff3e6}}</style></head><body>
<h1>Base, specificity, and residual-ensemble performance on active enhancers</h1>
<p class="warning">Use chr3R test metrics for generalization. “All” includes training and validation enhancers and is descriptive only.</p>
<p>ATAC and H3K27ac remain separate. Signal metrics use exact model-aligned observed summaries and <code>log1p</code>. Pleiotropy uses participation ratio after independently mapping every assay/context to its fixed training-background percentile excess; no ATAC/H3K27ac geometric mean is used.</p>
<h2>Untouched chr3R test set</h2>{html_table(test_summary, summary_columns)}
<h2>All 39k active enhancers</h2>{html_table(all_summary, summary_columns)}
<h2>Test performance for every context</h2>{html_table(test_contexts, context_columns)}
<h2>Test performance by observed hard breadth</h2>{html_table(test_breadth, breadth_columns)}
<h2>Files</h2><ul><li><a href="summary.tsv">All split/model/assay summaries</a></li><li><a href="per_context.tsv">Per-context signal and activity-ranking metrics</a></li><li><a href="by_breadth.tsv">Metrics for every hard activity breadth</a></li><li><a href="contrasts.tsv">All 28 context contrasts</a></li><li><a href="pca.tsv">Observed PCA-mode recovery</a></li><li><a href="metrics.json">Complete metrics and provenance</a></li></ul>
<h2>Interpretation</h2><p>Hard activity labels are used only as secondary ranking diagnostics because they inherit the catalog’s DHS-membership and H3K27ac-percentile thresholds. Observational agreement does not prove that an ISM attribution or motif effect is causal; robust sequence-determinant claims should be stable across checkpoints and perturbation controls.</p>
<p>Generated {provenance['analysis_date_utc']}; git commit <code>{provenance['git_commit']}</code>.</p>
</body></html>"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    with np.load(args.predictions, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    contexts = tuple(data["contexts"].astype(str))
    sources = tuple(data["sources"].astype(str))
    feature_names = tuple(data["feature_names"].astype(str))
    if contexts != CONTEXTS:
        raise ValueError("Prediction context order is incorrect")
    required_features = {"atac_mean_512", "h3k27ac_max_mean_512"}
    if not required_features <= set(feature_names):
        raise ValueError(f"Predictions lack required features: {required_features}")
    features = data["features"].astype(np.float64)
    expected = (len(data["ids"]), len(sources), len(CONTEXTS), len(feature_names))
    if features.shape != expected:
        raise ValueError(f"Prediction feature shape {features.shape} != {expected}")
    with np.load(args.background_reference, allow_pickle=False) as loaded:
        if tuple(loaded["contexts"].astype(str)) != CONTEXTS:
            raise ValueError("Background-reference context order is incorrect")
        backgrounds = {
            "atac": loaded["atac_sorted"].astype(np.float64),
            "h3k27ac": loaded["h3k27ac_sorted"].astype(np.float64),
        }
    observed_raw = {
        "atac": data["observed_atac_mean_512"].astype(np.float64),
        "h3k27ac": data["observed_h3k27ac_segment_means_512"]
        .astype(np.float64)
        .max(axis=2),
    }
    observed_state = {
        assay: calibrate(values, backgrounds[assay])[1]
        for assay, values in observed_raw.items()
    }
    feature_index = {
        "atac": feature_names.index("atac_mean_512"),
        "h3k27ac": feature_names.index("h3k27ac_max_mean_512"),
    }
    hard_activity = data["hard_activity"].astype(bool)
    split_values = data["split"].astype(str)
    split_masks = {
        "all": np.ones(len(split_values), dtype=bool),
        "train": split_values == "train",
        "validation": split_values == "validation",
        "test": split_values == "test",
    }
    summary_rows: list[dict[str, object]] = []
    per_context_rows: list[dict[str, object]] = []
    breadth_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    pca_rows: list[dict[str, object]] = []
    complete: dict[str, dict[str, object]] = {source: {} for source in sources}
    for source_index, source in enumerate(sources):
        for split_name, selected in split_masks.items():
            complete[source][split_name] = {}
            labels = hard_activity[selected]
            for assay in ASSAYS:
                observed = observed_raw[assay][selected]
                predicted = features[selected, source_index, :, feature_index[assay]]
                observed_calibrated = observed_state[assay][selected]
                predicted_calibrated = calibrate(predicted, backgrounds[assay])[1]
                signal = matrix_profile_metrics(observed, predicted, log1p=True)
                state = state_summary(observed_calibrated, predicted_calibrated)
                logged_observed = np.log1p(observed)
                logged_predicted = np.log1p(np.maximum(predicted, 0))
                structure = correlation_structure(
                    logged_observed, logged_predicted, CONTEXTS
                )
                ranking = hard_activity_ranking(labels, predicted_calibrated)
                specific = context_specific_top_metrics(
                    labels, predicted_calibrated, CONTEXTS, RELATED_GROUPS
                )
                contrasts = contrast_metrics(observed, predicted, CONTEXTS)
                pca = target_pca_projection_metrics(
                    logged_observed, logged_predicted, CONTEXTS
                )
                result = {
                    "elements": int(selected.sum()),
                    "signal": signal,
                    "background_percentile_state": state,
                    "correlation_structure": structure,
                    "hard_activity_ranking": ranking,
                    "context_specific_top": specific,
                    "contrasts": contrasts,
                    "target_pca": pca,
                }
                complete[source][split_name][assay] = result
                summary_rows.append(
                    {
                        "source": source,
                        "split": split_name,
                        "assay": assay,
                        "elements": int(selected.sum()),
                        "signal_macro_pearson": signal["macro_pearson"],
                        "signal_macro_spearman": signal["macro_spearman"],
                        "tissue_pattern_pearson": signal[
                            "tissue_pattern_mean_pearson"
                        ],
                        "tissue_pattern_spearman": signal[
                            "tissue_pattern_mean_spearman"
                        ],
                        "overcorrelation": structure[
                            "mean_prediction_minus_true"
                        ],
                        "pleiotropy_pearson": state["pleiotropy"]["pearson"],
                        "pleiotropy_spearman": state["pleiotropy"]["spearman"],
                        "pleiotropy_mae": state["pleiotropy"]["mae"],
                        "observed_pleiotropy_mean": state["pleiotropy"][
                            "observed_mean"
                        ],
                        "predicted_pleiotropy_mean": state["pleiotropy"][
                            "predicted_mean"
                        ],
                        "active_context_average_precision": ranking[
                            "average_precision"
                        ],
                        "active_context_positive_margin": ranking[
                            "positive_margin_fraction"
                        ],
                        "top_context_is_active": ranking[
                            "top_context_is_active_fraction"
                        ],
                        "specific_exact_context": specific["exact_fraction"],
                        "specific_group_tolerant": specific[
                            "group_tolerant_fraction"
                        ],
                        "specific_related_fraction_among_errors": specific[
                            "related_fraction_among_errors"
                        ],
                        "related_contrast_pearson": _mean_contrasts(
                            contrasts, True, "pearson"
                        ),
                        "unrelated_contrast_pearson": _mean_contrasts(
                            contrasts, False, "pearson"
                        ),
                    }
                )
                for context_index, context in enumerate(CONTEXTS):
                    context_curve = binary_curve(
                        labels[:, context_index],
                        predicted_calibrated[:, context_index],
                    )
                    per_context_rows.append(
                        {
                            "source": source,
                            "split": split_name,
                            "assay": assay,
                            "context": context,
                            **signal["by_context"][context],
                            "active_context_average_precision": context_curve[
                                "average_precision"
                            ],
                            "prevalence": float(labels[:, context_index].mean()),
                        }
                    )
                for breadth in range(1, len(CONTEXTS) + 1):
                    breadth_selected = labels.sum(axis=1) == breadth
                    if not breadth_selected.any():
                        continue
                    breadth_state = state_summary(
                        observed_calibrated[breadth_selected],
                        predicted_calibrated[breadth_selected],
                    )
                    breadth_rows.append(
                        {
                            "source": source,
                            "split": split_name,
                            "assay": assay,
                            "hard_breadth": breadth,
                            "elements": int(breadth_selected.sum()),
                            "tissue_pattern_pearson": breadth_state[
                                "tissue_pattern_mean_pearson"
                            ],
                            "tissue_pattern_spearman": breadth_state[
                                "tissue_pattern_mean_spearman"
                            ],
                            "pleiotropy_pearson": breadth_state["pleiotropy"][
                                "pearson"
                            ],
                            "pleiotropy_spearman": breadth_state["pleiotropy"][
                                "spearman"
                            ],
                            "pleiotropy_mae": breadth_state["pleiotropy"]["mae"],
                            "observed_pleiotropy_mean": breadth_state[
                                "pleiotropy"
                            ]["observed_mean"],
                            "predicted_pleiotropy_mean": breadth_state[
                                "pleiotropy"
                            ]["predicted_mean"],
                        }
                    )
                contrast_rows.extend(
                    {
                        "source": source,
                        "split": split_name,
                        "assay": assay,
                        **row,
                    }
                    for row in contrasts
                )
                pca_rows.extend(
                    {
                        "source": source,
                        "split": split_name,
                        "assay": assay,
                        "PC": item["component"],
                        "target_variance": item[
                            "target_explained_variance_ratio"
                        ],
                        "prediction_pearson": item["prediction_pearson"],
                        "prediction_r2": item["prediction_r2"],
                    }
                    for item in pca["components"]
                )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    write_tsv(
        args.output_directory / "summary.tsv",
        summary_rows,
        list(summary_rows[0]),
    )
    write_tsv(
        args.output_directory / "per_context.tsv",
        per_context_rows,
        list(per_context_rows[0]),
    )
    write_tsv(
        args.output_directory / "by_breadth.tsv",
        breadth_rows,
        list(breadth_rows[0]),
    )
    write_tsv(
        args.output_directory / "contrasts.tsv",
        contrast_rows,
        list(contrast_rows[0]),
    )
    write_tsv(args.output_directory / "pca.tsv", pca_rows, list(pca_rows[0]))
    provenance = {
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "predictions": {
            "path": str(args.predictions.resolve()),
            "sha256": sha256_file(args.predictions),
        },
        "background_reference": {
            "path": str(args.background_reference.resolve()),
            "sha256": sha256_file(args.background_reference),
        },
        "elements": len(data["ids"]),
        "split_counts": {
            split: int(mask.sum()) for split, mask in split_masks.items()
        },
    }
    payload = {
        "definitions": {
            "signal": "exact ATAC 512-bp mean or maximum of three H3K27ac 512-bp means; correlations after log1p",
            "pleiotropy": "participation ratio across context-wise training-background percentile excess; assays kept separate",
            "overcorrelation": "predicted minus observed mean off-diagonal context correlation",
            "hard_activity": "catalog context membership and H3K27ac background percentile > 0.6; secondary diagnostic only",
        },
        "provenance": provenance,
        "summary": summary_rows,
        "sources": complete,
    }
    atomic_write_json(args.output_directory / "metrics.json", finite_or_none(payload))
    render_report(
        args.output_directory / "index.html",
        summary_rows,
        per_context_rows,
        breadth_rows,
        provenance,
    )
    print(
        json.dumps(
            {
                "event": "enhancer_model_comparison_report_complete",
                "elements": len(data["ids"]),
                "report": str(args.output_directory / "index.html"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
