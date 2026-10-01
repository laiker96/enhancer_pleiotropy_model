"""Compare multiple predicted browser-track sets with aligned observations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .browser_report import (
    assay_metrics,
    extended_regression_metrics,
    finite_or_none,
    git_commit,
    html_table,
    interval_overlap_mask,
    load_track_matrix,
    resolve_recorded_path,
    write_tsv,
)
from .constants import ASSAYS, CONTEXTS
from .io import atomic_write_json, read_bed_intervals, sha256_file
from .metrics import correlation_structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-metadata", required=True, type=Path)
    parser.add_argument("--master-dhs-bed", required=True, type=Path)
    parser.add_argument("--h3k27ac-peaks-bed", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--active-quantile", default=0.90, type=float)
    parser.add_argument("--variable-quantile", default=0.75, type=float)
    return parser.parse_args()


def structure_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    if int(mask.sum()) < 2:
        return {
            "bins": int(mask.sum()),
            "mean_prediction_minus_true": float("nan"),
            "mean_absolute_error": float("nan"),
        }
    result = correlation_structure(observed[mask], predicted[mask], CONTEXTS)
    return {
        "bins": int(mask.sum()),
        "mean_prediction_minus_true": result["mean_prediction_minus_true"],
        "mean_absolute_error": result["mean_absolute_error"],
    }


def summary_row(
    source: str,
    assay: str,
    result: dict[str, object],
    regulatory_structure: dict[str, float | int],
    informative_structure: dict[str, float | int],
) -> dict[str, object]:
    macro = result["per_context"]["macro"]
    all_structure = result["correlation_structure"]
    return {
        "source": source,
        "assay": assay,
        "bins": result["bins"],
        "macro_pearson": macro["pearson"],
        "macro_spearman": macro["spearman"],
        "macro_rmse": macro["rmse"],
        "macro_mae": macro["mae"],
        "macro_r2": macro["r2"],
        "macro_bias": macro["bias"],
        "prediction_to_target_sd": macro["prediction_to_target_sd"],
        "tissue_pattern_all": result["tissue_pattern_all_variable"]["mean_pearson"],
        "tissue_pattern_informative": result["tissue_pattern_informative"][
            "mean_pearson"
        ],
        "top_context_accuracy_informative": result[
            "top_context_accuracy_informative"
        ],
        "informative_bins": result["informative_bins"],
        "overcorrelation_all": all_structure["mean_prediction_minus_true"],
        "correlation_matrix_mae_all": all_structure["mean_absolute_error"],
        "overcorrelation_regulatory": regulatory_structure[
            "mean_prediction_minus_true"
        ],
        "overcorrelation_informative": informative_structure[
            "mean_prediction_minus_true"
        ],
    }


def render_report(
    path: Path,
    metadata: dict[str, object],
    summary_rows: list[dict[str, object]],
    overall_rows: list[dict[str, object]],
    per_context_rows: list[dict[str, object]],
    strata_rows: list[dict[str, object]],
    pca_rows: list[dict[str, object]],
) -> None:
    summary_columns = [
        "source",
        "assay",
        "macro_pearson",
        "macro_spearman",
        "macro_rmse",
        "macro_r2",
        "prediction_to_target_sd",
        "tissue_pattern_informative",
        "top_context_accuracy_informative",
        "overcorrelation_regulatory",
    ]
    context_columns = [
        "source",
        "assay",
        "context",
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "r2",
        "bias",
        "prediction_to_target_sd",
        "raw_pearson",
    ]
    strata_columns = [
        "source",
        "assay",
        "stratum",
        "bins",
        "macro_pearson",
        "macro_spearman",
        "macro_rmse",
        "tissue_pattern_mean_pearson",
        "top_context_accuracy",
    ]
    pca_columns = [
        "source",
        "assay",
        "PC",
        "target_variance",
        "prediction_pearson",
        "prediction_r2",
    ]
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Three-model track comparison</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse;margin:1rem 0;font-size:.82rem}}th,td{{border:1px solid #ccc;padding:.3rem .45rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eee;position:sticky;top:0}}code{{background:#eee;padding:.1rem .25rem}}.warning{{border-left:5px solid #d95f02;padding:.7rem;background:#fff3e6}}
</style></head><body>
<h1>Observed versus three model-derived track sets</h1>
<p class="warning">This chr2L interval was used for validation and checkpoint selection. These are development metrics, not untouched test estimates.</p>
<p>Primary metrics compare <code>log1p(max(signal, 0))</code> on aligned, supported native bins (16 bp ATAC; 64 bp H3K27ac). Lower RMSE and context-correlation matrix error are better. Over-correlation is predicted minus observed mean context-pair correlation; values nearer zero are better.</p>
<h2>Overall mean across assays</h2>
{html_table(overall_rows, ['source','mean_macro_pearson','mean_macro_spearman','mean_macro_rmse','mean_tissue_pattern_informative','mean_overcorrelation_regulatory'])}
<h2>Model and assay summary</h2>
{html_table(summary_rows, summary_columns)}
<h2>Every predicted track versus its observation</h2>
{html_table(per_context_rows, context_columns)}
<h2>Genomic strata</h2>
{html_table(strata_rows, strata_columns)}
<h2>Observed-target PCA modes</h2>
{html_table(pca_rows, pca_columns)}
<h2>Files</h2><ul>
<li><a href="model_assay_summary.tsv">Model/assay summary</a></li>
<li><a href="per_context_metrics.tsv">All 48 track comparisons</a></li>
<li><a href="stratified_metrics.tsv">Genomic-stratum metrics</a></li>
<li><a href="pca_metrics.tsv">PCA-mode metrics</a></li>
<li><a href="metrics.json">Complete metrics and provenance</a></li></ul>
<h2>Interpretation limits</h2><p>Sliding-window predictions overlap, so bins are not independent. Correlation measures shape/rank agreement but not calibration; RMSE, bias, R2, and prediction-to-target standard deviation describe magnitude behavior. Informative bins and strata were defined from observations only and are shared by all three models.</p>
<p>Source metadata: <code>{metadata['method']}</code>.</p>
</body></html>"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not 0 < args.active_quantile < 1 or not 0 < args.variable_quantile < 1:
        raise ValueError("Activity and variability quantiles must lie between 0 and 1")
    metadata = json.loads(args.browser_metadata.read_text(encoding="utf-8"))
    contexts = tuple(metadata["contexts"])
    if contexts != CONTEXTS:
        raise ValueError(f"Expected context order {CONTEXTS}, found {contexts}")
    prediction_sources = tuple(
        source for source in metadata["output_bigwigs"] if source != "observed"
    )
    if not prediction_sources:
        raise ValueError("Comparison metadata contains no predicted sources")
    chromosome = metadata["chromosome"]
    dhs = read_bed_intervals(args.master_dhs_bed).get(chromosome, ())
    h3_peaks = read_bed_intervals(args.h3k27ac_peaks_bed).get(chromosome, ())
    paths = {
        source: {
            assay: {
                context: resolve_recorded_path(
                    metadata["output_bigwigs"][source][assay][context],
                    args.browser_metadata,
                )
                for context in contexts
            }
            for assay in ASSAYS
        }
        for source in ("observed", *prediction_sources)
    }

    complete: dict[str, dict[str, object]] = {source: {} for source in prediction_sources}
    summary_rows: list[dict[str, object]] = []
    per_context_rows: list[dict[str, object]] = []
    strata_rows: list[dict[str, object]] = []
    pca_rows: list[dict[str, object]] = []
    for assay in ASSAYS:
        starts, ends, observed, header = load_track_matrix(
            [paths["observed"][assay][context] for context in contexts], chromosome
        )
        dhs_mask = interval_overlap_mask(starts, ends, dhs)
        h3_mask = interval_overlap_mask(starts, ends, h3_peaks)
        masks = {
            "DHS_and_H3K27ac_peak": dhs_mask & h3_mask,
            "DHS_only": dhs_mask & ~h3_mask,
            "H3K27ac_peak_only": ~dhs_mask & h3_mask,
            "background": ~dhs_mask & ~h3_mask,
        }
        regulatory = dhs_mask | h3_mask
        logged_observed = np.log1p(np.maximum(observed, 0))
        for source in prediction_sources:
            pred_starts, pred_ends, predicted, pred_header = load_track_matrix(
                [paths[source][assay][context] for context in contexts], chromosome
            )
            if not (
                np.array_equal(starts, pred_starts)
                and np.array_equal(ends, pred_ends)
                and header == pred_header
            ):
                raise ValueError(f"{source}/{assay}: track geometry differs")
            if np.any(observed < 0) or np.any(predicted < 0):
                raise ValueError(f"{source}/{assay}: signal must be nonnegative")
            result, source_strata, informative = assay_metrics(
                observed,
                predicted,
                masks,
                args.active_quantile,
                args.variable_quantile,
                contexts,
            )
            logged_predicted = np.log1p(np.maximum(predicted, 0))
            raw = extended_regression_metrics(observed, predicted, contexts)
            regulatory_structure = structure_metrics(
                logged_observed, logged_predicted, regulatory
            )
            informative_structure = structure_metrics(
                logged_observed, logged_predicted, informative
            )
            result["regulatory_correlation_structure_summary"] = regulatory_structure
            result["informative_correlation_structure_summary"] = informative_structure
            result["raw_per_context"] = raw
            complete[source][assay] = result
            summary_rows.append(
                summary_row(
                    source,
                    assay,
                    result,
                    regulatory_structure,
                    informative_structure,
                )
            )
            for context in contexts:
                values = result["per_context"]["by_context"][context]
                raw_values = raw["by_context"][context]
                per_context_rows.append(
                    {
                        "source": source,
                        "assay": assay,
                        "context": context,
                        **values,
                        "raw_pearson": raw_values["pearson"],
                        "raw_rmse": raw_values["rmse"],
                    }
                )
            strata_rows.extend(
                {"source": source, "assay": assay, **row}
                for row in source_strata
            )
            pca_rows.extend(
                {
                    "source": source,
                    "assay": assay,
                    "PC": item["component"],
                    "target_variance": item["target_explained_variance_ratio"],
                    "prediction_pearson": item["prediction_pearson"],
                    "prediction_r2": item["prediction_r2"],
                }
                for item in result["target_pca"]["components"]
            )

    overall_rows = []
    for source in prediction_sources:
        rows = [row for row in summary_rows if row["source"] == source]
        overall_rows.append(
            {
                "source": source,
                "mean_macro_pearson": float(np.mean([row["macro_pearson"] for row in rows])),
                "mean_macro_spearman": float(np.mean([row["macro_spearman"] for row in rows])),
                "mean_macro_rmse": float(np.mean([row["macro_rmse"] for row in rows])),
                "mean_tissue_pattern_informative": float(
                    np.mean([row["tissue_pattern_informative"] for row in rows])
                ),
                "mean_overcorrelation_regulatory": float(
                    np.mean([row["overcorrelation_regulatory"] for row in rows])
                ),
            }
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    summary_columns = list(summary_rows[0])
    context_columns = list(per_context_rows[0])
    strata_columns = list(strata_rows[0])
    pca_columns = list(pca_rows[0])
    write_tsv(args.output_directory / "model_assay_summary.tsv", summary_rows, summary_columns)
    write_tsv(args.output_directory / "per_context_metrics.tsv", per_context_rows, context_columns)
    write_tsv(args.output_directory / "stratified_metrics.tsv", strata_rows, strata_columns)
    write_tsv(args.output_directory / "pca_metrics.tsv", pca_rows, pca_columns)
    provenance = {
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "browser_metadata": {
            "path": str(args.browser_metadata.resolve()),
            "sha256": sha256_file(args.browser_metadata),
        },
        "master_dhs_bed": {
            "path": str(args.master_dhs_bed.resolve()),
            "sha256": sha256_file(args.master_dhs_bed),
        },
        "h3k27ac_peaks_bed": {
            "path": str(args.h3k27ac_peaks_bed.resolve()),
            "sha256": sha256_file(args.h3k27ac_peaks_bed),
        },
        "active_quantile": args.active_quantile,
        "variable_quantile": args.variable_quantile,
    }
    payload = {
        "definitions": {
            "primary_transform": "log1p(max(signal, 0))",
            "overcorrelation": "mean predicted-minus-observed off-diagonal context correlation",
            "regulatory_bins": "native bins overlapping master DHS or consensus H3K27ac peaks",
            "informative_bins": "top active-quantile context mean and top variable-quantile context SD among active bins",
        },
        "provenance": provenance,
        "overall": overall_rows,
        "summary": summary_rows,
        "sources": complete,
    }
    atomic_write_json(args.output_directory / "metrics.json", finite_or_none(payload))
    render_report(
        args.output_directory / "index.html",
        metadata,
        summary_rows,
        overall_rows,
        per_context_rows,
        strata_rows,
        pca_rows,
    )
    print(
        json.dumps(
            {
                "event": "browser_comparison_report_complete",
                "sources": list(prediction_sources),
                "track_comparisons": len(per_context_rows),
                "report": str(args.output_directory / "index.html"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
