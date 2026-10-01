"""Evaluate additive assay activity breadth directly from frozen signal outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy

from .breadth_metrics import (
    GROUPS, GROUP_INDICES, GROUP_SIZES, count_metrics, group_counts, reliability_rows,
)
from .browser_report import finite_or_none, git_commit, html_table, write_tsv
from .constants import CONTEXTS
from .continuous_activity import participation_ratio
from .enhancer_model_comparison_inference import atomic_save_npz
from .expected_breadth import TRAINING_CHROMOSOMES, calibration_partition, calibration_svg
from .io import atomic_write_json, atomic_write_text, sha256_file
from .master_element_calibration import calibrate


ASSAYS = ("atac", "h3k27ac")
MODELS = ("base", "fine_tuned")
SCALES = ("percentile_excess", "log1p_signal")
EVALUATIONS = ("validation_evaluation", "chr3R_descriptive")
WARNING = (
    "Known active enhancers only; no evaluation of genome-wide negatives or mutation effects. "
    "chr2L selected the sequence checkpoints; chr3R influenced prior development. "
    "Neither evaluation is an untouched final test. No model settings are selected here."
)


def validate_activity(observed: np.ndarray, predicted: np.ndarray) -> None:
    if observed.shape != predicted.shape or observed.ndim != 2 or observed.shape[1] != 8 or not len(observed):
        raise ValueError("Activity arrays must align as nonempty [enhancers, 8]")
    if any(not np.isfinite(x).all() or np.any(x < 0) for x in (observed, predicted)):
        raise ValueError("Activity must be finite and nonnegative")


def evaluate_activity(observed: np.ndarray, predicted: np.ndarray) -> dict:
    """Continuous regression metrics; no catalog labels or thresholded predictions."""
    validate_activity(observed, predicted)
    truth, estimate = observed.sum(axis=1), predicted.sum(axis=1)
    true_groups, predicted_groups = group_counts(observed), group_counts(predicted)
    eligible = (truth > 0) & (estimate > 0)
    allocation = 0.5 * np.abs(
        true_groups[eligible] / truth[eligible, None]
        - predicted_groups[eligible] / estimate[eligible, None]
    ).sum(axis=1)
    contexts = [dict(context=c, **count_metrics(observed[:, i], predicted[:, i])) for i, c in enumerate(CONTEXTS)]
    valid_correlations = [row["pearson"] for row in contexts if np.isfinite(row["pearson"])]
    groups = [dict(group=g, size=int(GROUP_SIZES[i]),
                   **count_metrics(true_groups[:, i], predicted_groups[:, i]),
                   normalized_mae=float(np.abs(true_groups[:, i] - predicted_groups[:, i]).mean() / GROUP_SIZES[i]))
              for i, g in enumerate(GROUPS)]
    pairs = []
    for group, indices in zip(GROUPS, GROUP_INDICES, strict=True):
        for i, first in enumerate(indices):
            for second in indices[i + 1:]:
                actual = observed[:, first] - observed[:, second]
                prediction = predicted[:, first] - predicted[:, second]
                nonzero = actual != 0
                credit = (actual[nonzero] * prediction[nonzero] > 0).astype(float) + 0.5 * (prediction[nonzero] == 0)
                pairs.append(dict(group=group, context_a=CONTEXTS[first], context_b=CONTEXTS[second],
                                  **count_metrics(actual, prediction), direction_n=int(nonzero.sum()),
                                  direction_accuracy=float(credit.mean()) if len(credit) else float("nan")))
    return {
        "total": count_metrics(truth, estimate),
        "participation_ratio": count_metrics(participation_ratio(observed), participation_ratio(predicted)),
        "context_rmse": float(np.sqrt(np.mean((predicted - observed)**2))),
        "context_macro_pearson": float(np.mean(valid_correlations)) if valid_correlations else float("nan"),
        "context_correlation_n": len(valid_correlations),
        "group_normalized_mae": float(np.mean(np.abs(predicted_groups - true_groups) / GROUP_SIZES)),
        "group_allocation_tv": float(allocation.mean()) if len(allocation) else float("nan"),
        "allocation_n": int(eligible.sum()), "zero_observed_n": int((truth == 0).sum()),
        "zero_predicted_n": int((estimate == 0).sum()),
        "contexts": contexts, "groups": groups, "pairs": pairs,
    }


def bootstrap_activity(observed: np.ndarray, predictions: dict[str, np.ndarray], blocks: np.ndarray,
                       *, replicates: int, seed: int) -> list[dict]:
    """Paired 1-Mb block resampling, conditional on fixed predictions and normalization."""
    if replicates < 2 or len(blocks) != len(observed) or "fine_tuned" not in predictions:
        raise ValueError("Require aligned blocks, fine_tuned reference and at least two replicates")
    _, inverse = np.unique(blocks, return_inverse=True)
    block_n = len(np.unique(inverse))
    if block_n < 2:
        return []
    weights = np.random.default_rng(seed).multinomial(block_n, np.full(block_n, 1 / block_n), size=replicates)
    denominator = weights @ np.bincount(inverse)
    samples, points = {}, {}
    for model, predicted in predictions.items():
        validate_activity(observed, predicted)
        error = predicted.sum(axis=1) - observed.sum(axis=1)
        loss = np.column_stack((error**2, np.abs(error), error,
                                np.mean((predicted - observed)**2, axis=1),
                                np.mean(np.abs(group_counts(predicted) - group_counts(observed)) / GROUP_SIZES, axis=1)))
        totals = np.column_stack([np.bincount(inverse, weights=column) for column in loss.T])
        samples[model] = (weights @ totals) / denominator[:, None]
        points[model] = loss.mean(axis=0)
        samples[model][:, [0, 3]] = np.sqrt(samples[model][:, [0, 3]])
        points[model][[0, 3]] = np.sqrt(points[model][[0, 3]])
    rows = []
    for model in predictions:
        for i, metric in enumerate(("total_rmse", "total_mae", "total_bias", "context_rmse", "group_normalized_mae")):
            for kind in ("estimate", "difference_from_fine_tuned"):
                values, point = samples[model][:, i], points[model][i]
                if kind != "estimate":
                    values, point = values - samples["fine_tuned"][:, i], point - points["fine_tuned"][i]
                low, high = np.quantile(values, [0.025, 0.975])
                rows.append(dict(model=model, metric=metric, kind=kind, estimate=float(point),
                                 lower_95=float(low), upper_95=float(high), blocks=block_n, replicates=replicates))
    return rows


def saturation_metrics(observed_raw: np.ndarray, predicted_raw: np.ndarray,
                       observed: np.ndarray, predicted: np.ndarray, background: np.ndarray) -> dict:
    """The fixed 0.99 cutoff is diagnostic only, never an activity label."""
    saturated = observed >= 0.99
    log_error = np.log1p(predicted_raw) - np.log1p(observed_raw)
    return {
        "context_values_n": observed.size,
        "observed_ge_099_fraction": float(saturated.mean()),
        "predicted_ge_099_fraction": float(np.mean(predicted >= 0.99)),
        "both_ge_099_fraction": float(np.mean(saturated & (predicted >= 0.99))),
        "observed_above_background_max_fraction": float(np.mean(observed_raw > background[-1])),
        "predicted_above_background_max_fraction": float(np.mean(predicted_raw > background[-1])),
        "saturated_context_values_n": int(saturated.sum()),
        "saturated_log1p_rmse": float(np.sqrt(np.mean(log_error[saturated]**2))) if saturated.any() else float("nan"),
        "saturated_log1p_bias": float(log_error[saturated].mean()) if saturated.any() else float("nan"),
    }


def write_report(output: Path, tables: dict[str, list[dict]]) -> None:
    columns = ["assay", "model", "n", "observed_mean", "predicted_mean", "rmse", "mae", "bias", "spearman", "context_rmse", "group_allocation_tv"]
    definition = (
        "Primary activity a = max(0, 2 × training-background percentile − 1); breadth B = sum(a) across eight contexts. "
        "Groups sum exactly to B. This is activity-weighted breadth, not a probability or catalog context count. "
        "ATAC and H3K27ac remain separate; no joint-assay weights are imposed. "
        "Participation ratio measures evenness. Log1p signal totals are unbounded signal-burden diagnostics, not 0–8 breadth. "
        "Percentile saturation can hide signal errors; inspect log-signal and related-context contrasts too."
    )
    sections = ["<h1>Continuous assay activity breadth</h1>", f"<p>{WARNING}</p><p>{definition}</p>"]
    markdown = ["# Continuous assay activity breadth", "", WARNING, "", definition, ""]
    for split in EVALUATIONS:
        sections.append(f"<h2>{split}</h2>")
        markdown += [f"## {split}", "", "| Assay | Model | RMSE | MAE | Bias | Spearman |", "|---|---|---:|---:|---:|---:|"]
        for scale in SCALES:
            rows = [r for r in tables["summary"] if r["split"] == split and r["scale"] == scale]
            sections.append(f"<h3>{scale}</h3>" + html_table(rows, columns))
            if scale == SCALES[0]:
                markdown += [f'| {r["assay"]} | {r["model"]} | {r["rmse"]:.4f} | {r["mae"]:.4f} | {r["bias"]:.4f} | {r["spearman"]:.4f} |' for r in rows]
        markdown.append("")
        primary = lambda r: r["split"] == split and r["model"] == "fine_tuned"
        broad = [r for r in tables["stratified"] if primary(r) and r["stratum"] == "catalog_count" and r["value"] == 8]
        sections.append("<h3>Catalog-ubiquitous cohort (secondary diagnostic)</h3>" + html_table(broad, ["assay", "n", "observed_mean", "predicted_mean", "observed_pr_mean", "predicted_pr_mean", "log1p_context_rmse"]))
        for row in broad:
            markdown.append(f'{row["assay"]}, catalog count 8 (n={row["n"]}): observed activity breadth {row["observed_mean"]:.4f}; predicted {row["predicted_mean"]:.4f}.')
        markdown.append("")
        for table, title, cols in (
            ("per_group", "Additive group contributions", ["assay", "group", "observed_mean", "predicted_mean", "mae", "normalized_mae", "spearman"]),
            ("per_context", "Individual context contributions", ["assay", "context", "rmse", "mae", "bias", "pearson", "spearman"]),
            ("participation_ratio", "Participation-ratio evenness", ["assay", "observed_mean", "predicted_mean", "mae", "spearman"]),
        ):
            rows = [r for r in tables[table] if primary(r) and r["scale"] == SCALES[0]]
            sections.append(f"<h3>{title}</h3>" + html_table(rows, cols))
        rows = [r for r in tables["related_pairs"] if primary(r) and r["scale"] == "log1p_signal"]
        sections.append("<h3>Related-context log-signal contrasts</h3>" + html_table(rows, ["assay", "context_a", "context_b", "pearson", "mae", "direction_n", "direction_accuracy"]))
        sections.append("<h3>Percentile saturation</h3>" + html_table([r for r in tables["saturation"] if primary(r)], ["assay", "observed_ge_099_fraction", "predicted_ge_099_fraction", "saturated_log1p_rmse", "saturated_log1p_bias"]))
        sections.append("<h3>Continuous calibration: bins defined by predictions</h3><div class='plots'>")
        for assay in ASSAYS:
            for target, maximum in (("total", 8), *((g, len(c)) for g, c in GROUPS.items())):
                rows = [r for r in tables["reliability"] if primary(r) and r["assay"] == assay and r["target"] == target]
                sections.append(calibration_svg(rows, maximum, f"{assay}: {target}"))
        sections.append("</div>")
    sections.append("<h2>Downloads and provenance</h2><p>500 paired occupied 1-Mb block bootstrap replicates by default; conditional on frozen models/background. Few validation blocks limit precision. Constant means use only left chr2L calibration-fit rows, independently for each assay/scale.</p><ul>")
    files = [*(f"{name}.tsv" for name in tables), "element_predictions.tsv.gz", "continuous_breadth_predictions.npz", "provenance.json", "summary.md"]
    sections += [f'<li><a href="{name}">{name}</a></li>' for name in files]
    sections.append("</ul>")
    atomic_write_text(output / "index.html", '<!doctype html><html lang="en"><meta charset="utf-8"><title>Continuous assay breadth</title><style>body{font:16px sans-serif;margin:2em;max-width:1400px}table{border-collapse:collapse;margin-bottom:1em}td,th{padding:.4em;border:1px solid #ddd}th{background:#eee}.plots{display:flex;flex-wrap:wrap}figure{margin:.5em}</style><body>' + "\n".join(sections) + "</body></html>")
    atomic_write_text(output / "summary.md", "\n".join(markdown) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.bootstrap_replicates < 2:
        parser.error("At least two bootstrap replicates required")
    if args.output_directory.exists():
        raise FileExistsError("Use a new output directory to preserve previous results")
    metadata_path = args.predictions.with_suffix(".metadata.json")
    prediction_metadata = json.loads(metadata_path.read_text())
    prediction_hash = sha256_file(args.predictions)
    if prediction_metadata["output"]["sha256"] != prediction_hash:
        raise ValueError("Prediction cache hash disagrees with metadata")
    background_metadata_path = args.background.parent / "metrics.json"
    background_metadata = json.loads(background_metadata_path.read_text())
    background_chromosomes = background_metadata["selection"]["background_chromosomes"]
    if not background_chromosomes or not set(background_chromosomes).issubset(TRAINING_CHROMOSOMES):
        raise ValueError("Background must use only final training chromosomes, excluding chr2L/chr3R")
    with np.load(args.predictions, allow_pickle=False) as cache:
        data = {key: cache[key] for key in cache.files}
    with np.load(args.background, allow_pickle=False) as cache:
        background = {key: cache[key] for key in cache.files}
    if any(tuple(d["contexts"].astype(str)) != CONTEXTS for d in (data, background)):
        raise ValueError("Unexpected context order")
    n = len(data["ids"])
    if not n or len(np.unique(data["ids"])) != n or any(data[k].shape != (n,) for k in ("ids", "chrom", "summit", "split")):
        raise ValueError("Unique IDs and aligned coordinates/splits required")
    if not np.issubdtype(data["summit"].dtype, np.integer) or np.any(data["summit"] < 0):
        raise ValueError("Summits must be nonnegative integers")
    labels = data["hard_activity"]
    if labels.shape != (n, 8) or not np.isin(labels, (0, 1)).all() or not labels.any(axis=1).all():
        raise ValueError("Expected active-any binary catalog labels for secondary strata")
    segments = data["observed_h3k27ac_segment_means_512"]
    if segments.shape != (n, 8, 3) or not np.isfinite(segments).all() or np.any(segments < 0):
        raise ValueError("Invalid observed H3K27ac segments")
    observed_raw = np.stack((data["observed_atac_mean_512"], segments.max(axis=2))).astype(float)
    sources, features = data["sources"].tolist(), data["feature_names"].tolist()
    if len(set(sources)) != len(sources) or len(set(features)) != len(features) or data["features"].shape != (n, len(sources), 8, len(features)):
        raise ValueError("Invalid cached feature axes")
    predicted_raw = np.stack([np.stack([data["features"][:, sources.index(model), :, features.index(feature)]
                                       for feature in ("atac_mean_512", "h3k27ac_max_mean_512")]) for model in MODELS]).astype(float)
    partitions = calibration_partition(data["chrom"], data["summit"], data["split"])
    fit = partitions == "calibration_fit"
    observed_activity, predicted_activity = np.empty_like(observed_raw), np.empty_like(predicted_raw)
    for a, assay in enumerate(ASSAYS):
        reference = background[f"{assay}_sorted"]
        if len(reference) != background_metadata["selection"]["background_windows"]:
            raise ValueError("Background row count disagrees with metadata")
        for m in range(len(MODELS)):
            validate_activity(observed_raw[a], predicted_raw[m, a])
            predicted_activity[m, a] = calibrate(predicted_raw[m, a], reference)[1]
        observed_activity[a] = calibrate(observed_raw[a], reference)[1]
    observed_log, predicted_log = np.log1p(observed_raw), np.log1p(predicted_raw)
    args.output_directory.mkdir(parents=True)
    print(json.dumps(dict(event="continuous_breadth_start", device="cpu", cached_inference_device=prediction_metadata.get("inference", {}).get("device"),
                          elements=n, partitions={p: int(np.sum(partitions == p)) for p in np.unique(partitions)})), flush=True)
    tables = {k: [] for k in ("summary", "per_context", "per_group", "related_pairs", "participation_ratio", "reliability", "stratified", "saturation", "bootstrap")}
    constant_means = np.stack((observed_activity[:, fit].mean(axis=1), observed_log[:, fit].mean(axis=1)))
    for split in EVALUATIONS:
        selected = partitions == split
        blocks = np.asarray([f"{c}:{s // 1_000_000}" for c, s in zip(data["chrom"][selected], data["summit"][selected], strict=True)])
        for a, assay in enumerate(ASSAYS):
            for scale_i, (scale, observed_all, predicted_all) in enumerate(zip(SCALES, (observed_activity, observed_log), (predicted_activity, predicted_log), strict=True)):
                observed = observed_all[a, selected]
                predictions = {model: predicted_all[m, a, selected] for m, model in enumerate(MODELS)}
                predictions["constant_mean"] = np.broadcast_to(constant_means[scale_i, a], observed.shape)
                common = dict(split=split, assay=assay, scale=scale)
                for model, predicted in predictions.items():
                    key = dict(**common, model=model)
                    result = evaluate_activity(observed, predicted)
                    tables["summary"].append(dict(**key, **result["total"], **{k: v for k, v in result.items() if not isinstance(v, (list, dict))}))
                    tables["participation_ratio"].append(dict(**key, **result["participation_ratio"]))
                    for table, field in (("per_context", "contexts"), ("per_group", "groups"), ("related_pairs", "pairs")):
                        tables[table].extend(dict(**key, **row) for row in result[field])
                    if scale == "percentile_excess":
                        true_groups, pred_groups = group_counts(observed), group_counts(predicted)
                        targets = [("total", 8, observed.sum(axis=1), predicted.sum(axis=1))]
                        targets += [(g, int(GROUP_SIZES[i]), true_groups[:, i], pred_groups[:, i]) for i, g in enumerate(GROUPS)]
                        targets += [(c, 1, observed[:, i], predicted[:, i]) for i, c in enumerate(CONTEXTS)]
                        for target, maximum, truth, estimate in targets:
                            tables["reliability"].extend(dict(**key, target=target, **row) for row in reliability_rows(truth, estimate, maximum))
                tables["bootstrap"].extend(dict(**common, **row) for row in bootstrap_activity(observed, predictions, blocks, replicates=args.bootstrap_replicates, seed=args.seed))
            observed = observed_activity[a, selected]
            catalog_count = labels[selected].sum(axis=1)
            breadth_bin = np.minimum(observed.sum(axis=1).astype(int), 7)
            for m, model in enumerate(MODELS):
                predicted = predicted_activity[m, a, selected]
                key = dict(split=split, assay=assay, model=model)
                tables["saturation"].append(dict(**key, **saturation_metrics(observed_raw[a, selected], predicted_raw[m, a, selected], observed, predicted, background[f"{assay}_sorted"])))
                for stratum, values, categories in (("observed_activity_bin", breadth_bin, range(8)), ("catalog_count", catalog_count, range(1, 9))):
                    for category in categories:
                        mask = values == category
                        if not mask.any():
                            continue
                        row = dict(**key, stratum=stratum, value=category,
                                   **count_metrics(observed[mask].sum(axis=1), predicted[mask].sum(axis=1)),
                                   observed_pr_mean=float(participation_ratio(observed[mask]).mean()),
                                   predicted_pr_mean=float(participation_ratio(predicted[mask]).mean()),
                                   log1p_context_rmse=float(np.sqrt(np.mean((predicted_log[m, a, selected][mask] - observed_log[a, selected][mask])**2))))
                        tables["stratified"].append(row)
        print(json.dumps(dict(event="evaluation_complete", split=split, elements=int(selected.sum()))), flush=True)
    output = args.output_directory
    for name, rows in tables.items():
        columns = list(dict.fromkeys(k for row in rows for k in row))
        write_tsv(output / f"{name}.tsv", rows, columns)
    observed_groups = np.stack([group_counts(x) for x in observed_activity])
    predicted_groups = np.stack([np.stack([group_counts(x) for x in model]) for model in predicted_activity])
    atomic_save_npz(output / "continuous_breadth_predictions.npz", ids=data["ids"], chrom=data["chrom"], summit=data["summit"],
                    sequence_model_split=data["split"], evaluation_partition=partitions, catalog_labels=labels,
                    contexts=np.asarray(CONTEXTS), assays=np.asarray(ASSAYS), model_names=np.asarray(MODELS),
                    group_names=np.asarray(list(GROUPS)), baseline_scales=np.asarray(SCALES), constant_means=constant_means,
                    observed_signal=observed_raw, predicted_signal=predicted_raw,
                    observed_activity=observed_activity, predicted_activity=predicted_activity,
                    observed_breadth=observed_activity.sum(axis=-1), predicted_breadth=predicted_activity.sum(axis=-1),
                    observed_group_breadth=observed_groups, predicted_group_breadth=predicted_groups,
                    observed_group_mean_activity=observed_groups / GROUP_SIZES,
                    predicted_group_mean_activity=predicted_groups / GROUP_SIZES,
                    observed_participation_ratio=participation_ratio(observed_activity),
                    predicted_participation_ratio=participation_ratio(predicted_activity))
    readable = {"id": data["ids"], "chrom": data["chrom"], "summit": data["summit"], "partition": partitions, "catalog_count": labels.sum(axis=1)}
    for a, assay in enumerate(ASSAYS):
        for source, activity in (("observed", observed_activity[a]), *((model, predicted_activity[m, a]) for m, model in enumerate(MODELS))):
            prefix = f"{assay}__{source}"
            readable[f"{prefix}__breadth"] = activity.sum(axis=1)
            readable[f"{prefix}__participation_ratio"] = participation_ratio(activity)
            readable.update({f"{prefix}__{c}": activity[:, i] for i, c in enumerate(CONTEXTS)})
            readable.update({f"{prefix}__group_{g}": group_counts(activity)[:, i] for i, g in enumerate(GROUPS)})
    pd.DataFrame(readable).to_csv(output / "element_predictions.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    source_files = [Path(__file__), *(Path(__file__).with_name(f"{name}.py") for name in ("breadth_metrics", "expected_breadth", "master_element_calibration", "continuous_activity", "browser_report", "constants", "io", "enhancer_model_comparison_inference"))]
    provenance = dict(method="continuous_assay_breadth_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                      command=sys.argv, device="cpu", sequence_inference_reused=True, no_new_model_fit=True,
                      warning=WARNING, normalization="max(0, 2 * training-background empirical midrank percentile - 1)",
                      groups=GROUPS, joint_assay_score=None, baseline="per-context means on calibration_fit, independently for each assay and scale",
                      saturation_cutoff=0.99, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
                      bootstrap="paired occupied chromosome-anchored 1-Mb blocks; conditional on frozen models/background/constant means",
                      partitions={p: int(np.sum(partitions == p)) for p in np.unique(partitions)},
                      input_hashes={str(p): sha256_file(p) for p in (args.predictions, metadata_path, args.background, background_metadata_path)},
                      prediction_metadata=prediction_metadata,
                      background_metadata={k: background_metadata[k] for k in ("method", "definition", "selection", "inputs")},
                      source_hashes={p.name: sha256_file(p) for p in source_files}, git_commit=git_commit(),
                      versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))
    write_report(output, tables)
    provenance["output_hashes"] = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    atomic_write_json(output / "provenance.json", finite_or_none(provenance))
    print(json.dumps(dict(event="continuous_breadth_complete", output=str(output))), flush=True)


if __name__ == "__main__":
    main()
