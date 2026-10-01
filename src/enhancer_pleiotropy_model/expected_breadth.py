"""Fit activity probabilities on frozen outputs and report expected context breadth."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from .breadth_metrics import (
    GROUPS, GROUP_SIZES, block_bootstrap, count_metrics, evaluate_probabilities,
    group_counts, probability_metrics, reliability_rows, validate_probabilities,
)
from .browser_report import finite_or_none, git_commit, html_table, write_tsv
from .constants import CONTEXTS
from .enhancer_model_comparison_inference import atomic_save_npz
from .io import atomic_write_json, atomic_write_text, sha256_file


VALIDATION_END = 11_751_856
TRAINING_START = 11_761_856
PRIMARY_MODEL = "fine_tuned_joint"
TRAINING_CHROMOSOMES = ("chrX", "chr2R", "chr3L", "chr4", "chrY", "chrUn_CP007081v1", "chrUn_CP007120v1")


def calibration_partition(chrom: np.ndarray, summit: np.ndarray, split: np.ndarray) -> np.ndarray:
    """Use fixed chr2L regions and require complete 2,048-bp inputs inside them."""
    result = np.full(len(chrom), "not_evaluated", dtype="<U24")
    midpoint = VALIDATION_END // 2
    validation = (chrom == "chr2L") & (split == "validation")
    result[validation] = "validation_buffer"
    result[validation & (summit >= 1024) & (summit + 1024 <= midpoint - 5000)] = "calibration_fit"
    result[validation & (summit - 1024 >= midpoint + 5000) & (summit + 1024 <= VALIDATION_END)] = "validation_evaluation"
    result[(chrom == "chr3R") & (split == "test")] = "chr3R_descriptive"
    if not all(np.any(result == name) for name in ("calibration_fit", "validation_evaluation", "chr3R_descriptive")):
        raise ValueError("Final-model regional fit/evaluation/test partitions are required")
    return result


def fit_probability_heads(features: np.ndarray, labels: np.ndarray, *, l2: float, device: str) -> tuple[np.ndarray, dict]:
    """Fit eight independent regularized logistic heads together using Torch L-BFGS."""
    if features.ndim != 2 or labels.shape != (len(features), 8):
        raise ValueError("Expected aligned features and eight labels")
    validate_probabilities(labels, labels.astype(float))
    if not np.isfinite(features).all() or l2 < 0:
        raise ValueError("Finite features and nonnegative L2 required")
    prevalence = labels.mean(axis=0)
    if np.any((prevalence == 0) | (prevalence == 1)):
        raise ValueError("Each fit context needs positive and negative examples")
    design = torch.as_tensor(np.column_stack((np.ones(len(features)), features)), dtype=torch.float64, device=device)
    target = torch.as_tensor(labels, dtype=torch.float64, device=device)
    weights = torch.zeros((design.shape[1], 8), dtype=torch.float64, device=device)
    weights[0] = torch.as_tensor(np.log(prevalence / (1 - prevalence)), device=device)
    weights.requires_grad_()
    optimizer = torch.optim.LBFGS([weights], max_iter=500, tolerance_grad=1e-9,
                                  tolerance_change=1e-12, line_search_fn="strong_wolfe")
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad()
        logits = design @ weights
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss = loss + 0.5 * l2 * weights[1:].square().sum() / 8
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)
    objective = float(closure().detach().cpu())
    gradient = float(weights.grad.abs().max().cpu()) * 8
    if not np.isfinite(objective) or gradient > 1e-5:
        raise RuntimeError(f"Logistic optimization did not converge: maximum head gradient {gradient}")
    return weights.detach().cpu().numpy(), {
        "objective": objective, "maximum_head_gradient": gradient,
        "function_evaluations": evaluations, "device": device, "dtype": "float64",
        "l2": l2, "converged": True,
    }


def predict_probability_heads(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or weights.shape != (features.shape[1] + 1, 8):
        raise ValueError("Features and saved probability heads do not align")
    return expit(weights[0] + features @ weights[1:])


def catalog_ambiguity(catalog: Path, data: dict) -> tuple[np.ndarray, dict]:
    """Screen high-ATAC nonmembers using medians fitted on training catalog members."""
    fields = ("context_membership", "atac_normalized_cpm_per_kb", "h3k27ac_max_500_background_percentile")
    columns = ["master_dhs_id", "chrom", "summit"] + [f"{context}__{field}" for context in CONTEXTS for field in fields]
    frame = pd.read_csv(catalog, sep="\t", usecols=columns)
    if frame["master_dhs_id"].duplicated().any() or frame[columns].isna().any().any():
        raise ValueError("Catalog identifiers must be unique and selected fields complete")
    chrom = frame["chrom"].astype(str).to_numpy()
    summit = frame["summit"].to_numpy(int)
    train = np.isin(chrom, TRAINING_CHROMOSOMES) | ((chrom == "chr2L") & (summit - 1024 >= TRAINING_START))
    membership = frame[[f"{c}__{fields[0]}" for c in CONTEXTS]].to_numpy(float)
    signal = frame[[f"{c}__{fields[1]}" for c in CONTEXTS]].to_numpy(float)
    percentile = frame[[f"{c}__{fields[2]}" for c in CONTEXTS]].to_numpy(float)
    if not np.isin(membership, (0, 1)).all() or not np.isfinite(signal).all() or np.any(signal < 0):
        raise ValueError("Invalid catalog membership or signal")
    if not np.isfinite(percentile).all() or np.any((percentile < 0) | (percentile > 1)):
        raise ValueError("Invalid catalog H3K27ac percentiles")
    cutoffs = np.asarray([np.median(signal[train & (membership[:, i] == 1), i]) for i in range(8)])
    if not np.isfinite(cutoffs).all():
        raise ValueError("Missing training members for ambiguity screening")
    lookup = pd.Index(frame["master_dhs_id"]).get_indexer(data["ids"].astype(str))
    if np.any(lookup < 0):
        raise ValueError("Prediction IDs missing from catalog")
    if not np.array_equal(chrom[lookup], data["chrom"].astype(str)) or not np.array_equal(summit[lookup], data["summit"]):
        raise ValueError("Catalog and prediction coordinates disagree")
    hard = (membership[lookup] == 1) & (percentile[lookup] > 0.6)
    if not np.array_equal(hard, data["hard_activity"]):
        raise ValueError("Cached labels disagree with membership AND H3 percentile > 0.6")
    ambiguity = (membership[lookup] == 0) & (signal[lookup] >= cutoffs) & (percentile[lookup] > 0.6)
    return ambiguity, {
        "definition": "nonmember AND ATAC >= training-member median AND H3K27ac background percentile > 0.6",
        "interpretation": "sensitivity screen, not proof that a catalog label is incorrect",
        "atac_medians": dict(zip(CONTEXTS, cutoffs.tolist(), strict=True)),
        "training_member_n": {c: int(np.sum(train & (membership[:, i] == 1))) for i, c in enumerate(CONTEXTS)},
        "training_chromosomes": list(TRAINING_CHROMOSOMES),
        "training_chr2L_start": TRAINING_START,
        "ambiguous_pairs": int(ambiguity.sum()), "affected_elements": int(ambiguity.any(axis=1).sum()),
    }


def calibration_svg(rows: list[dict], maximum: int, title: str) -> str:
    """Standalone calibration plot; no plotting dependency required."""
    points = [r for r in rows if r["n"]]
    content = []
    for row in points:
        x, y = 40 + 230 * row["predicted_mean"] / maximum, 265 - 230 * row["observed_mean"] / maximum
        content.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2166ac"><title>n={row["n"]}; predicted={row["predicted_mean"]:.3f}; observed={row["observed_mean"]:.3f}</title></circle>')
    return f'''<figure><figcaption>{html.escape(title)}</figcaption><svg viewBox="0 0 310 310" width="310" role="img" aria-label="{html.escape(title)}">
<path d="M40 35 V265 H270" fill="none" stroke="#333"/><path d="M40 265 L270 35" stroke="#999" stroke-dasharray="4"/>
<text x="33" y="282">0</text><text x="265" y="282">{maximum}</text><text x="15" y="40">{maximum}</text>
<text x="90" y="305">Predicted mean</text><text transform="translate(13,210) rotate(-90)">Observed mean</text>{''.join(content)}</svg></figure>'''


def write_report(directory: Path, summary: list[dict], details: dict, reliability: list[dict], provenance: dict, stratified: list[dict]) -> None:
    columns = ["model", "n", "breadth_rmse", "breadth_mae", "breadth_bias", "breadth_spearman", "brier", "group_normalized_mae", "allocation_error", "related_pair_macro_accuracy"]
    warning = (
        "Exploratory evaluation on known active enhancers. chr2L was used to select the sequence checkpoints; "
        "chr3R has already influenced earlier development. Neither evaluation is an untouched final test. "
        "No classifier thresholds or model settings are selected from these evaluation results."
    )
    sections = []
    markdown = ["# Expected enhancer activity breadth", "", warning, "",
                "Breadth is the sum of eight fitted activity probabilities. Groups partition the contexts; group contributions sum exactly to total breadth.", ""]
    for split in ("validation_evaluation", "chr3R_descriptive"):
        rows = [r for r in summary if r["split"] == split]
        sections.append(f"<h2>{split}</h2>" + html_table(rows, columns))
        markdown += [f"## {split}", "", "| Model | RMSE | MAE | Bias | Spearman | Brier |", "|---|---:|---:|---:|---:|---:|"]
        markdown += [f'| {r["model"]} | {r["breadth_rmse"]:.4f} | {r["breadth_mae"]:.4f} | {r["breadth_bias"]:.4f} | {r["breadth_spearman"]:.4f} | {r["brier"]:.4f} |' for r in rows]
        markdown.append("")
        primary = details[PRIMARY_MODEL][split]
        breadth_rows = [r for r in stratified if r["model"] == PRIMARY_MODEL and r["split"] == split and r["stratum"] == "observed_breadth"]
        sections.append("<h3>Fine-tuned joint model: errors across observed breadth</h3>" + html_table(breadth_rows, ["value", "n", "observed_mean", "predicted_mean", "rmse", "mae", "bias"]))
        markdown += ["Fine-tuned joint model by observed breadth:", "", "| Observed breadth | n | Predicted mean | MAE |", "|---|---:|---:|---:|"]
        markdown += [f'| {r["value"]} | {r["n"]} | {r["predicted_mean"]:.4f} | {r["mae"]:.4f} |' for r in breadth_rows]
        markdown.append("")
        group_rows = [{"group": group, **values} for group, values in primary["groups"].items()]
        sections.append("<h3>Fine-tuned joint model: groups and related contexts</h3>" + html_table(group_rows, ["group", "n", "observed_mean", "predicted_mean", "rmse", "mae", "bias", "normalized_mae"]) + html_table(primary["related_pairs"], ["group", "context_a", "context_b", "n", "accuracy", "tie_fraction"]))
        sections.append("<h3>Fine-tuned joint model: calibration</h3><div class='plots'>")
        for level, targets, maximum in (("total", ("total",), 8), ("group", tuple(GROUPS), None), ("context", CONTEXTS, 1)):
            for target in targets:
                selected = [r for r in reliability if r["model"] == PRIMARY_MODEL and r["split"] == split and r["level"] == level and r["target"] == target]
                limit = len(GROUPS[target]) if level == "group" else maximum
                sections.append(calibration_svg(selected, limit, f"{level}: {target}"))
        sections.append("</div>")
    notes = (
        "All context outputs and group sums are saved in expected_breadth_predictions.npz and element_predictions.tsv.gz. "
        "Group MAE is divided by group size for comparable scales. Allocation error compares each group's fraction of total activity, excluding zero totals. "
        "Related-pair ranking uses discordant catalog labels and gives ties half credit. "
        "Confidence intervals use paired 1-Mb genomic-block bootstrap, conditional on fitted models; they do not cover model-training or label uncertainty. "
        "Ambiguity-screened metrics exclude flagged pairs for context scores and entire affected enhancers for exact breadth. "
        "Interval-distance diagnostics retain affected enhancers with observed breadth between the catalog count and that count plus ambiguous pairs. "
        "The screen does not establish biological ground truth. No mutation-effect accuracy is assessed."
    )
    links = ["summary.tsv", "per_context.tsv", "per_group.tsv", "related_pairs.tsv", "stratified.tsv", "reliability.tsv", "bootstrap.tsv", "sensitivity.tsv", "metrics.json", "provenance.json", "calibrator_parameters.npz", "expected_breadth_predictions.npz", "element_predictions.tsv.gz"]
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Expected enhancer breadth</title>
<style>body{{font:15px system-ui;max-width:1450px;margin:2rem auto;padding:0 1rem;color:#222}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ddd;padding:.4rem;text-align:right}}.warning{{background:#fff3d9;padding:1rem}}.plots{{display:flex;flex-wrap:wrap}}figure{{margin:1rem}}figcaption{{font-weight:600}}</style>
<h1>Expected enhancer activity breadth</h1><p class="warning">{warning}</p><p>{html.escape(notes)}</p>{''.join(sections)}
<h2>Data and provenance</h2><ul>{''.join(f'<li><a href="{link}">{link}</a></li>' for link in links)}</ul>
<pre>{html.escape(json.dumps(provenance, indent=2))}</pre></html>'''
    atomic_write_text(directory / "index.html", report)
    atomic_write_text(directory / "summary.md", "\n".join(markdown) + notes + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.bootstrap_replicates < 2 or args.l2 < 0:
        parser.error("Require nonnegative L2 and at least two bootstrap replicates")
    if args.output_directory.exists():
        raise FileExistsError("Use a new output directory to preserve previous results")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU fallback is not automatic")
    prediction_hash = sha256_file(args.predictions)
    metadata_path = args.predictions.with_suffix(".metadata.json")
    input_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if input_metadata and input_metadata["output"]["sha256"] != prediction_hash:
        raise ValueError("Prediction cache hash disagrees with its metadata")
    with np.load(args.predictions, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    if tuple(data["contexts"].astype(str)) != CONTEXTS:
        raise ValueError("Unexpected prediction context order")
    if len(np.unique(data["ids"])) != len(data["ids"]):
        raise ValueError("Prediction IDs must be unique")
    sources = data["sources"].astype(str).tolist()
    feature_names = data["feature_names"].astype(str).tolist()
    features = data["features"].astype(float)
    if features.shape != (len(data["ids"]), len(sources), 8, len(feature_names)) or not np.isfinite(features).all() or np.any(features < 0):
        raise ValueError("Invalid cached feature dimensions or values")
    labels = data["hard_activity"].astype(float)
    validate_probabilities(labels, labels)
    if not labels.any(axis=1).all():
        raise ValueError("This analysis expects the active-any enhancer comparison cache")
    partitions = calibration_partition(data["chrom"].astype(str), data["summit"], data["split"].astype(str))
    fit = partitions == "calibration_fit"
    ambiguity, ambiguity_metadata = catalog_ambiguity(args.catalog, data)
    args.output_directory.mkdir(parents=True)
    print(json.dumps({"event": "breadth_start", "device": args.device,
                      "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                      "partitions": {key: int(np.sum(partitions == key)) for key in np.unique(partitions)}}), flush=True)
    probability_sets = {"prevalence": np.broadcast_to(labels[fit].mean(axis=0), labels.shape).copy()}
    parameters = {"contexts": np.asarray(CONTEXTS), "prevalence": labels[fit].mean(axis=0)}
    optimizations = {}
    for source in ("base", "fine_tuned"):
        feature_matrix = np.concatenate([
            features[:, sources.index(source), :, feature_names.index(name)]
            for name in ("atac_mean_512", "h3k27ac_max_mean_512")
        ], axis=1)
        transformed = np.log1p(feature_matrix)
        for variant, selected_columns in (("atac_only", np.arange(8)), ("h3k27ac_only", np.arange(8, 16)), ("joint", np.arange(16))):
            name = f"{source}_{variant}"
            values = transformed[:, selected_columns]
            mean, scale = values[fit].mean(axis=0), values[fit].std(axis=0)
            scale = np.where(scale > 1e-8, scale, 1)
            standardized = (values - mean) / scale
            print(json.dumps({"event": "fitting_probability_heads", "model": name}), flush=True)
            weights, metadata = fit_probability_heads(standardized[fit], labels[fit], l2=args.l2, device=args.device)
            probability_sets[name] = predict_probability_heads(standardized, weights)
            parameters.update({f"{name}__weights": weights, f"{name}__mean": mean,
                               f"{name}__scale": scale, f"{name}__feature_indices": selected_columns})
            optimizations[name] = metadata
            print(json.dumps({"event": "probability_heads_fitted", "model": name, **metadata}), flush=True)
    atomic_save_npz(args.output_directory / "calibrator_parameters.npz", **parameters)
    observed_signal = np.log1p(data["observed_h3k27ac_segment_means_512"].max(axis=2)).max(axis=1)
    signal_edges = np.quantile(observed_signal[fit], [0.25, 0.5, 0.75])
    signal_quartile = np.searchsorted(signal_edges, observed_signal, side="right") + 1
    summary, context_rows, group_rows, pair_rows, stratified, reliability, sensitivity, bootstrap = ([] for _ in range(8))
    details = {name: {} for name in probability_sets}
    for split in ("calibration_fit", "validation_evaluation", "chr3R_descriptive"):
        selected = partitions == split
        truth, uncertain = labels[selected], ambiguity[selected]
        for name, all_probabilities in probability_sets.items():
            probabilities = all_probabilities[selected]
            result = evaluate_probabilities(truth, probabilities)
            details[name][split] = result
            prefix = {"model": name, "split": split}
            summary.append({**prefix, **{f"breadth_{k}": v for k, v in result["breadth"].items() if k != "n"},
                            "n": len(truth), **{key: result[key] for key in ("brier", "macro_average_precision", "group_normalized_mae", "allocation_error", "related_pair_macro_accuracy")}})
            context_rows.extend({**prefix, "context": context, **values} for context, values in result["contexts"].items())
            group_rows.extend({**prefix, "group": group, **values} for group, values in result["groups"].items())
            pair_rows.extend({**prefix, **row} for row in result["related_pairs"])
            for level, names, observed, predicted, limits in (
                ("total", ("total",), truth.sum(axis=1)[:, None], probabilities.sum(axis=1)[:, None], (8,)),
                ("group", tuple(GROUPS), group_counts(truth), group_counts(probabilities), GROUP_SIZES),
                ("context", CONTEXTS, truth, probabilities, (1,) * 8),
            ):
                for index, target in enumerate(names):
                    reliability.extend({**prefix, "level": level, "target": target, **row} for row in reliability_rows(observed[:, index], predicted[:, index], int(limits[index])))
            for variable, groups in (("observed_breadth", truth.sum(axis=1)), ("observed_h3_signal_quartile", signal_quartile[selected])):
                for value in np.unique(groups):
                    keep = groups == value
                    metrics = count_metrics(truth[keep].sum(axis=1), probabilities[keep].sum(axis=1))
                    stratified.append({**prefix, "stratum": variable, "value": int(value), **metrics})
            clean = ~uncertain.any(axis=1)
            if clean.any():
                metrics = count_metrics(truth[clean].sum(axis=1), probabilities[clean].sum(axis=1))
                sensitivity.append({**prefix, "scope": "breadth_unflagged_elements", **metrics})
            for index, context in enumerate(CONTEXTS):
                keep = ~uncertain[:, index]
                sensitivity.append({**prefix, "scope": f"context_unflagged_pairs_{context}", **probability_metrics(truth[keep, index], probabilities[keep, index])})
            lower, upper = truth.sum(axis=1), truth.sum(axis=1) + uncertain.sum(axis=1)
            expected = probabilities.sum(axis=1)
            distance = np.maximum(np.maximum(lower - expected, expected - upper), 0)
            sensitivity.append({**prefix, "scope": "ambiguity_interval_distance", "n": len(truth),
                                "mae": float(distance.mean()), "rmse": float(np.sqrt(np.mean(distance**2))),
                                "affected_n": int((~clean).sum())})
        if split != "calibration_fit":
            blocks = np.asarray([f"{chrom}:{int(position) // 1_000_000}" for chrom, position in zip(data["chrom"][selected], data["summit"][selected], strict=True)])
            bootstrap.extend({"split": split, **row} for row in block_bootstrap(
                truth, {name: p[selected] for name, p in probability_sets.items()}, blocks,
                replicates=args.bootstrap_replicates, seed=args.seed, reference=PRIMARY_MODEL,
            ))
        print(json.dumps({"event": "breadth_split_evaluated", "split": split, "n": int(selected.sum())}), flush=True)
    tables = {"summary": summary, "per_context": context_rows, "per_group": group_rows, "related_pairs": pair_rows,
              "stratified": stratified, "reliability": reliability, "sensitivity": sensitivity, "bootstrap": bootstrap}
    for name, rows in tables.items():
        columns = list(dict.fromkeys(key for row in rows for key in row))
        write_tsv(args.output_directory / f"{name}.tsv", [{key: row.get(key, "") for key in columns} for row in rows], columns)
    names = tuple(probability_sets)
    probability_array = np.stack(list(probability_sets.values()))
    group_array = np.stack([group_counts(p) for p in probability_sets.values()])
    atomic_save_npz(args.output_directory / "expected_breadth_predictions.npz",
                    ids=data["ids"], chrom=data["chrom"], summit=data["summit"], contexts=data["contexts"],
                    sequence_model_split=data["split"], evaluation_partition=partitions, labels=labels.astype(np.uint8),
                    model_names=np.asarray(names), probabilities=probability_array, expected_breadth=probability_array.sum(axis=2),
                    group_names=np.asarray(tuple(GROUPS)), group_expected_breadth=group_array,
                    group_mean_probability=group_array / GROUP_SIZES, ambiguity=ambiguity)
    elements = {"id": data["ids"], "chrom": data["chrom"], "summit": data["summit"], "partition": partitions,
                "observed_breadth": labels.sum(axis=1), "ambiguous_contexts": ambiguity.sum(axis=1)}
    for context_index, context in enumerate(CONTEXTS):
        elements[f"observed_{context}"] = labels[:, context_index]
    for model_index, name in enumerate(names):
        elements[f"{name}__breadth"] = probability_array[model_index].sum(axis=1)
        for context_index, context in enumerate(CONTEXTS):
            elements[f"{name}__p_{context}"] = probability_array[model_index, :, context_index]
        for group_index, group in enumerate(GROUPS):
            elements[f"{name}__{group}_breadth"] = group_array[model_index, :, group_index]
    pd.DataFrame(elements).to_csv(args.output_directory / "element_predictions.tsv.gz", sep="\t", index=False, compression="gzip")
    provenance = {
        "method": "expected_activity_breadth_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": sys.argv, "git_commit": git_commit(), "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "inputs": {"predictions": {"path": str(args.predictions), "sha256": prediction_hash},
                   "catalog": {"path": str(args.catalog), "sha256": sha256_file(args.catalog)}},
        "prediction_metadata": input_metadata,
        "source_hashes": {file.name: sha256_file(file) for file in (Path(__file__), Path(__file__).with_name("breadth_metrics.py"))},
        "fit_definition": "chr2L complete inputs in [0,5870928); evaluation complete inputs in [5880928,11751856)",
        "partitions": {key: int(np.sum(partitions == key)) for key in np.unique(partitions)},
        "l2": args.l2, "seed": args.seed, "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_block_bp": 1_000_000, "bootstrap_reference": PRIMARY_MODEL,
        "bootstrap_limit": "conditional on fitted models; chr2L has few blocks; no model refitting",
        "activity_definition": "catalog context membership AND H3K27ac background percentile > 0.6",
        "groups": GROUPS, "ambiguity_screen": ambiguity_metadata,
        "signal_quartile_edges": signal_edges.tolist(),
        "signal_definition": "maximum across contexts of log1p observed H3K27ac max-segment mean; quartiles fitted on calibration-fit only",
        "optimizations": optimizations,
        "limitations": ["Active-any enhancers only: no zero-breadth generalization assessed",
                        "chr2L previously selected sequence checkpoints; chr3R previously inspected: all evaluation exploratory",
                        "No model selection or threshold tuning on chr3R", "Catalog ambiguity remains", "No mutation-effect validation"],
    }
    atomic_write_json(args.output_directory / "provenance.json", finite_or_none(provenance))
    atomic_write_json(args.output_directory / "metrics.json", finite_or_none({"models": details, "summary": summary}))
    write_report(args.output_directory, summary, details, reliability, provenance, stratified)
    print(json.dumps({"event": "expected_breadth_complete", "report": str(args.output_directory / "index.html")}), flush=True)


if __name__ == "__main__":
    main()
