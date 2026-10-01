"""Evaluate a frozen joint regressor on the active v3 enhancer catalog."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .constants import CONTEXTS
from .inference import load_model, predict_sequences
from .io import atomic_write_json, read_fasta, sha256_file


FEATURE_NAMES = (
    "atac_mean_512",
    "atac_max_16",
    "h3k27ac_left_mean_512",
    "h3k27ac_center_mean_512",
    "h3k27ac_right_mean_512",
    "h3k27ac_max_mean_512",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--catalog-long", required=True, type=Path)
    parser.add_argument("--split-dataset", required=True, type=Path)
    parser.add_argument("--reference-fasta", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--activity-threshold", default=0.6, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--chunk-size", default=512, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument(
        "--no-reverse-complement-ensemble",
        action="store_true",
        help="Disable the forward/reverse-complement inference average.",
    )
    return parser.parse_args()


def _split_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"record_id", "source", "split"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing {sorted(required - set(reader.fieldnames or ())) }")
        for row in reader:
            if row["source"] == "catalog_enhancer_like_dhs":
                mapping[row["record_id"]] = row["split"]
    if not mapping:
        raise ValueError(f"{path}: no catalog enhancer-like DHS rows")
    return mapping


def load_active_enhancers(
    catalog: Path,
    split_dataset: Path,
    reference_fasta: Path,
    activity_threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load enhancer-like, non-blacklisted DHSs active in >=1 retained context."""
    general = [
        "master_dhs_id",
        "chrom",
        "start",
        "end",
        "summit",
        "blacklist_overlap",
        "regulatory_class",
    ]
    context_columns: list[str] = []
    for context in CONTEXTS:
        context_columns.extend(
            (
                f"{context}__context_membership",
                f"{context}__atac_normalized_cpm_per_kb",
                f"{context}__h3k27ac_max_500_normalized_cpm_per_kb",
                f"{context}__h3k27ac_max_500_background_percentile",
                f"{context}__h3k27ac_active",
            )
        )
    frame = pd.read_csv(
        catalog,
        sep="\t",
        compression="infer",
        usecols=general + context_columns,
    )
    frame = frame[
        frame["regulatory_class"].isin(
            ("distal_enhancer_like", "proximal_enhancer_like")
        )
        & (frame["blacklist_overlap"] == 0)
    ].copy()
    membership = frame[
        [f"{context}__context_membership" for context in CONTEXTS]
    ].to_numpy(float)
    percentiles = frame[
        [f"{context}__h3k27ac_max_500_background_percentile" for context in CONTEXTS]
    ].to_numpy(float)
    labels = (membership > 0.5) & (percentiles > activity_threshold)
    if np.isclose(activity_threshold, 0.6):
        catalog_labels = frame[
            [f"{context}__h3k27ac_active" for context in CONTEXTS]
        ].to_numpy(float) > 0.5
        if not np.array_equal(labels, catalog_labels):
            raise RuntimeError("Recomputed activity labels disagree with the v3 catalog")
    active = labels.any(axis=1)
    frame = frame.loc[active].reset_index(drop=True)
    labels = labels[active]
    observed_atac = frame[
        [f"{context}__atac_normalized_cpm_per_kb" for context in CONTEXTS]
    ].to_numpy(np.float32)
    observed_h3 = frame[
        [f"{context}__h3k27ac_max_500_normalized_cpm_per_kb" for context in CONTEXTS]
    ].to_numpy(np.float32)

    mapping = _split_map(split_dataset)
    frame["split"] = frame["master_dhs_id"].map(mapping)
    if frame["split"].isna().any():
        missing = int(frame["split"].isna().sum())
        raise ValueError(f"Split dataset lacks {missing} selected enhancer IDs")

    genome, _order = read_fasta(reference_fasta)
    sequences: list[str] = []
    valid = np.ones(len(frame), dtype=bool)
    for index, row in enumerate(frame.itertuples(index=False)):
        start = int(row.summit) - 1024
        chromosome = genome.get(str(row.chrom), "")
        sequence = chromosome[start : start + 2048] if start >= 0 else ""
        if len(sequence) != 2048 or set(sequence) - set("ACGT"):
            valid[index] = False
        sequences.append(sequence)
    if not valid.all():
        invalid_sequence_n = int((~valid).sum())
        frame = frame.loc[valid].reset_index(drop=True)
        labels = labels[valid]
        observed_atac = observed_atac[valid]
        observed_h3 = observed_h3[valid]
        sequences = [sequence for sequence, keep in zip(sequences, valid, strict=True) if keep]
    else:
        invalid_sequence_n = 0
    frame.attrs["invalid_sequence_n"] = invalid_sequence_n
    return frame, labels, observed_atac, observed_h3, sequences


def load_observed_h3_regions(
    catalog_long: Path, enhancer_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return matched whole-window means and central-window H3K27ac signals."""
    regional_columns = [
        f"h3k27ac_{region}_500_normalized_cpm_per_kb"
        for region in ("left", "center", "right")
    ]
    frame = pd.read_csv(
        catalog_long,
        sep="\t",
        compression="infer",
        usecols=["master_dhs_id", "context", *regional_columns],
    )
    frame = frame[
        frame["context"].isin(CONTEXTS)
        & frame["master_dhs_id"].isin(enhancer_ids)
    ].copy()
    if frame.duplicated(["master_dhs_id", "context"]).any():
        raise ValueError("Long catalog contains duplicate enhancer/context rows")
    expected_index = pd.MultiIndex.from_product(
        (enhancer_ids, CONTEXTS), names=("master_dhs_id", "context")
    )
    aligned = frame.set_index(["master_dhs_id", "context"]).reindex(expected_index)
    if aligned[regional_columns].isna().any().any():
        raise ValueError("Long catalog lacks an H3K27ac region for a selected enhancer")
    regions = aligned[regional_columns].to_numpy(np.float32).reshape(
        len(enhancer_ids), len(CONTEXTS), len(regional_columns)
    )
    return regions.mean(axis=-1), regions[:, :, 1]


def summarize_profiles(atac: np.ndarray, h3: np.ndarray) -> np.ndarray:
    if atac.ndim != 3 or h3.ndim != 3 or atac.shape[2] != len(CONTEXTS):
        raise ValueError("Unexpected profile shapes")
    if h3.shape[1] % 3:
        raise ValueError("H3K27ac profile cannot be divided into three equal windows")
    h3_segments = h3.reshape(len(h3), 3, h3.shape[1] // 3, len(CONTEXTS)).mean(axis=2)
    return np.stack(
        (
            atac.mean(axis=1),
            atac.max(axis=1),
            h3_segments[:, 0],
            h3_segments[:, 1],
            h3_segments[:, 2],
            h3_segments.max(axis=1),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def binary_curve(labels: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray | float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if labels.ndim != 1 or scores.shape != labels.shape or not 0 < positives < len(labels):
        raise ValueError("Binary metrics require aligned labels containing both classes")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    distinct = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    true_positive = np.cumsum(sorted_labels)[distinct].astype(float)
    false_positive = (np.cumsum(1 - sorted_labels)[distinct]).astype(float)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / positives
    recall_step = np.diff(np.r_[0.0, recall])
    average_precision = float(np.sum(recall_step * precision))
    auprc = float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))
    return {
        "thresholds": sorted_scores[distinct],
        "precision": precision,
        "recall": recall,
        "average_precision": average_precision,
        "auprc_trapezoidal": auprc,
    }


def choose_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    curve = binary_curve(labels, scores)
    precision = np.asarray(curve["precision"])
    recall = np.asarray(curve["recall"])
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    best = int(np.argmax(f1))
    return float(np.asarray(curve["thresholds"])[best]), float(f1[best])


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    curve = binary_curve(labels, scores)
    calls = scores >= threshold
    true_positive = int(np.sum(calls & (labels == 1)))
    predicted_positive = int(calls.sum())
    positive = int(labels.sum())
    return {
        "n": int(len(labels)),
        "positive_n": positive,
        "negative_n": int(len(labels) - positive),
        "prevalence": float(labels.mean()),
        "threshold": float(threshold),
        "precision": true_positive / predicted_positive if predicted_positive else 0.0,
        "recall": true_positive / positive,
        "average_precision": float(curve["average_precision"]),
        "auprc_trapezoidal": float(curve["auprc_trapezoidal"]),
    }


def empirical_percentiles(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def correlation_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    observed = np.log1p(np.maximum(observed, 0))
    predicted = np.log1p(np.maximum(predicted, 0))
    by_context = {}
    for index, context in enumerate(CONTEXTS):
        by_context[context] = {
            "pearson": float(pearsonr(observed[:, index], predicted[:, index]).statistic),
            "spearman": float(spearmanr(observed[:, index], predicted[:, index]).statistic),
        }
    pattern_pearson = []
    pattern_spearman = []
    for true_row, predicted_row in zip(observed, predicted, strict=True):
        if np.std(true_row) > 0 and np.std(predicted_row) > 0:
            pattern_pearson.append(float(np.corrcoef(true_row, predicted_row)[0, 1]))
            pattern_spearman.append(
                float(spearmanr(true_row, predicted_row).statistic)
            )
    return {
        "by_context": by_context,
        "macro_pearson": float(np.mean([value["pearson"] for value in by_context.values()])),
        "macro_spearman": float(np.mean([value["spearman"] for value in by_context.values()])),
        "tissue_pattern_mean_pearson": float(np.mean(pattern_pearson)),
        "tissue_pattern_median_pearson": float(np.median(pattern_pearson)),
        "tissue_pattern_mean_spearman": float(np.mean(pattern_spearman)),
        "tissue_pattern_median_spearman": float(np.median(pattern_spearman)),
        "tissue_pattern_n": len(pattern_pearson),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.chunk_size < 1:
        raise ValueError("Batch and chunk sizes must be positive")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    partial_directory = args.output_directory / "partial"
    partial_directory.mkdir(exist_ok=True)
    frame, labels, observed_atac, observed_h3, sequences = load_active_enhancers(
        args.catalog, args.split_dataset, args.reference_fasta, args.activity_threshold
    )
    identifiers = np.asarray(frame["master_dhs_id"].astype(str), dtype=np.str_)
    splits = np.asarray(frame["split"].astype(str), dtype=np.str_)
    observed_h3_mean, observed_h3_center = load_observed_h3_regions(
        args.catalog_long, identifiers
    )
    model, metadata = load_model(args.checkpoint, args.device)
    feature_chunks: list[np.ndarray] = []
    start_time = time.monotonic()
    for start in range(0, len(frame), args.chunk_size):
        end = min(start + args.chunk_size, len(frame))
        partial_path = partial_directory / f"features_{start:06d}_{end:06d}.npz"
        if partial_path.is_file():
            try:
                saved = np.load(partial_path, allow_pickle=False)
                saved_ids = saved["ids"]
                saved_features = saved["features"]
            except ValueError:
                # Migrate caches produced before IDs were stored as Unicode.
                saved = np.load(partial_path, allow_pickle=True)
                saved_ids = np.asarray(saved["ids"], dtype=np.str_)
                saved_features = saved["features"]
                np.savez_compressed(
                    partial_path,
                    ids=saved_ids,
                    features=saved_features,
                )
            if not np.array_equal(saved_ids, identifiers[start:end]):
                raise RuntimeError(f"Cached IDs disagree: {partial_path}")
            features = saved_features
            event = "enhancer_inference_chunk_reused"
        else:
            atac, h3 = predict_sequences(
                model,
                sequences[start:end],
                batch_size=args.batch_size,
                device=args.device,
                reverse_complement_ensemble=not args.no_reverse_complement_ensemble,
                mixed_precision=args.mixed_precision,
            )
            features = summarize_profiles(atac, h3)
            np.savez_compressed(partial_path, ids=identifiers[start:end], features=features)
            event = "enhancer_inference_chunk_complete"
        feature_chunks.append(features)
        elapsed = time.monotonic() - start_time
        print(json.dumps({"event": event, "complete": end, "total": len(frame), "elapsed_seconds": elapsed}), flush=True)
    features = np.concatenate(feature_chunks)
    np.savez_compressed(
        args.output_directory / "predicted_features.npz",
        ids=identifiers,
        chrom=np.asarray(frame["chrom"].astype(str), dtype=np.str_),
        split=splits,
        labels=labels.astype(np.uint8),
        observed_atac=observed_atac,
        observed_h3k27ac=observed_h3,
        observed_h3k27ac_mean_1500=observed_h3_mean,
        observed_h3k27ac_center_500=observed_h3_center,
        features=features,
        feature_names=np.asarray(FEATURE_NAMES),
        contexts=np.asarray(CONTEXTS),
    )

    train = splits == "train"
    scores = {
        "predicted_atac_mean": features[:, :, 0].astype(float),
        "predicted_h3k27ac_max500": features[:, :, 5].astype(float),
        "geometric_mean_predicted_percentiles": np.empty(labels.shape, dtype=float),
    }
    for context_index in range(len(CONTEXTS)):
        atac_percentile = empirical_percentiles(
            scores["predicted_atac_mean"][train, context_index],
            scores["predicted_atac_mean"][:, context_index],
        )
        h3_percentile = empirical_percentiles(
            scores["predicted_h3k27ac_max500"][train, context_index],
            scores["predicted_h3k27ac_max500"][:, context_index],
        )
        scores["geometric_mean_predicted_percentiles"][:, context_index] = np.sqrt(
            atac_percentile * h3_percentile
        )

    conditions: dict[str, object] = {}
    validation = splits == "validation"
    for name, values in scores.items():
        thresholds = np.empty(len(CONTEXTS), dtype=float)
        validation_f1 = {}
        for index, context in enumerate(CONTEXTS):
            thresholds[index], validation_f1[context] = choose_f1_threshold(
                labels[validation, index], values[validation, index]
            )
        split_results = {}
        for split in ("train", "validation", "test", "all"):
            mask = np.ones(len(frame), dtype=bool) if split == "all" else splits == split
            by_context = {
                context: binary_metrics(labels[mask, index], values[mask, index], thresholds[index])
                for index, context in enumerate(CONTEXTS)
            }
            split_results[split] = {
                "by_context": by_context,
                "macro": {
                    metric: float(np.mean([entry[metric] for entry in by_context.values()]))
                    for metric in ("precision", "recall", "average_precision", "auprc_trapezoidal")
                },
            }
        conditions[name] = {
            "threshold_selection": {
                "split": "validation",
                "criterion": "maximum F1 per context",
                "thresholds": dict(zip(CONTEXTS, thresholds.tolist(), strict=True)),
                "f1": validation_f1,
            },
            **split_results,
        }

    continuous = {}
    for split in ("train", "validation", "test", "all"):
        mask = np.ones(len(frame), dtype=bool) if split == "all" else splits == split
        continuous[split] = {
            "atac": correlation_metrics(observed_atac[mask], features[mask, :, 0]),
            "h3k27ac": correlation_metrics(observed_h3[mask], features[mask, :, 5]),
            "h3k27ac_mean_1536": correlation_metrics(
                observed_h3_mean[mask], features[mask, :, 2:5].mean(axis=-1)
            ),
            "h3k27ac_center_mean_512": correlation_metrics(
                observed_h3_center[mask], features[mask, :, 3]
            ),
        }
    result = {
        "method": "active_v3_distal_enhancer_frozen_joint_regressor_evaluation_v1",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": metadata.checkpoint_sha256,
            "epoch": metadata.epoch,
            "architecture": metadata.architecture,
        },
        "inputs": {
            "catalog": {"path": str(args.catalog), "sha256": sha256_file(args.catalog)},
            "catalog_long": {
                "path": str(args.catalog_long),
                "sha256": sha256_file(args.catalog_long),
            },
            "split_dataset": {"path": str(args.split_dataset), "sha256": sha256_file(args.split_dataset)},
            "reference_fasta": {"path": str(args.reference_fasta), "sha256": sha256_file(args.reference_fasta)},
        },
        "selection": {
            "regulatory_class": ["distal_enhancer_like", "proximal_enhancer_like"],
            "blacklist_overlap": 0,
            "activity_definition": f"context membership and H3K27ac background percentile > {args.activity_threshold}",
            "set_definition": "active in at least one of the eight retained contexts",
            "enhancers": len(frame),
            "excluded_invalid_2048bp_sequences": int(frame.attrs["invalid_sequence_n"]),
            "split_counts": {split: int(np.sum(splits == split)) for split in ("train", "validation", "test")},
        },
        "inference": {
            "reverse_complement_ensemble": not args.no_reverse_complement_ensemble,
            "device": args.device,
            "mixed_precision": args.mixed_precision,
            "batch_size": args.batch_size,
            "chunk_size": args.chunk_size,
            "elapsed_seconds": time.monotonic() - start_time,
        },
        "continuous_signal": continuous,
        "enhancer_activity_classification": conditions,
        "metric_notes": {
            "negative_class": "enhancers inactive in this context but active in at least one other retained context",
            "thresholds": "chosen independently per context on validation only",
            "continuous_transform": "Pearson and Spearman calculated after log1p",
            "h3k27ac_prediction": "maximum of the mean prediction in the left, center, and right 512-bp regions",
            "h3k27ac_alternative_summaries": {
                "h3k27ac": "predicted maximum of three 512-bp means versus the observed maximum of three 500-bp means",
                "h3k27ac_mean_1536": "predicted mean over 1,536 bp versus the observed mean over the matching three 500-bp regions",
                "h3k27ac_center_mean_512": "predicted central 512-bp mean versus the observed central 500-bp mean",
            },
        },
    }
    atomic_write_json(args.output_directory / "metrics.json", result)
    print(json.dumps({"event": "enhancer_catalog_evaluation_complete", "output": str(args.output_directory / "metrics.json")}), flush=True)


if __name__ == "__main__":
    main()
