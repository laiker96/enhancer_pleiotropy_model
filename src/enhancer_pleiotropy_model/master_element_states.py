"""Build assay-separated, background-standardized master-element states."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import pearsonr, spearmanr

from .constants import CONTEXTS
from .continuous_activity import participation_ratio
from .enhancer_catalog_evaluation import _split_map
from .io import atomic_write_json, atomic_write_text, open_text, sha256_file
from .preprocessing.profiles import binned_means


ENHANCER_CLASSES = ("distal_enhancer_like", "proximal_enhancer_like")
DEFAULT_BACKGROUND_CHROMOSOMES = (
    "chr2R",
    "chr3L",
    "chr4",
    "chrY",
    "chrUn_CP007081v1",
    "chrUn_CP007120v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--split-dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--atac-train-profiles", required=True, type=Path)
    parser.add_argument("--h3k27ac-train-profiles", required=True, type=Path)
    parser.add_argument("--bigwig-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--activity-threshold", default=0.6, type=float)
    parser.add_argument("--profile-chunk-size", default=5000, type=int)
    parser.add_argument(
        "--background-chromosomes",
        nargs="+",
        default=DEFAULT_BACKGROUND_CHROMOSOMES,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug-only limit applied after deterministic catalog selection.",
    )
    return parser.parse_args()


def background_training_indices(
    windows: Path,
    chromosomes: set[str],
) -> np.ndarray:
    """Return split-local train indices for selected genomic-background rows."""
    selected: list[int] = []
    train_index = 0
    with open_text(windows) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"split", "source", "chrom"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{windows}: missing columns: {sorted(missing)}")
        for row in reader:
            if row["split"] != "train":
                continue
            if row["source"] == "genomic_background" and row["chrom"] in chromosomes:
                selected.append(train_index)
            train_index += 1
    if not selected:
        raise ValueError("No eligible genomic-background training rows were found")
    return np.asarray(selected, dtype=np.int64)


def summarize_background_profiles(
    atac_path: Path,
    h3_path: Path,
    indices: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Summarize model-aligned profile arrays without loading them in full."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    atac = np.load(atac_path, mmap_mode="r")
    h3 = np.load(h3_path, mmap_mode="r")
    if atac.ndim != 3 or atac.shape[1:] != (32, len(CONTEXTS)):
        raise ValueError(f"Unexpected ATAC profile shape: {atac.shape}")
    if h3.ndim != 3 or h3.shape[1:] != (96, len(CONTEXTS)):
        raise ValueError(f"Unexpected H3K27ac profile shape: {h3.shape}")
    if len(atac) != len(h3) or indices.max(initial=-1) >= len(atac):
        raise ValueError("Profile rows and selected training indices do not align")

    atac_summary = np.empty((len(indices), len(CONTEXTS)), dtype=np.float32)
    h3_summary = np.empty_like(atac_summary)
    for start in range(0, len(indices), chunk_size):
        end = min(start + chunk_size, len(indices))
        selected = indices[start:end]
        atac_chunk = np.asarray(atac[selected], dtype=np.float32)
        h3_chunk = np.asarray(h3[selected], dtype=np.float32)
        atac_summary[start:end] = atac_chunk.mean(axis=1)
        h3_segments = h3_chunk.reshape(
            len(selected), 3, 32, len(CONTEXTS)
        ).mean(axis=2)
        h3_summary[start:end] = h3_segments.max(axis=1)
    return atac_summary, h3_summary, tuple(atac.shape), tuple(h3.shape)


def fit_background_transform(values: np.ndarray) -> dict[str, np.ndarray]:
    """Fit a robust per-context transform in log1p signal space."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Background values must be a finite nonnegative matrix")
    transformed = np.log1p(values)
    q25, median, q75 = np.quantile(transformed, (0.25, 0.5, 0.75), axis=0)
    mad = np.median(np.abs(transformed - median), axis=0)
    mad_scale = 1.4826 * mad
    iqr_scale = (q75 - q25) / 1.349
    scale = np.maximum(mad_scale, iqr_scale)
    standard_deviation = transformed.std(axis=0)
    scale = np.where(scale > 1e-8, scale, standard_deviation)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return {
        "median": median,
        "scale": scale,
        "q25": q25,
        "q75": q75,
        "mad": mad,
        "standard_deviation": standard_deviation,
    }


def apply_background_transform(
    values: np.ndarray,
    transform: dict[str, np.ndarray],
) -> np.ndarray:
    """Return positive robust-z excess above the background median."""
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Signals must be finite and nonnegative")
    median = np.asarray(transform["median"], dtype=np.float64)
    scale = np.asarray(transform["scale"], dtype=np.float64)
    if values.shape[-1] != len(median) or median.shape != scale.shape:
        raise ValueError("Signal contexts and background transform do not align")
    return np.maximum(0.0, (np.log1p(values) - median) / scale)


def noncompensatory_joint_activity(atac: np.ndarray, h3: np.ndarray) -> np.ndarray:
    """Require both standardized assays by taking their elementwise minimum."""
    atac = np.asarray(atac, dtype=np.float64)
    h3 = np.asarray(h3, dtype=np.float64)
    if atac.shape != h3.shape:
        raise ValueError("ATAC and H3K27ac states must align")
    if not np.all(np.isfinite(atac)) or not np.all(np.isfinite(h3)):
        raise ValueError("Standardized states must be finite")
    if np.any(atac < 0) or np.any(h3 < 0):
        raise ValueError("Standardized states must be nonnegative")
    return np.minimum(atac, h3)


def _safe_correlation(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if len(observed) < 2 or np.std(observed) == 0 or np.std(predicted) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(pearsonr(observed, predicted).statistic),
        "spearman": float(spearmanr(observed, predicted).statistic),
    }


def state_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    """Evaluate context profiles and their participation-ratio pleiotropy."""
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise ValueError("Observed and predicted states must be aligned matrices")
    per_context = {
        context: _safe_correlation(observed[:, index], predicted[:, index])
        for index, context in enumerate(CONTEXTS)
    }
    pattern_pearson: list[float] = []
    pattern_spearman: list[float] = []
    for true_row, predicted_row in zip(observed, predicted, strict=True):
        if np.std(true_row) == 0 or np.std(predicted_row) == 0:
            continue
        pattern_pearson.append(float(pearsonr(true_row, predicted_row).statistic))
        pattern_spearman.append(float(spearmanr(true_row, predicted_row).statistic))
    observed_pleiotropy = participation_ratio(observed)
    predicted_pleiotropy = participation_ratio(predicted)
    return {
        "by_context": per_context,
        "macro_pearson": float(np.nanmean([x["pearson"] for x in per_context.values()])),
        "macro_spearman": float(np.nanmean([x["spearman"] for x in per_context.values()])),
        "tissue_pattern_mean_pearson": float(np.mean(pattern_pearson)),
        "tissue_pattern_mean_spearman": float(np.mean(pattern_spearman)),
        "tissue_pattern_n": len(pattern_pearson),
        "pleiotropy": {
            **_safe_correlation(observed_pleiotropy, predicted_pleiotropy),
            "mae": float(np.mean(np.abs(observed_pleiotropy - predicted_pleiotropy))),
            "observed_mean": float(observed_pleiotropy.mean()),
            "predicted_mean": float(predicted_pleiotropy.mean()),
        },
    }


def _load_master_elements(
    catalog: Path,
    split_dataset: Path,
    activity_threshold: float,
    chromosome_sizes: dict[str, int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, int]:
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
        frame["regulatory_class"].isin(ENHANCER_CLASSES)
        & (frame["blacklist_overlap"] == 0)
    ].copy()
    membership = frame[
        [f"{context}__context_membership" for context in CONTEXTS]
    ].to_numpy(float) > 0.5
    percentiles = frame[
        [f"{context}__h3k27ac_max_500_background_percentile" for context in CONTEXTS]
    ].to_numpy(float)
    labels = membership & (percentiles > activity_threshold)
    if np.isclose(activity_threshold, 0.6):
        stored = frame[
            [f"{context}__h3k27ac_active" for context in CONTEXTS]
        ].to_numpy(float) > 0.5
        if not np.array_equal(labels, stored):
            raise RuntimeError("Recomputed activity labels disagree with the catalog")
    table_atac = frame[
        [f"{context}__atac_normalized_cpm_per_kb" for context in CONTEXTS]
    ].to_numpy(np.float32)
    table_h3 = frame[
        [f"{context}__h3k27ac_max_500_normalized_cpm_per_kb" for context in CONTEXTS]
    ].to_numpy(np.float32)

    valid = np.asarray(
        [
            str(chrom) in chromosome_sizes
            and int(summit) - 1024 >= 0
            and int(summit) + 1024 <= chromosome_sizes[str(chrom)]
            for chrom, summit in zip(frame["chrom"], frame["summit"], strict=True)
        ],
        dtype=bool,
    )
    invalid_n = int((~valid).sum())
    frame = frame.loc[valid].reset_index(drop=True)
    labels = labels[valid]
    table_atac = table_atac[valid]
    table_h3 = table_h3[valid]

    split_mapping = _split_map(split_dataset)
    frame["split"] = frame["master_dhs_id"].map(split_mapping)
    if frame["split"].isna().any():
        raise ValueError(
            f"Split dataset lacks {int(frame['split'].isna().sum())} master elements"
        )
    return frame, labels, table_atac, table_h3, invalid_n


def extract_observed_signals(
    frame: pd.DataFrame,
    bigwig_directory: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Path]]]:
    """Extract exact ATAC-512 and three H3K27ac-512 means from BigWigs."""
    count = len(frame)
    observed_atac = np.empty((count, len(CONTEXTS)), dtype=np.float32)
    observed_h3_segments = np.empty((count, len(CONTEXTS), 3), dtype=np.float32)
    groups = list(frame.groupby("chrom", sort=False).groups.items())
    paths: dict[str, dict[str, Path]] = {"atac": {}, "h3k27ac": {}}
    for context_index, context in enumerate(CONTEXTS):
        atac_path = bigwig_directory / f"{context}.atac.mean.background_tmm.bw"
        h3_path = bigwig_directory / f"{context}.h3k27ac.mean.background_tmm.bw"
        if not atac_path.is_file() or not h3_path.is_file():
            raise FileNotFoundError(f"Missing BigWig: {atac_path} or {h3_path}")
        paths["atac"][context] = atac_path
        paths["h3k27ac"][context] = h3_path
        with pyBigWig.open(str(atac_path)) as atac_bw, pyBigWig.open(str(h3_path)) as h3_bw:
            for chromosome, row_indices in groups:
                row_indices = np.asarray(row_indices, dtype=np.int64)
                summits = frame.loc[row_indices, "summit"].to_numpy(np.int64)
                observed_atac[row_indices, context_index] = binned_means(
                    atac_bw,
                    str(chromosome),
                    summits - 256,
                    summits + 256,
                    512,
                )[:, 0]
                observed_h3_segments[row_indices, context_index] = binned_means(
                    h3_bw,
                    str(chromosome),
                    summits - 768,
                    summits + 768,
                    512,
                )
        print(
            json.dumps(
                {
                    "event": "master_element_bigwigs_complete",
                    "context": context,
                    "complete": context_index + 1,
                    "total": len(CONTEXTS),
                }
            ),
            flush=True,
        )
    return observed_atac, observed_h3_segments, paths


def _align_predictions(
    predictions: Path,
    identifiers: np.ndarray,
    require_complete_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(predictions, allow_pickle=False) as cached:
        if tuple(str(value) for value in cached["contexts"]) != CONTEXTS:
            raise ValueError("Cached prediction context order is incorrect")
        feature_names = tuple(str(value) for value in cached["feature_names"])
        required = ("atac_mean_512", "h3k27ac_max_mean_512")
        if not set(required) <= set(feature_names):
            raise ValueError(f"Cached predictions lack {required}")
        cached_ids = cached["ids"].astype(np.str_)
        features = cached["features"].astype(np.float32)
    if len(set(cached_ids.tolist())) != len(cached_ids):
        raise ValueError("Cached prediction IDs are not unique")
    lookup = {identifier: index for index, identifier in enumerate(cached_ids)}
    positions = np.asarray([lookup.get(identifier, -1) for identifier in identifiers])
    has_prediction = positions >= 0
    predicted_atac = np.full((len(identifiers), len(CONTEXTS)), np.nan, np.float32)
    predicted_h3 = np.full_like(predicted_atac, np.nan)
    matched = positions[has_prediction]
    predicted_atac[has_prediction] = features[
        matched, :, feature_names.index("atac_mean_512")
    ]
    predicted_h3[has_prediction] = features[
        matched, :, feature_names.index("h3k27ac_max_mean_512")
    ]
    if require_complete_cache and not np.array_equal(
        np.sort(identifiers[has_prediction]), np.sort(cached_ids)
    ):
        raise ValueError("Some cached prediction IDs are absent from the master-element set")
    return predicted_atac, predicted_h3, has_prediction


def _qc_correlations(table: np.ndarray, exact: np.ndarray) -> dict[str, object]:
    by_context = {
        context: _safe_correlation(
            np.log1p(table[:, index]), np.log1p(exact[:, index])
        )
        for index, context in enumerate(CONTEXTS)
    }
    return {
        "by_context": by_context,
        "macro_pearson": float(np.mean([x["pearson"] for x in by_context.values()])),
        "macro_spearman": float(np.mean([x["spearman"] for x in by_context.values()])),
    }


def _transform_json(transform: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        field: dict(zip(CONTEXTS, values.astype(float).tolist(), strict=True))
        for field, values in transform.items()
    }


def _write_summary_table(
    path: Path,
    frame: pd.DataFrame,
    labels: np.ndarray,
    observed_atac: np.ndarray,
    observed_h3: np.ndarray,
    predicted_atac: np.ndarray,
    predicted_h3: np.ndarray,
    observed_atac_excess: np.ndarray,
    observed_h3_excess: np.ndarray,
    predicted_atac_excess: np.ndarray,
    predicted_h3_excess: np.ndarray,
    has_prediction: np.ndarray,
) -> None:
    columns = frame[
        [
            "master_dhs_id",
            "chrom",
            "start",
            "end",
            "summit",
            "regulatory_class",
            "split",
        ]
    ].copy()
    columns["active_context_count"] = labels.sum(axis=1)
    columns["active_in_any_context"] = labels.any(axis=1)
    columns["has_model_prediction"] = has_prediction
    observed_joint = noncompensatory_joint_activity(
        observed_atac_excess, observed_h3_excess
    )
    columns["observed_atac_pleiotropy"] = participation_ratio(observed_atac_excess)
    columns["observed_h3k27ac_pleiotropy"] = participation_ratio(observed_h3_excess)
    columns["observed_joint_min_pleiotropy"] = participation_ratio(observed_joint)
    predicted_atac_pleiotropy = np.full(len(frame), np.nan)
    predicted_h3_pleiotropy = np.full(len(frame), np.nan)
    predicted_joint_pleiotropy = np.full(len(frame), np.nan)
    predicted_atac_pleiotropy[has_prediction] = participation_ratio(
        predicted_atac_excess[has_prediction]
    )
    predicted_h3_pleiotropy[has_prediction] = participation_ratio(
        predicted_h3_excess[has_prediction]
    )
    predicted_joint_pleiotropy[has_prediction] = participation_ratio(
        noncompensatory_joint_activity(
            predicted_atac_excess[has_prediction], predicted_h3_excess[has_prediction]
        )
    )
    columns["predicted_atac_pleiotropy"] = predicted_atac_pleiotropy
    columns["predicted_h3k27ac_pleiotropy"] = predicted_h3_pleiotropy
    columns["predicted_joint_min_pleiotropy"] = predicted_joint_pleiotropy
    for index, context in enumerate(CONTEXTS):
        prefix = f"{context}__"
        columns[prefix + "active"] = labels[:, index]
        columns[prefix + "observed_atac_mean_512"] = observed_atac[:, index]
        columns[prefix + "observed_h3k27ac_max_mean_512"] = observed_h3[:, index]
        columns[prefix + "predicted_atac_mean_512"] = predicted_atac[:, index]
        columns[prefix + "predicted_h3k27ac_max_mean_512"] = predicted_h3[:, index]
        columns[prefix + "observed_atac_background_excess"] = observed_atac_excess[:, index]
        columns[prefix + "observed_h3k27ac_background_excess"] = observed_h3_excess[:, index]
        columns[prefix + "predicted_atac_background_excess"] = predicted_atac_excess[:, index]
        columns[prefix + "predicted_h3k27ac_background_excess"] = predicted_h3_excess[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tsv.gz", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        columns.to_csv(temporary, sep="\t", index=False, compression="gzip", na_rep="nan")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _summary_markdown(metrics: dict[str, object]) -> str:
    selection = metrics["selection"]
    lines = [
        "# Master-element continuous states",
        "",
        f"- Enhancer-like master elements: {selection['elements']:,}",
        f"- Active in at least one retained context: {selection['active_elements']:,}",
        f"- Elements with cached frozen-model predictions: {selection['predicted_elements']:,}",
        f"- Background training windows: {selection['background_windows']:,}",
        "- ATAC summary: exact summit-centered 512-bp BigWig mean.",
        "- H3K27ac summary: maximum of three adjacent 512-bp BigWig means over 1,536 bp.",
        "- Per assay/context transform: positive robust-z excess in log1p space relative to independent genomic-background training windows.",
        "- Joint activity: elementwise minimum of ATAC and H3K27ac excess; a high value in one assay cannot rescue a low value in the other.",
        "- ATAC, H3K27ac, and joint participation-ratio pleiotropy are retained separately.",
        "",
        "Hard catalog activity calls are descriptive annotations, not fitted thresholds for the continuous states.",
        "See `metrics.json`, `background_transform.json`, `master_element_summary.tsv.gz`, and `master_element_states.npz`.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    if not 0 <= args.activity_threshold <= 1:
        raise ValueError("activity-threshold must lie between zero and one")
    if args.profile_chunk_size < 1:
        raise ValueError("profile-chunk-size must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    background_chromosomes = tuple(dict.fromkeys(args.background_chromosomes))
    if not background_chromosomes:
        raise ValueError("At least one background chromosome is required")
    args.output_directory.mkdir(parents=True, exist_ok=True)

    first_bigwig = args.bigwig_directory / f"{CONTEXTS[0]}.atac.mean.background_tmm.bw"
    with pyBigWig.open(str(first_bigwig)) as bigwig:
        chromosome_sizes = {str(key): int(value) for key, value in bigwig.chroms().items()}
    frame, labels, table_atac, table_h3, invalid_n = _load_master_elements(
        args.catalog,
        args.split_dataset,
        args.activity_threshold,
        chromosome_sizes,
    )
    if args.limit is not None:
        frame = frame.iloc[: args.limit].reset_index(drop=True)
        labels = labels[: args.limit]
        table_atac = table_atac[: args.limit]
        table_h3 = table_h3[: args.limit]

    indices = background_training_indices(args.windows, set(background_chromosomes))
    background_atac, background_h3, atac_shape, h3_shape = summarize_background_profiles(
        args.atac_train_profiles,
        args.h3k27ac_train_profiles,
        indices,
        args.profile_chunk_size,
    )
    atac_transform = fit_background_transform(background_atac)
    h3_transform = fit_background_transform(background_h3)
    print(
        json.dumps(
            {"event": "background_transform_complete", "windows": len(indices)}
        ),
        flush=True,
    )

    observed_atac, observed_h3_segments, bigwig_paths = extract_observed_signals(
        frame, args.bigwig_directory
    )
    observed_h3 = observed_h3_segments.max(axis=2)
    identifiers = frame["master_dhs_id"].astype(str).to_numpy(dtype=np.str_)
    predicted_atac, predicted_h3, has_prediction = _align_predictions(
        args.predictions, identifiers, require_complete_cache=args.limit is None
    )

    observed_atac_excess = apply_background_transform(observed_atac, atac_transform)
    observed_h3_excess = apply_background_transform(observed_h3, h3_transform)
    predicted_atac_excess = np.full_like(predicted_atac, np.nan, dtype=np.float64)
    predicted_h3_excess = np.full_like(predicted_h3, np.nan, dtype=np.float64)
    predicted_atac_excess[has_prediction] = apply_background_transform(
        predicted_atac[has_prediction], atac_transform
    )
    predicted_h3_excess[has_prediction] = apply_background_transform(
        predicted_h3[has_prediction], h3_transform
    )
    observed_joint = noncompensatory_joint_activity(
        observed_atac_excess, observed_h3_excess
    )

    split_metrics: dict[str, object] = {}
    split_values = frame["split"].astype(str).to_numpy()
    for split in ("all", "train", "validation", "test"):
        selected = has_prediction & (
            np.ones(len(frame), dtype=bool) if split == "all" else split_values == split
        )
        if not selected.any():
            continue
        predicted_joint = noncompensatory_joint_activity(
            predicted_atac_excess[selected], predicted_h3_excess[selected]
        )
        split_metrics[split] = {
            "elements": int(selected.sum()),
            "atac": state_metrics(
                observed_atac_excess[selected], predicted_atac_excess[selected]
            ),
            "h3k27ac": state_metrics(
                observed_h3_excess[selected], predicted_h3_excess[selected]
            ),
            "joint_min": state_metrics(observed_joint[selected], predicted_joint),
        }

    active_count = labels.sum(axis=1)
    metrics: dict[str, object] = {
        "method": "master_element_background_standardized_states_v1",
        "selection": {
            "elements": int(len(frame)),
            "active_elements": int((active_count > 0).sum()),
            "inactive_enhancer_like_controls": int((active_count == 0).sum()),
            "predicted_elements": int(has_prediction.sum()),
            "excluded_invalid_2048bp_context": invalid_n,
            "activity_definition": (
                f"context membership and H3K27ac background percentile > {args.activity_threshold}"
            ),
            "background_windows": int(len(indices)),
            "background_chromosomes": list(background_chromosomes),
        },
        "state_definition": {
            "transform": "max(0, (log1p(signal) - background_median) / robust_scale)",
            "robust_scale": "max(1.4826*MAD, IQR/1.349), with SD then 1.0 fallback",
            "joint": "min(atac_background_excess, h3k27ac_background_excess)",
            "pleiotropy": "participation ratio across eight retained contexts",
            "note": "Assay-specific states and pleiotropy remain primary; joint-min is noncompensatory.",
        },
        "table_vs_exact_bigwig_qc": {
            "atac_catalog_variable_interval_vs_exact_central_512": _qc_correlations(
                table_atac, observed_atac
            ),
            "h3k27ac_catalog_max_500_vs_exact_max_of_three_512": _qc_correlations(
                table_h3, observed_h3
            ),
        },
        "frozen_model_performance": split_metrics,
        "inputs": {
            "catalog": {"path": str(args.catalog), "sha256": sha256_file(args.catalog)},
            "split_dataset": {
                "path": str(args.split_dataset),
                "sha256": sha256_file(args.split_dataset),
            },
            "predictions": {
                "path": str(args.predictions),
                "sha256": sha256_file(args.predictions),
            },
            "windows": {"path": str(args.windows), "sha256": sha256_file(args.windows)},
            "profiles": {
                "atac": {"path": str(args.atac_train_profiles), "shape": atac_shape},
                "h3k27ac": {"path": str(args.h3k27ac_train_profiles), "shape": h3_shape},
            },
            "bigwigs": {
                assay: {
                    context: {"path": str(path), "sha256": sha256_file(path)}
                    for context, path in paths.items()
                }
                for assay, paths in bigwig_paths.items()
            },
        },
    }
    transform_metadata = {
        "method": "robust_log1p_background_transform_v1",
        "background_windows": int(len(indices)),
        "background_chromosomes": list(background_chromosomes),
        "atac": _transform_json(atac_transform),
        "h3k27ac": _transform_json(h3_transform),
    }

    np.savez_compressed(
        args.output_directory / "master_element_states.npz",
        ids=identifiers,
        chrom=frame["chrom"].astype(str).to_numpy(dtype=np.str_),
        start=frame["start"].to_numpy(np.int64),
        end=frame["end"].to_numpy(np.int64),
        summit=frame["summit"].to_numpy(np.int64),
        split=split_values.astype(np.str_),
        contexts=np.asarray(CONTEXTS),
        hard_activity=labels.astype(np.uint8),
        has_prediction=has_prediction.astype(np.uint8),
        observed_atac_mean_512=observed_atac,
        observed_h3k27ac_segment_means_512=observed_h3_segments,
        predicted_atac_mean_512=predicted_atac,
        predicted_h3k27ac_max_mean_512=predicted_h3,
        observed_atac_background_excess=observed_atac_excess.astype(np.float32),
        observed_h3k27ac_background_excess=observed_h3_excess.astype(np.float32),
        predicted_atac_background_excess=predicted_atac_excess.astype(np.float32),
        predicted_h3k27ac_background_excess=predicted_h3_excess.astype(np.float32),
        observed_joint_min=observed_joint.astype(np.float32),
    )
    _write_summary_table(
        args.output_directory / "master_element_summary.tsv.gz",
        frame,
        labels,
        observed_atac,
        observed_h3,
        predicted_atac,
        predicted_h3,
        observed_atac_excess,
        observed_h3_excess,
        predicted_atac_excess,
        predicted_h3_excess,
        has_prediction,
    )
    atomic_write_json(args.output_directory / "background_transform.json", transform_metadata)
    atomic_write_json(args.output_directory / "metrics.json", metrics)
    atomic_write_text(args.output_directory / "summary.md", _summary_markdown(metrics))
    print(
        json.dumps(
            {
                "event": "master_element_states_complete",
                "elements": len(frame),
                "predicted_elements": int(has_prediction.sum()),
                "output": str(args.output_directory),
            }
        ),
        flush=True,
    )
    return metrics


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
