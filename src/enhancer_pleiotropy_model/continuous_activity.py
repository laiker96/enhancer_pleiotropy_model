"""Evaluate model-aligned continuous enhancer activity on a held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import pearsonr, spearmanr

from .constants import CONTEXTS
from .enhancer_catalog_evaluation import correlation_metrics
from .io import atomic_write_json, atomic_write_text, sha256_file
from .preprocessing.profiles import binned_means


RELATED_GROUPS = {
    "brain": ("ab", "lb"),
    "embryo": ("e5", "e13"),
    "imaginal_disc": ("ead", "hid", "wid"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--bigwig-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--activity-threshold", default=0.6, type=float)
    parser.add_argument("--atac-window-bp", default=512, type=int)
    return parser.parse_args()


def combined_activity(atac: np.ndarray, h3k27ac: np.ndarray) -> np.ndarray:
    """Geometric mean of aligned nonnegative ATAC and H3K27ac signals."""
    atac = np.asarray(atac, dtype=np.float64)
    h3k27ac = np.asarray(h3k27ac, dtype=np.float64)
    if atac.shape != h3k27ac.shape:
        raise ValueError("ATAC and H3K27ac arrays must align")
    if not np.all(np.isfinite(atac)) or not np.all(np.isfinite(h3k27ac)):
        raise ValueError("Activity inputs must be finite")
    if np.any(atac < 0) or np.any(h3k27ac < 0):
        raise ValueError("Activity inputs must be nonnegative")
    return np.sqrt(atac * h3k27ac)


def participation_ratio(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return continuous context breadth along the final array dimension."""
    values = np.abs(np.asarray(values, dtype=np.float64))
    if values.ndim < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Participation-ratio input must be a finite array")
    numerator = np.square(values.sum(axis=-1))
    denominator = np.square(values).sum(axis=-1)
    return np.divide(
        numerator,
        denominator + epsilon,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def _correlation(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("Correlation inputs must be aligned vectors")
    if len(observed) < 2 or np.std(observed) == 0 or np.std(predicted) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(pearsonr(observed, predicted).statistic),
        "spearman": float(spearmanr(observed, predicted).statistic),
    }


def contrast_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    contexts: tuple[str, ...] = CONTEXTS,
) -> list[dict[str, object]]:
    """Measure recovery of every signed context contrast on log1p signal."""
    if observed.shape != predicted.shape or observed.shape[1] != len(contexts):
        raise ValueError("Contrast arrays and contexts must align")
    observed_log = np.log1p(np.maximum(observed, 0))
    predicted_log = np.log1p(np.maximum(predicted, 0))
    related_pairs = {
        frozenset((first, second))
        for group in RELATED_GROUPS.values()
        for index, first in enumerate(group)
        for second in group[index + 1 :]
    }
    rows: list[dict[str, object]] = []
    for first_index, first in enumerate(contexts):
        for second_index in range(first_index + 1, len(contexts)):
            second = contexts[second_index]
            true_difference = observed_log[:, first_index] - observed_log[:, second_index]
            predicted_difference = (
                predicted_log[:, first_index] - predicted_log[:, second_index]
            )
            nonzero = true_difference != 0
            correlations = _correlation(true_difference, predicted_difference)
            rows.append(
                {
                    "context_a": first,
                    "context_b": second,
                    "related_group": next(
                        (
                            name
                            for name, group in RELATED_GROUPS.items()
                            if first in group and second in group
                        ),
                        "unrelated",
                    ),
                    "is_related": frozenset((first, second)) in related_pairs,
                    "n": int(len(observed)),
                    "nonzero_true_difference_n": int(nonzero.sum()),
                    **correlations,
                    "direction_accuracy": float(
                        np.mean(
                            np.sign(true_difference[nonzero])
                            == np.sign(predicted_difference[nonzero])
                        )
                    )
                    if nonzero.any()
                    else float("nan"),
                    "mean_absolute_log_difference_error": float(
                        np.mean(np.abs(true_difference - predicted_difference))
                    ),
                }
            )
    return rows


def _load_catalog(
    path: Path,
    identifiers: np.ndarray,
    stored_chromosomes: np.ndarray,
    activity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = ["master_dhs_id", "chrom", "summit"]
    for context in CONTEXTS:
        columns.extend(
            (
                f"{context}__context_membership",
                f"{context}__h3k27ac_max_500_background_percentile",
            )
        )
    frame = pd.read_csv(path, sep="\t", compression="infer", usecols=columns)
    if frame["master_dhs_id"].duplicated().any():
        raise ValueError("Catalog contains duplicate enhancer IDs")
    aligned = frame.set_index("master_dhs_id").reindex(identifiers)
    if aligned.isna().any().any():
        raise ValueError("Catalog metadata are missing for cached predictions")
    chromosomes = aligned["chrom"].astype(str).to_numpy(dtype=np.str_)
    if not np.array_equal(chromosomes, stored_chromosomes):
        raise ValueError("Catalog chromosome order disagrees with predictions")
    membership = aligned[
        [f"{context}__context_membership" for context in CONTEXTS]
    ].to_numpy(float) > 0.5
    h3_high = aligned[
        [f"{context}__h3k27ac_max_500_background_percentile" for context in CONTEXTS]
    ].to_numpy(float) > activity_threshold
    return aligned["summit"].to_numpy(np.int64), membership, h3_high


def extract_atac_window_means(
    bigwig_directory: Path,
    chromosome: str,
    summits: np.ndarray,
    window_bp: int,
) -> tuple[np.ndarray, dict[str, Path]]:
    """Extract exact central-window ATAC means using one range read per context."""
    if window_bp < 1 or window_bp % 2:
        raise ValueError("ATAC window size must be a positive even integer")
    starts = np.asarray(summits, dtype=np.int64) - window_bp // 2
    ends = starts + window_bp
    output = np.empty((len(summits), len(CONTEXTS)), dtype=np.float32)
    paths: dict[str, Path] = {}
    for index, context in enumerate(CONTEXTS):
        path = bigwig_directory / f"{context}.atac.mean.background_tmm.bw"
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[context] = path
        with pyBigWig.open(str(path)) as bigwig:
            output[:, index] = binned_means(
                bigwig, chromosome, starts, ends, window_bp
            )[:, 0]
        print(
            json.dumps(
                {
                    "event": "observed_atac_context_complete",
                    "context": context,
                    "complete": index + 1,
                    "total": len(CONTEXTS),
                }
            ),
            flush=True,
        )
    return output, paths


def _stratum_rows(
    membership: np.ndarray,
    h3_high: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, object]]:
    definitions = {
        "atac_peak_h3_high": membership & h3_high,
        "atac_peak_h3_low": membership & ~h3_high,
        "no_atac_peak_h3_high": ~membership & h3_high,
        "neither": ~membership & ~h3_high,
    }
    rows = []
    for name, selected in definitions.items():
        true_values = observed[selected]
        predicted_values = predicted[selected]
        rows.append(
            {
                "stratum": name,
                "n": int(selected.sum()),
                "observed_median": float(np.median(true_values)),
                "predicted_median": float(np.median(predicted_values)),
                "observed_mean": float(np.mean(true_values)),
                "predicted_mean": float(np.mean(predicted_values)),
                "mean_absolute_log_error": float(
                    np.mean(np.abs(np.log1p(true_values) - np.log1p(predicted_values)))
                ),
            }
        )
    return rows


def _subset_metrics(
    labels: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    breadth = labels.sum(axis=1)
    subsets = {
        "context_specific_breadth_1": breadth == 1,
        "intermediate_breadth_2_to_4": (breadth >= 2) & (breadth <= 4),
        "broad_breadth_5_to_8": breadth >= 5,
    }
    result = {}
    for name, selected in subsets.items():
        true_pleiotropy = participation_ratio(observed[selected])
        predicted_pleiotropy = participation_ratio(predicted[selected])
        pattern = correlation_metrics(observed[selected], predicted[selected])
        flattened = _correlation(
            np.log1p(observed[selected]).ravel(),
            np.log1p(predicted[selected]).ravel(),
        )
        pleiotropy = _correlation(true_pleiotropy, predicted_pleiotropy)
        result[name] = {
            "enhancers": int(selected.sum()),
            "flattened_log1p_pearson": flattened["pearson"],
            "flattened_log1p_spearman": flattened["spearman"],
            "tissue_pattern_mean_pearson": pattern["tissue_pattern_mean_pearson"],
            "tissue_pattern_mean_spearman": pattern["tissue_pattern_mean_spearman"],
            "pleiotropy_pearson": pleiotropy["pearson"],
            "pleiotropy_spearman": pleiotropy["spearman"],
            "pleiotropy_mae": float(
                np.mean(np.abs(true_pleiotropy - predicted_pleiotropy))
            ),
        }
    return result


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, na_rep="nan")


def _summary_markdown(metrics: dict[str, object]) -> str:
    continuous = metrics["continuous_combined_activity"]
    pleiotropy = metrics["continuous_pleiotropy"]
    lines = [
        "# Continuous enhancer-activity evaluation",
        "",
        f"- Split: `{metrics['selection']['split']}`",
        f"- Enhancers: {metrics['selection']['enhancers']:,}",
        "- Score: `sqrt(ATAC central-512 mean x H3K27ac max of three 500-bp means)`",
        "- Correlations use `log1p` signal except the scale-invariant pleiotropy score.",
        "",
        "## Main results",
        "",
        f"- Macro per-context Pearson: {continuous['macro_pearson']:.4f}",
        f"- Macro per-context Spearman: {continuous['macro_spearman']:.4f}",
        f"- Mean tissue-pattern Pearson: {continuous['tissue_pattern_mean_pearson']:.4f}",
        f"- Mean tissue-pattern Spearman: {continuous['tissue_pattern_mean_spearman']:.4f}",
        f"- Enhancer pleiotropy Pearson: {pleiotropy['pearson']:.4f}",
        f"- Enhancer pleiotropy Spearman: {pleiotropy['spearman']:.4f}",
        f"- Enhancer pleiotropy MAE: {pleiotropy['mae']:.4f}",
        "",
        "## Interpretation boundary",
        "",
        "This evaluates continuous observed signal, not agreement with the catalog's hard activity call. "
        "The hard breadth labels are used only to define descriptive subsets.",
        "",
        "See `metrics.json`, `per_context.tsv`, `context_contrasts.tsv`, "
        "`assay_strata.tsv`, and `continuous_activity.npz` for full results.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not 0 <= args.activity_threshold <= 1:
        raise ValueError("activity-threshold must lie between zero and one")
    args.output_directory.mkdir(parents=True, exist_ok=True)

    with np.load(args.predictions, allow_pickle=False) as cached:
        contexts = tuple(str(value) for value in cached["contexts"])
        if contexts != CONTEXTS:
            raise ValueError(f"Cached context order differs from {CONTEXTS}")
        feature_names = tuple(str(value) for value in cached["feature_names"])
        required_features = ("atac_mean_512", "h3k27ac_max_mean_512")
        missing = set(required_features) - set(feature_names)
        if missing:
            raise ValueError(f"Cached predictions lack features: {sorted(missing)}")
        selected = cached["split"] == args.split
        if not selected.any():
            raise ValueError(f"Cached predictions contain no split {args.split!r}")
        identifiers = cached["ids"][selected].astype(np.str_)
        chromosomes = cached["chrom"][selected].astype(np.str_)
        labels = cached["labels"][selected].astype(bool)
        observed_h3 = cached["observed_h3k27ac"][selected].astype(np.float64)
        features = cached["features"][selected].astype(np.float64)

    unique_chromosomes = np.unique(chromosomes)
    if len(unique_chromosomes) != 1:
        raise ValueError("Exact ATAC extraction currently requires one split chromosome")
    chromosome = str(unique_chromosomes[0])
    summits, membership, h3_high = _load_catalog(
        args.catalog, identifiers, chromosomes, args.activity_threshold
    )
    if not np.array_equal(labels, membership & h3_high):
        raise ValueError("Cached labels disagree with catalog activity states")

    observed_atac, bigwig_paths = extract_atac_window_means(
        args.bigwig_directory, chromosome, summits, args.atac_window_bp
    )
    predicted_atac = features[:, :, feature_names.index("atac_mean_512")]
    predicted_h3 = features[:, :, feature_names.index("h3k27ac_max_mean_512")]
    observed_activity = combined_activity(observed_atac, observed_h3)
    predicted_activity = combined_activity(predicted_atac, predicted_h3)
    observed_pleiotropy = participation_ratio(observed_activity)
    predicted_pleiotropy = participation_ratio(predicted_activity)

    continuous = correlation_metrics(observed_activity, predicted_activity)
    per_context_rows = [
        {"context": context, **continuous["by_context"][context]}
        for context in CONTEXTS
    ]
    contrast_rows = contrast_metrics(observed_activity, predicted_activity)
    stratum_rows = _stratum_rows(
        membership, h3_high, observed_activity, predicted_activity
    )
    pleiotropy_correlations = _correlation(observed_pleiotropy, predicted_pleiotropy)

    metrics: dict[str, object] = {
        "method": "model_aligned_continuous_enhancer_activity_v1",
        "selection": {
            "split": args.split,
            "chromosome": chromosome,
            "enhancers": int(len(identifiers)),
            "catalog_active_in_at_least_one_context": True,
            "hard_label_use": "descriptive breadth subsets only",
        },
        "score_definition": {
            "formula": "sqrt(atac * h3k27ac)",
            "observed_atac": f"exact mean normalized signal in summit-centered {args.atac_window_bp} bp",
            "predicted_atac": "mean prediction over central 512 bp",
            "observed_h3k27ac": "maximum of left, center, right 500-bp normalized means",
            "predicted_h3k27ac": "maximum of left, center, right 512-bp predicted means",
            "activity_threshold": args.activity_threshold,
        },
        "continuous_combined_activity": continuous,
        "continuous_pleiotropy": {
            **pleiotropy_correlations,
            "mae": float(np.mean(np.abs(observed_pleiotropy - predicted_pleiotropy))),
            "observed_mean": float(observed_pleiotropy.mean()),
            "predicted_mean": float(predicted_pleiotropy.mean()),
        },
        "hard_breadth_subsets": _subset_metrics(
            labels, observed_activity, predicted_activity
        ),
        "context_contrasts": {
            "related_pair_macro_pearson": float(
                np.mean([row["pearson"] for row in contrast_rows if row["is_related"]])
            ),
            "unrelated_pair_macro_pearson": float(
                np.mean([row["pearson"] for row in contrast_rows if not row["is_related"]])
            ),
            "related_pair_direction_accuracy": float(
                np.mean(
                    [row["direction_accuracy"] for row in contrast_rows if row["is_related"]]
                )
            ),
            "unrelated_pair_direction_accuracy": float(
                np.mean(
                    [row["direction_accuracy"] for row in contrast_rows if not row["is_related"]]
                )
            ),
        },
        "inputs": {
            "predictions": {
                "path": str(args.predictions),
                "sha256": sha256_file(args.predictions),
            },
            "catalog": {"path": str(args.catalog), "sha256": sha256_file(args.catalog)},
            "atac_bigwigs": {
                context: {"path": str(path), "sha256": sha256_file(path)}
                for context, path in bigwig_paths.items()
            },
        },
    }

    np.savez_compressed(
        args.output_directory / "continuous_activity.npz",
        ids=identifiers,
        chrom=chromosomes,
        summit=summits,
        contexts=np.asarray(CONTEXTS),
        labels=labels.astype(np.uint8),
        context_membership=membership.astype(np.uint8),
        h3k27ac_high=h3_high.astype(np.uint8),
        observed_atac_central_512=observed_atac,
        predicted_atac_central_512=predicted_atac.astype(np.float32),
        observed_h3k27ac_max_500=observed_h3.astype(np.float32),
        predicted_h3k27ac_max_512=predicted_h3.astype(np.float32),
        observed_combined_activity=observed_activity.astype(np.float32),
        predicted_combined_activity=predicted_activity.astype(np.float32),
        observed_pleiotropy=observed_pleiotropy.astype(np.float32),
        predicted_pleiotropy=predicted_pleiotropy.astype(np.float32),
    )
    _write_tsv(args.output_directory / "per_context.tsv", per_context_rows)
    _write_tsv(args.output_directory / "context_contrasts.tsv", contrast_rows)
    _write_tsv(args.output_directory / "assay_strata.tsv", stratum_rows)
    atomic_write_json(args.output_directory / "metrics.json", metrics)
    atomic_write_text(args.output_directory / "summary.md", _summary_markdown(metrics))
    print(
        json.dumps(
            {
                "event": "continuous_activity_evaluation_complete",
                "enhancers": len(identifiers),
                "output": str(args.output_directory / "metrics.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
