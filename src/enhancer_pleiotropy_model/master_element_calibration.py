"""Calibrate master-element assay summaries against training genomic background."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .constants import CONTEXTS
from .continuous_activity import participation_ratio
from .io import atomic_write_json, atomic_write_text, sha256_file
from .master_element_states import (
    DEFAULT_BACKGROUND_CHROMOSOMES,
    background_training_indices,
    noncompensatory_joint_activity,
    state_metrics,
    summarize_background_profiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-states", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--atac-train-profiles", required=True, type=Path)
    parser.add_argument("--h3k27ac-train-profiles", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--profile-chunk-size", default=5000, type=int)
    parser.add_argument(
        "--background-chromosomes",
        nargs="+",
        default=DEFAULT_BACKGROUND_CHROMOSOMES,
    )
    return parser.parse_args()


def fit_empirical_background(values: np.ndarray) -> np.ndarray:
    """Return independently sorted background summaries for every context."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(CONTEXTS):
        raise ValueError("Background values must be [windows, contexts]")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Background values must be finite and nonnegative")
    return np.sort(values.astype(np.float64), axis=0)


def empirical_background_percentiles(
    values: np.ndarray,
    sorted_background: np.ndarray,
) -> np.ndarray:
    """Map values to per-context mid-rank plotting positions in training background."""
    values = np.asarray(values, dtype=np.float64)
    sorted_background = np.asarray(sorted_background, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CONTEXTS):
        raise ValueError("Values must be [elements, contexts]")
    if sorted_background.ndim != 2 or sorted_background.shape[1] != len(CONTEXTS):
        raise ValueError("Sorted background must be [windows, contexts]")
    if len(sorted_background) < 1:
        raise ValueError("At least one background window is required")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Values must be finite and nonnegative")
    if not np.all(np.isfinite(sorted_background)) or np.any(sorted_background < 0):
        raise ValueError("Sorted background must be finite and nonnegative")
    if np.any(np.diff(sorted_background, axis=0) < 0):
        raise ValueError("Background columns must be sorted")

    count = len(sorted_background)
    percentiles = np.empty_like(values)
    for context_index in range(len(CONTEXTS)):
        reference = sorted_background[:, context_index]
        left = np.searchsorted(reference, values[:, context_index], side="left")
        right = np.searchsorted(reference, values[:, context_index], side="right")
        midrank = (left + right) / 2.0
        percentiles[:, context_index] = (midrank + 0.5) / (count + 1.0)
    return percentiles


def positive_percentile_excess(percentiles: np.ndarray) -> np.ndarray:
    """Map the upper half of a background percentile distribution onto [0, 1]."""
    percentiles = np.asarray(percentiles, dtype=np.float64)
    if not np.all(np.isfinite(percentiles)):
        raise ValueError("Percentiles must be finite")
    if np.any(percentiles < 0) or np.any(percentiles > 1):
        raise ValueError("Percentiles must lie in [0, 1]")
    return np.maximum(0.0, 2.0 * percentiles - 1.0)


def calibrate(
    values: np.ndarray,
    sorted_background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    percentiles = empirical_background_percentiles(values, sorted_background)
    return percentiles, positive_percentile_excess(percentiles)


def _distribution(values: np.ndarray, above_maximum: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "zero_fraction": float(np.mean(values == 0)),
        "above_background_max_fraction": float(np.mean(above_maximum)),
    }


def _limiter(atac: np.ndarray, h3k27ac: np.ndarray) -> dict[str, float | int]:
    both_positive = (atac > 0) & (h3k27ac > 0)
    result: dict[str, float | int] = {
        "atac_lower_fraction_all": float(np.mean(atac < h3k27ac)),
        "h3k27ac_lower_fraction_all": float(np.mean(h3k27ac < atac)),
        "equal_fraction_all": float(np.mean(atac == h3k27ac)),
        "both_positive_fraction": float(np.mean(both_positive)),
        "both_positive_count": int(both_positive.sum()),
    }
    if both_positive.any():
        result.update(
            {
                "atac_lower_fraction_both_positive": float(
                    np.mean(atac[both_positive] < h3k27ac[both_positive])
                ),
                "h3k27ac_lower_fraction_both_positive": float(
                    np.mean(h3k27ac[both_positive] < atac[both_positive])
                ),
            }
        )
    return result


def _performance_by_split(
    split: np.ndarray,
    has_prediction: np.ndarray,
    observed_atac: np.ndarray,
    observed_h3: np.ndarray,
    predicted_atac: np.ndarray,
    predicted_h3: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("all", "train", "validation", "test"):
        selected = has_prediction & (
            np.ones(len(split), dtype=bool) if name == "all" else split == name
        )
        if not selected.any():
            continue
        observed_joint = noncompensatory_joint_activity(
            observed_atac[selected], observed_h3[selected]
        )
        predicted_joint = noncompensatory_joint_activity(
            predicted_atac[selected], predicted_h3[selected]
        )
        result[name] = {
            "elements": int(selected.sum()),
            "atac": state_metrics(observed_atac[selected], predicted_atac[selected]),
            "h3k27ac": state_metrics(observed_h3[selected], predicted_h3[selected]),
            "joint_min": state_metrics(observed_joint, predicted_joint),
        }
    return result


def _write_element_table(
    path: Path,
    states: dict[str, np.ndarray],
    observed_atac: np.ndarray,
    observed_h3: np.ndarray,
    predicted_atac: np.ndarray,
    predicted_h3: np.ndarray,
) -> None:
    has_prediction = states["has_prediction"].astype(bool)
    observed_joint = noncompensatory_joint_activity(observed_atac, observed_h3)
    predicted_pleiotropy = {
        "atac": np.full(len(has_prediction), np.nan),
        "h3k27ac": np.full(len(has_prediction), np.nan),
        "joint_min": np.full(len(has_prediction), np.nan),
    }
    predicted_pleiotropy["atac"][has_prediction] = participation_ratio(
        predicted_atac[has_prediction]
    )
    predicted_pleiotropy["h3k27ac"][has_prediction] = participation_ratio(
        predicted_h3[has_prediction]
    )
    predicted_pleiotropy["joint_min"][has_prediction] = participation_ratio(
        noncompensatory_joint_activity(
            predicted_atac[has_prediction], predicted_h3[has_prediction]
        )
    )
    table = pd.DataFrame(
        {
            "master_dhs_id": states["ids"].astype(str),
            "chrom": states["chrom"].astype(str),
            "start": states["start"],
            "end": states["end"],
            "split": states["split"].astype(str),
            "active_context_count": states["hard_activity"].sum(axis=1),
            "has_model_prediction": has_prediction,
            "observed_atac_pleiotropy": participation_ratio(observed_atac),
            "observed_h3k27ac_pleiotropy": participation_ratio(observed_h3),
            "observed_joint_min_pleiotropy": participation_ratio(observed_joint),
            "predicted_atac_pleiotropy": predicted_pleiotropy["atac"],
            "predicted_h3k27ac_pleiotropy": predicted_pleiotropy["h3k27ac"],
            "predicted_joint_min_pleiotropy": predicted_pleiotropy["joint_min"],
        }
    )
    for index, context in enumerate(CONTEXTS):
        prefix = f"{context}__"
        table[prefix + "observed_atac_percentile_excess"] = observed_atac[:, index]
        table[prefix + "observed_h3k27ac_percentile_excess"] = observed_h3[:, index]
        table[prefix + "predicted_atac_percentile_excess"] = predicted_atac[:, index]
        table[prefix + "predicted_h3k27ac_percentile_excess"] = predicted_h3[:, index]
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tsv.gz", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        table.to_csv(temporary, sep="\t", index=False, compression="gzip", na_rep="nan")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.profile_chunk_size < 1:
        raise ValueError("profile-chunk-size must be positive")
    background_chromosomes = tuple(dict.fromkeys(args.background_chromosomes))
    if not background_chromosomes:
        raise ValueError("At least one background chromosome is required")
    args.output_directory.mkdir(parents=True, exist_ok=True)

    with np.load(args.master_states, allow_pickle=False) as loaded:
        states = {name: loaded[name] for name in loaded.files}
    if tuple(states["contexts"].astype(str)) != CONTEXTS:
        raise ValueError("Master-state context order is incorrect")
    indices = background_training_indices(args.windows, set(background_chromosomes))
    background_atac, background_h3, atac_shape, h3_shape = summarize_background_profiles(
        args.atac_train_profiles,
        args.h3k27ac_train_profiles,
        indices,
        args.profile_chunk_size,
    )
    reference_atac = fit_empirical_background(background_atac)
    reference_h3 = fit_empirical_background(background_h3)

    observed_atac_raw = states["observed_atac_mean_512"].astype(np.float64)
    observed_h3_raw = states["observed_h3k27ac_segment_means_512"].max(axis=2).astype(np.float64)
    has_prediction = states["has_prediction"].astype(bool)
    observed_atac_percentile, observed_atac = calibrate(observed_atac_raw, reference_atac)
    observed_h3_percentile, observed_h3 = calibrate(observed_h3_raw, reference_h3)
    predicted_atac_percentile = np.full_like(observed_atac, np.nan)
    predicted_h3_percentile = np.full_like(observed_h3, np.nan)
    predicted_atac = np.full_like(observed_atac, np.nan)
    predicted_h3 = np.full_like(observed_h3, np.nan)
    predicted_atac_percentile[has_prediction], predicted_atac[has_prediction] = calibrate(
        states["predicted_atac_mean_512"][has_prediction], reference_atac
    )
    predicted_h3_percentile[has_prediction], predicted_h3[has_prediction] = calibrate(
        states["predicted_h3k27ac_max_mean_512"][has_prediction], reference_h3
    )
    observed_joint = noncompensatory_joint_activity(observed_atac, observed_h3)
    split = states["split"].astype(str)
    labels = states["hard_activity"].astype(bool)

    robust_atac = states["observed_atac_background_excess"].astype(np.float64)
    robust_h3 = states["observed_h3k27ac_background_excess"].astype(np.float64)
    metrics: dict[str, object] = {
        "method": "training_background_empirical_percentile_excess_v1",
        "definition": {
            "percentile": "per-context mid-rank plotting position in training genomic background",
            "positive_excess": "max(0, 2 * percentile - 1)",
            "joint": "min(atac_percentile_excess, h3k27ac_percentile_excess)",
            "pleiotropy": "participation ratio across eight retained contexts",
        },
        "selection": {
            "elements": int(len(split)),
            "predicted_elements": int(has_prediction.sum()),
            "background_windows": int(len(indices)),
            "background_chromosomes": list(background_chromosomes),
        },
        "distribution_comparability": {
            "empirical_percentile_excess": {
                "all_element_contexts": {
                    "atac": _distribution(
                        observed_atac, observed_atac_raw > reference_atac[-1]
                    ),
                    "h3k27ac": _distribution(
                        observed_h3, observed_h3_raw > reference_h3[-1]
                    ),
                    "limiter": _limiter(observed_atac, observed_h3),
                },
                "catalog_active_element_contexts": {
                    "count": int(labels.sum()),
                    "atac": _distribution(
                        observed_atac[labels],
                        (observed_atac_raw > reference_atac[-1])[labels],
                    ),
                    "h3k27ac": _distribution(
                        observed_h3[labels],
                        (observed_h3_raw > reference_h3[-1])[labels],
                    ),
                },
            },
            "previous_robust_log1p_excess": {
                "all_element_contexts": {
                    "atac": _distribution(robust_atac, np.zeros_like(robust_atac, bool)),
                    "h3k27ac": _distribution(robust_h3, np.zeros_like(robust_h3, bool)),
                    "limiter": _limiter(robust_atac, robust_h3),
                }
            },
        },
        "frozen_model_performance": _performance_by_split(
            split,
            has_prediction,
            observed_atac,
            observed_h3,
            predicted_atac,
            predicted_h3,
        ),
        "inputs": {
            "master_states": {
                "path": str(args.master_states),
                "sha256": sha256_file(args.master_states),
            },
            "windows": {"path": str(args.windows), "sha256": sha256_file(args.windows)},
            "profiles": {
                "atac": {"path": str(args.atac_train_profiles), "shape": atac_shape},
                "h3k27ac": {"path": str(args.h3k27ac_train_profiles), "shape": h3_shape},
            },
        },
    }

    np.savez_compressed(
        args.output_directory / "background_reference.npz",
        contexts=np.asarray(CONTEXTS),
        atac_sorted=reference_atac.astype(np.float32),
        h3k27ac_sorted=reference_h3.astype(np.float32),
    )
    np.savez_compressed(
        args.output_directory / "calibrated_states.npz",
        ids=states["ids"],
        contexts=np.asarray(CONTEXTS),
        split=split.astype(np.str_),
        hard_activity=labels.astype(np.uint8),
        has_prediction=has_prediction.astype(np.uint8),
        observed_atac_percentile=observed_atac_percentile.astype(np.float32),
        observed_h3k27ac_percentile=observed_h3_percentile.astype(np.float32),
        predicted_atac_percentile=predicted_atac_percentile.astype(np.float32),
        predicted_h3k27ac_percentile=predicted_h3_percentile.astype(np.float32),
        observed_atac_percentile_excess=observed_atac.astype(np.float32),
        observed_h3k27ac_percentile_excess=observed_h3.astype(np.float32),
        predicted_atac_percentile_excess=predicted_atac.astype(np.float32),
        predicted_h3k27ac_percentile_excess=predicted_h3.astype(np.float32),
        observed_joint_min=observed_joint.astype(np.float32),
    )
    _write_element_table(
        args.output_directory / "element_summary.tsv.gz",
        states,
        observed_atac,
        observed_h3,
        predicted_atac,
        predicted_h3,
    )
    atomic_write_json(args.output_directory / "metrics.json", metrics)
    test = metrics["frozen_model_performance"].get("test")
    lines = [
        "# Empirically calibrated master-element states",
        "",
        f"- Master elements: {len(split):,}",
        f"- Elements with cached predictions: {has_prediction.sum():,}",
        f"- Training genomic-background windows: {len(indices):,}",
        "- Calibration: per-assay/context empirical background percentile.",
        "- Positive excess: `max(0, 2 * percentile - 1)`.",
        "- Joint activity: elementwise minimum after percentile calibration.",
        "",
    ]
    if test is not None:
        lines.extend(
            [
                "## Held-out test performance",
                "",
                f"- ATAC macro Pearson: {test['atac']['macro_pearson']:.4f}",
                f"- ATAC tissue-pattern Pearson: {test['atac']['tissue_pattern_mean_pearson']:.4f}",
                f"- ATAC pleiotropy Pearson: {test['atac']['pleiotropy']['pearson']:.4f}",
                f"- H3K27ac macro Pearson: {test['h3k27ac']['macro_pearson']:.4f}",
                f"- H3K27ac tissue-pattern Pearson: {test['h3k27ac']['tissue_pattern_mean_pearson']:.4f}",
                f"- H3K27ac pleiotropy Pearson: {test['h3k27ac']['pleiotropy']['pearson']:.4f}",
                f"- Joint-min macro Pearson: {test['joint_min']['macro_pearson']:.4f}",
                f"- Joint-min tissue-pattern Pearson: {test['joint_min']['tissue_pattern_mean_pearson']:.4f}",
                f"- Joint-min pleiotropy Pearson: {test['joint_min']['pleiotropy']['pearson']:.4f}",
                "",
            ]
        )
    atomic_write_text(args.output_directory / "summary.md", "\n".join(lines))
    print(
        json.dumps(
            {
                "event": "master_element_calibration_complete",
                "elements": len(split),
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
