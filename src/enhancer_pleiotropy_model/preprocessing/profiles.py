#!/usr/bin/env python3
"""Build dense central epigenomic profiles aligned to a sequence dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import tempfile
from typing import TextIO

import numpy as np
import pyBigWig

from ..constants import CONTEXTS
from ..io import atomic_write_json, sha256_file


VALID_SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--bigwig-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--bin-size", default=16, type=int)
    parser.add_argument(
        "--assay", choices=("atac", "h3k27ac"), default="atac"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        help=(
            "Optional centered target size. By default, preserve target_start/target_end "
            "from the dataset."
        ),
    )
    parser.add_argument(
        "--bigwig-template",
        help=(
            "Filename template relative to --bigwig-directory. Defaults to the "
            "background-TMM track for --assay."
        ),
    )
    parser.add_argument(
        "--contexts", nargs="+", choices=CONTEXTS, default=tuple(CONTEXTS)
    )
    return parser.parse_args()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def read_profile_coordinates(
    dataset: Path,
    centered_target_size: int | None = None,
) -> tuple[
    dict[str, int],
    dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    int,
]:
    """Read row-ordered target coordinates, grouped by split and chromosome."""
    grouped_lists: dict[str, dict[str, tuple[list[int], list[int], list[int]]]] = {
        split: {} for split in VALID_SPLITS
    }
    counts = {split: 0 for split in VALID_SPLITS}
    target_sizes: set[int] = set()
    with open_text(dataset) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chrom", "target_start", "target_end", "split"}
        if centered_target_size is not None:
            required.update(("start", "end"))
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{dataset}: missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            split = row["split"]
            if split not in grouped_lists:
                raise ValueError(f"{dataset}: invalid split {split!r}")
            start, end = int(row["target_start"]), int(row["target_end"])
            if start < 0 or end <= start:
                raise ValueError(f"{dataset}: invalid target interval {start}-{end}")
            if centered_target_size is not None:
                center = (start + end) // 2
                start = center - centered_target_size // 2
                end = start + centered_target_size
                sequence_start, sequence_end = int(row["start"]), int(row["end"])
                if start < sequence_start or end > sequence_end:
                    raise ValueError(
                        f"{dataset}: centered target {start}-{end} is outside "
                        f"sequence interval {sequence_start}-{sequence_end}"
                    )
            target_sizes.add(end - start)
            index = counts[split]
            counts[split] += 1
            indices, starts, ends = grouped_lists[split].setdefault(
                row["chrom"], ([], [], [])
            )
            indices.append(index)
            starts.append(start)
            ends.append(end)
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Dataset has an empty split: {counts}")
    if len(target_sizes) != 1:
        raise ValueError(f"Targets must have one fixed size, found {sorted(target_sizes)}")

    grouped: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        split: {} for split in VALID_SPLITS
    }
    for split, by_chromosome in grouped_lists.items():
        for chromosome, (indices, starts, ends) in by_chromosome.items():
            grouped[split][chromosome] = (
                np.asarray(indices, dtype=np.int64),
                np.asarray(starts, dtype=np.int64),
                np.asarray(ends, dtype=np.int64),
            )
    return counts, grouped, target_sizes.pop()


def binned_means(
    bigwig: pyBigWig.pyBigWig,
    chromosome: str,
    starts: np.ndarray,
    ends: np.ndarray,
    bin_size: int,
) -> np.ndarray:
    """Return exact base-resolution means without one BigWig call per window."""
    chromosome_size = bigwig.chroms(chromosome)
    if chromosome_size is None:
        raise ValueError(f"BigWig is missing chromosome {chromosome}")
    if starts.min() < 0 or ends.max() > chromosome_size:
        raise ValueError(f"Targets on {chromosome} fall outside the BigWig")
    target_sizes = ends - starts
    if np.any(target_sizes != target_sizes[0]) or target_sizes[0] % bin_size:
        raise ValueError("Target sizes must be fixed and divisible by bin-size")

    range_start, range_end = int(starts.min()), int(ends.max())
    values = np.asarray(
        bigwig.values(chromosome, range_start, range_end, numpy=True),
        dtype=np.float64,
    )
    values = np.nan_to_num(values, nan=0.0, posinf=np.nan, neginf=np.nan)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{chromosome}: BigWig contains invalid nonnegative coverage")
    cumulative = np.empty(len(values) + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(values, out=cumulative[1:])

    relative_starts = starts - range_start
    bin_count = int(target_sizes[0] // bin_size)
    output = np.empty((len(starts), bin_count), dtype=np.float32)
    for bin_index in range(bin_count):
        left = relative_starts + bin_index * bin_size
        right = left + bin_size
        output[:, bin_index] = (cumulative[right] - cumulative[left]) / bin_size
    return output


def build_profiles(args: argparse.Namespace) -> dict[str, object]:
    contexts = tuple(args.contexts)
    if args.bin_size < 1:
        raise ValueError("bin-size must be positive")
    target_size_override = getattr(args, "target_size", None)
    assay = getattr(args, "assay", "atac")
    if target_size_override is not None and target_size_override < 1:
        raise ValueError("target-size must be positive")
    if target_size_override is not None and target_size_override % 2:
        raise ValueError("target-size must be even")
    if not contexts or len(set(contexts)) != len(contexts):
        raise ValueError("contexts must be non-empty and unique")

    counts, grouped, target_size = read_profile_coordinates(
        args.dataset, target_size_override
    )
    if target_size % args.bin_size:
        raise ValueError("Target size must be divisible by bin-size")
    bin_count = target_size // args.bin_size
    bigwig_template = getattr(args, "bigwig_template", None) or (
        f"{{context}}.{assay}.mean.background_tmm.bw"
    )
    bigwig_paths = {
        context: args.bigwig_directory
        / bigwig_template.format(context=context)
        for context in contexts
    }
    missing = [str(path) for path in bigwig_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing BigWigs: " + ", ".join(missing))

    args.output_directory.mkdir(parents=True, exist_ok=True)
    final_paths = {
        split: args.output_directory / f"{split}_profiles.npy"
        for split in VALID_SPLITS
    }
    temporary_paths: dict[str, Path] = {}
    arrays: dict[str, np.memmap] = {}
    statistics = {
        split: {
            context: {"sum": 0.0, "sum_squares": 0.0, "maximum": 0.0, "count": 0}
            for context in contexts
        }
        for split in VALID_SPLITS
    }
    try:
        for split, final_path in final_paths.items():
            with tempfile.NamedTemporaryFile(
                dir=args.output_directory,
                prefix=f".{final_path.name}.",
                suffix=".npy",
                delete=False,
            ) as temporary:
                temporary_paths[split] = Path(temporary.name)
            arrays[split] = np.lib.format.open_memmap(
                temporary_paths[split],
                mode="w+",
                dtype=np.float32,
                shape=(counts[split], bin_count, len(contexts)),
            )

        for context_index, context in enumerate(contexts):
            with pyBigWig.open(str(bigwig_paths[context])) as bigwig:
                for split in VALID_SPLITS:
                    for chromosome, (indices, starts, ends) in grouped[split].items():
                        values = binned_means(
                            bigwig, chromosome, starts, ends, args.bin_size
                        )
                        arrays[split][indices, :, context_index] = values
                        summary = statistics[split][context]
                        values64 = values.astype(np.float64, copy=False)
                        summary["sum"] += float(values64.sum())
                        summary["sum_squares"] += float(np.square(values64).sum())
                        summary["maximum"] = max(
                            float(summary["maximum"]), float(values64.max(initial=0.0))
                        )
                        summary["count"] += int(values64.size)

        for split in VALID_SPLITS:
            arrays[split].flush()
            del arrays[split]
            temporary_paths[split].replace(final_paths[split])
        temporary_paths.clear()
    finally:
        arrays.clear()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    reported_statistics: dict[str, dict[str, dict[str, float | int]]] = {}
    for split in VALID_SPLITS:
        reported_statistics[split] = {}
        for context in contexts:
            summary = statistics[split][context]
            count = int(summary["count"])
            mean = float(summary["sum"]) / count
            variance = max(0.0, float(summary["sum_squares"]) / count - mean**2)
            reported_statistics[split][context] = {
                "count": count,
                "mean": mean,
                "standard_deviation": variance**0.5,
                "maximum": float(summary["maximum"]),
            }

    metadata: dict[str, object] = {
        "method": f"dense_central_{assay}_profiles_v1",
        "assay": assay,
        "dataset": {"path": str(args.dataset), "sha256": sha256_file(args.dataset)},
        "contexts": list(contexts),
        "context_order": list(contexts),
        "target_window_size_bp": target_size,
        "bin_size_bp": args.bin_size,
        "bins_per_target": bin_count,
        "target_definition": (
            f"mean raw nonnegative background-TMM normalized {assay.upper()} coverage in "
            f"each {args.bin_size}-bp bin"
        ),
        "split_counts": counts,
        "statistics": reported_statistics,
        "bigwigs": {
            context: {"path": str(path), "sha256": sha256_file(path)}
            for context, path in bigwig_paths.items()
        },
        "outputs": {
            split: {"path": str(path), "sha256": sha256_file(path)}
            for split, path in final_paths.items()
        },
    }
    atomic_write_json(args.output_directory / "profiles.metadata.json", metadata)
    return metadata


def main() -> None:
    metadata = build_profiles(parse_args())
    print(
        json.dumps(
            {
                "event": f"dense_{metadata['assay']}_profiles_complete",
                "split_counts": metadata["split_counts"],
                "bins_per_target": metadata["bins_per_target"],
                "contexts": metadata["contexts"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
