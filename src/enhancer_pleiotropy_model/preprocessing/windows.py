#!/usr/bin/env python3
"""Build blocked sliding DNA windows with multi-context epigenomic targets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
from typing import TextIO

import numpy as np

from ..constants import CONTEXTS
from ..io import (
    MutableIntervalIndex,
    atomic_write_text,
    read_bed_intervals,
    read_fasta,
    sha256_file,
)


PEAK_SOURCE = "atac_peak_overlap"
H3K27AC_PEAK_SOURCE = "h3k27ac_peak_overlap"
JOINT_PEAK_SOURCE = "atac_h3k27ac_peak_overlap"
BACKGROUND_SOURCE = "genomic_background"
REGULATORY_SOURCES = frozenset(
    (PEAK_SOURCE, H3K27AC_PEAK_SOURCE, JOINT_PEAK_SOURCE)
)
SIGNAL_PREFIX = "atac_signal_"
SUPPORTED_SIGNAL_ASSAYS = ("atac", "h3k27ac")
DNA_ALPHABET = frozenset("ACGT")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class CandidateWindow:
    chrom: str
    input_start: int
    input_end: int
    target_start: int
    target_end: int
    split: str
    block_id: str
    source: str
    sampling: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-fasta", required=True, type=Path)
    parser.add_argument("--blacklist-bed", required=True, type=Path)
    parser.add_argument("--master-dhs-bed", required=True, type=Path)
    parser.add_argument("--master-dhs-summits-bed", type=Path)
    parser.add_argument("--h3k27ac-peaks-bed", type=Path)
    parser.add_argument("--bigwig-directory", required=True, type=Path)
    parser.add_argument(
        "--signal-assays",
        nargs="+",
        choices=SUPPORTED_SIGNAL_ASSAYS,
        default=("atac",),
        help="BigWig assays to emit; the default preserves ATAC-only behavior.",
    )
    parser.add_argument(
        "--omit-signal-summaries",
        action="store_true",
        help=(
            "Do not redundantly store per-window BigWig means. Dense profile "
            "targets should be generated separately with the profiles command."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument(
        "--context-flank-size",
        type=int,
        default=0,
        help="Genomic input context added on each side of the central target window.",
    )
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument(
        "--validation-stride",
        type=int,
        help="Validation stride; defaults to the training stride.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("blocked", "chromosome", "regions"),
        default="blocked",
        help=(
            "Use hashed blocks within supervised training chromosomes or the "
            "supervised chromosome validation split."
        ),
    )
    parser.add_argument("--block-size", type=int, default=1_000_000)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--background-to-peak-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--train-chromosomes", nargs="*", default=[])
    parser.add_argument("--validation-chromosomes", nargs="*", default=[])
    parser.add_argument("--test-chromosomes", nargs="*", default=[])
    for split in SPLITS:
        parser.add_argument(
            f"--{split}-regions",
            nargs="*",
            default=[],
            metavar="CHROM:START-END",
        )
    return parser.parse_args()


def parse_region(value: str) -> tuple[str, int, int]:
    try:
        chrom, coordinates = value.rsplit(":", 1)
        start_text, end_text = coordinates.split("-", 1)
        start, end = int(start_text), int(end_text)
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"Invalid region {value!r}; expected CHROM:START-END"
        ) from error
    if not chrom or start < 0 or end <= start:
        raise ValueError(f"Invalid region {value!r}")
    return chrom, start, end


def build_split_regions(
    genome: dict[str, str],
    chromosome_splits: dict[str, set[str]],
    region_values: dict[str, list[str]],
) -> dict[str, list[tuple[int, int, str]]]:
    """Resolve full chromosomes and intervals into disjoint split regions."""
    regions: dict[str, list[tuple[int, int, str]]] = {}
    for split in SPLITS:
        for chrom in chromosome_splits[split]:
            if chrom not in genome:
                raise ValueError(f"Reference FASTA lacks chromosome {chrom}")
            regions.setdefault(chrom, []).append((0, len(genome[chrom]), split))
        for value in region_values[split]:
            chrom, start, end = parse_region(value)
            if chrom not in genome:
                raise ValueError(f"Reference FASTA lacks chromosome {chrom}")
            if end > len(genome[chrom]):
                raise ValueError(
                    f"Region {value!r} exceeds chromosome length {len(genome[chrom])}"
                )
            regions.setdefault(chrom, []).append((start, end, split))

    split_counts = {split: 0 for split in SPLITS}
    for chrom, chrom_regions in regions.items():
        previous_end = -1
        for start, end, split in sorted(chrom_regions):
            if start < previous_end:
                raise ValueError(f"Split regions overlap on {chrom}")
            previous_end = end
            split_counts[split] += 1
    if any(count == 0 for count in split_counts.values()):
        raise ValueError(f"Train, validation, and test regions are required: {split_counts}")
    return regions


def containing_split(
    regions: dict[str, list[tuple[int, int, str]]],
    chrom: str,
    start: int,
    end: int,
) -> str | None:
    matches = [
        split
        for region_start, region_end, split in regions.get(chrom, ())
        if region_start <= start and end <= region_end
    ]
    if len(matches) > 1:
        raise ValueError(f"Input interval {chrom}:{start}-{end} belongs to multiple splits")
    return matches[0] if matches else None


def block_is_validation(
    chrom: str, block_index: int, seed: int, validation_fraction: float
) -> bool:
    digest = hashlib.sha256(f"{seed}\t{chrom}\t{block_index}".encode()).digest()
    value = int.from_bytes(digest[:8], byteorder="big") / 2**64
    return value < validation_fraction


def output_handle(path: Path, temporary: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(temporary, "wt", encoding="utf-8", newline="")
    return temporary.open("w", encoding="utf-8", newline="")


def select_balanced_windows(
    candidates: list[CandidateWindow], background_to_peak_ratio: float, seed: int
) -> tuple[list[CandidateWindow], dict[str, dict[str, int]]]:
    selected: list[CandidateWindow] = []
    counts: dict[str, dict[str, int]] = {}
    for split_index, split in enumerate(SPLITS):
        peaks = [
            window
            for window in candidates
            if window.split == split and window.source in REGULATORY_SOURCES
        ]
        backgrounds = [
            window
            for window in candidates
            if window.split == split and window.source == BACKGROUND_SOURCE
        ]
        if not peaks or not backgrounds:
            raise ValueError(
                f"Both regulatory-overlap and background windows are required in {split}"
            )
        requested_background = math.ceil(len(peaks) * background_to_peak_ratio)
        retained_background = min(len(backgrounds), requested_background)
        rng = random.Random(seed + split_index)
        selected_background = rng.sample(backgrounds, retained_background)
        selected.extend(peaks)
        selected.extend(selected_background)
        counts[split] = {
            "candidate_peak_overlap": len(peaks),
            "candidate_background": len(backgrounds),
            "selected_peak_overlap": len(peaks),
            "selected_background": retained_background,
            "candidate_by_source": {
                source: sum(window.source == source for window in candidates if window.split == split)
                for source in sorted(REGULATORY_SOURCES | {BACKGROUND_SOURCE})
            },
        }
    return selected, counts


def mean_signal_by_window(
    bigwig: pyBigWig.pyBigWig,
    chrom: str,
    chromosome_length: int,
    stride: int,
    window_size: int,
    starts: list[int],
) -> np.ndarray:
    bin_count = chromosome_length // stride
    if not bin_count:
        raise ValueError(f"Chromosome {chrom} is shorter than stride {stride}")
    bin_end = bin_count * stride
    bin_sums = bigwig.stats(
        chrom,
        0,
        bin_end,
        nBins=bin_count,
        type="sum",
        exact=True,
    )
    sums = np.fromiter(
        (0.0 if value is None else value for value in bin_sums),
        dtype=np.float64,
        count=bin_count,
    )
    if not np.all(np.isfinite(sums)) or np.any(sums < -1e-6):
        raise ValueError(f"BigWig contains invalid signal values on {chrom}")
    sums = np.maximum(sums, 0.0)
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(sums)))
    bins_per_window = window_size // stride
    start_bins = np.asarray(starts, dtype=np.int64) // stride
    window_sums = cumulative[start_bins + bins_per_window] - cumulative[start_bins]
    return window_sums / window_size


def mean_signal_by_arbitrary_window(
    bigwig: pyBigWig.pyBigWig,
    chrom: str,
    window_size: int,
    starts: list[int],
) -> np.ndarray:
    """Integrate sparse BigWig intervals exactly at arbitrary window starts."""
    if not starts:
        return np.empty(0, dtype=np.float64)
    positions = sorted({*starts, *(start + window_size for start in starts)})
    intervals = bigwig.intervals(chrom) or ()
    integrals: dict[int, float] = {}
    interval_index = 0
    completed_sum = 0.0
    for position in positions:
        while interval_index < len(intervals) and intervals[interval_index][1] <= position:
            start, end, value = intervals[interval_index]
            if not math.isfinite(value) or value < -1e-6:
                raise ValueError(f"BigWig contains invalid signal values on {chrom}")
            completed_sum += (end - start) * max(0.0, value)
            interval_index += 1
        integral = completed_sum
        if interval_index < len(intervals):
            start, end, value = intervals[interval_index]
            if not math.isfinite(value) or value < -1e-6:
                raise ValueError(f"BigWig contains invalid signal values on {chrom}")
            if start < position < end:
                integral += (position - start) * max(0.0, value)
        integrals[position] = integral
    return np.asarray(
        [
            (integrals[start + window_size] - integrals[start]) / window_size
            for start in starts
        ],
        dtype=np.float64,
    )


def mean_signal_for_starts(
    bigwig: pyBigWig.pyBigWig,
    chrom: str,
    chromosome_length: int,
    aligned_stride: int,
    window_size: int,
    starts: list[int],
) -> np.ndarray:
    """Use fast binned queries for grid starts and exact integration otherwise."""
    result = np.empty(len(starts), dtype=np.float64)
    aligned_indices = [index for index, start in enumerate(starts) if start % aligned_stride == 0]
    arbitrary_indices = [index for index, start in enumerate(starts) if start % aligned_stride]
    if aligned_indices:
        aligned_values = mean_signal_by_window(
            bigwig,
            chrom,
            chromosome_length,
            aligned_stride,
            window_size,
            [starts[index] for index in aligned_indices],
        )
        result[aligned_indices] = aligned_values
    if arbitrary_indices:
        arbitrary_values = mean_signal_by_arbitrary_window(
            bigwig,
            chrom,
            window_size,
            [starts[index] for index in arbitrary_indices],
        )
        result[arbitrary_indices] = arbitrary_values
    return result


def build_windows(args: argparse.Namespace) -> dict[str, object]:
    try:
        import pyBigWig
    except ImportError as error:
        raise RuntimeError(
            "Building signal windows requires pyBigWig; training an existing dataset does not"
        ) from error

    requested_signal_assays = tuple(getattr(args, "signal_assays", ("atac",)))
    context_flank_size = getattr(args, "context_flank_size", 0)
    validation_stride = getattr(args, "validation_stride", None) or args.stride
    split_strategy = getattr(args, "split_strategy", "blocked")
    h3k27ac_peaks_bed = getattr(args, "h3k27ac_peaks_bed", None)
    master_dhs_summits_bed = getattr(args, "master_dhs_summits_bed", None)
    if not requested_signal_assays or len(set(requested_signal_assays)) != len(
        requested_signal_assays
    ):
        raise ValueError("signal-assays must be a non-empty list without duplicates")
    signal_assays = (
        () if getattr(args, "omit_signal_summaries", False) else requested_signal_assays
    )
    positive_integers = {
        "window-size": args.window_size,
        "stride": args.stride,
        "validation-stride": validation_stride,
        "block-size": args.block_size,
    }
    invalid = [name for name, value in positive_integers.items() if value < 1]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if context_flank_size < 0:
        raise ValueError("context-flank-size must be non-negative")
    if args.window_size % args.stride or args.window_size % validation_stride:
        raise ValueError("window-size must be an integer multiple of both strides")
    input_window_size = args.window_size + 2 * context_flank_size
    if args.block_size < input_window_size:
        raise ValueError("block-size must be at least the complete input window size")
    if split_strategy == "blocked" and not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be strictly between zero and one")
    if args.background_to_peak_ratio <= 0:
        raise ValueError("background-to-peak-ratio must be positive")

    chromosome_splits = {
        "train": set(args.train_chromosomes),
        "validation": set(args.validation_chromosomes),
        "test": set(args.test_chromosomes),
    }
    region_values = {
        split: list(getattr(args, f"{split}_regions", ())) for split in SPLITS
    }
    if split_strategy != "regions" and any(region_values.values()):
        raise ValueError("Explicit regions require --split-strategy regions")
    if split_strategy != "regions" and any(
        not values for key, values in chromosome_splits.items() if key != "test"
    ):
        raise ValueError("Training and validation chromosome sets must be non-empty")
    if (
        chromosome_splits["train"] & chromosome_splits["validation"]
        or chromosome_splits["train"] & chromosome_splits["test"]
        or chromosome_splits["validation"] & chromosome_splits["test"]
    ):
        raise ValueError("Chromosome splits must be disjoint")
    genome, chromosome_order = read_fasta(args.reference_fasta)
    split_regions = (
        build_split_regions(genome, chromosome_splits, region_values)
        if split_strategy == "regions"
        else {}
    )
    training_chromosomes = chromosome_splits["train"]
    validation_chromosomes = (
        chromosome_splits["validation"] if split_strategy == "chromosome" else set()
    )
    allowed_chromosomes = (
        set(split_regions)
        if split_strategy == "regions"
        else training_chromosomes
        | validation_chromosomes
        | chromosome_splits["test"]
    )
    missing_reference = allowed_chromosomes - set(genome)
    if missing_reference:
        raise ValueError(f"Reference FASTA lacks chromosomes: {sorted(missing_reference)}")

    blacklist = {
        chrom: MutableIntervalIndex(intervals)
        for chrom, intervals in read_bed_intervals(args.blacklist_bed).items()
    }
    master_dhs = {
        chrom: MutableIntervalIndex(intervals)
        for chrom, intervals in read_bed_intervals(args.master_dhs_bed).items()
    }
    h3k27ac_peaks = (
        {
            chrom: MutableIntervalIndex(intervals)
            for chrom, intervals in read_bed_intervals(h3k27ac_peaks_bed).items()
        }
        if h3k27ac_peaks_bed is not None
        else {}
    )
    master_dhs_summits = (
        read_bed_intervals(master_dhs_summits_bed)
        if master_dhs_summits_bed is not None
        else {}
    )
    bigwig_paths = {
        assay: {
            context: args.bigwig_directory
            / f"{context}.{assay}.mean.background_tmm.bw"
            for context in CONTEXTS
        }
        for assay in signal_assays
    }
    missing_bigwigs = [
        str(path)
        for assay_paths in bigwig_paths.values()
        for path in assay_paths.values()
        if not path.is_file()
    ]
    if missing_bigwigs:
        raise FileNotFoundError(f"Missing signal BigWigs: {missing_bigwigs}")

    handles: dict[str, dict[str, pyBigWig.pyBigWig]] = {
        assay: {} for assay in signal_assays
    }
    try:
        for assay, assay_paths in bigwig_paths.items():
            for context, path in assay_paths.items():
                handle = pyBigWig.open(str(path))
                if handle is None or not handle.isBigWig():
                    raise ValueError(f"{path}: not a BigWig")
                chromosome_lengths = handle.chroms()
                for chrom in allowed_chromosomes:
                    if chromosome_lengths.get(chrom) != len(genome[chrom]):
                        raise ValueError(
                            f"{path}: chromosome length mismatch for {chrom}: "
                            f"{chromosome_lengths.get(chrom)} != {len(genome[chrom])}"
                        )
                handles[assay][context] = handle

        candidate_by_coordinate: dict[tuple[str, int, int], CandidateWindow] = {}
        skipped = {
            "crosses_chromosome_boundary": 0,
            "crosses_block_boundary": 0,
            "crosses_split_boundary_or_unassigned": 0,
            "blacklist_overlap": 0,
            "ambiguous_sequence": 0,
            "duplicate_coordinate": 0,
        }

        def add_candidate(chrom: str, target_start: int, sampling: str) -> None:
            chromosome_sequence = genome[chrom]
            target_end = target_start + args.window_size
            input_start = target_start - context_flank_size
            input_end = target_end + context_flank_size
            if input_start < 0 or input_end > len(chromosome_sequence):
                skipped["crosses_chromosome_boundary"] += 1
                return
            region_split = (
                containing_split(split_regions, chrom, input_start, input_end)
                if split_strategy == "regions"
                else None
            )
            if split_strategy == "regions" and region_split is None:
                skipped["crosses_split_boundary_or_unassigned"] += 1
                return
            block_index = target_start // args.block_size
            block_start = block_index * args.block_size
            block_end = (block_index + 1) * args.block_size
            if (
                split_strategy == "blocked"
                and (input_start < block_start or input_end > block_end)
            ):
                skipped["crosses_block_boundary"] += 1
                return
            coordinate = (chrom, target_start, target_end)
            existing = candidate_by_coordinate.get(coordinate)
            if existing is not None:
                skipped["duplicate_coordinate"] += 1
                if sampling not in existing.sampling.split("+"):
                    candidate_by_coordinate[coordinate] = replace(
                        existing, sampling=f"{existing.sampling}+{sampling}"
                    )
                return
            blacklist_index = blacklist.get(chrom)
            if (
                blacklist_index is not None
                and blacklist_index.overlaps(input_start, input_end)
            ):
                skipped["blacklist_overlap"] += 1
                return
            sequence = chromosome_sequence[input_start:input_end]
            if set(sequence) - DNA_ALPHABET:
                skipped["ambiguous_sequence"] += 1
                return
            if split_strategy == "regions":
                split = region_split
            elif chrom in chromosome_splits["test"]:
                split = "test"
            elif split_strategy == "chromosome":
                split = "validation" if chrom in validation_chromosomes else "train"
            else:
                split = (
                    "validation"
                    if block_is_validation(
                        chrom, block_index, args.seed, args.validation_fraction
                    )
                    else "train"
                )
            assert split is not None
            atac_index = master_dhs.get(chrom)
            h3k27ac_index = h3k27ac_peaks.get(chrom)
            atac_overlap = atac_index is not None and atac_index.overlaps(
                target_start, target_end
            )
            h3k27ac_overlap = h3k27ac_index is not None and h3k27ac_index.overlaps(
                target_start, target_end
            )
            if atac_overlap and h3k27ac_overlap:
                source = JOINT_PEAK_SOURCE
            elif atac_overlap:
                source = PEAK_SOURCE
            elif h3k27ac_overlap:
                source = H3K27AC_PEAK_SOURCE
            else:
                source = BACKGROUND_SOURCE
            candidate_by_coordinate[coordinate] = CandidateWindow(
                chrom=chrom,
                input_start=input_start,
                input_end=input_end,
                target_start=target_start,
                target_end=target_end,
                split=split,
                block_id=f"{split}:{chrom}:{block_index}",
                source=source,
                sampling=sampling,
            )

        for chrom in chromosome_order:
            if chrom not in allowed_chromosomes:
                continue
            chromosome_sequence = genome[chrom]
            chromosome_stride = (
                math.gcd(args.stride, validation_stride)
                if split_strategy == "regions"
                else (
                    validation_stride
                    if chrom in validation_chromosomes
                    or chrom in chromosome_splits["test"]
                    else args.stride
                )
            )
            for target_start in range(
                0,
                len(chromosome_sequence) - args.window_size + 1,
                chromosome_stride,
            ):
                if split_strategy == "regions":
                    input_start = target_start - context_flank_size
                    input_end = target_start + args.window_size + context_flank_size
                    split = containing_split(
                        split_regions, chrom, input_start, input_end
                    )
                    desired_stride = (
                        args.stride if split == "train" else validation_stride
                    )
                    if split is None or target_start % desired_stride:
                        continue
                add_candidate(chrom, target_start, "sliding_grid")
            for summit_start, _summit_end in master_dhs_summits.get(chrom, []):
                add_candidate(
                    chrom,
                    summit_start - args.window_size // 2,
                    "dhs_summit_centered",
                )

        candidates = list(candidate_by_coordinate.values())
        selected, sampling_counts = select_balanced_windows(
            candidates, args.background_to_peak_ratio, args.seed
        )
        chromosome_rank = {chrom: index for index, chrom in enumerate(chromosome_order)}
        selected.sort(
            key=lambda window: (
                chromosome_rank[window.chrom],
                window.target_start,
                window.source,
            )
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=args.output.parent, prefix=f".{args.output.name}.", delete=False
        ) as raw_handle:
            temporary = Path(raw_handle.name)
        output_counts = {
            split: {
                source: 0
                for source in sorted(REGULATORY_SOURCES | {BACKGROUND_SOURCE})
            }
            for split in SPLITS
        }
        signal_summary = {
            split: {
                assay: {
                    context: {"sum": 0.0, "maximum": 0.0, "nonzero": 0}
                    for context in CONTEXTS
                }
                for assay in signal_assays
            }
            for split in output_counts
        }
        try:
            with output_handle(args.output, temporary) as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(
                    (
                        "record_id",
                        "source",
                        "chrom",
                        "start",
                        "end",
                        "target_start",
                        "target_end",
                        "block_id",
                        "split",
                        "sampling",
                        "sequence",
                        *(
                            f"{assay}_signal_{context}"
                            for assay in signal_assays
                            for context in CONTEXTS
                        ),
                    )
                )
                selected_by_chromosome = {
                    chrom: [window for window in selected if window.chrom == chrom]
                    for chrom in chromosome_order
                    if chrom in allowed_chromosomes
                }
                for chrom in chromosome_order:
                    chromosome_windows = selected_by_chromosome.get(chrom, [])
                    if not chromosome_windows:
                        continue
                    starts = [window.target_start for window in chromosome_windows]
                    aligned_stride = math.gcd(args.stride, validation_stride)
                    signals = {
                        assay: {
                            context: mean_signal_for_starts(
                                handles[assay][context],
                                chrom,
                                len(genome[chrom]),
                                aligned_stride,
                                args.window_size,
                                starts,
                            )
                            for context in CONTEXTS
                        }
                        for assay in signal_assays
                    }
                    for index, window in enumerate(chromosome_windows):
                        values = [
                            float(signals[assay][context][index])
                            for assay in signal_assays
                            for context in CONTEXTS
                        ]
                        writer.writerow(
                            (
                                f"ATAC_{chrom}_{window.target_start}_{window.target_end}",
                                window.source,
                                chrom,
                                window.input_start,
                                window.input_end,
                                window.target_start,
                                window.target_end,
                                window.block_id,
                                window.split,
                                window.sampling,
                                genome[chrom][window.input_start : window.input_end],
                                *(format(value, ".9g") for value in values),
                            )
                        )
                        output_counts[window.split][window.source] += 1
                        value_index = 0
                        for assay in signal_assays:
                            for context in CONTEXTS:
                                value = values[value_index]
                                value_index += 1
                                summary = signal_summary[window.split][assay][context]
                                summary["sum"] += value
                                summary["maximum"] = max(summary["maximum"], value)
                                summary["nonzero"] += int(value > 0)
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        for assay_handles in handles.values():
            for handle in assay_handles.values():
                handle.close()

    total_counts = {
        split: sum(source_counts.values()) for split, source_counts in output_counts.items()
    }
    if any(not count for count in total_counts.values()):
        raise ValueError(f"Pretraining split is empty: {total_counts}")
    for split, assay_summaries in signal_summary.items():
        for context_summaries in assay_summaries.values():
            for summary in context_summaries.values():
                summary["mean"] = summary.pop("sum") / total_counts[split]

    metadata: dict[str, object] = {
        "method": (
            "region_split_peak_enriched_epigenomic_signal_v4_central_target"
            if split_strategy == "regions"
            else "chromosome_validation_peak_enriched_epigenomic_signal_v3_central_target"
            if split_strategy == "chromosome"
            and (h3k27ac_peaks_bed is not None or master_dhs_summits_bed is not None)
            else (
                "blocked_sliding_window_multicontext_atac_signal_v2_central_target"
                if signal_assays == ("atac",)
                else "blocked_sliding_window_multimodal_epigenomic_signal_v2_central_target"
            )
        )
        if context_flank_size
        else (
            "blocked_sliding_window_multicontext_atac_signal_v1"
            if signal_assays == ("atac",)
            else "blocked_sliding_window_multimodal_epigenomic_signal_v1"
        ),
        "signal_assays": list(signal_assays),
        "window_signal_summaries_omitted": not bool(signal_assays),
        "window_size_bp": args.window_size,
        "target_window_size_bp": args.window_size,
        "context_flank_size_bp": context_flank_size,
        "input_window_size_bp": input_window_size,
        "stride_bp": args.stride,
        "training_stride_bp": args.stride,
        "validation_stride_bp": validation_stride,
        "block_size_bp": args.block_size,
        "split_strategy": split_strategy,
        "validation_fraction": (
            args.validation_fraction if split_strategy == "blocked" else None
        ),
        "background_to_peak_ratio": args.background_to_peak_ratio,
        "seed": args.seed,
        "window_counts": total_counts,
        "window_counts_by_source": output_counts,
        "candidate_and_sampling_counts": sampling_counts,
        "candidate_counts_by_sampling": {
            split: {
                sampling: sum(
                    window.split == split and window.sampling == sampling
                    for window in candidates
                )
                for sampling in sorted({window.sampling for window in candidates})
            }
            for split in SPLITS
        },
        "signal_summary": signal_summary,
        "target_definition": (
            "Dense assay labels are stored separately in profile arrays; the window table "
            "contains sequence and sampling metadata only"
            if not signal_assays
            else "mean basewise background-TMM-normalized context-average coverage for each "
            "requested assay over the central target window only; bases absent from a sparse BigWig "
            "contribute zero"
        ),
        "leakage_control": {
            "training_chromosomes": sorted(training_chromosomes),
            "pretraining_validation_chromosomes": sorted(validation_chromosomes),
            "excluded_supervised_test_chromosomes": sorted(chromosome_splits["test"]),
            "pretraining_validation_partition": (
                "explicit disjoint genomic regions; complete input windows crossing a region boundary are excluded"
                if split_strategy == "regions"
                else "supervised chromosome validation split; no validation chromosome occurs in training"
                if split_strategy == "chromosome"
                else (
                    "deterministically hashed genomic blocks; complete input-context windows crossing "
                    "block boundaries are excluded"
                )
            ),
        },
        "augmentation": {
            "sliding_windows": True,
            "dhs_summit_centered_windows": master_dhs_summits_bed is not None,
            "reverse_complement": "applied as a virtual second training view by the trainer",
        },
        "skipped_candidate_counts": skipped,
        "inputs": {
            "chromosome_splits": {
                split: sorted(chromosomes)
                for split, chromosomes in chromosome_splits.items()
            },
            "region_splits": region_values,
            "reference_fasta": {
                "path": str(args.reference_fasta),
                "sha256": sha256_file(args.reference_fasta),
            },
            "blacklist_bed": {
                "path": str(args.blacklist_bed),
                "sha256": sha256_file(args.blacklist_bed),
            },
            "master_dhs_bed": {
                "path": str(args.master_dhs_bed),
                "sha256": sha256_file(args.master_dhs_bed),
            },
            **(
                {
                    "master_dhs_summits_bed": {
                        "path": str(master_dhs_summits_bed),
                        "sha256": sha256_file(master_dhs_summits_bed),
                    }
                }
                if master_dhs_summits_bed is not None
                else {}
            ),
            **(
                {
                    "h3k27ac_peaks_bed": {
                        "path": str(h3k27ac_peaks_bed),
                        "sha256": sha256_file(h3k27ac_peaks_bed),
                    }
                }
                if h3k27ac_peaks_bed is not None
                else {}
            ),
            "signal_bigwigs": {
                assay: {
                    context: {"path": str(path), "sha256": sha256_file(path)}
                    for context, path in assay_paths.items()
                }
                for assay, assay_paths in bigwig_paths.items()
            },
        },
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
    }
    atomic_write_text(args.metadata, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> None:
    metadata = build_windows(parse_args())
    print(
        json.dumps(
            {
                "event": "epigenomic_signal_windows_complete",
                "training_chromosomes": metadata["leakage_control"][
                    "training_chromosomes"
                ],
                "window_counts": metadata["window_counts"],
                "window_counts_by_source": metadata["window_counts_by_source"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
