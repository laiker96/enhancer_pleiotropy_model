#!/usr/bin/env python3
"""Build a merged H3K27ac peak union with replicate support when available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from ..constants import CONTEXTS
from ..io import (
    MutableIntervalIndex,
    atomic_write_text,
    read_bed_intervals,
    sha256_file,
)


PEAK_NAME = re.compile(r"^(?P<context>[^_]+)_h3k27ac_rep\d+_peaks\.broadPeak$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--contexts", nargs="+", default=list(CONTEXTS))
    return parser.parse_args()


def merge_intervals(
    intervals_by_chromosome: dict[str, list[tuple[int, int]]]
) -> dict[str, list[tuple[int, int]]]:
    merged: dict[str, list[tuple[int, int]]] = {}
    for chrom, intervals in intervals_by_chromosome.items():
        chromosome_merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if chromosome_merged and start <= chromosome_merged[-1][1]:
                previous_start, previous_end = chromosome_merged[-1]
                chromosome_merged[-1] = (previous_start, max(previous_end, end))
            else:
                chromosome_merged.append((start, end))
        merged[chrom] = chromosome_merged
    return merged


def replicate_supported_intervals(
    replicate_intervals: list[dict[str, list[tuple[int, int]]]],
) -> dict[str, list[tuple[int, int]]]:
    if len(replicate_intervals) == 1:
        return replicate_intervals[0]
    supported: dict[str, list[tuple[int, int]]] = {}
    for replicate_index, intervals_by_chromosome in enumerate(replicate_intervals):
        other_indices: dict[str, list[MutableIntervalIndex]] = {}
        for other_index, other_intervals in enumerate(replicate_intervals):
            if other_index == replicate_index:
                continue
            for chrom, intervals in other_intervals.items():
                other_indices.setdefault(chrom, []).append(MutableIntervalIndex(intervals))
        for chrom, intervals in intervals_by_chromosome.items():
            comparison_indices = other_indices.get(chrom, [])
            for start, end in intervals:
                if comparison_indices and all(
                    index.overlaps(start, end) for index in comparison_indices
                ):
                    supported.setdefault(chrom, []).append((start, end))
    return merge_intervals(supported)


def build_consensus(args: argparse.Namespace) -> dict[str, object]:
    contexts = tuple(args.contexts)
    if not contexts or len(set(contexts)) != len(contexts):
        raise ValueError("Contexts must be non-empty and unique")
    paths_by_context: dict[str, list[Path]] = {context: [] for context in contexts}
    for path in sorted(args.peak_directory.glob("*_h3k27ac_rep*_peaks.broadPeak")):
        match = PEAK_NAME.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected H3K27ac peak filename: {path.name}")
        context = match.group("context")
        if context in paths_by_context:
            paths_by_context[context].append(path)
    missing = [context for context, paths in paths_by_context.items() if not paths]
    if missing:
        raise FileNotFoundError(f"Missing H3K27ac peak files for contexts: {missing}")

    union_intervals: dict[str, list[tuple[int, int]]] = {}
    context_summary: dict[str, object] = {}
    input_metadata: dict[str, list[dict[str, str]]] = {}
    for context, paths in paths_by_context.items():
        replicate_intervals = [read_bed_intervals(path) for path in paths]
        supported = replicate_supported_intervals(replicate_intervals)
        for chrom, intervals in supported.items():
            union_intervals.setdefault(chrom, []).extend(intervals)
        context_summary[context] = {
            "replicates": len(paths),
            "raw_peak_counts": [
                sum(map(len, intervals.values())) for intervals in replicate_intervals
            ],
            "supported_merged_peak_count": sum(map(len, supported.values())),
            "support_rule": (
                "overlaps a peak in every other replicate"
                if len(paths) > 1
                else "single available replicate retained"
            ),
        }
        input_metadata[context] = [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ]

    merged_union = merge_intervals(union_intervals)
    lines = [
        f"{chrom}\t{start}\t{end}"
        for chrom in sorted(merged_union)
        for start, end in merged_union[chrom]
    ]
    if not lines:
        raise ValueError("The H3K27ac consensus peak union is empty")
    atomic_write_text(args.output, "\n".join(lines) + "\n")
    metadata: dict[str, object] = {
        "method": "context_h3k27ac_replicate_supported_merged_union_v1",
        "contexts": list(contexts),
        "context_summary": context_summary,
        "union_peak_count": len(lines),
        "inputs": input_metadata,
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
    }
    atomic_write_text(args.metadata, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> None:
    metadata = build_consensus(parse_args())
    print(
        json.dumps(
            {
                "event": "h3k27ac_consensus_peaks_complete",
                "union_peak_count": metadata["union_peak_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
