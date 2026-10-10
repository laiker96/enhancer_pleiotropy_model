"""Replicate-supported H3K27ac peak union used to pick regulatory windows."""

from __future__ import annotations

from pathlib import Path

from ..io import IntervalIndex, atomic_write_text, read_bed_intervals, sha256_file, write_json

Intervals = dict[str, list[tuple[int, int]]]


def merge_intervals(intervals_by_chromosome: Intervals) -> Intervals:
    merged: Intervals = {}
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


def replicate_supported(replicates: list[Intervals]) -> Intervals:
    """Keep a peak only if it overlaps a peak in every other replicate."""
    if len(replicates) == 1:
        return replicates[0]
    supported: Intervals = {}
    for index, intervals_by_chromosome in enumerate(replicates):
        others: dict[str, list[IntervalIndex]] = {}
        for other_index, other in enumerate(replicates):
            if other_index != index:
                for chrom, intervals in other.items():
                    others.setdefault(chrom, []).append(IntervalIndex(intervals))
        for chrom, intervals in intervals_by_chromosome.items():
            comparison = others.get(chrom, [])
            for start, end in intervals:
                if comparison and all(other.overlaps(start, end) for other in comparison):
                    supported.setdefault(chrom, []).append((start, end))
    return merge_intervals(supported)


def build_h3k27ac_consensus(
    peak_files: dict[str, list[Path]], output: Path, metadata_path: Path
) -> dict:
    """Merge replicate-supported peaks of every context into one BED3 union."""
    union: Intervals = {}
    summary, inputs = {}, {}
    for context, paths in peak_files.items():
        if not paths:
            raise FileNotFoundError(f"No H3K27ac peak files configured for {context}")
        replicates = [read_bed_intervals(path) for path in paths]
        supported = replicate_supported(replicates)
        for chrom, intervals in supported.items():
            union.setdefault(chrom, []).extend(intervals)
        summary[context] = {
            "replicates": len(paths),
            "raw_peak_counts": [sum(map(len, r.values())) for r in replicates],
            "supported_merged_peak_count": sum(map(len, supported.values())),
        }
        inputs[context] = [{"path": str(path), "sha256": sha256_file(path)} for path in paths]
    merged = merge_intervals(union)
    lines = [f"{chrom}\t{start}\t{end}" for chrom in sorted(merged) for start, end in merged[chrom]]
    if not lines:
        raise ValueError("The H3K27ac consensus peak union is empty")
    atomic_write_text(output, "\n".join(lines) + "\n")
    metadata = {
        "method": "per-context replicate-supported peaks, merged across contexts",
        "support_rule": "a peak must overlap a peak in every other replicate of its context",
        "contexts": summary, "union_peak_count": len(lines), "inputs": inputs,
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    write_json(metadata_path, metadata)
    return metadata
