"""Genome-wide 2,048-bp training windows for the ATAC/H3K27ac regressors.

Each window is a 512-bp target plus 768 bp of sequence context on each side.
Candidates come from a sliding grid (128-bp stride in training, 256 bp in
validation/test) and from windows centred on every master-DHS summit. A
candidate is *regulatory* if its 512-bp target overlaps a master DHS and/or the
H3K27ac consensus peaks, otherwise *background*. Every regulatory window is
kept, plus an equal number of randomly drawn background windows per split.
Windows touching the blacklist, containing non-ACGT bases or crossing a split
boundary are discarded.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import gzip
import io
import math
from pathlib import Path
import random
import tempfile

from ..constants import SPLITS
from ..io import IntervalIndex, read_bed_intervals, read_fasta, sha256_file, write_json

PEAK_SOURCE = "atac_peak_overlap"
H3K27AC_PEAK_SOURCE = "h3k27ac_peak_overlap"
JOINT_PEAK_SOURCE = "atac_h3k27ac_peak_overlap"
BACKGROUND_SOURCE = "genomic_background"
REGULATORY_SOURCES = frozenset((PEAK_SOURCE, H3K27AC_PEAK_SOURCE, JOINT_PEAK_SOURCE))
ALL_SOURCES = sorted(REGULATORY_SOURCES | {BACKGROUND_SOURCE})
DNA = frozenset("ACGT")
COLUMNS = ("record_id", "source", "chrom", "start", "end", "target_start", "target_end",
           "block_id", "split", "sampling", "sequence")


@dataclass(frozen=True)
class Candidate:
    chrom: str
    input_start: int
    input_end: int
    target_start: int
    target_end: int
    split: str
    block_id: str
    source: str
    sampling: str


def parse_region(value: str) -> tuple[str, int, int]:
    chrom, coordinates = value.rsplit(":", 1)
    start, end = (int(x) for x in coordinates.split("-", 1))
    if not chrom or start < 0 or end <= start:
        raise ValueError(f"Invalid region {value!r}; expected CHROM:START-END")
    return chrom, start, end


def split_regions(genome: dict[str, str], splits: dict) -> dict[str, list[tuple[int, int, str]]]:
    """Resolve whole chromosomes and explicit intervals into disjoint split regions."""
    regions: dict[str, list[tuple[int, int, str]]] = {}
    for split in SPLITS:
        for chrom in splits[split].get("chromosomes", []):
            if chrom not in genome:
                raise ValueError(f"Reference FASTA lacks chromosome {chrom}")
            regions.setdefault(chrom, []).append((0, len(genome[chrom]), split))
        for value in splits[split].get("regions", []):
            chrom, start, end = parse_region(value)
            if chrom not in genome or end > len(genome[chrom]):
                raise ValueError(f"Region {value!r} is outside the reference")
            regions.setdefault(chrom, []).append((start, end, split))
    present = {split for intervals in regions.values() for *_, split in intervals}
    if present != set(SPLITS):
        raise ValueError("Train, validation and test regions are all required")
    for chrom, intervals in regions.items():
        previous_end = -1
        for start, end, _ in sorted(intervals):
            if start < previous_end:
                raise ValueError(f"Split regions overlap on {chrom}")
            previous_end = end
    return regions


def containing_split(regions, chrom: str, start: int, end: int) -> str | None:
    matches = [split for lo, hi, split in regions.get(chrom, ()) if lo <= start and end <= hi]
    if len(matches) > 1:
        raise ValueError(f"{chrom}:{start}-{end} belongs to multiple splits")
    return matches[0] if matches else None


def balance(candidates: list[Candidate], ratio: float, seed: int):
    """Keep all regulatory windows and ceil(ratio * n) random background windows per split."""
    selected, counts = [], {}
    for split_index, split in enumerate(SPLITS):
        peaks = [w for w in candidates if w.split == split and w.source in REGULATORY_SOURCES]
        backgrounds = [w for w in candidates if w.split == split and w.source == BACKGROUND_SOURCE]
        if not peaks or not backgrounds:
            raise ValueError(f"Both regulatory and background windows are required in {split}")
        retained = min(len(backgrounds), math.ceil(len(peaks) * ratio))
        selected.extend(peaks)
        selected.extend(random.Random(seed + split_index).sample(backgrounds, retained))
        counts[split] = {
            "candidate_peak_overlap": len(peaks), "candidate_background": len(backgrounds),
            "selected_peak_overlap": len(peaks), "selected_background": retained,
            "candidate_by_source": {s: sum(w.source == s for w in candidates if w.split == split)
                                    for s in ALL_SOURCES},
        }
    return selected, counts


def build_windows(
    *, reference_fasta: Path, blacklist: Path, master_dhs: Path, master_dhs_summits: Path,
    h3k27ac_peaks: Path, bigwig_paths: list[Path], splits: dict, output: Path,
    metadata_path: Path, target_bp: int, flank_bp: int, train_stride_bp: int,
    validation_stride_bp: int, background_to_peak_ratio: float, block_size_bp: int, seed: int,
) -> dict:
    import pyBigWig

    if target_bp % train_stride_bp or target_bp % validation_stride_bp:
        raise ValueError("target_bp must be a multiple of both strides")
    genome, chromosome_order = read_fasta(reference_fasta)
    regions = split_regions(genome, splits)
    for path in bigwig_paths:  # the profiles step reads these; fail early on a mismatch
        with pyBigWig.open(str(path)) as handle:
            lengths = handle.chroms()
            for chrom in regions:
                if lengths.get(chrom) != len(genome[chrom]):
                    raise ValueError(f"{path}: chromosome length mismatch for {chrom}")
    excluded = {c: IntervalIndex(v) for c, v in read_bed_intervals(blacklist).items()}
    dhs = {c: IntervalIndex(v) for c, v in read_bed_intervals(master_dhs).items()}
    h3 = {c: IntervalIndex(v) for c, v in read_bed_intervals(h3k27ac_peaks).items()}
    summits = read_bed_intervals(master_dhs_summits)

    candidates: dict[tuple[str, int, int], Candidate] = {}
    skipped = dict.fromkeys(("crosses_chromosome_boundary", "crosses_split_boundary_or_unassigned",
                             "blacklist_overlap", "ambiguous_sequence", "duplicate_coordinate"), 0)

    def add(chrom: str, target_start: int, sampling: str) -> None:
        sequence = genome[chrom]
        target_end = target_start + target_bp
        input_start, input_end = target_start - flank_bp, target_end + flank_bp
        if input_start < 0 or input_end > len(sequence):
            skipped["crosses_chromosome_boundary"] += 1
            return
        split = containing_split(regions, chrom, input_start, input_end)
        if split is None:
            skipped["crosses_split_boundary_or_unassigned"] += 1
            return
        key = (chrom, target_start, target_end)
        if key in candidates:
            skipped["duplicate_coordinate"] += 1
            existing = candidates[key]
            if sampling not in existing.sampling.split("+"):
                candidates[key] = replace(existing, sampling=f"{existing.sampling}+{sampling}")
            return
        if chrom in excluded and excluded[chrom].overlaps(input_start, input_end):
            skipped["blacklist_overlap"] += 1
            return
        if set(sequence[input_start:input_end]) - DNA:
            skipped["ambiguous_sequence"] += 1
            return
        atac = chrom in dhs and dhs[chrom].overlaps(target_start, target_end)
        acetyl = chrom in h3 and h3[chrom].overlaps(target_start, target_end)
        source = (JOINT_PEAK_SOURCE if atac and acetyl else PEAK_SOURCE if atac
                  else H3K27AC_PEAK_SOURCE if acetyl else BACKGROUND_SOURCE)
        candidates[key] = Candidate(chrom, input_start, input_end, target_start, target_end, split,
                                    f"{split}:{chrom}:{target_start // block_size_bp}", source, sampling)

    grid = math.gcd(train_stride_bp, validation_stride_bp)
    for chrom in chromosome_order:
        if chrom not in regions:
            continue
        for target_start in range(0, len(genome[chrom]) - target_bp + 1, grid):
            split = containing_split(regions, chrom, target_start - flank_bp,
                                     target_start + target_bp + flank_bp)
            stride = train_stride_bp if split == "train" else validation_stride_bp
            if split is not None and target_start % stride == 0:
                add(chrom, target_start, "sliding_grid")
        for summit, _ in summits.get(chrom, []):
            add(chrom, summit - target_bp // 2, "dhs_summit_centered")

    pool = list(candidates.values())
    selected, sampling_counts = balance(pool, background_to_peak_ratio, seed)
    rank = {chrom: index for index, chrom in enumerate(chromosome_order)}
    selected.sort(key=lambda w: (rank[w.chrom], w.target_start, w.source))

    counts = {split: dict.fromkeys(ALL_SOURCES, 0) for split in SPLITS}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as raw:
        temporary = Path(raw.name)
    try:
        # mtime=0 and no embedded file name make the gzip bytes reproducible.
        with (temporary.open("wb") as raw_output,
              gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as binary,
              io.TextIOWrapper(binary, encoding="utf-8", newline="") as handle):
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(COLUMNS)
            for w in selected:
                writer.writerow((f"ATAC_{w.chrom}_{w.target_start}_{w.target_end}", w.source, w.chrom,
                                 w.input_start, w.input_end, w.target_start, w.target_end,
                                 w.block_id, w.split, w.sampling,
                                 genome[w.chrom][w.input_start:w.input_end]))
                counts[w.split][w.source] += 1
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    metadata = {
        "method": "region-split peak-enriched 2,048-bp windows (512-bp target + 768-bp flanks)",
        "target_bp": target_bp, "flank_bp": flank_bp, "input_bp": target_bp + 2 * flank_bp,
        "train_stride_bp": train_stride_bp, "validation_stride_bp": validation_stride_bp,
        "background_to_peak_ratio": background_to_peak_ratio, "seed": seed,
        "splits": splits,
        "window_counts": {split: sum(c.values()) for split, c in counts.items()},
        "window_counts_by_source": counts,
        "candidate_and_sampling_counts": sampling_counts,
        "skipped_candidate_counts": skipped,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in (
            ("reference_fasta", reference_fasta), ("blacklist", blacklist), ("master_dhs", master_dhs),
            ("master_dhs_summits", master_dhs_summits), ("h3k27ac_peaks", h3k27ac_peaks))},
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    write_json(metadata_path, metadata)
    return metadata
