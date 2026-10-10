"""Dense per-context signal profiles (regression targets) for every window.

For each window, the bigWig mean in consecutive 16-bp bins is read over a
target centred on the window: 512 bp for ATAC (32 bins) and 1,536 bp for
H3K27ac (96 bins; pooled to 24 x 64 bp during training). Arrays are float32
``[windows, bins, contexts]`` in the order of ``windows.tsv.gz``.
"""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile

import numpy as np

from ..constants import SPLITS
from ..io import open_text, sha256_file, write_json


def bigwig_path(directory: Path, context: str, assay: str) -> Path:
    return Path(directory) / f"{context}.{assay}.mean.background_tmm.bw"


def read_target_coordinates(windows: Path, target_bp: int):
    """Row-ordered centred target intervals, grouped by split and chromosome."""
    grouped = {split: {} for split in SPLITS}
    counts = dict.fromkeys(SPLITS, 0)
    with open_text(windows) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            split = row["split"]
            centre = (int(row["target_start"]) + int(row["target_end"])) // 2
            start = centre - target_bp // 2
            if start < int(row["start"]) or start + target_bp > int(row["end"]):
                raise ValueError(f"{windows}: profile target outside window {row['record_id']}")
            indices, starts = grouped[split].setdefault(row["chrom"], ([], []))
            indices.append(counts[split])
            starts.append(start)
            counts[split] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Dataset has an empty split: {counts}")
    return counts, {split: {chrom: (np.asarray(i, np.int64), np.asarray(s, np.int64))
                            for chrom, (i, s) in by_chrom.items()}
                    for split, by_chrom in grouped.items()}


def binned_means(bigwig, chrom: str, starts: np.ndarray, target_bp: int, bin_bp: int) -> np.ndarray:
    """Exact base-resolution bin means from one bigWig read per chromosome."""
    size = bigwig.chroms(chrom)
    if size is None or starts.min() < 0 or starts.max() + target_bp > size:
        raise ValueError(f"Targets on {chrom} fall outside the bigWig")
    lo, hi = int(starts.min()), int(starts.max()) + target_bp
    values = np.asarray(bigwig.values(chrom, lo, hi, numpy=True), dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=np.nan, neginf=np.nan)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{chrom}: bigWig contains invalid coverage")
    cumulative = np.empty(len(values) + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(values, out=cumulative[1:])
    relative = starts - lo
    output = np.empty((len(starts), target_bp // bin_bp), dtype=np.float32)
    for index in range(output.shape[1]):
        left = relative + index * bin_bp
        output[:, index] = (cumulative[left + bin_bp] - cumulative[left]) / bin_bp
    return output


def build_profiles(*, windows: Path, bigwig_dir: Path, output_dir: Path, assay: str,
                   contexts: tuple[str, ...], target_bp: int, bin_bp: int) -> dict:
    import pyBigWig

    if target_bp % bin_bp or target_bp % 2:
        raise ValueError("Profile target must be even and divisible by the bin size")
    counts, grouped = read_target_coordinates(windows, target_bp)
    paths = {context: bigwig_path(bigwig_dir, context, assay) for context in contexts}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing bigWigs: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    finals = {split: output_dir / f"{split}_profiles.npy" for split in SPLITS}
    temporaries, arrays = {}, {}
    try:
        for split, final in finals.items():
            with tempfile.NamedTemporaryFile(dir=output_dir, prefix=f".{final.name}.",
                                             suffix=".npy", delete=False) as handle:
                temporaries[split] = Path(handle.name)
            arrays[split] = np.lib.format.open_memmap(
                temporaries[split], mode="w+", dtype=np.float32,
                shape=(counts[split], target_bp // bin_bp, len(contexts)))
        for index, context in enumerate(contexts):
            with pyBigWig.open(str(paths[context])) as bigwig:
                for split in SPLITS:
                    for chrom, (rows, starts) in grouped[split].items():
                        arrays[split][rows, :, index] = binned_means(bigwig, chrom, starts, target_bp, bin_bp)
        for split in SPLITS:
            arrays[split].flush()
            del arrays[split]
            temporaries[split].replace(finals[split])
        temporaries.clear()
    finally:
        arrays.clear()
        for path in temporaries.values():
            path.unlink(missing_ok=True)
    metadata = {
        "assay": assay, "contexts": list(contexts), "target_bp": target_bp, "bin_bp": bin_bp,
        "bins": target_bp // bin_bp, "split_counts": counts,
        "definition": f"mean normalized {assay} coverage per {bin_bp}-bp bin, centred on the window",
        "windows": {"path": str(windows), "sha256": sha256_file(windows)},
        "bigwigs": {c: {"path": str(p), "sha256": sha256_file(p)} for c, p in paths.items()},
        "outputs": {s: {"path": str(p), "sha256": sha256_file(p)} for s, p in finals.items()},
    }
    write_json(output_dir / "profiles.metadata.json", metadata)
    return metadata
