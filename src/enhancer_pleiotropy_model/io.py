"""Small deterministic I/O and genomic interval helpers."""

from __future__ import annotations

from bisect import bisect_left
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, TextIO


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_fasta(path: Path) -> tuple[dict[str, str], list[str]]:
    sequences: dict[str, list[str]] = {}
    order: list[str] = []
    current: str | None = None
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                current = stripped[1:].split()[0]
                if not current or current in sequences:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate or blank FASTA record"
                    )
                sequences[current] = []
                order.append(current)
            elif current is None:
                raise ValueError(f"{path}:{line_number}: sequence before FASTA header")
            else:
                sequences[current].append(stripped.upper())
    if not sequences:
        raise ValueError(f"{path}: no FASTA records")
    return {chrom: "".join(parts) for chrom, parts in sequences.items()}, order


def read_bed_intervals(path: Path) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = {}
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "track ", "browser ")):
                continue
            fields = stripped.split("\t")
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_number}: expected BED3+")
            chrom, start_text, end_text = fields[:3]
            start, end = int(start_text), int(end_text)
            if start < 0 or end <= start:
                raise ValueError(f"{path}:{line_number}: invalid interval")
            intervals.setdefault(chrom, []).append((start, end))
    return intervals


class MutableIntervalIndex:
    """Small sorted interval index used during deterministic window sampling."""

    def __init__(self, intervals: Iterable[tuple[int, int]]) -> None:
        merged: list[list[int]] = []
        for start, end in sorted(intervals):
            if start < 0 or end <= start:
                raise ValueError(f"Invalid interval: {start}-{end}")
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self.starts = [interval[0] for interval in merged]
        self.ends = [interval[1] for interval in merged]

    def overlaps(self, start: int, end: int) -> bool:
        index = bisect_left(self.starts, start)
        if index and self.ends[index - 1] > start:
            return True
        return index < len(self.starts) and self.starts[index] < end

    def add(self, start: int, end: int) -> None:
        if self.overlaps(start, end):
            raise ValueError("Cannot add an overlapping interval")
        index = bisect_left(self.starts, start)
        self.starts.insert(index, start)
        self.ends.insert(index, end)
