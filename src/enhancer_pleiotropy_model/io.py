"""Small deterministic file and genomic-interval helpers."""

from __future__ import annotations

from bisect import bisect_left
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, TextIO

import numpy as np


def open_text(path: Path) -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    """Atomically write JSON; NaN/inf become null and NumPy values become Python."""
    atomic_write_text(path, json.dumps(_json_ready(value), indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def read_fasta(path: Path) -> tuple[dict[str, str], list[str]]:
    """Return upper-case sequences by name and the record order."""
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
                    raise ValueError(f"{path}:{line_number}: duplicate or blank FASTA record")
                sequences[current] = []
                order.append(current)
            elif current is None:
                raise ValueError(f"{path}:{line_number}: sequence before FASTA header")
            else:
                sequences[current].append(stripped.upper())
    if not sequences:
        raise ValueError(f"{path}: no FASTA records")
    return {name: "".join(parts) for name, parts in sequences.items()}, order


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
            start, end = int(fields[1]), int(fields[2])
            if start < 0 or end <= start:
                raise ValueError(f"{path}:{line_number}: invalid interval")
            intervals.setdefault(fields[0], []).append((start, end))
    return intervals


class IntervalIndex:
    """Sorted, merged half-open intervals with a fast overlap query."""

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
