"""Predict eight-context enhancer activity for DNA sequences or genomic regions."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .calibration import apply, load_calibration
from .constants import CONTEXTS, INPUT_BP
from .data.enhancers import LOOKUP
from .io import open_text, read_fasta


def sequences_from_fasta(path: Path) -> tuple[list[str], np.ndarray]:
    records, order = read_fasta(path)
    bad = [name for name in order if len(records[name]) != INPUT_BP]
    if bad:
        raise ValueError(f"Sequences must be exactly {INPUT_BP} bp; e.g. {bad[0]} is {len(records[bad[0]])} bp. "
                         "Use --bed with --genome to extract centred windows instead.")
    return order, _encode([records[name] for name in order])


def sequences_from_bed(bed: Path, genome_fasta: Path) -> tuple[list[str], np.ndarray]:
    """2,048-bp windows centred on each BED interval (name column used as the ID when present)."""
    genome, _ = read_fasta(genome_fasta)
    names, sequences = [], []
    with open_text(bed) as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if not fields or fields[0].startswith(("#", "track", "browser")):
                continue
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            left = (start + end) // 2 - INPUT_BP // 2
            if chrom not in genome or left < 0 or left + INPUT_BP > len(genome[chrom]):
                raise ValueError(f"{chrom}:{start}-{end} cannot be extended to {INPUT_BP} bp")
            names.append(fields[3] if len(fields) > 3 else f"{chrom}:{start}-{end}")
            sequences.append(genome[chrom][left:left + INPUT_BP])
    return names, _encode(sequences)


def _encode(sequences: list[str]) -> np.ndarray:
    codes = LOOKUP[np.frombuffer("".join(sequences).upper().encode("ascii"), dtype=np.uint8)]
    if (codes > 3).any():
        raise ValueError("Sequences may only contain A, C, G and T")
    return codes.reshape(len(sequences), INPUT_BP)


def predict(checkpoint: Path, names: list[str], codes: np.ndarray, output: Path, *,
            calibration: Path | None = None, device_name: str = "auto", batch_size: int = 64) -> Path:
    from .models import load_classifier
    from .training.classifier import predict_probabilities
    from .training.common import resolve_device

    device = resolve_device(device_name)
    model, _ = load_classifier(checkpoint)
    probabilities = predict_probabilities(model.to(device), codes, device, batch_size, half=False)
    calibration = calibration or (Path(checkpoint).parent / "calibration.json")
    calibrated = Path(calibration).is_file()
    if calibrated:
        probabilities = apply(probabilities, load_calibration(calibration))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", *(f"p_{c}" for c in CONTEXTS), "predicted_breadth"])
        for name, row in zip(names, probabilities, strict=True):
            writer.writerow([name, *(f"{v:.6f}" for v in row), f"{row.sum():.6f}"])
    print(f"wrote {len(names)} predictions ({'calibrated' if calibrated else 'uncalibrated'}) to {output}")
    return output
