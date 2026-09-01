"""Generate aligned observed/predicted BigWigs and an IGV session."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import pyBigWig

from .constants import (
    ASSAYS,
    ATAC_TARGET_BP,
    CONTEXTS,
    DNA_ALPHABET,
    H3K27AC_TARGET_BP,
    INPUT_BP,
    SOURCE_BIN_BP,
)
from .inference import load_model, predict_sequences, resolve_device
from .io import (
    MutableIntervalIndex,
    atomic_write_json,
    read_bed_intervals,
    read_fasta,
    sha256_file,
)


@dataclass(frozen=True)
class GridWindow:
    input_start: int
    input_end: int
    target_start: int
    sequence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-fasta", required=True, type=Path)
    parser.add_argument("--observed-bigwig-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--blacklist-bed", type=Path)
    parser.add_argument("--chromosome", default="chr2L")
    parser.add_argument("--region-start", default=0, type=int)
    parser.add_argument("--region-end", required=True, type=int)
    parser.add_argument("--stride", default=256, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--progress-every-batches", default=100, type=int)
    parser.add_argument("--checkpoint-every-batches", default=100, type=int)
    parser.add_argument("--igv-genome", default="dm6")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="no"
    )
    parser.add_argument(
        "--no-reverse-complement-ensemble",
        action="store_false",
        dest="reverse_complement_ensemble",
    )
    parser.set_defaults(reverse_complement_ensemble=True)
    return parser.parse_args()


def build_grid_windows(
    sequence: str,
    region_start: int,
    region_end: int,
    stride: int,
    blacklist: MutableIntervalIndex | None = None,
) -> tuple[list[GridWindow], dict[str, int]]:
    if stride < 1 or stride % 64:
        raise ValueError("Stride must be a positive multiple of 64 bp")
    if not 0 <= region_start < region_end <= len(sequence):
        raise ValueError("Region lies outside the chromosome")
    flank = (INPUT_BP - ATAC_TARGET_BP) // 2
    first_target = math.ceil((region_start + flank) / stride) * stride
    last_target = region_end - ATAC_TARGET_BP - flank
    windows: list[GridWindow] = []
    skipped = {"ambiguous_sequence": 0, "blacklist_overlap": 0}
    for target_start in range(first_target, last_target + 1, stride):
        input_start = target_start - flank
        input_end = input_start + INPUT_BP
        if blacklist is not None and blacklist.overlaps(input_start, input_end):
            skipped["blacklist_overlap"] += 1
            continue
        input_sequence = sequence[input_start:input_end]
        if len(input_sequence) != INPUT_BP or set(input_sequence) - DNA_ALPHABET:
            skipped["ambiguous_sequence"] += 1
            continue
        windows.append(
            GridWindow(
                input_start=input_start,
                input_end=input_end,
                target_start=target_start,
                sequence=input_sequence,
            )
        )
    if not windows:
        raise ValueError("No valid sliding windows remain")
    return windows, skipped


def output_start(window: GridWindow, target_bp: int) -> int:
    return window.input_start + (INPUT_BP - target_bp) // 2


def accumulation_geometry(
    windows: list[GridWindow], target_bp: int, bin_size: int
) -> tuple[int, int]:
    starts = [output_start(window, target_bp) for window in windows]
    ends = [start + target_bp for start in starts]
    if min(starts) % bin_size or max(ends) % bin_size:
        raise ValueError("Output geometry is not aligned to its native bin size")
    return min(starts) // bin_size, max(ends) // bin_size


def accumulate_profile(
    totals: np.ndarray,
    support: np.ndarray | None,
    prediction: np.ndarray,
    global_start_bin: int,
    output_start_bp: int,
    bin_size: int,
) -> None:
    if output_start_bp % bin_size:
        raise ValueError("Output start is not bin-aligned")
    local_start = output_start_bp // bin_size - global_start_bin
    local_end = local_start + len(prediction)
    if local_start < 0 or local_end > len(totals):
        raise ValueError("Prediction falls outside the accumulation array")
    if prediction.shape[1:] != totals.shape[1:]:
        raise ValueError("Prediction contexts do not match accumulation array")
    totals[local_start:local_end] += prediction
    if support is not None:
        support[local_start:local_end] += 1


def load_observed_bins(
    path: Path,
    chromosome: str,
    start_bin: int,
    end_bin: int,
    bin_size: int,
) -> np.ndarray:
    start, end = start_bin * bin_size, end_bin * bin_size
    with pyBigWig.open(str(path)) as bigwig:
        chromosome_size = bigwig.chroms(chromosome)
        if chromosome_size is None or end > chromosome_size:
            raise ValueError(f"{path}: requested bins lie outside {chromosome}")
        values = np.asarray(bigwig.values(chromosome, start, end, numpy=True))
    values = np.nan_to_num(values, nan=0.0, posinf=np.nan, neginf=np.nan)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{path}: observed signal must be finite and nonnegative")
    return values.reshape(-1, bin_size).mean(axis=1, dtype=np.float64).astype(
        np.float32
    )


def atomic_save_state(
    path: Path,
    signature: str,
    next_window: int,
    totals: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez(
            handle,
            signature=np.asarray(signature),
            next_window=np.asarray(next_window, dtype=np.int64),
            **{f"{assay}_totals": values for assay, values in totals.items()},
        )
    temporary.replace(path)


def load_state(
    path: Path,
    signature: str,
    expected_shapes: dict[str, tuple[int, int]],
) -> tuple[int, dict[str, np.ndarray]]:
    with np.load(path) as state:
        if str(state["signature"].item()) != signature:
            raise ValueError("Partial browser-track state belongs to a different run")
        totals = {
            assay: np.asarray(state[f"{assay}_totals"], dtype=np.float32)
            for assay in ASSAYS
        }
        for assay, shape in expected_shapes.items():
            if totals[assay].shape != shape:
                raise ValueError(f"Partial {assay} totals have the wrong shape")
        return int(state["next_window"].item()), totals


def write_bigwig(
    path: Path,
    chromosome_header: list[tuple[str, int]],
    chromosome: str,
    start_bin: int,
    bin_size: int,
    values: np.ndarray,
    support: np.ndarray,
) -> None:
    covered = np.flatnonzero(support > 0)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite track: {path}")
    with pyBigWig.open(str(temporary), "w") as bigwig:
        bigwig.addHeader(chromosome_header)
        for offset in range(0, len(covered), 100_000):
            indices = covered[offset : offset + 100_000]
            starts = (indices + start_bin) * bin_size
            bigwig.addEntries(
                [chromosome] * len(indices),
                starts.astype(np.int64).tolist(),
                ends=(starts + bin_size).astype(np.int64).tolist(),
                values=values[indices].astype(np.float64).tolist(),
            )
    temporary.replace(path)


def write_igv_session(
    path: Path,
    genome: str,
    chromosome: str,
    region_start: int,
    region_end: int,
    output_paths: dict[str, dict[str, dict[str, Path]]],
) -> None:
    root = ET.Element(
        "Session",
        genome=genome,
        locus=f"{chromosome}:{region_start + 1}-{region_end}",
        version="3",
    )
    resources = ET.SubElement(root, "Resources")
    panel = ET.SubElement(root, "Panel", height="1200", name="DataPanel", width="1400")
    colors = {"observed": "44,123,182", "predicted": "215,25,28"}
    for assay in ASSAYS:
        for context in CONTEXTS:
            for source in ("observed", "predicted"):
                track_path = output_paths[source][assay][context]
                relative = os.path.relpath(track_path, path.parent)
                name = f"{assay.upper()} {context} {source}"
                ET.SubElement(resources, "Resource", name=name, path=relative)
                ET.SubElement(
                    panel,
                    "Track",
                    color=colors[source],
                    displayMode="COLLAPSED",
                    height="35",
                    id=relative,
                    name=name,
                    renderer="BAR_CHART",
                    visible="true",
                    windowFunction="mean",
                )
    ET.indent(root, space="  ")
    temporary = path.with_suffix(path.suffix + ".tmp")
    ET.ElementTree(root).write(temporary, encoding="UTF-8", xml_declaration=True)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if (
        args.batch_size < 1
        or args.progress_every_batches < 1
        or args.checkpoint_every_batches < 1
    ):
        raise ValueError("Batch size and progress/checkpoint intervals must be positive")
    device = resolve_device(args.device)
    if args.mixed_precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 inference requires CUDA")

    genome, chromosome_order = read_fasta(args.reference_fasta)
    if args.chromosome not in genome:
        raise ValueError(f"Reference FASTA lacks {args.chromosome}")
    chromosome_sizes = [(chrom, len(genome[chrom])) for chrom in chromosome_order]
    blacklist = None
    if args.blacklist_bed is not None:
        blacklist = MutableIntervalIndex(
            read_bed_intervals(args.blacklist_bed).get(args.chromosome, ())
        )
    windows, skipped = build_grid_windows(
        genome[args.chromosome],
        args.region_start,
        args.region_end,
        args.stride,
        blacklist,
    )

    model, model_metadata = load_model(args.checkpoint, device)
    h3_bin_size = SOURCE_BIN_BP * int(model.h3k27ac_output_pool_size)
    geometry = {
        "atac": {"target_bp": ATAC_TARGET_BP, "bin_size": SOURCE_BIN_BP},
        "h3k27ac": {"target_bp": H3K27AC_TARGET_BP, "bin_size": h3_bin_size},
    }
    ranges = {
        assay: accumulation_geometry(
            windows, values["target_bp"], values["bin_size"]
        )
        for assay, values in geometry.items()
    }
    supports: dict[str, np.ndarray] = {}
    for assay in ASSAYS:
        start_bin, end_bin = ranges[assay]
        support = np.zeros(end_bin - start_bin, dtype=np.uint16)
        target_bp, bin_size = (
            geometry[assay]["target_bp"],
            geometry[assay]["bin_size"],
        )
        output_bins = target_bp // bin_size
        for window in windows:
            local_start = output_start(window, target_bp) // bin_size - start_bin
            support[local_start : local_start + output_bins] += 1
        supports[assay] = support

    signature_payload = {
        "checkpoint_sha256": model_metadata.checkpoint_sha256,
        "chromosome": args.chromosome,
        "region": [args.region_start, args.region_end],
        "stride": args.stride,
        "windows": len(windows),
        "rc_ensemble": args.reverse_complement_ensemble,
        "mixed_precision": args.mixed_precision,
        "geometry": geometry,
    }
    signature = json.dumps(signature_payload, sort_keys=True)
    expected_shapes = {
        assay: (len(supports[assay]), len(CONTEXTS)) for assay in ASSAYS
    }
    state_path = args.output_directory / ".partial_predictions.npz"
    if state_path.is_file():
        next_window, totals = load_state(state_path, signature, expected_shapes)
        print(json.dumps({"event": "browser_tracks_resumed", "window": next_window}))
    else:
        next_window = 0
        totals = {
            assay: np.zeros(shape, dtype=np.float32)
            for assay, shape in expected_shapes.items()
        }

    started = time.monotonic()
    batches = 0
    for start in range(next_window, len(windows), args.batch_size):
        items = windows[start : start + args.batch_size]
        atac, h3k27ac = predict_sequences(
            model,
            [window.sequence for window in items],
            batch_size=len(items),
            device=device,
            reverse_complement_ensemble=args.reverse_complement_ensemble,
            mixed_precision=args.mixed_precision,
        )
        predictions = {"atac": atac, "h3k27ac": h3k27ac}
        for item_index, window in enumerate(items):
            for assay in ASSAYS:
                accumulate_profile(
                    totals[assay],
                    None,
                    predictions[assay][item_index],
                    ranges[assay][0],
                    output_start(window, geometry[assay]["target_bp"]),
                    geometry[assay]["bin_size"],
                )
        next_window = start + len(items)
        batches += 1
        if batches % args.checkpoint_every_batches == 0:
            atomic_save_state(state_path, signature, next_window, totals)
        if batches % args.progress_every_batches == 0:
            print(
                json.dumps(
                    {
                        "event": "browser_tracks_progress",
                        "windows": next_window,
                        "total_windows": len(windows),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    atomic_save_state(state_path, signature, next_window, totals)

    observed_paths = {
        assay: {
            context: args.observed_bigwig_directory
            / f"{context}.{assay}.mean.background_tmm.bw"
            for context in CONTEXTS
        }
        for assay in ASSAYS
    }
    observed: dict[str, np.ndarray] = {}
    for assay in ASSAYS:
        start_bin, end_bin = ranges[assay]
        columns = [
            load_observed_bins(
                observed_paths[assay][context],
                args.chromosome,
                start_bin,
                end_bin,
                geometry[assay]["bin_size"],
            )
            for context in CONTEXTS
        ]
        observed[assay] = np.column_stack(columns)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, dict[str, dict[str, Path]]] = {
        source: {assay: {} for assay in ASSAYS}
        for source in ("observed", "predicted")
    }
    for assay in ASSAYS:
        support = supports[assay]
        predicted = totals[assay] / np.maximum(support[:, None], 1)
        for context_index, context in enumerate(CONTEXTS):
            for source, values in (
                ("observed", observed[assay]),
                ("predicted", predicted),
            ):
                path = args.output_directory / f"{source}.{context}.{assay}.bw"
                write_bigwig(
                    path,
                    chromosome_sizes,
                    args.chromosome,
                    ranges[assay][0],
                    geometry[assay]["bin_size"],
                    values[:, context_index],
                    support,
                )
                output_paths[source][assay][context] = path

    session_path = args.output_directory / "igv_session.xml"
    write_igv_session(
        session_path,
        args.igv_genome,
        args.chromosome,
        args.region_start,
        args.region_end,
        output_paths,
    )
    metadata = {
        "method": "native_bin_per_base_sliding_overlap_mean_v1",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": model_metadata.checkpoint_sha256,
            "epoch": model_metadata.epoch,
        },
        "reference_fasta": {
            "path": str(args.reference_fasta),
            "sha256": sha256_file(args.reference_fasta),
        },
        "blacklist_bed": (
            {"path": str(args.blacklist_bed), "sha256": sha256_file(args.blacklist_bed)}
            if args.blacklist_bed is not None
            else None
        ),
        "chromosome": args.chromosome,
        "region": {"start": args.region_start, "end": args.region_end},
        "contexts": list(CONTEXTS),
        "geometry": geometry,
        "stride_bp": args.stride,
        "windows": len(windows),
        "skipped_windows": skipped,
        "reverse_complement_ensemble": args.reverse_complement_ensemble,
        "aggregation": (
            "Arithmetic mean of every model-output contribution covering each base; "
            "stored as native 16-bp ATAC and 64-bp H3K27ac intervals. Observed "
            "tracks use the identical output bins and model-support mask."
        ),
        "support": {
            assay: {
                str(int(value)): int(count)
                for value, count in zip(
                    *np.unique(supports[assay], return_counts=True), strict=True
                )
            }
            for assay in ASSAYS
        },
        "source_observed_bigwigs": {
            assay: {context: str(path) for context, path in paths.items()}
            for assay, paths in observed_paths.items()
        },
        "output_bigwigs": {
            source: {
                assay: {context: str(path) for context, path in paths.items()}
                for assay, paths in assays.items()
            }
            for source, assays in output_paths.items()
        },
        "igv_session": str(session_path),
    }
    atomic_write_json(args.output_directory / "browser_tracks.metadata.json", metadata)
    state_path.unlink()
    print(
        json.dumps(
            {
                "event": "browser_tracks_complete",
                "windows": len(windows),
                "tracks": 2 * len(ASSAYS) * len(CONTEXTS),
                "igv_session": str(session_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
