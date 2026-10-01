"""Generate aligned observed/base/fine-tuned/ensemble BigWigs for IGV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

from .browser_tracks import (
    accumulate_profile,
    accumulation_geometry,
    build_grid_windows,
    load_observed_bins,
    output_start,
    write_bigwig,
)
from .constants import (
    ASSAYS,
    ATAC_TARGET_BP,
    CONTEXTS,
    H3K27AC_TARGET_BP,
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
from .residual_ensemble import residual_ensemble_predictions


PREDICTION_SOURCES = ("base", "fine_tuned", "ensemble")
TRACK_SOURCES = ("observed", *PREDICTION_SOURCES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--specific-checkpoint", required=True, type=Path)
    parser.add_argument("--alpha-atac", required=True, type=float)
    parser.add_argument("--alpha-h3k27ac", required=True, type=float)
    parser.add_argument("--reference-fasta", required=True, type=Path)
    parser.add_argument("--observed-bigwig-directory", required=True, type=Path)
    parser.add_argument(
        "--observed-filename-template",
        default="{context}.{assay}.mean.background_tmm.bw",
        help="Filename template under --observed-bigwig-directory.",
    )
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--blacklist-bed", type=Path)
    parser.add_argument("--chromosome", default="chr2L")
    parser.add_argument("--region-start", default=0, type=int)
    parser.add_argument("--region-end", required=True, type=int)
    parser.add_argument("--stride", default=256, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--progress-every-batches", default=50, type=int)
    parser.add_argument("--checkpoint-every-batches", default=50, type=int)
    parser.add_argument("--igv-genome", default="dm6")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16"
    )
    parser.add_argument(
        "--no-reverse-complement-ensemble",
        action="store_false",
        dest="reverse_complement_ensemble",
    )
    parser.set_defaults(reverse_complement_ensemble=True)
    return parser.parse_args()


def atomic_save_comparison_state(
    path: Path,
    signature: str,
    next_window: int,
    totals: dict[str, dict[str, np.ndarray]],
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
            **{
                f"{source}_{assay}_totals": totals[source][assay]
                for source in PREDICTION_SOURCES
                for assay in ASSAYS
            },
        )
    temporary.replace(path)


def load_comparison_state(
    path: Path,
    signature: str,
    expected_shapes: dict[str, tuple[int, int]],
) -> tuple[int, dict[str, dict[str, np.ndarray]]]:
    with np.load(path) as state:
        if str(state["signature"].item()) != signature:
            raise ValueError("Partial comparison state belongs to a different run")
        next_window = int(state["next_window"].item())
        totals = {
            source: {
                assay: np.asarray(
                    state[f"{source}_{assay}_totals"], dtype=np.float32
                )
                for assay in ASSAYS
            }
            for source in PREDICTION_SOURCES
        }
    for source in PREDICTION_SOURCES:
        for assay, shape in expected_shapes.items():
            if totals[source][assay].shape != shape:
                raise ValueError(f"Partial {source} {assay} totals have the wrong shape")
    return next_window, totals


def write_comparison_igv_session(
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
    panel = ET.SubElement(root, "Panel", height="1600", name="DataPanel", width="1600")
    colors = {
        "observed": "44,123,182",
        "base": "215,25,28",
        "fine_tuned": "255,127,0",
        "ensemble": "35,139,69",
    }
    labels = {
        "observed": "observed",
        "base": "base e31",
        "fine_tuned": "specificity e1",
        "ensemble": "residual ensemble",
    }
    for assay in ASSAYS:
        for context in CONTEXTS:
            autoscale_group = f"{assay}_{context}"
            for source in TRACK_SOURCES:
                track_path = output_paths[source][assay][context]
                relative = os.path.relpath(track_path, path.parent)
                name = f"{assay.upper()} {context} {labels[source]}"
                ET.SubElement(resources, "Resource", name=name, path=relative)
                ET.SubElement(
                    panel,
                    "Track",
                    autoScale="true",
                    autoscaleGroup=autoscale_group,
                    color=colors[source],
                    displayMode="COLLAPSED",
                    height="32",
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
    if not 0 <= args.alpha_atac <= 1 or not 0 <= args.alpha_h3k27ac <= 1:
        raise ValueError("Ensemble alphas must be in [0,1]")
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

    base_model, base_metadata = load_model(args.base_checkpoint, device)
    specific_model, specific_metadata = load_model(args.specific_checkpoint, device)
    if base_metadata.contexts != specific_metadata.contexts:
        raise ValueError("Comparison checkpoints have different context orders")
    if base_metadata.architecture != specific_metadata.architecture:
        raise ValueError("Comparison checkpoints have different architectures")
    h3_bin_size = SOURCE_BIN_BP * int(base_model.h3k27ac_output_pool_size)
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
        target_bp = geometry[assay]["target_bp"]
        bin_size = geometry[assay]["bin_size"]
        output_bins = target_bp // bin_size
        for window in windows:
            local_start = output_start(window, target_bp) // bin_size - start_bin
            support[local_start : local_start + output_bins] += 1
        supports[assay] = support

    signature_payload = {
        "base_checkpoint_sha256": base_metadata.checkpoint_sha256,
        "specific_checkpoint_sha256": specific_metadata.checkpoint_sha256,
        "ensemble_alphas": {
            "atac": args.alpha_atac,
            "h3k27ac": args.alpha_h3k27ac,
        },
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
        next_window, totals = load_comparison_state(
            state_path, signature, expected_shapes
        )
        print(json.dumps({"event": "browser_comparison_resumed", "window": next_window}))
    else:
        next_window = 0
        totals = {
            source: {
                assay: np.zeros(shape, dtype=np.float32)
                for assay, shape in expected_shapes.items()
            }
            for source in PREDICTION_SOURCES
        }

    started = time.monotonic()
    batches = 0
    alpha_by_assay = {
        "atac": args.alpha_atac,
        "h3k27ac": args.alpha_h3k27ac,
    }
    for start in range(next_window, len(windows), args.batch_size):
        items = windows[start : start + args.batch_size]
        sequences = [window.sequence for window in items]
        base_outputs = predict_sequences(
            base_model,
            sequences,
            batch_size=len(items),
            device=device,
            reverse_complement_ensemble=args.reverse_complement_ensemble,
            mixed_precision=args.mixed_precision,
        )
        specific_outputs = predict_sequences(
            specific_model,
            sequences,
            batch_size=len(items),
            device=device,
            reverse_complement_ensemble=args.reverse_complement_ensemble,
            mixed_precision=args.mixed_precision,
        )
        batch_predictions = {
            "base": dict(zip(ASSAYS, base_outputs, strict=True)),
            "fine_tuned": dict(zip(ASSAYS, specific_outputs, strict=True)),
            "ensemble": {},
        }
        for assay in ASSAYS:
            batch_predictions["ensemble"][assay], _ = residual_ensemble_predictions(
                batch_predictions["base"][assay],
                batch_predictions["fine_tuned"][assay],
                alpha_by_assay[assay],
            )
        for item_index, window in enumerate(items):
            for source in PREDICTION_SOURCES:
                for assay in ASSAYS:
                    accumulate_profile(
                        totals[source][assay],
                        None,
                        batch_predictions[source][assay][item_index],
                        ranges[assay][0],
                        output_start(window, geometry[assay]["target_bp"]),
                        geometry[assay]["bin_size"],
                    )
        next_window = start + len(items)
        batches += 1
        if batches % args.checkpoint_every_batches == 0:
            atomic_save_comparison_state(state_path, signature, next_window, totals)
        if batches % args.progress_every_batches == 0:
            print(
                json.dumps(
                    {
                        "event": "browser_comparison_progress",
                        "windows": next_window,
                        "total_windows": len(windows),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    atomic_save_comparison_state(state_path, signature, next_window, totals)

    observed_paths = {
        assay: {
            context: args.observed_bigwig_directory
            / args.observed_filename_template.format(
                context=context,
                assay=assay,
            )
            for context in CONTEXTS
        }
        for assay in ASSAYS
    }
    observed = {}
    for assay in ASSAYS:
        start_bin, end_bin = ranges[assay]
        observed[assay] = np.column_stack(
            [
                load_observed_bins(
                    observed_paths[assay][context],
                    args.chromosome,
                    start_bin,
                    end_bin,
                    geometry[assay]["bin_size"],
                )
                for context in CONTEXTS
            ]
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {
        source: {assay: {} for assay in ASSAYS} for source in TRACK_SOURCES
    }
    for assay in ASSAYS:
        support = supports[assay]
        averaged_predictions = {
            source: totals[source][assay] / np.maximum(support[:, None], 1)
            for source in PREDICTION_SOURCES
        }
        values_by_source = {"observed": observed[assay], **averaged_predictions}
        for context_index, context in enumerate(CONTEXTS):
            for source in TRACK_SOURCES:
                path = args.output_directory / f"{source}.{context}.{assay}.bw"
                write_bigwig(
                    path,
                    chromosome_sizes,
                    args.chromosome,
                    ranges[assay][0],
                    geometry[assay]["bin_size"],
                    values_by_source[source][:, context_index],
                    support,
                )
                output_paths[source][assay][context] = path

    session_path = args.output_directory / "igv_session.xml"
    write_comparison_igv_session(
        session_path,
        args.igv_genome,
        args.chromosome,
        args.region_start,
        args.region_end,
        output_paths,
    )
    metadata = {
        "method": "three_model_native_bin_sliding_overlap_mean_v1",
        "checkpoints": {
            "base": {
                "path": str(args.base_checkpoint),
                "sha256": base_metadata.checkpoint_sha256,
                "epoch": base_metadata.epoch,
            },
            "fine_tuned": {
                "path": str(args.specific_checkpoint),
                "sha256": specific_metadata.checkpoint_sha256,
                "epoch": specific_metadata.epoch,
            },
        },
        "ensemble": {
            "method": "log1p common-mean/context-residual ensemble",
            "alpha_atac": args.alpha_atac,
            "alpha_h3k27ac": args.alpha_h3k27ac,
            "combination_stage": "each sliding-window prediction before genomic averaging",
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
            "Arithmetic mean of all model-output contributions covering each native bin. "
            "Observed tracks use the same native bins and model-support mask."
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
    atomic_write_json(
        args.output_directory / "browser_comparison.metadata.json", metadata
    )
    state_path.unlink()
    print(
        json.dumps(
            {
                "event": "browser_comparison_complete",
                "windows": len(windows),
                "tracks": len(TRACK_SOURCES) * len(ASSAYS) * len(CONTEXTS),
                "igv_session": str(session_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
