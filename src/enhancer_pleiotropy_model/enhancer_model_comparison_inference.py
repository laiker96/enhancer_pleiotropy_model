"""Run exact base/fine-tuned/residual-ensemble inference on enhancer elements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from .constants import CONTEXTS, INPUT_BP
from .enhancer_catalog_evaluation import FEATURE_NAMES, summarize_profiles
from .inference import load_model, predict_sequences, resolve_device
from .io import atomic_write_json, read_fasta, sha256_file
from .residual_ensemble import residual_ensemble_predictions


SOURCES = ("base", "fine_tuned", "ensemble")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-states", required=True, type=Path)
    parser.add_argument("--reference-fasta", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--specific-checkpoint", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--validation-chromosome", default="chr2L")
    parser.add_argument("--validation-end", default=11_751_856, type=int)
    parser.add_argument("--training-start", default=11_761_856, type=int)
    parser.add_argument("--test-chromosome", default="chr3R")
    parser.add_argument("--alpha-atac", default=0.9, type=float)
    parser.add_argument("--alpha-h3k27ac", default=0.3, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--chunk-size", default=512, type=int)
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


def assign_final_splits(
    chromosomes: np.ndarray,
    summits: np.ndarray,
    validation_chromosome: str,
    validation_end: int,
    training_start: int,
    test_chromosome: str,
) -> np.ndarray:
    """Assign the final model's regional validation and chromosome test split."""
    chromosomes = np.asarray(chromosomes).astype(str)
    summits = np.asarray(summits, dtype=np.int64)
    if chromosomes.shape != summits.shape:
        raise ValueError("Chromosomes and summits must align")
    if not 0 < validation_end < training_start:
        raise ValueError("Regional split requires a positive validation/training buffer")
    splits = np.full(len(chromosomes), "train", dtype="<U12")
    splits[chromosomes == test_chromosome] = "test"
    on_validation_chromosome = chromosomes == validation_chromosome
    splits[on_validation_chromosome & (summits < validation_end)] = "validation"
    splits[
        on_validation_chromosome
        & (summits >= validation_end)
        & (summits < training_start)
    ] = "buffer"
    return splits


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.chunk_size < 1:
        raise ValueError("Batch and chunk sizes must be positive")
    if not 0 <= args.alpha_atac <= 1 or not 0 <= args.alpha_h3k27ac <= 1:
        raise ValueError("Ensemble alphas must lie in [0, 1]")
    device = resolve_device(args.device)
    genome, _ = read_fasta(args.reference_fasta)
    with np.load(args.master_states, allow_pickle=False) as loaded:
        states = {name: loaded[name] for name in loaded.files}
    if tuple(states["contexts"].astype(str)) != CONTEXTS:
        raise ValueError("Master-state context order is incorrect")
    hard_activity = states["hard_activity"].astype(bool)
    selected = hard_activity.any(axis=1)
    identifiers = states["ids"].astype(str)[selected]
    chromosomes = states["chrom"].astype(str)[selected]
    summits = states["summit"].astype(np.int64)[selected]
    hard_activity = hard_activity[selected]
    observed_atac = states["observed_atac_mean_512"].astype(np.float32)[selected]
    observed_h3_segments = states["observed_h3k27ac_segment_means_512"].astype(
        np.float32
    )[selected]
    sequences: list[str] = []
    valid = np.ones(len(identifiers), dtype=bool)
    for index, (chromosome, summit) in enumerate(
        zip(chromosomes, summits, strict=True)
    ):
        start = int(summit) - INPUT_BP // 2
        sequence = genome.get(chromosome, "")[start : start + INPUT_BP] if start >= 0 else ""
        if len(sequence) != INPUT_BP or set(sequence) - set("ACGT"):
            valid[index] = False
        sequences.append(sequence)
    invalid_sequence_n = int((~valid).sum())
    identifiers = identifiers[valid]
    chromosomes = chromosomes[valid]
    summits = summits[valid]
    hard_activity = hard_activity[valid]
    observed_atac = observed_atac[valid]
    observed_h3_segments = observed_h3_segments[valid]
    sequences = [sequence for sequence, keep in zip(sequences, valid, strict=True) if keep]
    splits = assign_final_splits(
        chromosomes,
        summits,
        args.validation_chromosome,
        args.validation_end,
        args.training_start,
        args.test_chromosome,
    )
    keep = splits != "buffer"
    buffered_n = int((~keep).sum())
    identifiers = identifiers[keep]
    chromosomes = chromosomes[keep]
    summits = summits[keep]
    hard_activity = hard_activity[keep]
    observed_atac = observed_atac[keep]
    observed_h3_segments = observed_h3_segments[keep]
    splits = splits[keep]
    sequences = [sequence for sequence, retain in zip(sequences, keep, strict=True) if retain]

    base_model, base_metadata = load_model(args.base_checkpoint, device)
    specific_model, specific_metadata = load_model(args.specific_checkpoint, device)
    if base_metadata.architecture != specific_metadata.architecture:
        raise ValueError("Base and specificity checkpoints have different architectures")
    signature = json.dumps(
        {
            "base": base_metadata.checkpoint_sha256,
            "specific": specific_metadata.checkpoint_sha256,
            "alpha_atac": args.alpha_atac,
            "alpha_h3k27ac": args.alpha_h3k27ac,
            "reverse_complement_ensemble": args.reverse_complement_ensemble,
            "mixed_precision": args.mixed_precision,
        },
        sort_keys=True,
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    partial_directory = args.output_directory / "partial"
    partial_directory.mkdir(exist_ok=True)
    chunks: list[np.ndarray] = []
    started = time.monotonic()
    clipped = {"atac": [], "h3k27ac": []}
    for start in range(0, len(identifiers), args.chunk_size):
        end = min(start + args.chunk_size, len(identifiers))
        partial = partial_directory / f"features_{start:06d}_{end:06d}.npz"
        if partial.is_file():
            with np.load(partial, allow_pickle=False) as saved:
                if str(saved["signature"].item()) != signature:
                    raise RuntimeError(f"Cached inference signature differs: {partial}")
                if not np.array_equal(saved["ids"].astype(str), identifiers[start:end]):
                    raise RuntimeError(f"Cached enhancer IDs differ: {partial}")
                features = saved["features"].astype(np.float32)
                clipped["atac"].append(float(saved["clipped_atac"].item()))
                clipped["h3k27ac"].append(float(saved["clipped_h3k27ac"].item()))
            event = "enhancer_comparison_chunk_reused"
        else:
            batch_sequences = sequences[start:end]
            base_atac, base_h3 = predict_sequences(
                base_model,
                batch_sequences,
                batch_size=args.batch_size,
                device=device,
                reverse_complement_ensemble=args.reverse_complement_ensemble,
                mixed_precision=args.mixed_precision,
            )
            specific_atac, specific_h3 = predict_sequences(
                specific_model,
                batch_sequences,
                batch_size=args.batch_size,
                device=device,
                reverse_complement_ensemble=args.reverse_complement_ensemble,
                mixed_precision=args.mixed_precision,
            )
            ensemble_atac, clipped_atac = residual_ensemble_predictions(
                base_atac, specific_atac, args.alpha_atac
            )
            ensemble_h3, clipped_h3 = residual_ensemble_predictions(
                base_h3, specific_h3, args.alpha_h3k27ac
            )
            features = np.stack(
                (
                    summarize_profiles(base_atac, base_h3),
                    summarize_profiles(specific_atac, specific_h3),
                    summarize_profiles(ensemble_atac, ensemble_h3),
                ),
                axis=1,
            )
            clipped["atac"].append(clipped_atac)
            clipped["h3k27ac"].append(clipped_h3)
            atomic_save_npz(
                partial,
                ids=identifiers[start:end].astype(np.str_),
                signature=np.asarray(signature),
                features=features,
                clipped_atac=np.asarray(clipped_atac),
                clipped_h3k27ac=np.asarray(clipped_h3),
            )
            event = "enhancer_comparison_chunk_complete"
        expected = (end - start, len(SOURCES), len(CONTEXTS), len(FEATURE_NAMES))
        if features.shape != expected:
            raise ValueError(f"Unexpected feature shape {features.shape}; expected {expected}")
        chunks.append(features)
        print(
            json.dumps(
                {
                    "event": event,
                    "complete": end,
                    "total": len(identifiers),
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    features = np.concatenate(chunks)
    output = args.output_directory / "enhancer_predictions.npz"
    atomic_save_npz(
        output,
        ids=identifiers.astype(np.str_),
        chrom=chromosomes.astype(np.str_),
        summit=summits,
        split=splits.astype(np.str_),
        contexts=np.asarray(CONTEXTS),
        sources=np.asarray(SOURCES),
        feature_names=np.asarray(FEATURE_NAMES),
        hard_activity=hard_activity.astype(np.uint8),
        observed_atac_mean_512=observed_atac,
        observed_h3k27ac_segment_means_512=observed_h3_segments,
        features=features,
    )
    metadata = {
        "method": "exact_active_enhancer_three_model_inference_v1",
        "selection": {
            "active_in_at_least_one_context": True,
            "activity_definition": "catalog context membership and H3K27ac background percentile > 0.6",
            "elements": len(identifiers),
            "invalid_sequence_elements": invalid_sequence_n,
            "buffered_elements": buffered_n,
            "split_counts": {
                split: int(np.sum(splits == split))
                for split in ("train", "validation", "test")
            },
        },
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
            "method": "per-bin log1p context-residual blend before enhancer summarization",
            "alpha_atac": args.alpha_atac,
            "alpha_h3k27ac": args.alpha_h3k27ac,
            "clipped_fraction_mean": {
                assay: float(np.mean(values)) for assay, values in clipped.items()
            },
        },
        "split_definition": {
            "validation": f"{args.validation_chromosome}:0-{args.validation_end}",
            "buffer": f"{args.validation_chromosome}:{args.validation_end}-{args.training_start}",
            "test": args.test_chromosome,
            "train": "all remaining retained loci",
        },
        "inference": {
            "device": str(device),
            "mixed_precision": args.mixed_precision,
            "reverse_complement_ensemble": args.reverse_complement_ensemble,
            "batch_size": args.batch_size,
            "chunk_size": args.chunk_size,
            "elapsed_seconds": time.monotonic() - started,
        },
        "inputs": {
            "master_states": {
                "path": str(args.master_states),
                "sha256": sha256_file(args.master_states),
            },
            "reference_fasta": {
                "path": str(args.reference_fasta),
                "sha256": sha256_file(args.reference_fasta),
            },
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    atomic_write_json(output.with_suffix(".metadata.json"), metadata)
    print(
        json.dumps(
            {
                "event": "enhancer_comparison_inference_complete",
                "elements": len(identifiers),
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
