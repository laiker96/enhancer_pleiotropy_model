"""Load production checkpoints and run profile inference."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .constants import ATAC_TARGET_BP, CONTEXTS, H3K27AC_TARGET_BP, INPUT_BP
from .io import atomic_write_json, open_text, sha256_file
from .model import EnformerLikeJointProfileRegressor, infer_model_preset
from .sequence import one_hot_batch, reverse_complement


@dataclass(frozen=True)
class ModelMetadata:
    checkpoint: Path
    checkpoint_sha256: str
    contexts: tuple[str, ...]
    architecture: dict[str, object]
    epoch: int | None


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[EnformerLikeJointProfileRegressor, ModelMetadata]:
    """Load a released or historical production joint-profile checkpoint."""
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "enformer_like_dense_atac_h3k27ac_profile_regressor":
        raise ValueError(f"{path}: unsupported checkpoint kind")
    contexts = tuple(str(value) for value in checkpoint.get("contexts", ()))
    architecture = dict(checkpoint.get("architecture", {}))
    if not contexts or not architecture:
        raise ValueError(f"{path}: missing contexts or architecture")
    if contexts != CONTEXTS:
        raise ValueError(f"{path}: expected context order {CONTEXTS}, found {contexts}")
    decoder = dict(architecture.get("h3k27ac_decoder", {}))
    if int(decoder.get("layers", 0)) != 0:
        raise ValueError("H3K27ac decoder checkpoints are not supported in v0.1")
    cross_attention = dict(architecture.get("h3k27ac_atac_cross_attention", {}))
    if bool(cross_attention.get("enabled", False)):
        raise ValueError("Cross-attention checkpoints are not supported in v0.1")
    model = EnformerLikeJointProfileRegressor(
        context_count=len(contexts),
        model_size=infer_model_preset(architecture),
        h3k27ac_output_pool_size=int(
            architecture.get("h3k27ac_output_pool_size", 4)
        ),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    resolved = resolve_device(device) if isinstance(device, str) else device
    model.to(resolved)
    metadata = ModelMetadata(
        checkpoint=path,
        checkpoint_sha256=sha256_file(path),
        contexts=contexts,
        architecture=architecture,
        epoch=(int(checkpoint["epoch"]) if checkpoint.get("epoch") is not None else None),
    )
    return model, metadata


def centered_target_mask(
    batch_size: int, sequence_length: int, target_length: int, device: torch.device
) -> torch.Tensor:
    if target_length > sequence_length or (sequence_length - target_length) % 2:
        raise ValueError("Target must be symmetrically centered in the sequence")
    start = (sequence_length - target_length) // 2
    mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)
    mask[:, start : start + target_length] = True
    return mask


@torch.no_grad()
def predict_sequences(
    model: EnformerLikeJointProfileRegressor,
    sequences: Iterable[str],
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    reverse_complement_ensemble: bool = True,
    mixed_precision: str = "no",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict ATAC and H3K27ac profiles in forward genomic coordinates."""
    sequence_list = [sequence.upper() for sequence in sequences]
    if not sequence_list:
        raise ValueError("No sequences were provided")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be one of: no, fp16, bf16")
    if any(len(sequence) != INPUT_BP for sequence in sequence_list):
        raise ValueError(f"Every sequence must contain exactly {INPUT_BP} bases")
    resolved = resolve_device(device) if isinstance(device, str) else device
    if mixed_precision == "fp16" and resolved.type != "cuda":
        raise ValueError("FP16 inference requires CUDA")
    model.to(resolved).eval()
    atac_batches: list[np.ndarray] = []
    h3_batches: list[np.ndarray] = []
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    autocast_enabled = mixed_precision in {"fp16", "bf16"}
    for start in range(0, len(sequence_list), batch_size):
        batch_sequences = sequence_list[start : start + batch_size]
        one_hot, attention_mask = one_hot_batch(batch_sequences)
        one_hot = one_hot.to(resolved)
        attention_mask = attention_mask.to(resolved)
        atac_mask = centered_target_mask(
            len(batch_sequences), INPUT_BP, ATAC_TARGET_BP, resolved
        )
        h3_mask = centered_target_mask(
            len(batch_sequences), INPUT_BP, H3K27AC_TARGET_BP, resolved
        )
        with torch.autocast(
            device_type=resolved.type,
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            predictions = model(one_hot, attention_mask, atac_mask, h3_mask)
            if reverse_complement_ensemble:
                reverse_sequences = [
                    reverse_complement(sequence) for sequence in batch_sequences
                ]
                reverse_one_hot, reverse_attention = one_hot_batch(reverse_sequences)
                reverse_predictions = model(
                    reverse_one_hot.to(resolved),
                    reverse_attention.to(resolved),
                    atac_mask,
                    h3_mask,
                )
                predictions = tuple(
                    0.5 * (forward + reverse.flip(1))
                    for forward, reverse in zip(
                        predictions, reverse_predictions, strict=True
                    )
                )
        atac_batches.append(predictions[0].float().cpu().numpy())
        h3_batches.append(predictions[1].float().cpu().numpy())
    return np.concatenate(atac_batches), np.concatenate(h3_batches)


def read_sequence_table(path: Path) -> tuple[list[str], list[str]]:
    identifiers: list[str] = []
    sequences: list[str] = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = {"id", "sequence"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            identifiers.append(row["id"])
            sequences.append(row["sequence"])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Sequence identifiers must be unique")
    return identifiers, sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--reverse-complement-ensemble", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    identifiers, sequences = read_sequence_table(args.sequences)
    model, metadata = load_model(args.checkpoint, device)
    atac, h3k27ac = predict_sequences(
        model,
        sequences,
        batch_size=args.batch_size,
        device=device,
        reverse_complement_ensemble=args.reverse_complement_ensemble,
        mixed_precision=args.mixed_precision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=np.asarray(identifiers),
        contexts=np.asarray(metadata.contexts),
        atac=atac,
        h3k27ac=h3k27ac,
    )
    atomic_write_json(
        args.output.with_suffix(args.output.suffix + ".metadata.json"),
        {
            "checkpoint": str(metadata.checkpoint),
            "checkpoint_sha256": metadata.checkpoint_sha256,
            "contexts": list(metadata.contexts),
            "sequence_count": len(sequences),
            "reverse_complement_ensemble": args.reverse_complement_ensemble,
            "output_shapes": {
                "atac": list(atac.shape),
                "h3k27ac": list(h3k27ac.shape),
            },
        },
    )
    print(json.dumps({"event": "prediction_complete", "sequences": len(sequences)}))


if __name__ == "__main__":
    main()
