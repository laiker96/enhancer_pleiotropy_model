"""Prepared dataset validation and deterministic PyTorch loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .constants import ATAC_TARGET_BP, H3K27AC_TARGET_BP
from .io import open_text, sha256_file
from .sequence import one_hot_batch, reverse_complement


SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class WindowRecord:
    identifier: str
    source: str
    chrom: str
    start: int
    end: int
    target_start: int
    target_end: int
    block_id: str
    split: str
    sequence: str


def read_windows(path: Path) -> dict[str, list[WindowRecord]]:
    records = {split: [] for split in SPLITS}
    required = {
        "record_id",
        "source",
        "chrom",
        "start",
        "end",
        "target_start",
        "target_end",
        "block_id",
        "split",
        "sequence",
    }
    block_splits: dict[str, str] = {}
    genomic_intervals: dict[str, list[tuple[int, int, str]]] = {}
    identifiers: set[str] = set()
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            split = row["split"]
            if split not in records:
                raise ValueError(f"{path}:{line_number}: unsupported split {split}")
            identifier = row["record_id"]
            if identifier in identifiers:
                raise ValueError(f"{path}:{line_number}: duplicate record {identifier}")
            identifiers.add(identifier)
            start, end = int(row["start"]), int(row["end"])
            target_start, target_end = int(row["target_start"]), int(row["target_end"])
            sequence = row["sequence"].upper()
            if start < 0 or end - start != len(sequence):
                raise ValueError(f"{path}:{line_number}: coordinate/sequence mismatch")
            if not start <= target_start < target_end <= end:
                raise ValueError(f"{path}:{line_number}: target outside sequence")
            if target_end - target_start != ATAC_TARGET_BP:
                raise ValueError(f"{path}:{line_number}: ATAC target is not 512 bp")
            expected_target_start = start + (len(sequence) - ATAC_TARGET_BP) // 2
            if target_start != expected_target_start:
                raise ValueError(
                    f"{path}:{line_number}: ATAC target is not centered in the sequence"
                )
            previous = block_splits.setdefault(row["block_id"], split)
            if previous != split:
                raise ValueError(f"Block {row['block_id']} occurs in multiple splits")
            genomic_intervals.setdefault(row["chrom"], []).append((start, end, split))
            records[split].append(
                WindowRecord(
                    identifier=identifier,
                    source=row["source"],
                    chrom=row["chrom"],
                    start=start,
                    end=end,
                    target_start=target_start,
                    target_end=target_end,
                    block_id=row["block_id"],
                    split=split,
                    sequence=sequence,
                )
            )
    if any(not values for values in records.values()):
        raise ValueError("Training, validation, and test splits must be non-empty")
    for chrom, intervals in genomic_intervals.items():
        maximum_end_by_split = {split: -1 for split in SPLITS}
        for start, end, split in sorted(intervals):
            if any(
                maximum_end_by_split[other_split] > start
                for other_split in SPLITS
                if other_split != split
            ):
                raise ValueError(
                    f"Input windows from different splits overlap on {chrom}"
                )
            maximum_end_by_split[split] = max(maximum_end_by_split[split], end)
    sequence_lengths = {
        len(record.sequence) for values in records.values() for record in values
    }
    if len(sequence_lengths) != 1:
        raise ValueError(f"Sequences do not have one fixed length: {sequence_lengths}")
    return records


def load_profile_metadata(directory: Path) -> dict[str, object]:
    path = directory / "profiles.metadata.json"
    metadata = json.loads(path.read_text())
    required = {"bin_size_bp", "bins_per_target", "contexts", "dataset", "outputs"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"{path}: missing metadata keys {sorted(missing)}")
    return metadata


def load_profiles(
    directory: Path,
    dataset_path: Path,
    expected_counts: dict[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    metadata = load_profile_metadata(directory)
    expected_hash = sha256_file(dataset_path)
    if metadata["dataset"]["sha256"] != expected_hash:
        raise ValueError(f"{directory}: profile dataset hash does not match windows")
    arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        path = directory / f"{split}_profiles.npy"
        expected_output = metadata["outputs"].get(split, {})
        if expected_output.get("sha256") != sha256_file(path):
            raise ValueError(f"{path}: profile hash does not match metadata")
        array = np.load(path, mmap_mode="r")
        expected_shape = (
            expected_counts[split],
            int(metadata["bins_per_target"]),
            len(metadata["contexts"]),
        )
        if array.shape != expected_shape or array.dtype != np.float32:
            raise ValueError(
                f"{path}: expected float32 {expected_shape}, found {array.dtype} {array.shape}"
            )
        arrays[split] = array
    return arrays, metadata


def pool_adjacent(profile: np.ndarray, pool_size: int) -> np.ndarray:
    if profile.shape[0] % pool_size:
        raise ValueError("Profile bins are not divisible by pool size")
    if pool_size == 1:
        return np.asarray(profile, dtype=np.float32)
    return profile.reshape(
        profile.shape[0] // pool_size, pool_size, profile.shape[1]
    ).mean(axis=1, dtype=np.float32)


class JointProfileDataset(Dataset):
    def __init__(
        self,
        records: Sequence[WindowRecord],
        atac: np.ndarray,
        h3k27ac: np.ndarray,
        h3k27ac_pool_size: int,
        *,
        training: bool,
        rc_probability: float,
        seed: int,
        indices: Sequence[int] | None = None,
    ) -> None:
        if not (len(records) == len(atac) == len(h3k27ac)):
            raise ValueError("Record and profile counts differ")
        self.records = records
        self.atac = atac
        self.h3k27ac = h3k27ac
        self.h3k27ac_pool_size = h3k27ac_pool_size
        self.training = training
        self.rc_probability = rc_probability
        self.seed = seed
        self.epoch = 0
        self.indices = (
            tuple(range(len(records)))
            if indices is None
            else tuple(int(index) for index in indices)
        )
        if any(index < 0 or index >= len(records) for index in self.indices):
            raise ValueError("Dataset subset index is outside the prepared records")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def _reverse(self, index: int) -> bool:
        if not self.training or self.rc_probability == 0:
            return False
        digest = hashlib.sha256(
            f"{self.seed}\t{self.epoch}\t{index}".encode()
        ).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        return fraction < self.rc_probability

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, torch.Tensor, bool]:
        source_index = self.indices[index]
        sequence = self.records[source_index].sequence
        atac = np.array(self.atac[source_index], dtype=np.float32, copy=True)
        h3k27ac = pool_adjacent(
            np.asarray(self.h3k27ac[source_index], dtype=np.float32),
            self.h3k27ac_pool_size,
        )
        reverse = self._reverse(index)
        if reverse:
            sequence = reverse_complement(sequence)
            atac = atac[::-1].copy()
            h3k27ac = h3k27ac[::-1].copy()
        return sequence, torch.from_numpy(atac), torch.from_numpy(h3k27ac), reverse


class EpochShuffleSampler(Sampler[int]):
    """Epoch-specific deterministic permutation with restart offset."""

    def __init__(self, size: int, seed: int, epoch: int, start_index: int = 0) -> None:
        if not 0 <= start_index <= size:
            raise ValueError("Sampler restart offset is outside the dataset")
        generator = torch.Generator().manual_seed(seed + epoch)
        self.indices = torch.randperm(size, generator=generator).tolist()[start_index:]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class JointProfileCollator:
    def __call__(self, items: list[tuple[str, torch.Tensor, torch.Tensor, bool]]) -> dict[str, torch.Tensor]:
        sequences = [item[0] for item in items]
        one_hot, attention_mask = one_hot_batch(sequences)
        sequence_length = one_hot.shape[-1]
        atac_mask = torch.zeros_like(attention_mask)
        h3_mask = torch.zeros_like(attention_mask)
        atac_start = (sequence_length - ATAC_TARGET_BP) // 2
        h3_start = (sequence_length - H3K27AC_TARGET_BP) // 2
        atac_mask[:, atac_start : atac_start + ATAC_TARGET_BP] = True
        h3_mask[:, h3_start : h3_start + H3K27AC_TARGET_BP] = True
        return {
            "one_hot": one_hot,
            "attention_mask": attention_mask,
            "atac_target_mask": atac_mask,
            "h3k27ac_target_mask": h3_mask,
            "atac_labels": torch.stack([item[1] for item in items]),
            "h3k27ac_labels": torch.stack([item[2] for item in items]),
        }


def make_loader(
    dataset: JointProfileDataset,
    *,
    batch_size: int,
    workers: int,
    epoch: int,
    seed: int,
    training: bool,
    start_batch: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset.set_epoch(epoch)
    sampler = (
        EpochShuffleSampler(
            len(dataset),
            seed,
            epoch,
            start_index=min(start_batch * batch_size, len(dataset)),
        )
        if training
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=JointProfileCollator(),
        drop_last=False,
    )


def streamed_h3_log_statistics(
    profiles: np.ndarray, pool_size: int, chunk_size: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(profiles.shape[-1], dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    count = 0
    for start in range(0, len(profiles), chunk_size):
        chunk = np.asarray(profiles[start : start + chunk_size], dtype=np.float32)
        if pool_size > 1:
            chunk = chunk.reshape(
                len(chunk), chunk.shape[1] // pool_size, pool_size, chunk.shape[2]
            ).mean(axis=2)
        transformed = np.log1p(chunk.astype(np.float64))
        sums += transformed.sum(axis=(0, 1))
        sums_sq += np.square(transformed).sum(axis=(0, 1))
        count += transformed.shape[0] * transformed.shape[1]
    means = sums / count
    variances = np.maximum(sums_sq / count - np.square(means), 0.0)
    standard_deviations = np.sqrt(variances)
    if np.any(~np.isfinite(means)) or np.any(standard_deviations <= 0):
        raise ValueError("H3K27ac training standardization is invalid")
    return means, standard_deviations


def streamed_profile_means(
    profiles: np.ndarray, pool_size: int = 1, chunk_size: int = 4096
) -> np.ndarray:
    sums = np.zeros(profiles.shape[-1], dtype=np.float64)
    count = 0
    for start in range(0, len(profiles), chunk_size):
        chunk = np.asarray(profiles[start : start + chunk_size], dtype=np.float32)
        if pool_size > 1:
            chunk = chunk.reshape(
                len(chunk), chunk.shape[1] // pool_size, pool_size, chunk.shape[2]
            ).mean(axis=2)
        sums += chunk.sum(axis=(0, 1), dtype=np.float64)
        count += chunk.shape[0] * chunk.shape[1]
    return (sums / count).astype(np.float32)
