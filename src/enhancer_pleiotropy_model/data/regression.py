"""Loading prepared regression windows and profiles for PyTorch training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..constants import ATAC_TARGET_BP, H3K27AC_TARGET_BP, SPLITS
from ..io import open_text, read_json, sha256_file

COMPLEMENT = str.maketrans("ACGT", "TGCA")
ONE_HOT = np.zeros((256, 4), dtype=np.float32)
for _index, _base in enumerate("ACGT"):
    ONE_HOT[ord(_base), _index] = 1.0


@dataclass(frozen=True)
class Window:
    identifier: str
    source: str
    chrom: str
    start: int
    end: int
    split: str
    sequence: str


def read_windows(path: Path) -> dict[str, list[Window]]:
    records = {split: [] for split in SPLITS}
    with open_text(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sequence = row["sequence"].upper()
            start, end = int(row["start"]), int(row["end"])
            if end - start != len(sequence):
                raise ValueError(f"{path}: coordinate/sequence mismatch for {row['record_id']}")
            records[row["split"]].append(Window(row["record_id"], row["source"], row["chrom"],
                                                start, end, row["split"], sequence))
    if any(not values for values in records.values()):
        raise ValueError("Training, validation and test splits must be non-empty")
    return records


def load_profiles(directory: Path, windows_path: Path, counts: dict[str, int]):
    """Memory-map the per-split profile arrays after checking they match the windows."""
    metadata = read_json(Path(directory) / "profiles.metadata.json")
    if metadata["windows"]["sha256"] != sha256_file(windows_path):
        raise ValueError(f"{directory}: profiles were built from a different windows file")
    arrays = {}
    for split in SPLITS:
        array = np.load(Path(directory) / f"{split}_profiles.npy", mmap_mode="r")
        expected = (counts[split], int(metadata["bins"]), len(metadata["contexts"]))
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError(f"{directory}/{split}: expected float32 {expected}, found {array.shape}")
        arrays[split] = array
    return arrays, metadata


def pool_adjacent(profile: np.ndarray, pool_size: int) -> np.ndarray:
    if pool_size == 1:
        return np.asarray(profile, dtype=np.float32)
    return profile.reshape(profile.shape[0] // pool_size, pool_size, profile.shape[1]).mean(
        axis=1, dtype=np.float32)


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


class ProfileDataset(Dataset):
    """Windows with ATAC and pooled H3K27ac targets; optional stochastic reverse complement.

    The reverse-complement decision is a hash of (seed, epoch, index), so it is
    reproducible and independent of the data order.
    """

    def __init__(self, windows: Sequence[Window], atac: np.ndarray, h3k27ac: np.ndarray,
                 h3k27ac_pool_size: int, *, training: bool, rc_probability: float, seed: int) -> None:
        if not len(windows) == len(atac) == len(h3k27ac):
            raise ValueError("Window and profile counts differ")
        self.windows, self.atac, self.h3k27ac = windows, atac, h3k27ac
        self.pool_size = h3k27ac_pool_size
        self.training, self.rc_probability, self.seed = training, rc_probability, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.windows)

    def _reverse(self, index: int) -> bool:
        if not self.training or self.rc_probability == 0:
            return False
        digest = hashlib.sha256(f"{self.seed}\t{self.epoch}\t{index}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self.rc_probability

    def __getitem__(self, index: int):
        sequence = self.windows[index].sequence
        atac = np.array(self.atac[index], dtype=np.float32, copy=True)
        h3 = pool_adjacent(np.asarray(self.h3k27ac[index], dtype=np.float32), self.pool_size)
        if self._reverse(index):
            sequence = reverse_complement(sequence)
            atac, h3 = atac[::-1].copy(), h3[::-1].copy()
        return sequence, torch.from_numpy(atac), torch.from_numpy(h3)


class EpochShuffleSampler(Sampler[int]):
    """Epoch-specific deterministic permutation that can restart mid-epoch."""

    def __init__(self, size: int, seed: int, epoch: int, start_index: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed + epoch)
        self.indices = torch.randperm(size, generator=generator).tolist()[start_index:]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def one_hot(sequences: Sequence[str]) -> torch.Tensor:
    codes = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
    encoded = ONE_HOT[codes].reshape(len(sequences), -1, 4).transpose(0, 2, 1)
    return torch.from_numpy(np.ascontiguousarray(encoded))


def collate(items) -> dict[str, torch.Tensor]:
    """Batch tensors in the layout the regressors expect (inputs are never padded)."""
    x = one_hot([item[0] for item in items])
    length = x.shape[-1]
    mask = torch.ones((len(items), length), dtype=torch.bool)
    atac_mask, h3_mask = torch.zeros_like(mask), torch.zeros_like(mask)
    atac_start, h3_start = (length - ATAC_TARGET_BP) // 2, (length - H3K27AC_TARGET_BP) // 2
    atac_mask[:, atac_start:atac_start + ATAC_TARGET_BP] = True
    h3_mask[:, h3_start:h3_start + H3K27AC_TARGET_BP] = True
    return {"one_hot": x, "attention_mask": mask, "atac_target_mask": atac_mask,
            "h3k27ac_target_mask": h3_mask,
            "atac_labels": torch.stack([item[1] for item in items]),
            "h3k27ac_labels": torch.stack([item[2] for item in items])}


def make_loader(dataset: ProfileDataset, *, batch_size: int, epoch: int, seed: int, training: bool,
                start_batch: int = 0, pin_memory: bool = False) -> DataLoader:
    dataset.set_epoch(epoch)
    sampler = (EpochShuffleSampler(len(dataset), seed, epoch, min(start_batch * batch_size, len(dataset)))
               if training else None)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0,
                      pin_memory=pin_memory, collate_fn=collate, drop_last=False)


def profile_means(profiles: np.ndarray, pool_size: int = 1, chunk: int = 4096) -> np.ndarray:
    """Mean of every context's bins over the training split (initial output bias)."""
    sums, count = np.zeros(profiles.shape[-1], dtype=np.float64), 0
    for start in range(0, len(profiles), chunk):
        values = np.asarray(profiles[start:start + chunk], dtype=np.float32)
        if pool_size > 1:
            values = values.reshape(len(values), values.shape[1] // pool_size, pool_size,
                                    values.shape[2]).mean(axis=2)
        sums += values.sum(axis=(0, 1), dtype=np.float64)
        count += values.shape[0] * values.shape[1]
    return (sums / count).astype(np.float32)
