"""DNA encoding and reverse-complement utilities."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from .constants import COMPLEMENT_INDICES, DNA_ALPHABET


COMPLEMENT = str.maketrans("ACGT", "TGCA")
ONE_HOT_LOOKUP = np.zeros((256, 4), dtype=np.float32)
for _index, _base in enumerate("ACGT"):
    ONE_HOT_LOOKUP[ord(_base), _index] = 1.0


def reverse_complement(sequence: str) -> str:
    sequence = sequence.upper()
    if set(sequence) - DNA_ALPHABET:
        raise ValueError("Sequence contains a non-ACGT base")
    return sequence.translate(COMPLEMENT)[::-1]


def one_hot_batch(
    sequences: Sequence[str], minimum_multiple: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Cannot encode an empty sequence batch")
    maximum = max(map(len, sequences))
    padded = math.ceil(maximum / minimum_multiple) * minimum_multiple
    encoded = np.zeros((len(sequences), 4, padded), dtype=np.float32)
    mask = np.zeros((len(sequences), padded), dtype=np.bool_)
    for index, sequence in enumerate(sequences):
        sequence = sequence.upper()
        if not sequence or set(sequence) - DNA_ALPHABET:
            raise ValueError(f"Sequence {index} contains non-ACGT bases")
        codes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        encoded[index, :, : len(sequence)] = ONE_HOT_LOOKUP[codes].T
        mask[index, : len(sequence)] = True
    return torch.from_numpy(encoded), torch.from_numpy(mask)


def reverse_complement_one_hot(
    one_hot: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        one_hot[:, COMPLEMENT_INDICES].flip(-1),
        attention_mask.flip(-1),
    )
