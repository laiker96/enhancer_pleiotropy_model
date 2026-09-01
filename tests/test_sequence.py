import pytest
import torch

from enhancer_pleiotropy_model.sequence import (
    one_hot_batch,
    reverse_complement,
    reverse_complement_one_hot,
)


def test_reverse_complement_sequence_and_one_hot_agree():
    sequences = ["ACGTACGTACGTACGT", "TTGGCATTGGCATTGG"]
    encoded, mask = one_hot_batch(sequences)
    reversed_encoded, reversed_mask = reverse_complement_one_hot(encoded, mask)
    expected, expected_mask = one_hot_batch(
        [reverse_complement(sequence) for sequence in sequences]
    )
    assert torch.equal(reversed_encoded, expected)
    assert torch.equal(reversed_mask, expected_mask)


def test_sequence_encoding_rejects_ambiguous_bases():
    with pytest.raises(ValueError, match="non-ACGT"):
        one_hot_batch(["ACNG"])
