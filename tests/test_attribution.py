import numpy as np
import torch

from enhancer_pleiotropy_model.attribution import CalibratedTargets, dinucleotide_shuffle, integrate, one_hot, seed_for
from enhancer_pleiotropy_model.models import make_classifier, use_interleaved_dilation


def dinucleotides(codes):
    return np.bincount(codes[:-1] * 4 + codes[1:], minlength=16)


def test_shuffle_keeps_dinucleotides_and_is_seeded():
    codes = np.random.default_rng(0).integers(0, 4, 2048).astype(np.uint8)
    first = dinucleotide_shuffle(codes, seed_for(1, "DHS1", 0))
    assert np.array_equal(dinucleotides(first), dinucleotides(codes))
    assert np.array_equal(first, dinucleotide_shuffle(codes, seed_for(1, "DHS1", 0)))
    assert not np.array_equal(first, dinucleotide_shuffle(codes, seed_for(1, "DHS1", 1)))


def test_integrated_gradients_are_complete():
    torch.manual_seed(0)
    model = use_interleaved_dilation(make_classifier("cnn", "assay_heads", 1, [.3] * 8).eval()).requires_grad_(False)
    target = CalibratedTargets(model, {"a": [1.] * 8, "b": [0.] * 8, "probability_clip": 1e-7})
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 4, (1, 2048))
    x = one_hot(codes, "cpu")
    baseline = one_hot(dinucleotide_shuffle(codes[0], 3)[None], "cpu")
    result = integrate(target, x, baseline, steps=32, internal_batch=32)
    assert torch.allclose(result["actual"].sum(2), result["target_difference"], atol=2e-4)


def test_interleaved_dilation_matches_native():
    torch.manual_seed(0)
    model = make_classifier("cnn", "assay_heads", 1, [.3] * 8).eval()
    x = one_hot(np.random.default_rng(2).integers(0, 4, (2, 2048)), "cpu")
    native = model(x)
    assert torch.allclose(native, use_interleaved_dilation(model)(x), atol=1e-5)
