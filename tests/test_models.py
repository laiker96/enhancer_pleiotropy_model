import pytest
import torch

from enhancer_pleiotropy_model.data.regression import collate
from enhancer_pleiotropy_model.models import (ARCHITECTURES, MANUSCRIPT_READOUT, load_classifier, make_classifier,
                                              make_regressor)
from enhancer_pleiotropy_model.training.common import atomic_torch_save

PREVALENCE = [.3] * 8


def batch(n=2):
    torch.manual_seed(0)
    sequences = ["".join("ACGT"[i] for i in torch.randint(0, 4, (2048,)).tolist()) for _ in range(n)]
    return collate([(s, torch.zeros(32, 8), torch.zeros(24, 8)) for s in sequences])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_regressor_output_shapes(architecture):
    b = batch()
    model = make_regressor(architecture, {"atac": [1.0] * 8, "h3k27ac": [2.0] * 8}).eval()
    atac, h3 = model(b["one_hot"], b["attention_mask"], b["atac_target_mask"], b["h3k27ac_target_mask"])
    assert atac.shape == (2, 32, 8) and h3.shape == (2, 24, 8)
    assert torch.all(atac >= 0) and torch.all(h3 >= 0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_classifier_outputs_eight_logits(architecture):
    model = make_classifier(architecture, MANUSCRIPT_READOUT[architecture], 1, PREVALENCE).eval()
    assert model(batch()["one_hot"]).shape == (2, 8)


def test_manuscript_classifier_parameter_count():
    model = make_classifier("cnn", "assay_heads", 20260914, PREVALENCE)
    assert sum(p.numel() for p in model.parameters()) == 1_244_168


def test_scratch_and_pretrained_heads_start_identical(tmp_path):
    from enhancer_pleiotropy_model.constants import CONTEXTS
    regressor = make_regressor("cnn")
    scratch = make_classifier("cnn", "assay_heads", 5, PREVALENCE)
    atomic_torch_save({"architecture": "cnn", "contexts": list(CONTEXTS), "state_dict": regressor.state_dict()},
                      tmp_path / "parent.pt")
    pretrained = make_classifier("cnn", "assay_heads", 5, PREVALENCE, tmp_path / "parent.pt")
    assert torch.equal(scratch.head[0].weight, pretrained.head[0].weight)


def test_classifier_checkpoint_round_trip(tmp_path):
    model = make_classifier("attention", "new_head", 3, PREVALENCE).eval()
    atomic_torch_save({"kind": "classifier", "settings": {"architecture": "attention", "readout": "new_head"},
                       "state_dict": model.state_dict()}, tmp_path / "best_model.pt")
    loaded, _ = load_classifier(tmp_path / "best_model.pt")
    x = batch()["one_hot"]
    assert torch.allclose(model(x), loaded(x))


def test_assay_heads_rejected_for_dense():
    with pytest.raises(ValueError):
        make_classifier("dense", "assay_heads", 1, PREVALENCE)


def test_architecture_matches_docs_figure():
    """Every dimension drawn in docs/figures/best_model_architecture.svg."""
    model = make_classifier("cnn", "assay_heads", 1, PREVALENCE).eval()
    x = batch()["one_hot"]
    hidden, mask = x, torch.ones(2, 2048, dtype=torch.bool)
    shapes, kernels = [], []
    for block in model.convolutional_body.blocks:
        hidden, mask = block(hidden, mask)
        shapes.append(tuple(hidden.shape[1:][::-1]))
        kernels.append(block.convolution.kernel_size[0])
    assert shapes == [(1024, 96), (512, 128), (256, 160), (128, 192)]
    assert kernels == [15, 5, 5, 5]
    assert [b.conv.dilation[0] for b in model.transformer] == [1, 2, 4, 8, 16, 32]
    assert all(b.conv.kernel_size[0] == 3 for b in model.transformer)
    assert model.atac_features[1].out_features == model.h3k27ac_features[1].out_features == 384
    assert (model.head[0].in_features, model.head[0].out_features) == (1536, 8)
