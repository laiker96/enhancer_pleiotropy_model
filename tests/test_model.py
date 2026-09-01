import torch

from enhancer_pleiotropy_model.constants import CONTEXTS, INPUT_BP
from enhancer_pleiotropy_model.inference import centered_target_mask, load_model
from enhancer_pleiotropy_model.model import EnformerLikeJointProfileRegressor
from enhancer_pleiotropy_model.sequence import one_hot_batch
from enhancer_pleiotropy_model.training import architecture_metadata


def model_inputs():
    one_hot, attention_mask = one_hot_batch(["ACGT" * (INPUT_BP // 4)])
    atac_mask = centered_target_mask(1, INPUT_BP, 512, torch.device("cpu"))
    h3_mask = centered_target_mask(1, INPUT_BP, 1536, torch.device("cpu"))
    return one_hot, attention_mask, atac_mask, h3_mask


def test_default_4x_parameter_count_is_frozen():
    model = EnformerLikeJointProfileRegressor(model_size="4x")
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_358_704


def test_base_model_output_geometry():
    model = EnformerLikeJointProfileRegressor(model_size="base").eval()
    with torch.no_grad():
        atac, h3k27ac = model(*model_inputs())
    assert atac.shape == (1, 32, 8)
    assert h3k27ac.shape == (1, 24, 8)
    assert torch.all(atac > 0)
    assert torch.all(h3k27ac > 0)


def test_checkpoint_round_trip_uses_strict_loader(tmp_path):
    model = EnformerLikeJointProfileRegressor(model_size="base")
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "kind": "enformer_like_dense_atac_h3k27ac_profile_regressor",
            "version": 1,
            "state_dict": model.state_dict(),
            "architecture": architecture_metadata(model, CONTEXTS),
            "contexts": CONTEXTS,
            "epoch": 1,
        },
        checkpoint,
    )
    restored, metadata = load_model(checkpoint)
    assert metadata.contexts == CONTEXTS
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in model.state_dict().items()
    )
