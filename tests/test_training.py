import pytest
import torch

from enhancer_pleiotropy_model.training import (
    CrestedCosineMSELogLoss,
    WarmupPlateauScheduler,
    build_loss_criteria,
    calculate_losses,
)


def test_learning_rate_warmup_cosine_transition_and_hold():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = WarmupPlateauScheduler(
        optimizer,
        maximum_learning_rate=0.1,
        post_warmup_learning_rate=0.05,
        warmup_steps=2,
        decay_steps=2,
        plateau_factor=0.5,
        plateau_patience=3,
        plateau_threshold=1e-4,
        minimum_learning_rate=1e-3,
    )
    observed = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    assert observed == pytest.approx([0.05, 0.1, 0.075, 0.05, 0.05])
    assert scheduler.step_validation(1.0)["eligible_after_scheduled_decay"] is True


def test_crested_loss_matches_reference_formula():
    labels = torch.tensor([[[1.0, 2.0], [3.0, 0.5]]])
    predictions = torch.tensor([[[1.5, 1.0], [2.0, 1.0]]], requires_grad=True)
    criterion = CrestedCosineMSELogLoss(max_weight=100, multiplier=2)
    observed = criterion.components(predictions, labels)

    transformed_labels = torch.log1p(2 * labels)
    transformed_predictions = torch.log1p(2 * predictions)
    expected_mse = torch.square(transformed_predictions - transformed_labels).mean()
    expected_weight = expected_mse.abs().clamp(1, 100)
    expected_cosine = torch.nn.functional.cosine_similarity(
        labels, predictions, dim=-1
    ).mean()
    expected_total = expected_mse - expected_weight * expected_cosine

    assert observed["mse"].item() == pytest.approx(expected_mse.item())
    assert observed["cosine_similarity"].item() == pytest.approx(
        expected_cosine.item()
    )
    assert observed["cosine_weight"].item() == pytest.approx(expected_weight.item())
    assert observed["total"].item() == pytest.approx(expected_total.item())
    observed["total"].backward()
    assert torch.isfinite(predictions.grad).all()


def test_crested_loss_zero_vectors_and_optional_mask():
    values = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    exact = CrestedCosineMSELogLoss()(values, values)
    masked = CrestedCosineMSELogLoss(minimum_target_norm=0.01)(values, values)
    assert exact.item() == pytest.approx(-0.5)
    assert masked.item() == pytest.approx(-1.0)


def test_both_assays_use_independent_crested_losses():
    training = {
        "loss": {
            "name": "crested_cosine_mse_log_both",
            "max_weight": 100,
            "minimum_target_norm": 0,
            "multipliers": {"atac": 1.0, "h3k27ac": 1.0},
        }
    }
    atac, h3k27ac, metadata = build_loss_criteria(
        training,
        h3_means=torch.zeros(8).numpy(),
        h3_standard_deviations=torch.ones(8).numpy(),
        device=torch.device("cpu"),
    )
    assert isinstance(atac, CrestedCosineMSELogLoss)
    assert isinstance(h3k27ac, CrestedCosineMSELogLoss)
    labels = (torch.ones(2, 3, 8), torch.ones(2, 2, 8))
    losses = calculate_losses(labels, labels, atac, h3k27ac)
    assert losses["atac"].item() == pytest.approx(-1.0)
    assert losses["h3k27ac"].item() == pytest.approx(-1.0)
    assert losses["total"].item() == pytest.approx(-2.0)
    assert metadata["context_axis"] == -1
