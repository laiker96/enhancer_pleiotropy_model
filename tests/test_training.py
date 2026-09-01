import pytest
import torch

from enhancer_pleiotropy_model.training import WarmupPlateauScheduler


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
