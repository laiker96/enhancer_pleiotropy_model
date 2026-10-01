"""Paper pretraining schedule: 0 -> .004 in 5000, cosine -> 0 at 15000.

Counts successful optimizer updates. At zero completed updates the next LR is
zero, following a zero-indexed schedule; after 5000 it is the peak. This is not
AlphaGenome's separate distillation schedule. Parameters permit tiny CPU tests.
"""

import math


SCHEDULE_NAME = "alphagenome_pretraining"


class AlphaGenomeSchedule:
    def __init__(self, optimizer, maximum_learning_rate=.004, warmup_steps=5000, total_steps=15000):
        if not math.isfinite(maximum_learning_rate) or maximum_learning_rate <= 0 or not 0 < warmup_steps < total_steps:
            raise ValueError("Invalid AlphaGenome warmup/cosine schedule")
        self.optimizer = optimizer
        self.maximum_learning_rate = maximum_learning_rate
        self.warmup_steps = warmup_steps
        self.scheduled_steps = total_steps
        self.optimizer_steps = 0
        self._set_lr()

    def _set_lr(self):
        count = self.optimizer_steps
        if count < self.warmup_steps:
            lr = self.maximum_learning_rate * count / self.warmup_steps
        else:
            fraction = min(1., (count - self.warmup_steps) / (self.scheduled_steps - self.warmup_steps))
            lr = self.maximum_learning_rate * .5 * (1 + math.cos(math.pi * fraction))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self):
        self.optimizer_steps += 1
        self._set_lr()

    def step_validation(self, score):
        lr = self.optimizer.param_groups[0]["lr"]
        return {"score": score, "eligible_after_scheduled_decay": False,
                "learning_rate_before": lr, "learning_rate_after": lr, "reduced": False,
                "policy": "fixed step budget; validation does not change LR"}

    def state_dict(self):
        return {"name": SCHEDULE_NAME, "maximum_learning_rate": self.maximum_learning_rate,
                "warmup_steps": self.warmup_steps, "total_steps": self.scheduled_steps,
                "optimizer_steps": self.optimizer_steps}

    def load_state_dict(self, state):
        expected, observed = self.state_dict(), dict(state)
        expected.pop("optimizer_steps")
        count = observed.pop("optimizer_steps")
        if expected != observed or not isinstance(count, int) or not 0 <= count <= self.scheduled_steps:
            raise ValueError("AlphaGenome schedule configuration or step count changed")
        self.optimizer_steps = count
        self._set_lr()
