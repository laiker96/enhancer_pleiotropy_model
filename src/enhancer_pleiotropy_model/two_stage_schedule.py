"""Fast cosine to a knee, then a long cosine tail; training stops by epochs."""

import math

from .alphagenome_schedule import AlphaGenomeSchedule


SCHEDULE_NAME = "alphagenome_two_stage"


class AlphaGenomeTwoStageSchedule(AlphaGenomeSchedule):
    def __init__(self, optimizer, maximum_learning_rate=.004, *, epochs=40,
                 batches_per_epoch=11835, warmup_steps=5000, fast_decay_steps=10000,
                 knee_learning_rate=.0001, minimum_learning_rate=.000001):
        counts = (epochs, batches_per_epoch, warmup_steps, fast_decay_steps)
        rates = (maximum_learning_rate, knee_learning_rate, minimum_learning_rate)
        if (any(type(v) is not int or v < 1 for v in counts)
                or warmup_steps + fast_decay_steps >= epochs * batches_per_epoch
                or not all(math.isfinite(v) for v in rates)
                or not 0 < minimum_learning_rate < knee_learning_rate < maximum_learning_rate):
            raise ValueError("Invalid two-stage epoch schedule")
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch
        self.fast_decay_steps = fast_decay_steps
        self.knee_learning_rate = knee_learning_rate
        self.minimum_learning_rate = minimum_learning_rate
        super().__init__(optimizer, maximum_learning_rate, warmup_steps, epochs * batches_per_epoch)

    @classmethod
    def from_config(cls, optimizer, training, batches_per_epoch):
        schedule = dict(training["schedule"])
        if schedule.pop("name") != SCHEDULE_NAME:
            raise ValueError("Expected the two-stage schedule")
        # Frozen JSON is read by PyYAML, which can parse 1e-06 as text.
        for name in ("knee_learning_rate", "minimum_learning_rate"):
            if name in schedule:
                schedule[name] = float(schedule[name])
        return cls(optimizer, float(training["max_learning_rate"]),
                   epochs=training["epochs"], batches_per_epoch=batches_per_epoch, **schedule)

    @property
    def phase(self):
        if self.optimizer_steps < self.warmup_steps:
            return "warmup"
        if self.optimizer_steps < self.warmup_steps + self.fast_decay_steps:
            return "fast_decay"
        return "slow_decay"

    def _set_lr(self):
        count = self.optimizer_steps
        knee = self.warmup_steps + self.fast_decay_steps
        if count < self.warmup_steps:
            lr = self.maximum_learning_rate * count / self.warmup_steps
        elif count < knee:
            fraction = (count - self.warmup_steps) / self.fast_decay_steps
            lr = self.knee_learning_rate + .5 * (self.maximum_learning_rate - self.knee_learning_rate) * (
                1 + math.cos(math.pi * fraction))
        else:
            fraction = min(1., (count - knee) / (self.scheduled_steps - knee))
            lr = self.minimum_learning_rate + .5 * (self.knee_learning_rate - self.minimum_learning_rate) * (
                1 + math.cos(math.pi * fraction))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def state_dict(self):
        return {"name": SCHEDULE_NAME, "maximum_learning_rate": self.maximum_learning_rate,
                "knee_learning_rate": self.knee_learning_rate,
                "minimum_learning_rate": self.minimum_learning_rate,
                "warmup_steps": self.warmup_steps, "fast_decay_steps": self.fast_decay_steps,
                "epochs": self.epochs, "batches_per_epoch": self.batches_per_epoch,
                "total_steps": self.scheduled_steps, "optimizer_steps": self.optimizer_steps}

    def step_validation(self, score):
        result = super().step_validation(score)
        result["policy"] = "fixed full-epoch budget; validation does not change LR"
        return result
