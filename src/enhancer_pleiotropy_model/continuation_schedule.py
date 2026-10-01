"""User-approved warmup/hold/decay extension, counting lifetime optimizer updates."""

import math

from .alphagenome_schedule import AlphaGenomeSchedule


CONTINUATION_SCHEDULE_NAME = "alphagenome_continuation"


class AlphaGenomeContinuationSchedule(AlphaGenomeSchedule):
    def __init__(self, optimizer, maximum_learning_rate=.0005, *, start_step=15000,
                 warmup_steps=1000, hold_steps=24000, total_steps=65000,
                 minimum_learning_rate=.00001):
        if (any(not isinstance(v, int) for v in (start_step, warmup_steps, hold_steps, total_steps))
                or start_step < 0 or warmup_steps < 1 or hold_steps < 0
                or total_steps <= start_step + warmup_steps + hold_steps
                or not math.isfinite(maximum_learning_rate)
                or not math.isfinite(minimum_learning_rate)
                or not 0 < minimum_learning_rate < maximum_learning_rate):
            raise ValueError("Invalid continuation warmup/hold/decay schedule")
        self.optimizer = optimizer
        self.maximum_learning_rate = maximum_learning_rate
        self.minimum_learning_rate = minimum_learning_rate
        self.start_step = start_step
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.scheduled_steps = total_steps
        self.optimizer_steps = start_step
        self._set_lr()

    @classmethod
    def from_config(cls, optimizer, training):
        schedule = dict(training["schedule"])
        if schedule.pop("name") != CONTINUATION_SCHEDULE_NAME:
            raise ValueError("Expected the continuation schedule")
        # PyYAML can read JSON scientific notation (1e-05) as a string.
        if "minimum_learning_rate" in schedule:
            schedule["minimum_learning_rate"] = float(schedule["minimum_learning_rate"])
        return cls(optimizer, float(training["max_learning_rate"]), **schedule)

    @property
    def continuation_steps(self):
        return self.optimizer_steps - self.start_step

    def _set_lr(self):
        count = self.continuation_steps
        hold_end = self.warmup_steps + self.hold_steps
        if count < self.warmup_steps:
            lr = self.maximum_learning_rate * count / self.warmup_steps
        elif count < hold_end:
            lr = self.maximum_learning_rate
        else:
            fraction = min(1., (count - hold_end) / (self.scheduled_steps - self.start_step - hold_end))
            lr = self.minimum_learning_rate + .5 * (
                self.maximum_learning_rate - self.minimum_learning_rate) * (1 + math.cos(math.pi * fraction))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def state_dict(self):
        return {"name": CONTINUATION_SCHEDULE_NAME, "maximum_learning_rate": self.maximum_learning_rate,
                "minimum_learning_rate": self.minimum_learning_rate, "start_step": self.start_step,
                "warmup_steps": self.warmup_steps, "hold_steps": self.hold_steps,
                "total_steps": self.scheduled_steps, "optimizer_steps": self.optimizer_steps}

    def load_state_dict(self, state):
        expected, observed = self.state_dict(), dict(state)
        expected.pop("optimizer_steps")
        count = observed.pop("optimizer_steps")
        if expected != observed or not isinstance(count, int) or not self.start_step <= count <= self.scheduled_steps:
            raise ValueError("Continuation schedule configuration or step count changed")
        self.optimizer_steps = count
        self._set_lr()
