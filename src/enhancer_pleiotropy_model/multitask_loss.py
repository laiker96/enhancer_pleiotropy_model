"""Signal-anchored breadth/contrast loss and reproducible targeted sampling."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .breadth_metrics import GROUP_INDICES
from .constants import CONTEXTS
from .master_element_calibration import calibrate


ASSAYS = ("atac", "h3k27ac")
CLOSE_GROUPS = tuple(indices for indices in GROUP_INDICES if len(indices) > 1)
PAIRS = tuple((a, b) for group in CLOSE_GROUPS for i, a in enumerate(group) for b in group[i + 1:])
PAIR_GROUP = tuple(g for g, group in enumerate(CLOSE_GROUPS) for i, a in enumerate(group) for b in group[i + 1:])
PAIR_WEIGHTS = tuple(1 / (3 * PAIR_GROUP.count(g)) for g in PAIR_GROUP)


def summarize_profiles(values: torch.Tensor, assay: str) -> torch.Tensor:
    expected_bins = 32 if assay == "atac" else 24
    if assay not in ASSAYS or values.ndim != 3 or values.shape[1:] != (expected_bins, 8):
        raise ValueError("Expected production [batch, bins, 8] assay geometry")
    values = values.float()
    if assay == "atac":
        return values.mean(dim=1)
    return values.reshape(len(values), 3, 8, 8).mean(dim=2).amax(dim=1)


def summarize_numpy(atac: np.ndarray, h3: np.ndarray) -> np.ndarray:
    """Summaries from prepared 16-bp arrays or pooled 64-bp H3 predictions."""
    if atac.shape[1:] != (32, 8) or h3.shape[1:] not in ((96, 8), (24, 8)):
        raise ValueError("Unexpected profile geometry")
    if len(atac) != len(h3) or any(not np.isfinite(x).all() or np.any(x < 0) for x in (atac, h3)):
        raise ValueError("Profiles must be aligned, finite and nonnegative")
    return np.stack((atac.mean(axis=1, dtype=np.float64),
                     h3.reshape(len(h3), 3, -1, 8).mean(axis=2, dtype=np.float64).max(axis=1)))


class ActivityInterpolation(nn.Module):
    """Frozen monotonic interpolation in log1p signal, with empirical tails."""

    def __init__(self, background: np.ndarray, knots: int = 4097):
        super().__init__()
        if knots < 3:
            raise ValueError("At least three interpolation knots required")
        # Also validates nonnegative, finite, sorted, nonempty reference columns.
        calibrate(np.zeros((1, 8)), background)
        self.knots = knots
        self.max_reference_error = 0.0
        for c in range(8):
            reference = background[:, c].astype(float)
            # Real reference values avoid artificial rank plateaus between knots
            # when a small/discrete background has fewer unique values than knots.
            raw = np.unique(np.quantile(reference, np.linspace(0, 1, knots), method="nearest"))
            if len(raw) < 2:
                raise ValueError("Each context needs at least two distinct background values")
            excess = np.maximum(0, (np.searchsorted(reference, raw, "left") + np.searchsorted(reference, raw, "right") + 1) / (len(reference) + 1) - 1)
            self.register_buffer(f"x_{c}", torch.tensor(np.log1p(raw), dtype=torch.float32))
            self.register_buffer(f"y_{c}", torch.tensor(excess, dtype=torch.float32))
            actual = np.maximum(0, (np.searchsorted(reference, reference, "left") + np.searchsorted(reference, reference, "right") + 1) / (len(reference) + 1) - 1)
            self.max_reference_error = max(self.max_reference_error, float(np.max(np.abs(np.interp(np.log1p(reference), np.log1p(raw), excess) - actual))))
        self.upper_tail = 1 - 1 / (len(background) + 1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 2 or signal.shape[1] != 8:
            raise ValueError("Activity signal must be [batch, 8]")
        z = torch.log1p(signal.float().clamp_min(0))
        columns = []
        for c in range(8):
            x, y = getattr(self, f"x_{c}"), getattr(self, f"y_{c}")
            index = torch.searchsorted(x, z[:, c].contiguous()).clamp(1, len(x) - 1)
            fraction = ((z[:, c] - x[index - 1]) / (x[index] - x[index - 1])).clamp(0, 1)
            value = y[index - 1] + fraction * (y[index] - y[index - 1])
            value = torch.where(z[:, c] < x[0], torch.zeros_like(value), value)
            value = torch.where(z[:, c] > x[-1], torch.full_like(value, self.upper_tail), value)
            columns.append(value)
        return torch.stack(columns, dim=1)


class JointBreadthLoss(nn.Module):
    def __init__(self, backgrounds: dict[str, np.ndarray], *, breadth_weight: float,
                 group_weight: float, contrast_weight: float, knots: int = 4097):
        super().__init__()
        weights = (breadth_weight, group_weight, contrast_weight)
        if not all(np.isfinite(w) and w >= 0 for w in weights):
            raise ValueError("Loss weights must be finite and nonnegative")
        self.weights = weights
        self.activity = nn.ModuleDict({a: ActivityInterpolation(backgrounds[a], knots) for a in ASSAYS})
        self.register_buffer("pair_weights", torch.tensor(PAIR_WEIGHTS, dtype=torch.float32))

    def forward(self, predictions: tuple[torch.Tensor, ...], labels: tuple[torch.Tensor, ...],
                *, natural_n: int | None = None, targeted_assay: torch.Tensor | None = None,
                targeted_pair: torch.Tensor | None = None, contrast_ramp: float = 1.0) -> dict[str, torch.Tensor]:
        n = len(predictions[0])
        natural_n = n if natural_n is None else natural_n
        if len(predictions) != 2 or len(labels) != 2 or not 0 < natural_n <= n or not 0 <= contrast_ramp <= 1:
            raise ValueError("Invalid assay count, representative batch size or contrast ramp")
        if natural_n < n and (targeted_assay is None or targeted_pair is None or targeted_assay.shape != (n - natural_n,) or targeted_pair.shape != (n - natural_n,)):
            raise ValueError("Targeted examples require aligned assay and pair identifiers")
        if natural_n < n and (torch.any((targeted_assay < 0) | (targeted_assay > 1)) or torch.any((targeted_pair < 0) | (targeted_pair >= len(PAIRS)))):
            raise ValueError("Targeted assay/pair identifier outside range")
        losses, contrasts = {}, []
        for assay, pred, truth in zip(ASSAYS, predictions, labels, strict=True):
            if pred.shape != truth.shape or len(pred) != n:
                raise ValueError("Predicted and measured profile arrays must align")
            predicted_summary, summary = summarize_profiles(pred, assay), summarize_profiles(truth, assay)
            losses[f"{assay}_signal"] = F.mse_loss(torch.log1p(pred[:natural_n].float().clamp_min(0)), torch.log1p(truth[:natural_n].float()))
            activity, target = self.activity[assay](predicted_summary[:natural_n]), self.activity[assay](summary[:natural_n])
            difference = activity - target
            losses[f"{assay}_breadth"] = difference.mean(dim=1).square().mean()
            losses[f"{assay}_groups"] = torch.stack([difference[:, indices].mean(dim=1).square().mean() for indices in GROUP_INDICES]).mean()
            log_error = torch.log1p(predicted_summary) - torch.log1p(summary)
            pair_error = torch.stack([log_error[:, a] - log_error[:, b] for a, b in PAIRS], dim=1)
            pair_loss = F.smooth_l1_loss(pair_error, torch.zeros_like(pair_error), reduction="none", beta=1.0)
            losses[f"{assay}_contrast"] = (pair_loss[:natural_n] * self.pair_weights).sum(dim=1).mean()
            contrasts.append(pair_loss)
        for component in ("signal", "breadth", "groups", "contrast"):
            losses[component] = torch.stack([losses[f"{a}_{component}"] for a in ASSAYS]).mean()
        if natural_n < n:
            pair_losses = torch.stack(contrasts, dim=1)[natural_n:]
            targeted = pair_losses[torch.arange(n - natural_n, device=pair_losses.device), targeted_assay, targeted_pair].mean()
            losses["targeted_contrast"] = targeted
            losses["contrast"] = 0.5 * (losses["contrast"] + targeted)
        b, g, d = self.weights
        losses["total"] = losses["signal"] + b * losses["breadth"] + g * losses["groups"] + d * contrast_ramp * losses["contrast"]
        return losses


def build_target_pools(summaries: np.ndarray, activity: np.ndarray, peak_mask: np.ndarray) -> tuple[dict, dict]:
    """Training-only assay/group/pair/direction/breadth strata, not hard catalog calls."""
    if summaries.shape != activity.shape or summaries.shape != (2, len(peak_mask), 8) or not peak_mask.any():
        raise ValueError("Require aligned training summaries and nonempty peak selection")
    pools, metadata = {}, {}
    for a, assay in enumerate(ASSAYS):
        edges = np.quantile(activity[a, peak_mask].sum(axis=1), [0.25, 0.5, 0.75])
        breadth = np.searchsorted(edges, activity[a].sum(axis=1), side="right")
        metadata[assay] = dict(breadth_edges=edges.tolist(), pairs={})
        z = np.log1p(summaries[a])
        for p, (first, second) in enumerate(PAIRS):
            delta = z[:, first] - z[:, second]
            low, high = np.quantile(np.abs(delta[peak_mask]), [0.25, 0.75])
            metadata[assay]["pairs"][f"{CONTEXTS[first]}-{CONTEXTS[second]}"] = dict(near_equal_max=float(low), strong_min=float(high))
            for direction, mask in ((-1, (delta < 0) & (np.abs(delta) >= high)), (0, np.abs(delta) <= low), (1, (delta > 0) & (np.abs(delta) >= high))):
                for quartile in range(4):
                    indices = np.flatnonzero(peak_mask & mask & (breadth == quartile))
                    if len(indices):
                        pools[(a, PAIR_GROUP[p], p, direction, quartile)] = indices
    metadata["pool_counts"] = {",".join(map(str, key)): len(values) for key, values in pools.items()}
    return pools, metadata


def mixed_batch_plan(n: int, pools: dict, *, steps: int, natural_batch: int, targeted_batch: int,
                     seed: int, epoch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Natural sampling without replacement; hierarchical targeted sampling with replacement."""
    if min(steps, natural_batch, targeted_batch) < 1 or steps * natural_batch > n or not pools:
        raise ValueError("Invalid batch sizes, empty pools or more natural draws than training rows")
    rng = np.random.default_rng(seed + epoch)
    natural = rng.permutation(n)[:steps * natural_batch].reshape(steps, natural_batch)
    tree = {}
    for key, values in sorted(pools.items()):
        node = tree
        for component in key[:-1]:
            node = node.setdefault(component, {})
        if np.any(values < 0) or np.any(values >= n):
            raise ValueError("Targeted pool contains out-of-range training indices")
        node[key[-1]] = values
    targeted = np.empty((steps, targeted_batch), dtype=np.int64)
    assays, pairs = np.empty_like(targeted), np.empty_like(targeted)
    for index in np.ndindex(targeted.shape):
        node, key = tree, []
        for _ in range(5):
            choices = tuple(node)
            choice = choices[rng.integers(len(choices))]
            key.append(choice)
            node = node[choice]
        targeted[index] = node[rng.integers(len(node))]
        assays[index], pairs[index] = key[0], key[2]
    return np.concatenate((natural, targeted), axis=1), assays, pairs
