"""Validation-only breadth monitoring with frozen training-background references."""

from __future__ import annotations

import hashlib

import numpy as np

from .breadth_metrics import GROUPS, group_counts
from .constants import ASSAYS
from .continuous_breadth import evaluate_activity, saturation_metrics
from .master_element_calibration import calibrate, fit_empirical_background
from .multitask_loss import summarize_numpy


def activity_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    result = evaluate_activity(observed, predicted)

    def r2(truth, estimate):
        variance = np.square(truth - truth.mean()).sum()
        return float(1 - np.square(estimate - truth).sum() / variance) if variance > 0 else float("nan")

    result["total"]["r2"] = r2(observed.sum(axis=1), predicted.sum(axis=1))
    for i, row in enumerate(result["contexts"]):
        row["r2"] = r2(observed[:, i], predicted[:, i])
    truth_groups, predicted_groups = group_counts(observed), group_counts(predicted)
    for i, row in enumerate(result["groups"]):
        row["r2"] = r2(truth_groups[:, i], predicted_groups[:, i])
    return result


class TrainingBreadthMonitor:
    def __init__(self, records, atac, h3, chunk_size: int = 4096):
        if not len(records) or len(atac) != len(records) or len(h3) != len(records):
            raise ValueError("Breadth reference training records and profiles must align")
        if any(r.split != "train" for r in records):
            raise ValueError("Breadth references must use training records only")
        background = np.asarray([r.source == "genomic_background" for r in records])
        if not background.any() or background.all():
            raise ValueError("Breadth monitoring needs training background and regulatory windows")
        summaries = np.empty((2, len(records), 8), np.float64)
        for start in range(0, len(records), chunk_size):
            stop = min(start + chunk_size, len(records))
            summaries[:, start:stop] = summarize_numpy(atac[start:stop], h3[start:stop])
        self.references = {a: fit_empirical_background(summaries[i, background]) for i, a in enumerate(ASSAYS)}
        masks = {"all": np.ones(len(records), bool), "regulatory": ~background, "background": background}
        self.baselines = {}
        for i, assay in enumerate(ASSAYS):
            activity = calibrate(summaries[i], self.references[assay])[1]
            self.baselines[assay] = {name: activity[mask].mean(axis=0) for name, mask in masks.items()}
        self.metadata = {
            "fit_split": "train", "background_n": int(background.sum()),
            "definition": "sum of eight max(0, 2 * empirical training-background midrank percentile - 1)",
            "summaries": {"atac": "central 512-bp mean", "h3k27ac": "maximum of three 512-bp means"},
            "groups": GROUPS, "used_in_loss_or_checkpoint_selection": False,
            "reference_sha256": {a: hashlib.sha256(r.tobytes()).hexdigest() for a, r in self.references.items()},
            "constant_context_activities": {a: {c: v.tolist() for c, v in d.items()} for a, d in self.baselines.items()},
        }

    def evaluate(self, labels, predictions, regulatory_mask):
        observed = summarize_numpy(labels["atac"], labels["h3k27ac"])
        predicted = summarize_numpy(predictions["atac"], predictions["h3k27ac"])
        if observed.shape != predicted.shape or regulatory_mask.shape != (observed.shape[1],):
            raise ValueError("Breadth validation arrays and masks must align")
        masks = {"all": np.ones(len(regulatory_mask), bool),
                 "regulatory": regulatory_mask, "background": ~regulatory_mask}
        report = {"reference": self.metadata}
        for i, assay in enumerate(ASSAYS):
            truth = calibrate(observed[i], self.references[assay])[1]
            estimate = calibrate(predicted[i], self.references[assay])[1]
            report[assay] = {}
            for cohort, mask in masks.items():
                if not mask.any():
                    continue
                metrics = activity_metrics(truth[mask], estimate[mask])
                constant = np.broadcast_to(self.baselines[assay][cohort], truth[mask].shape)
                metrics["constant_mean"] = activity_metrics(truth[mask], constant)["total"]
                metrics["saturation"] = saturation_metrics(
                    observed[i, mask], predicted[i, mask], truth[mask], estimate[mask], self.references[assay]
                )
                report[assay][cohort] = metrics
        return report
