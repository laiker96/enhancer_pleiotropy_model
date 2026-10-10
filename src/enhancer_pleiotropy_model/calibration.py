"""Per-context probability calibration of a trained classifier.

Probabilities ``p`` are the mean of forward and reverse-complement sigmoid
outputs. Three per-context methods are compared:

* ``raw`` - no change;
* ``temperature`` - ``sigmoid(a * logit(p))``;
* ``sigmoid`` (Platt) - ``sigmoid(a * logit(p) + b)``;

with ``a > 0`` so rankings are preserved, each fitted by maximum likelihood on
one of two populations of validation examples:

* ``enhancers_only`` - the validation enhancers (used for all paper analyses);
* ``enhancers_plus_background_1to1`` - the enhancers plus their GC-matched
  genomic backgrounds (labelled inactive everywhere).

Methods are compared with five spatially blocked folds (contiguous blocks of
enhancer coordinates; matched backgrounds follow their enhancer; any training
pair whose 2,048-bp input overlaps a held-out input is removed). Within each
population, the method with the lowest out-of-fold log loss is refitted on all
of that population's validation examples. The test split is never used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata

from .constants import CONTEXTS, INPUT_BP
from .io import read_json, write_json

METHODS = ("raw", "temperature", "sigmoid")
POPULATIONS = ("enhancers_only", "enhancers_plus_background_1to1")
DEFAULT_POPULATION = "enhancers_only"
EPSILON = 1e-7
FOLDS = 5


def fit(probabilities: np.ndarray, labels: np.ndarray, method: str = "sigmoid") -> dict:
    x = logit(np.clip(np.asarray(probabilities, float), EPSILON, 1 - EPSILON))
    y = np.asarray(labels, float)
    a, b = [], []
    for j in range(x.shape[1]):
        if method == "raw":
            a.append(1.)
            b.append(0.)
            continue

        def objective(parameters):
            z = parameters[0] * x[:, j] + (parameters[1] if method == "sigmoid" else 0.)
            error = expit(z) - y[:, j]
            gradient = [float(np.mean(error * x[:, j]))] + ([float(error.mean())] if method == "sigmoid" else [])
            return float(np.mean(np.logaddexp(0, z) - y[:, j] * z)), np.asarray(gradient)

        initial = [1., 0.] if method == "sigmoid" else [1.]
        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          bounds=[(1e-6, None)] + ([(None, None)] if method == "sigmoid" else []),
                          options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
        # A line-search stop at a flat optimum (e.g. near-constant predictions) is accepted
        # only if the result is finite, better than the start and has a near-zero gradient.
        if not result.success and not (np.isfinite(result.x).all() and result.fun <= objective(initial)[0]
                                       and np.abs(objective(result.x)[1]).max() < 1e-5):
            raise RuntimeError(f"Calibration did not converge for {CONTEXTS[j]}: {result.message}")
        a.append(float(result.x[0]))
        b.append(float(result.x[1]) if method == "sigmoid" else 0.)
    return {"method": method, "a": a, "b": b, "probability_clip": EPSILON,
            "formula": "sigmoid(a * logit(clip(mean forward/RC probability, eps, 1 - eps)) + b)"}


def apply(probabilities: np.ndarray, calibration: dict) -> np.ndarray:
    if calibration.get("method") == "raw":
        return np.asarray(probabilities, float).copy()
    eps = calibration["probability_clip"]
    x = logit(np.clip(np.asarray(probabilities, float), eps, 1 - eps))
    return expit(x * np.asarray(calibration["a"]) + np.asarray(calibration["b"]))


def _overlaps(chrom, start, end, held_chrom, held_start, held_end) -> np.ndarray:
    """Whether each half-open interval overlaps the union of the held-out intervals."""
    found = np.zeros(len(start), bool)
    for c in np.unique(held_chrom):
        source = np.flatnonzero(held_chrom == c)
        merged = []
        for lo, hi in sorted(zip(held_start[source], held_end[source])):
            if not merged or lo > merged[-1][1]:
                merged.append([lo, hi])
            else:
                merged[-1][1] = max(hi, merged[-1][1])
        lower, upper = np.asarray(merged).T
        query = np.flatnonzero(chrom == c)
        index = np.searchsorted(lower, end[query], side="left") - 1
        good = index >= 0
        found[query[good]] = upper[index[good]] > start[query[good]]
    return found


def spatial_folds(chrom, summit, bg_chrom, bg_start, bg_end, bg_pair):
    """Fold of each enhancer and, per fold, the purged training mask over enhancers."""
    n = len(summit)
    fold = np.empty(n, int)
    order = np.lexsort((summit, chrom))
    for number, indices in enumerate(np.array_split(order, FOLDS)):
        fold[indices] = number
    half = INPUT_BP // 2
    masks, records = [], []
    for number in range(FOLDS):
        held = fold == number
        held_bg = held[bg_pair]
        hc = np.r_[chrom[held], bg_chrom[held_bg]]
        hs = np.r_[summit[held] - half, bg_start[held_bg]]
        he = np.r_[summit[held] + half, bg_end[held_bg]]
        conflict = _overlaps(chrom, summit - half, summit + half, hc, hs, he)
        conflict[bg_pair[_overlaps(bg_chrom, bg_start, bg_end, hc, hs, he)]] = True
        train = ~held & ~conflict
        masks.append(train)
        records.append({"fold": number, "heldout_pairs": int(held.sum()), "training_pairs": int(train.sum()),
                        "purged_pairs": int((~held & conflict).sum())})
    return fold, masks, records


def evaluate(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    y, p = np.asarray(labels, float), np.asarray(probabilities, float)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    nll = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    ece = []
    for j in range(p.shape[1]):
        bins = np.minimum((p[:, j] * 10).astype(int), 9)
        ece.append(sum((bins == k).sum() / len(y) * abs(p[bins == k, j].mean() - y[bins == k, j].mean())
                       for k in range(10) if (bins == k).any()))
    truth, prediction = y.sum(1), p.sum(1)
    return {"n": int(len(y)), "log_loss": float(nll.mean()), "brier": float(np.square(p - y).mean()),
            "macro_ece10": float(np.mean(ece)),
            "breadth": {"observed_mean": float(truth.mean()), "predicted_mean": float(prediction.mean()),
                        "bias": float((prediction - truth).mean()), "mae": float(np.abs(prediction - truth).mean()),
                        "spearman": float(np.corrcoef(rankdata(truth), rankdata(prediction))[0, 1])
                        if np.ptp(truth) else None}}


def calibration_study(enhancer_p, labels, chrom, summit, background_p, bg_chrom, bg_start, bg_end, bg_pair):
    """Cross-validated method comparison and final calibrators for both populations."""
    order = np.argsort(bg_pair, kind="stable")  # backgrounds in the order of their enhancers
    background_p, bg_chrom, bg_start, bg_end, bg_pair = (
        np.asarray(v)[order] for v in (background_p, bg_chrom, bg_start, bg_end, bg_pair))
    n, m = len(enhancer_p), len(background_p)
    probabilities = np.vstack((enhancer_p, background_p)).astype(float)
    all_labels = np.vstack((labels, np.zeros((m, labels.shape[1]), labels.dtype)))
    fold, masks, fold_records = spatial_folds(chrom, summit, bg_chrom, bg_start, bg_end, bg_pair)
    held_out = [np.r_[fold == k, fold[bg_pair] == k] for k in range(FOLDS)]
    views = {"enhancers_only": np.r_[np.ones(n, bool), np.zeros(m, bool)],
             "enhancers_plus_background_1to1": np.ones(n + m, bool)}
    results, calibrators = {}, {}
    for population in POPULATIONS:
        results[population] = {}
        for method in METHODS:
            oof = np.full_like(probabilities, np.nan)
            for k, mask in enumerate(masks):
                train = np.r_[mask, mask[bg_pair] if population.endswith("1to1") else np.zeros(m, bool)]
                oof[held_out[k]] = apply(probabilities[held_out[k]],
                                         fit(probabilities[train], all_labels[train], method))
            results[population][method] = {view: evaluate(all_labels[rows], oof[rows])
                                           for view, rows in views.items()}
        chosen = min(METHODS, key=lambda method: results[population][method][population]["log_loss"])
        rows = views[population]
        calibrators[population] = dict(fit(probabilities[rows], all_labels[rows], chosen),
                                       contexts=list(CONTEXTS), fitting_population=population,
                                       training_examples=int(rows.sum()))
    return calibrators, {"folds": fold_records, "results": results,
                         "selected": {p: c["method"] for p, c in calibrators.items()}}


def report_markdown(study: dict) -> str:
    lines = ["# Probability calibration", "",
             "Out-of-fold estimates from five spatially blocked validation folds (no test data).", "",
             "| Fitted on | Method | Evaluated on | Log loss | Brier | ECE (10 bins) | Breadth MAE | Breadth bias |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for population in POPULATIONS:
        for method in METHODS:
            for view, r in study["results"][population][method].items():
                lines.append(f"| {population} | {method} | {view} | {r['log_loss']:.5f} | {r['brier']:.5f} | "
                             f"{r['macro_ece10']:.5f} | {r['breadth']['mae']:.4f} | {r['breadth']['bias']:+.4f} |")
    lines += ["", f"Selected (lowest out-of-fold log loss): {study['selected']}.",
              f"`calibration.json` holds both final calibrators; predictions use `{DEFAULT_POPULATION}`."]
    return "\n".join(lines) + "\n"


def calibrate_run(run: Path, data: Path | None = None, device_name: str = "auto") -> Path:
    """Predict the validation enhancers and backgrounds, run the study and write calibration.json."""
    import torch

    from .data.background import load_background
    from .data.enhancers import load_split
    from .models import load_classifier, use_interleaved_dilation
    from .training.classifier import predict_probabilities
    from .training.common import resolve_device

    run = Path(run)
    if data is None:
        data = Path(read_json(run / "metrics.json")["config"]["data"])
    device = resolve_device(device_name)
    torch.use_deterministic_algorithms(True)
    model, checkpoint = load_classifier(run / "best_model.pt")
    model = use_interleaved_dilation(model.to(device))
    validation = load_split(data, "validation")
    background = load_background(data, "validation", validation)
    enhancer_p = predict_probabilities(model, validation["sequence"], device)
    background_p = predict_probabilities(model, background["sequence"], device)
    calibrators, study = calibration_study(
        enhancer_p, validation["labels"], validation["chrom"], validation["summit"], background_p,
        background["chrom"], background["start"], background["end"], background["enhancer_index"])
    write_json(run / "calibration.json", calibrators)
    write_json(run / "calibration_study.json", study)
    (run / "calibration_report.md").write_text(report_markdown(study))
    np.savez_compressed(run / "calibration_predictions.npz", enhancer_ids=validation["ids"],
                        enhancer_probabilities=enhancer_p, background_ids=background["ids"],
                        background_probabilities=background_p)
    return run / "calibration.json"


def load_calibration(path: Path, population: str = DEFAULT_POPULATION) -> dict:
    value = read_json(path)
    return value.get(population, value)
