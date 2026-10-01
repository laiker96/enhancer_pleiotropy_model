"""Small background batches and separate enhancer/background diagnostics."""
import json
from pathlib import Path

import numpy as np

from .data import CONTEXTS, digest
from .metrics import binary, binary_curve


def load_background(root, split, enhancers, input_bp=2048):
    if split not in {"train", "validation", "test"}: raise ValueError(split)
    root = Path(root)
    audit = json.loads((root/"audit.json").read_text())
    path = root/(split+".npz")
    if digest(path) != audit["outputs"][path.name]: raise ValueError("Background data hash mismatch")
    with np.load(path, allow_pickle=False) as f: data = dict(f)
    idx = data["enhancer_index"]
    n = len(idx)
    if (not n or data["sequence"].shape != (n,input_bp) or data["labels"].shape != (n,8)
            or not np.isin(data["sequence"], [0,1,2,3]).all() or data["labels"].any()
            or len(np.unique(data["ids"])) != n or len(np.unique(idx)) != n
            or np.any(idx < 0) or np.any(idx >= len(enhancers["ids"]))
            or not np.array_equal(data["enhancer_ids"], enhancers["ids"][idx])):
        raise ValueError("Invalid or unaligned background arrays")
    return data


def background_epoch(n_enhancers, n_background, seed, epoch):
    """Independent RNG preserves the baseline enhancer order and RC draws."""
    count = n_enhancers//3
    if n_background < count: raise ValueError("Insufficient unique training backgrounds")
    rng = np.random.default_rng(seed+10_000_000+epoch)
    return rng.permutation(n_background)[:count], rng.random(count) < .5


def background_slice(enhancer_begin, enhancer_end):
    # Cumulative rounding gives exactly floor(N_enhancers/3) examples per epoch.
    return slice(enhancer_begin//3, enhancer_end//3)


def background_metrics(labels, enhancer_probabilities, background_probabilities, paired_indices):
    y = np.asarray(labels)
    p, bg = np.asarray(enhancer_probabilities), np.asarray(background_probabilities)
    idx = np.asarray(paired_indices)
    if (y.ndim != 2 or y.shape[1] != 8 or p.shape != y.shape or bg.shape != (len(idx),8)
            or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or not np.isfinite(bg).all()
            or np.any((p<0)|(p>1)) or np.any((bg<0)|(bg>1))
            or np.any(idx<0) or np.any(idx>=len(y)) or len(np.unique(idx))!=len(idx)):
        raise ValueError("Invalid background evaluation arrays")
    result = {view:dict(contexts={}) for view in ("active_vs_background", "enhancers_plus_background")}
    result["background_rejection"] = dict(contexts={})
    for j,context in enumerate(CONTEXTS):
        active = y[:,j].astype(bool)
        if not active.any() or active.all(): raise ValueError("Both enhancer classes required")
        paired_active = y[idx,j].astype(bool)
        if not paired_active.any(): raise ValueError("Matched positives required in every context")
        ys = np.r_[np.ones(paired_active.sum()),np.zeros(paired_active.sum())]
        ps = np.r_[p[idx[paired_active],j],bg[paired_active,j]]
        combined_y = np.r_[y[:,j],np.zeros(len(bg))]
        combined_p = np.r_[p[:,j],bg[:,j]]
        for view,truth,scores in [("active_vs_background",ys,ps),
                                  ("enhancers_plus_background",combined_y,combined_p)]:
            curve = binary_curve(truth,scores)
            result[view]["contexts"][context] = dict(binary(truth,scores),n=len(truth),prevalence=float(truth.mean()),
                average_precision=float(curve["average_precision"]),auprc_trapezoidal=float(curve["auprc_trapezoidal"]))
        k = int(np.ceil(.8*active.sum()))
        threshold = float(np.sort(p[active,j])[-k])
        result["background_rejection"]["contexts"][context] = dict(
            diagnostic_threshold_at_80_enhancer_recall=threshold,
            enhancer_recall=float((p[active,j]>=threshold).mean()),
            background_fpr_at_80_recall=float((bg[:,j]>=threshold).mean()),
            other_enhancer_fpr_at_80_recall=float((p[~active,j]>=threshold).mean()),
            background_fpr_at_05=float((bg[:,j]>=.5).mean()))
    for view in ("active_vs_background", "enhancers_plus_background"):
        for out,key in (("macro_average_precision","average_precision"),("macro_auroc","auroc")):
            result[view][out] = float(np.mean([v[key] for v in result[view]["contexts"].values()]))
    result["background_rejection"]["macro_fpr_at_80_recall"] = float(np.mean([
        v["background_fpr_at_80_recall"] for v in result["background_rejection"]["contexts"].values()]))
    return result
