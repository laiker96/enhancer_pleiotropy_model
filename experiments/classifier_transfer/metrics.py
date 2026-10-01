"""Classification, additive expected breadth, calibration and specificity."""

import numpy as np
from scipy.stats import rankdata

from enhancer_pleiotropy_model.breadth_metrics import evaluate_probabilities, reliability_rows, GROUP_INDICES, GROUPS
from enhancer_pleiotropy_model.enhancer_catalog_evaluation import binary_curve, choose_f1_threshold
from .data import CONTEXTS


def r2(y, p):
    denom = np.square(y - np.mean(y)).sum()
    return float(1 - np.square(p-y).sum()/denom) if denom > 0 else float("nan")


def binary(y, p, threshold=.5):
    y = np.asarray(y, dtype=bool)
    calls = p >= threshold
    tp, fp, fn = int((y & calls).sum()), int((~y & calls).sum()), int((y & ~calls).sum())
    tn = int((~y & ~calls).sum())
    npos, nneg = int(y.sum()), int((~y).sum())
    auc = ((rankdata(p)[y].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else float("nan")
    denominator = float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** .5
    return dict(auroc=float(auc), threshold=float(threshold), tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=(tp+tn)/len(y) if len(y) else float("nan"),
                specificity=tn/nneg if nneg else float("nan"),
                balanced_accuracy=.5*(tp/npos+tn/nneg) if npos and nneg else float("nan"),
                mcc=(tp*tn-fp*fn)/denominator if denominator else float("nan"),
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else float("nan"),
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def thresholds_from_validation(y, p):
    return np.asarray([choose_f1_threshold(y[:, i], p[:, i])[0] for i in range(8)])


def summarize(y, p, thresholds=None):
    p = np.asarray(p, dtype=np.float64)
    report = evaluate_probabilities(y, p)
    report["breadth"]["r2"] = r2(y.sum(1), p.sum(1))
    clipped = np.clip(p, 1e-7, 1-1e-7)
    report["bce"] = float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean())
    for i, name in enumerate(CONTEXTS):
        report["contexts"][name].update(binary(y[:, i], p[:, i]))
        report["contexts"][name]["reliability"] = reliability_rows(y[:, i], p[:, i], 1)
        row = report["contexts"][name]
        row["auprc_trapezoidal"] = (float(binary_curve(y[:, i], p[:, i])["auprc_trapezoidal"])
                                      if 0 < y[:, i].sum() < len(y) else float("nan"))
        row["ece_10_equal_width"] = sum(r["n"] * abs(r["predicted_mean"]-r["observed_mean"])
                                          for r in row["reliability"] if r["n"]) / len(y)
        if thresholds is not None:
            report["contexts"][name]["validation_threshold"] = binary(y[:, i], p[:, i], thresholds[i])
    report["macro_auroc"] = float(np.mean([v["auroc"] for v in report["contexts"].values()]))
    report["macro_f1_at_05"] = float(np.mean([v["f1"] for v in report["contexts"].values()]))
    report["breadth_reliability"] = reliability_rows(y.sum(1), p.sum(1), 8)
    grouped_y = np.column_stack([y[:, idx].any(1) for idx in GROUP_INDICES])
    grouped_p = np.column_stack([p[:, idx].max(1) for idx in GROUP_INDICES])
    ties = grouped_p == grouped_p.max(1, keepdims=True)
    report["any_member_group_top_choice_hit"] = float(((ties/ties.sum(1, keepdims=True))*grouped_y).sum(1).mean())
    report["group_score_definition"] = "MAX context probability for ranking, not a calibrated OR probability"
    for name, idx in zip(GROUPS, GROUP_INDICES, strict=True):
        report["groups"][name]["r2"] = r2(y[:, idx].sum(1), p[:, idx].sum(1))
    report["by_observed_breadth"] = {
        str(k): dict(n=int((y.sum(1)==k).sum()), predicted_mean=float(p[y.sum(1)==k].sum(1).mean()))
        for k in range(1, 9) if np.any(y.sum(1)==k)}
    return report
