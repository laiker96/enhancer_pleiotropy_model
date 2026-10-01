"""CPU-only comprehensive reports from locked validation/test predictions. No refitting."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import scipy

from .data import CONTEXTS, digest, load_split, write_json
from .metrics import summarize, thresholds_from_validation


DEFINITIONS = {
    "average_precision": "Stepwise precision-recall area, summing recall increments times precision. Existing selection metric.",
    "auprc_trapezoidal": "Trapezoidal PR area including initial (recall=0, precision=1); not interchangeable with average precision.",
    "prevalence": "Fraction catalog-active in this context and split; random-ranking population AP baseline.",
    "thresholds": "Both fixed 0.5 and per-context maximum-F1 validation thresholds; never optimized on test.",
    "ece_10_equal_width": "Sample-weighted absolute calibration gap in 10 probability bins. Empty bins omitted.",
    "breadth": "Sum of 8 classifier probabilities, compared with catalog-active context count, not continuous signal breadth.",
    "seeds": "Mean and sample SD over classifier seeds, conditional on one pretrained regressor per architecture.",
    "missing": "JSON null means undefined/unavailable, not zero. MCC undefined for a constant call vector.",
    "regression": "Selected-checkpoint validation metrics in original log1p signal scale; all-bin and central views kept distinct. No new regressor test inference in this report.",
    "test_scope": "Historical chr3R holdout, previously examined in model development; not a pristine independent test.",
    "group_score": "Group MAX probability is only a ranking score; additive expected counts and any-member hit are separate.",
}


def cached_predictions(path, split):
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["ids"], split["ids"]):
            raise ValueError(f"Prediction IDs/order mismatch: {path}")
        p = np.asarray(saved["probabilities"], dtype=np.float64)
    if (p.shape != split["labels"].shape or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))):
        raise ValueError(f"Invalid probabilities: {path}")
    return p


def seed_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in ("suite", "architecture", "mode", "split", "context"))].append(row)
    output = []
    metadata = {"suite", "architecture", "mode", "split", "context", "seed", "model", "selected_epoch"}
    for key, values in grouped.items():
        if len({r["seed"] for r in values}) != len(values):
            raise ValueError("Duplicate classifier seed")
        item = dict(zip(("suite", "architecture", "mode", "split", "context"), key))
        item["seeds"] = [r["seed"] for r in values]
        item["metrics"] = {}
        for metric in sorted(values[0].keys() - metadata):
            data = np.asarray([r[metric] if r[metric] is not None else np.nan for r in values], float)
            valid = data[np.isfinite(data)]
            item["metrics"][metric] = dict(n=int(len(valid)), mean=float(valid.mean()) if len(valid) else None,
                                            sd=float(valid.std(ddof=1)) if len(valid)>1 else None)
        output.append(item)
    return output


def regression_rows(regressors, suite):
    rows = []
    def visit(value, path, architecture, epoch):
        if not isinstance(value, dict): return
        for context, metrics in value.get("by_context", {}).items():
            for metric, number in metrics.items():
                if isinstance(number, (int, float)) or number is None:
                    rows.append(dict(suite=suite, architecture=architecture, selected_epoch=epoch,
                                     split="validation", view="/".join(path), context=context,
                                     metric=metric, value=number))
        for key, child in value.items():
            if key != "by_context": visit(child, path+[key], architecture, epoch)
    for architecture, values in regressors.items():
        visit(values["validation"], [], architecture, values["epoch"])
    return rows


def generate(roots, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    all_results, rows, regression, sources = {}, [], [], []
    for root in map(Path, roots):
        source_path = root/"comparison/metrics.json"
        source = json.loads(source_path.read_text())
        if source["status"] != "complete": raise ValueError("Comparison is incomplete")
        completion = root/"complete.json"
        if completion.exists() and json.loads(completion.read_text())["comparison_sha256"] != digest(source_path):
            raise ValueError("Completed comparison hash mismatch")
        suite = root.name
        if suite in all_results: raise ValueError("Duplicate suite identifier")
        lock_path = root/"comparison/selection_lock.json"
        locks = json.loads(lock_path.read_text())
        if locks != source["selection"]: raise ValueError("Selection lock differs from comparison")
        data = {s: load_split(root/"data", s) for s in ("validation", "test")}
        for split in data.values():
            if len(np.unique(split["ids"])) != len(split["ids"]): raise ValueError("Duplicate element IDs")
        all_results[suite] = dict(classifiers={}, regressor_best_validation=source["regressor_best_validation"],
                                  original_seed_aggregates=source["seed_aggregates"],
                                  original_paired_differences=source["paired_finetune_minus_scratch"],
                                  original_block_bootstrap=source["block_bootstrap"])
        files = [source_path, lock_path, root/"data/audit.json"]
        for name, selection in locks.items():
            directory = root/"runs"/name
            if digest(directory/"best_model.pt") != selection["checkpoint_sha256"]:
                raise ValueError("Locked checkpoint hash mismatch")
            val_path = directory/"best_validation_predictions.npz"
            test_path = root/"comparison"/f"{name}.npz"
            files.extend([val_path, test_path])
            val_p = cached_predictions(val_path, data["validation"])
            thresholds = thresholds_from_validation(data["validation"]["labels"], val_p)
            if not np.array_equal(thresholds, np.asarray(selection["thresholds"])):
                raise ValueError("Validation threshold lock mismatch")
            by_split = {}
            for split, p in (("validation", val_p), ("test", cached_predictions(test_path, data["test"]))):
                by_split[split] = metrics = summarize(data[split]["labels"], p, thresholds)
                for context in CONTEXTS:
                    detail = metrics["contexts"][context]
                    row = dict(suite=suite, model=name, architecture=selection["architecture"],
                               mode=selection["mode"], seed=selection["seed"], split=split,
                               context=context, selected_epoch=selection["epoch"])
                    row.update({k:v for k,v in detail.items() if not isinstance(v, (dict, list))})
                    row.update({"validation_threshold_"+k:v for k,v in detail["validation_threshold"].items()})
                    rows.append(row)
            for metric in ("macro_average_precision", "macro_auroc", "brier"):
                if not np.isclose(by_split["test"][metric], source["results"][name][metric], rtol=0, atol=1e-12):
                    raise ValueError(f"Recomputed original metric differs: {name}/{metric}")
            all_results[suite]["classifiers"][name] = dict(selection=selection, metrics=by_split,
                parameters=source["results"][name]["parameters"])
        regression.extend(regression_rows(source["regressor_best_validation"], suite))
        sources.append(dict(suite=suite, root=str(root.resolve()),
                            files={str(p.relative_to(root)):digest(p) for p in files}))
    aggregates = seed_summary(rows)
    output.mkdir(parents=True)
    implementation = dict(numpy=np.__version__, scipy=scipy.__version__,
        code_sha256={name:digest(Path(__file__).parent/name) for name in ("report.py", "metrics.py", "data.py")})
    write_json(output/"metrics.json", dict(definitions=DEFINITIONS, contexts=CONTEXTS, sources=sources,
                                         implementation=implementation, suites=all_results))
    write_json(output/"classification_per_context.json", rows)
    write_json(output/"classification_per_context_seed_summary.json", aggregates)
    write_json(output/"regression_per_context_validation.json", regression)
    lines = ["# Per-context classifier performance", "", "Mean ± sample SD across classifier seeds. AP is stepwise PR area.",
             "Trapezoidal PR-AUC, AUROC, calibration, confusion counts and validation-threshold metrics are in the JSON tables.",
             "Validation is used for selection. Test is the historically examined chr3R holdout.", ""]
    for split in ("test", "validation"):
        lines += [f"## {split.title()} average precision", "",
                  "| Model | " + " | ".join(CONTEXTS) + " |", "|---|" + "---:|"*8]
        groups = defaultdict(dict)
        for row in aggregates:
            if row["split"] == split:
                groups[(row["suite"], row["architecture"], row["mode"])][row["context"]] = row
        for key, contexts in groups.items():
            cells = []
            for context in CONTEXTS:
                metric = contexts[context]["metrics"]["average_precision"]
                cells.append(f"{metric['mean']:.3f} ± {metric['sd']:.3f}" if metric["sd"] is not None else f"{metric['mean']:.3f}")
            lines.append("| " + "/".join(key) + " | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["Regressor per-context metrics are selected-checkpoint validation metrics only, with view and assay preserved.",
              "No additional regressor test inference is claimed. All source predictions and checkpoints remain unchanged.", ""]
    (output/"summary.md").write_text("\n".join(lines))
    write_json(output/"verification.json", dict(status="verified", classification_context_rows=len(rows),
        seed_summary_rows=len(aggregates), regression_metric_rows=len(regression),
        checks=["data hashes", "unique aligned IDs", "finite bounded probabilities", "checkpoint hashes",
                "validation thresholds match locks", "original test AP/AUROC/Brier reproduced"],
        output_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(f"Comprehensive metrics saved: {len(rows)} context rows, {len(aggregates)} seed summaries", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root, args.output)
