"""Refresh a compact inventory from receipts/history, never by opening test DNA."""
import csv
import json
import math
from pathlib import Path

from classifier_transfer.data import CONTEXTS
from .common import classifiers, digest, run_id


def population_metrics(row, prefix, metrics):
    """Keep the evaluation population in every background/test column name."""
    for key in ("macro_average_precision", "macro_auroc", "bce", "brier", "macro_f1_at_05",
                "related_pair_macro_accuracy", "any_member_group_top_choice_hit", "macro_fpr_at_80_recall"):
        row[prefix + "_" + ("macro_ap" if key == "macro_average_precision" else key)] = metrics.get(key)
    for key in ("pearson", "spearman", "r2", "mae"):
        row[prefix + "_breadth_" + key] = metrics.get("breadth", {}).get(key)
    for context in CONTEXTS:
        for key in ("average_precision", "auroc", "brier", "prevalence", "n"):
            row[f"{prefix}_{context}_{key}"] = metrics.get("contexts", {}).get(context, {}).get(key)


def regression_metrics(row, best):
    row["val_overcorrelation"] = best.get("regulatory_overcorrelation", {}).get("mean_positive_excess")
    for assay in ("atac", "h3k27ac"):
        metrics = best.get("validation", {}).get(assay, {})
        center = metrics.get("window_mean", {}) if assay == "atac" else metrics.get("segments", {}).get("center", {})
        for region, entry in (("all_bins", metrics.get("dense_profile", {})), ("center512", center)):
            for key in ("pearson", "spearman", "r2", "mae"):
                row[f"val_{assay}_{region}_{key}"] = entry.get("macro", {}).get(key)
        row[f"val_{assay}_context_pattern_pearson"] = metrics.get("tissue_pattern", {}).get("mean_pearson")
        for context in CONTEXTS:
            row[f"val_{assay}_{context}_center512_pearson"] = center.get("by_context", {}).get(context, {}).get("pearson")
        breadth = best.get("continuous_breadth", {}).get(assay, {}).get("regulatory", {}).get("total", {})
        for key in ("pearson", "spearman", "r2", "mae"):
            row[f"val_{assay}_regulatory_breadth_{key}"] = breadth.get(key)


def refresh(cfg):
    work = Path(cfg["paths"]["work"])
    rows = []
    tests = work / "evaluation/metrics.json"
    tests = json.loads(tests.read_text()) if tests.exists() else {}
    for architecture in cfg["architectures"]:
        root = work / "regressors" / architecture
        path = root / "model/metrics.json"
        metrics = json.loads(path.read_text()) if path.exists() else {}
        history_path = root / "model/history.json"
        history = metrics.get("history", json.loads(history_path.read_text()) if history_path.exists() else [])
        best_epoch = metrics.get("best_epoch")
        if best_epoch is None and history:
            best_epoch = max(history, key=lambda r: r["scientific_composite"])["epoch"]
        best = next((r for r in history if r["epoch"] == best_epoch), {})
        checkpoint = root / "model/best_model.pt"
        row = dict(model_id="regressor__"+architecture, type="regressor", architecture=architecture,
            training="scratch", input_bp=2048, epochs_completed=len(history), selected_epoch=best_epoch,
            status="complete" if (root / "complete.json").exists() else ("partial" if root.exists() else "planned"),
            checkpoint_path=str(checkpoint), checkpoint_sha256=digest(checkpoint) if checkpoint.exists() else "",
            val_scientific_composite=best.get("scientific_composite"),
            selected_lr_encoder=best.get("learning_rate"),
            lr_current_encoder=history[-1].get("learning_rate") if history else None,
            metrics_path=str(path if path.exists() else history_path))
        regression_metrics(row, best)
        rows.append(row)
    for entry in classifiers(cfg):
        name = run_id(**entry); root = work / "classifiers" / name
        path = root / "history.json"
        history = json.loads(path.read_text()) if path.exists() else []
        selected = [r for r in history if r["selected"]]
        best = selected[-1] if selected else {}
        metrics = best.get("metrics", {})
        checkpoint = root / "best_model.pt"
        lrs = history[-1]["learning_rates"] if history else [None,None]
        selected_lrs = best.get("learning_rates", [None, None])
        parent = work / "regressors" / entry["architecture"] / "model/best_model.pt"
        row = dict(model_id=name, type="classifier", architecture=entry["architecture"],
            training=entry["mode"], readout=entry["readout"], population=entry["population"], seed=entry["seed"],
            input_bp=2048, epochs_completed=len(history), selected_epoch=best.get("epoch"),
            status="complete" if (root / "complete.json").exists() else ("partial" if root.exists() else "planned"),
            checkpoint_path=str(checkpoint), checkpoint_sha256=digest(checkpoint) if checkpoint.exists() else "",
            val_macro_ap=metrics.get("macro_average_precision"), val_macro_auroc=metrics.get("macro_auroc"),
            val_bce=metrics.get("bce"), val_breadth_r2=metrics.get("breadth", {}).get("r2"),
            epochs_planned=cfg["classifier"]["epochs"],
            max_lr_encoder=cfg["classifier"]["max_learning_rate"] * (
                cfg["classifier"]["finetune_encoder_factor"] if entry["mode"] == "finetune" else 1),
            max_lr_head=cfg["classifier"]["max_learning_rate"],
            selected_lr_encoder=selected_lrs[0], selected_lr_head=selected_lrs[1],
            parent_regressor_model_id="regressor__"+entry["architecture"] if entry["mode"] == "finetune" else "",
            initialization_checkpoint_path=str(parent) if entry["mode"] == "finetune" else "",
            initialization_checkpoint_sha256=digest(parent) if entry["mode"] == "finetune" and parent.exists() else "",
            lr_current_encoder=lrs[0], lr_current_head=lrs[1], metrics_path=str(path))
        population_metrics(row, "val", metrics)
        for population, result in best.get("background_metrics", {}).items():
            population_metrics(row, "val_"+population, result)
        for population, result in tests.get(name, {}).items():
            population_metrics(row, "test_"+population, result)
        rows.append(row)
    rows = [{key: None if isinstance(value, float) and not math.isfinite(value) else value
             for key, value in row.items()} for row in rows]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    work.mkdir(parents=True, exist_ok=True)
    path = work / "model_inventory.tsv"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)
    print(path)
