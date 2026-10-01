#!/usr/bin/env python3
"""Render the frozen paper benchmark without datasets, checkpoints, or GPU code."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reproduce/benchmarks/models.json"
REPORT = ROOT / "docs/paper_benchmarks.md"
NAMES = {"cnn": "Dilated CNN", "attention": "CNN + attention", "dense": "Flatten/dense CNN",
         "enhancernet_cnn": "EnhancerNet CNN", "enhancernet_attention": "EnhancerNet + attention",
         "cnn_1024": "Dilated CNN 1,024", "cnn_512": "Dilated CNN 512"}
GROUP_FIELDS = ("architecture", "readout", "training_type", "training_population", "input_bp",
                "pretraining_input_bp", "epochs_planned", "classifier_schedule_epochs", "experiment_revision")


def validate(snapshot):
    rows = snapshot["models"]
    if snapshot["schema_version"] != 1 or len({r["model_id"] for r in rows}) != len(rows):
        raise ValueError("Unsupported snapshot or duplicate model IDs")
    for row in rows:
        if row["status"] != "complete" or not 1 <= row["selected_epoch"] <= row["epochs_completed"]:
            raise ValueError("Benchmark requires completed runs with a selected epoch")
        if not re.fullmatch(r"[0-9a-f]{64}", row["checkpoint_sha256"] or ""):
            raise ValueError("Missing checkpoint hash")
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Nonfinite benchmark value")
            if key.endswith("_artifact_path") and value:
                if Path(value).is_absolute() or ".." in Path(value).parts:
                    raise ValueError("Artifact paths must be relative")
            if isinstance(value, str) and "/home/" in value:
                raise ValueError("Machine-specific path in public snapshot")
        if row["model_type"] == "classifier":
            values = [row["val_" + c + "_ap"] for c in snapshot["context_order"]]
            if not all(v is not None and 0 <= v <= 1 for v in values):
                raise ValueError("Missing/invalid per-context validation AP")
            if not math.isclose(mean(values), row["val_macro_ap"], abs_tol=1e-12):
                raise ValueError("Per-context AP does not reproduce macro AP")
    return rows


def groups(rows):
    out = defaultdict(list)
    for row in rows:
        if row["model_type"] == "classifier":
            out[tuple(row.get(k) for k in GROUP_FIELDS)].append(row)
    return [sorted(value, key=lambda r: r["seed"])
            for _, value in sorted(out.items(), key=lambda item: tuple(str(x) for x in item[0]))]


def representative(rows):
    # Epochs were selected by enhancer validation AP during training. Choose
    # the displayed seed by the deployment population, never a test metric.
    key = "val_combined_macro_ap" if rows[0]["training_population"] != "enhancers only" else "val_macro_ap"
    if any(row.get(key) is None for row in rows):
        raise ValueError("Missing validation metric for representative seed selection")
    return max(rows, key=lambda row: (row[key], -row["val_loss"], -row["seed"]))


def number(value):
    return "—" if value is None else f"{value:.3f}"


def aggregate(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return "—"
    text = f"{mean(values):.4f}"
    if len(values) > 1:
        text += f" ± {stdev(values):.4f}"
    if len(values) != len(rows):
        text += f" (n={len(values)}/{len(rows)})"
    return text


def hidden(row):
    if row["readout"].startswith("legacy fresh"):
        return "Replace assay hidden"
    if row["readout"].startswith("retained_assay"):
        return "Keep assay hidden"
    return "Keep shared dense"


def initialization(row):
    if row["training_type"] == "scratch":
        return "Scratch"
    return f"Fine-tune ({row['pretraining_input_bp']} bp parent)"


def table(headers, rows):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(escape, headers)) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |"] +
                      ["| " + " | ".join(map(escape, row)) + " |" for row in rows])


def render(snapshot):
    rows = validate(snapshot)
    grouped = [(f"C{i:02}", group) for i, group in enumerate(groups(rows), 1)]
    portable = set(snapshot["portable_architectures"])
    out = ["# Paper model benchmarks", "",
        "Frozen reference measurements from the current transfer-learning comparison: "
        f"**{sum(r['model_type'] == 'regressor' for r in rows)} regressors and "
        f"{sum(r['model_type'] == 'classifier' for r in rows)} classifiers**. "
        "These are historical results, not measurements from a fresh portable-workflow run.", "",
        "The [machine-readable snapshot](../reproduce/benchmarks/models.json) retains all recorded "
        "validation/test metrics, per-context results, learning rates, training settings, parent lineage, "
        "relative artifact paths and full checkpoint SHA256 hashes. "
        "Missing values are null/—, not zeros.", "",
        "## How to read these tables", "",
        "- **E:** active-in-context enhancers versus other catalog enhancers. "
        "**E+BG:** active enhancers versus other enhancers plus matched genomic background. "
        "**A/BG:** active enhancers versus their matched background sequences only. "
        "AP means average precision, not trapezoidal PR area; it depends on class prevalence.",
        "- Classifier tables report mean ± sample SD across three classifier seeds, "
        "not confidence intervals. Fine-tuning seeds share one pretrained regressor per architecture.",
        "- Every checkpoint was selected on validation, not test. Classifier epoch selection: "
        "maximum validation enhancer-only macro AP, BCE tie-break. The representative seed shown "
        "below maximizes validation E+BG AP for background-trained models, otherwise validation E AP.",
        "- Validation is the first half of chr2L; test is chr3R. This historical test set "
        "has been inspected repeatedly and is not a new untouched holdout. No superiority test is claimed.",
        "- Replacement and retained hidden readouts differ in architecture as well as transferred "
        "weights. Scratch initializes the same corresponding classifier architecture randomly; "
        "fine-tuning updates all transferred layers, not a frozen encoder.", "",
        "## Recommended historical checkpoints", ""]
    known = [r for r in rows if r["model_type"] == "classifier" and r["architecture"] in portable
             and r["training_population"] == "enhancers only"]
    best_enhancer = representative(known)
    best_background = next(r for r in rows if r["model_id"] == snapshot["recommended_classifier_id"])
    out.append(table(["Purpose", "Checkpoint", "Epoch", "Val E AP", "Val E+BG AP", "Test E AP", "Test E+BG AP"], [
        [label, "`" + r["model_id"] + "`", r["selected_epoch"], number(r["val_macro_ap"]),
         number(r["val_combined_macro_ap"]), number(r["test_enhancer_macro_ap"]), number(r["test_combined_macro_ap"])]
        for label, r in [("Enhancer-only discrimination", best_enhancer),
                         ("Enhancer + background; calibrated IG analysis", best_background)]]))
    out += ["", "These single-checkpoint scores are not seed averages. A model that wins for E "
            "need not win for E+BG. Use the latter checkpoint for the current calibrated-attribution workflow.", ""]
    sections = [
        ("Primary architectures: enhancer-only training", lambda r: r["architecture"] in portable and r["training_population"] == "enhancers only"),
        ("Primary architectures: 25% background training", lambda r: r["architecture"] in portable and r["training_population"] != "enhancers only"),
        ("Additional historical architectures and shorter-input pilots", lambda r: r["architecture"] not in portable)]
    for title, choose in sections:
        out += ["## " + title, ""]
        if title.startswith("Additional"):
            out += ["These results are retained for completeness. EnhancerNet and 512/1,024-bp "
                    "runs are **not** supported by the portable training dispatcher. The 512-bp "
                    "classifiers ran only 15 epochs of the original 40-epoch LR curve; their test "
                    "metrics were not measured. Short-input regressors also changed H3K27ac "
                    "supervision to 512 bp, so this is not a pure input-length comparison.", ""]
        out += [table(["Ref", "Model", "Hidden policy", "Initialization", "Train population", "Val E AP", "Val E+BG AP", "Test E AP", "Test E+BG AP"], [
            [ref, NAMES[g[0]["architecture"]], hidden(g[0]), initialization(g[0]),
             "E" if g[0]["training_population"] == "enhancers only" else "E + BG",
             *[aggregate(g, key) for key in ("val_macro_ap", "val_combined_macro_ap", "test_enhancer_macro_ap", "test_combined_macro_ap")]]
            for ref, g in grouped if choose(g[0])]), ""]
    out += ["## Background rejection and enhancer discrimination", "",
        "Same seeds and selected checkpoints as above; all values are three-seed mean ± SD. "
        "The FPR column is a diagnostic threshold at 80% enhancer recall, not a deployment cutoff.", "",
        table(["Ref", "Val E AUROC", "Val E Brier", "Val A/BG AP", "Test A/BG AP", "Val BG FPR@80% recall"], [
            [ref, *[aggregate(g, key) for key in ("val_macro_auroc", "val_brier",
             "val_active_vs_background_macro_ap", "test_active_vs_background_macro_ap", "val_background_fpr_at_80_recall")]]
            for ref, g in grouped]), "",
        "## Validation breadth and context discrimination", "",
        "Breadth compares the sum of the original, uncalibrated classifier probabilities "
        "with observed catalog degree of pleiotropy. These are **not** the subsequently fitted "
        "calibrated-probability metrics. Close-context accuracy uses discordant labels within "
        "related context pairs; family hit is the any-active-member top-family diagnostic. "
        "All columns are enhancer-only, three-seed mean ± SD.", "",
        table(["Ref", "Breadth r", "Breadth ρ", "Breadth R²", "Breadth MAE", "Close-context accuracy", "Family hit"], [
            [ref, *[aggregate(g, key) for key in ("val_catalog_breadth_pearson",
             "val_catalog_breadth_spearman", "val_catalog_breadth_r2", "val_catalog_breadth_mae",
             "val_close_context_accuracy", "val_any_member_group_top_choice_hit")]] for ref, g in grouped]), "",
        "## Representative checkpoint settings", "",
        "IDs refer to the validation-selected representative seed per condition, not the best test seed. "
        "Encoder/head columns show **peak** LR. Full selected-epoch and final LRs are in the snapshot. "
        "All classifiers use BCE; scratch uses 1e-4, fine-tuning 1e-5 for transferred layers and "
        "1e-4 for the new head. The schedule is one-epoch warm-up then cosine to 10% of peak.", "",
        table(["Ref", "Representative model ID", "Selected / completed epochs", "Peak LR encoder / head", "Checkpoint SHA256 (prefix)"], [
            [ref, "`" + (r := representative(g))["model_id"] + "`",
             f"{r['selected_epoch']} / {r['epochs_completed']}",
             f"{r['max_lr_encoder']:.0e} / {r['max_lr_head']:.0e}", "`" + r["checkpoint_sha256"][:12] + "`"]
            for ref, g in grouped]), ""]
    for split, template in [("Validation", "val_{}_ap"), ("Test", "test_{}_enhancer_ap")]:
        out += ["## " + split + " AP per context: representative checkpoints", "",
            "Enhancer-only evaluation; values are for the same representative checkpoint selected above. "
            "`ab` adult brain, `e13/e5` embryo stages, `ead/hid/wid` eye-antennal/haltere/wing discs, "
            "`lb` larval brain, `o` ovary.", "",
            table(["Ref", *snapshot["context_order"]], [
                [ref, *[number(representative(g).get(template.format(c))) for c in snapshot["context_order"]]]
                for ref, g in grouped]), ""]
    regressors = [r for r in rows if r["model_type"] == "regressor"]
    out += ["## Regressors", "",
        "One selected checkpoint per architecture. Loss: AlphaGenome-style count/position loss "
        "plus a cross-context auxiliary term; training-only track scaling. All ran 40 epochs. "
        "The shared LR schedule warms to 1e-4 in one epoch, decays to 5e-5 over four epochs, "
        "then uses validation-plateau reductions down to 1e-6. Selection maximizes the "
        "scientific composite. See the [scientific contract](reproduction_science.md).", "",
        table(["ID", "Input bp", "ATAC / H3K27ac target bp", "Epoch", "Val composite", "Val overcorrelation"], [
            ["`" + r["model_id"] + "`", r["input_bp"],
             f"{r['regression_atac_target_bp']} / {r['regression_h3k27ac_target_bp']}", r["selected_epoch"],
             number(r["val_scientific_composite"]), number(r["val_overcorrelation"])] for r in regressors]), ""]
    for assay in ("atac", "h3k27ac"):
        keys = [f"val_{assay}_{key}" for key in ("all_bins_pearson", "all_bins_spearman", "all_bins_r2",
            "center512_pearson", "center512_spearman", "center512_r2", "context_pattern_pearson",
            "regulatory_continuous_breadth_spearman")]
        out += ["### " + assay.upper() + " validation", "",
            table(["Model", "All bins r", "All bins ρ", "All bins R²", "Center r", "Center ρ", "Center R²", "Context-pattern r", "Breadth ρ"], [
                [NAMES[r["architecture"]], *[number(r.get(k)) for k in keys]] for r in regressors]), ""]
    out += ["All-bin metrics pool individual bins within each context and then macro-average the "
        "eight context metrics. Center metrics use the central-512-bp mean. Both are measured "
        "after the original log1p transform; R² is not squared Pearson correlation. "
        "The snapshot retains additional per-context and breadth metrics. Compare matched target "
        "spans: 2048-bp models predict wider H3K27ac profiles than the shorter-input regressors.", "",
        "## Complete checkpoint index", "",
        "Paths below are relative to the **historical artifact root**, not this source checkout. "
        "They identify existing research artifacts but are not download links. Full hashes, "
        "last-checkpoint paths and initialization checkpoint hashes are in the JSON snapshot. "
        "Weights need a separate versioned deposit before this becomes a public model zoo.", "",
        "<details>", f"<summary>All {len(rows)} selected checkpoints and per-run validation results</summary>", "",
        table(["Model ID", "Type / training", "Epoch", "Val E AP / composite", "Checkpoint SHA256", "Artifact path"], [
            ["`" + r["model_id"] + "`", r["model_type"] + " / " + r["training_type"], r["selected_epoch"],
             number(r["val_macro_ap"] if r["model_type"] == "classifier" else r["val_scientific_composite"]),
             "`" + r["checkpoint_sha256"] + "`", "`" + r["checkpoint_artifact_path"] + "`"] for r in rows]), "",
        "</details>", "", "## Rebuild and provenance", "",
        "```bash", "python scripts/summarize_paper_benchmarks.py --check",
        "# After an explicitly reviewed snapshot update:",
        "python scripts/summarize_paper_benchmarks.py --write", "```", "",
        "This is CPU-only report generation from saved metrics. It does not evaluate models or "
        "change scientific results. New portable runs have their own `model_inventory.tsv`; "
        "they do not overwrite this historical benchmark.", "",
        "- Source inventory: `" + snapshot["source_inventory"] + "`.",
        "- Source inventory SHA256: `" + snapshot["source_inventory_sha256"] + "`.",
        "- Original source-snapshot SHA256: `" + snapshot["source_snapshot_sha256"] + "`.",
        "- Snapshot contains no home-directory paths, login names, credentials, sequences or weights.", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=REPORT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    content = render(json.loads(args.snapshot.read_text()))
    if args.check:
        if not args.output.is_file() or args.output.read_text() != content:
            raise SystemExit("Benchmark table is missing/stale; review snapshot, then use --write")
        print("Benchmark table matches all saved metrics and checkpoint metadata.")
    elif args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
        print(args.output)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
