"""Evaluate locked classifiers; never select models using test results."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from enhancer_pleiotropy_model.breadth_metrics import block_bootstrap, group_counts
from .data import digest, load_split, write_json
from .metrics import summarize, thresholds_from_validation
from .models import initialize_classifier
from .train import predict, require_cuda_allocation, event


def run(root, config):
    require_cuda_allocation()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    root = Path(root)
    validation = load_split(root/"data", "validation")
    architectures = config.get("architectures", ["attention", "cnn"])
    locks = {}
    for architecture in architectures:
        for mode in ("scratch", "finetune"):
            for seed in config["seeds"]:
                name = f"{architecture}_{mode}_{seed}"
                directory = root/"runs"/name
                complete = json.loads((directory/"complete.json").read_text())
                if complete["status"] != "complete" or complete["epochs"] != config["classifier"]["epochs"]:
                    raise ValueError(f"Incomplete classifier: {name}")
                if complete["checkpoint_sha256"] != digest(directory/"best_model.pt"):
                    raise ValueError("Selected classifier hash mismatch")
                settings=complete["settings"]
                if (settings["architecture"]!=architecture or settings["seed"]!=seed
                        or settings["training"]!=config["classifier"]
                        or (settings["initialization_sha256"] is None)!=(mode=="scratch")
                        or settings["data_audit_sha256"]!=digest(root/"data/audit.json")):
                    raise ValueError("Classifier comparison configuration mismatch")
                with np.load(directory/"best_validation_predictions.npz") as cached:
                    if not np.array_equal(cached["ids"], validation["ids"]): raise ValueError("Validation IDs differ")
                    thresholds = thresholds_from_validation(validation["labels"], cached["probabilities"])
                locks[name] = dict(checkpoint_sha256=complete["checkpoint_sha256"],
                                   epoch=complete["best_epoch"], thresholds=thresholds.tolist(),
                                   architecture=architecture, seed=seed, mode=mode)
    output = root/"comparison"
    output.mkdir(exist_ok=False)
    write_json(output/"selection_lock.json", locks)
    # No test sequence or labels are opened until every model/threshold is locked.
    test = load_split(root/"data", "test")
    results, probabilities = {}, {}
    for name, selected in locks.items():
        directory = root/"runs"/name
        saved = torch.load(directory/"best_model.pt", map_location="cpu", weights_only=False)
        model = initialize_classifier(selected["architecture"], selected["seed"], np.full(8,.5),
                                      None, config["classifier"].get("readout", "legacy")).cuda()
        model.load_state_dict(saved["state_dict"], strict=True)
        probabilities[name] = p = predict(model, test, torch.device("cuda"))
        results[name] = summarize(test["labels"], p, np.asarray(selected["thresholds"]))
        results[name]["selected_epoch"] = selected["epoch"]
        results[name]["parameters"] = sum(v.numel() for v in model.parameters())
        results[name]["elapsed_seconds"] = json.loads((root/"logs"/f"{name}.stage.json").read_text())["elapsed_seconds"]
        np.savez_compressed(output/f"{name}.npz", ids=test["ids"], probabilities=p,
                            expected_breadth=p.sum(1), group_contributions=group_counts(p))
        event("classifier_test_evaluated", name=name, macro_ap=results[name]["macro_average_precision"])
        del model
    intervals = {}
    blocks = np.asarray([f"{c}:{s//1000000}" for c,s in zip(test["chrom"],test["summit"])])
    for architecture in architectures:
        for seed in config["seeds"]:
            pair = {mode:probabilities[f"{architecture}_{mode}_{seed}"] for mode in ("scratch", "finetune")}
            intervals[f"{architecture}_{seed}"] = block_bootstrap(test["labels"], pair, blocks,
                                               reference="scratch", replicates=500, seed=seed)
    def compact(row):
        return dict(macro_ap=row["macro_average_precision"], macro_auroc=row["macro_auroc"],
                    brier=row["brier"], breadth_r2=row["breadth"]["r2"], breadth_mae=row["breadth"]["mae"],
                    close_context_accuracy=row["related_pair_macro_accuracy"],
                    group_hit=row["any_member_group_top_choice_hit"])
    aggregates = {}
    for architecture in architectures:
        for mode in ("scratch", "finetune"):
            rows = [compact(results[f"{architecture}_{mode}_{seed}"]) for seed in config["seeds"]]
            aggregates[f"{architecture}_{mode}"] = {k:dict(mean=float(np.mean([r[k] for r in rows])),
                                                          sd=float(np.std([r[k] for r in rows],ddof=1))) for k in rows[0]}
    differences = {}
    for architecture in architectures:
        rows = []
        for seed in config["seeds"]:
            a,b = (compact(results[f"{architecture}_{mode}_{seed}"]) for mode in ("finetune", "scratch"))
            rows.append({k:a[k]-b[k] for k in a})
        differences[architecture] = {k:dict(mean=float(np.mean([r[k] for r in rows])),
                                           sd=float(np.std([r[k] for r in rows],ddof=1))) for k in rows[0]}
    regression = {}
    for architecture in architectures:
        selected = json.loads((root/"parents"/f"{architecture}.json").read_text())
        history_path = root/"parents/attention_history.json" if architecture=="attention" else root/f"{architecture}_regression/model/history.json"
        history = json.loads(history_path.read_text())
        row = next(e for e in history if e["epoch"]==selected["epoch"])
        regression[architecture] = dict(epoch=selected["epoch"], validation_loss=row["validation_losses"]["total"],
                     composite=row["scientific_composite"], overcorrelation=row["regulatory_overcorrelation"],
                     validation=row["validation"], continuous_breadth=row["continuous_breadth"])
    report = dict(status="complete", test_n=len(test["labels"]), selection=locks, results=results,
                  seed_aggregates=aggregates, paired_finetune_minus_scratch=differences,
                  block_bootstrap=intervals, regressor_best_validation=regression,
                  seed_scope="Three classification seeds conditional on one regressor per architecture",
                  test_scope="Historical chr3R holdout; previously examined in project development")
    write_json(output/"metrics.json", report)
    lines = ["# Classifier transfer comparison", "", "Historical chr3R test; all selections frozen on validation.", "",
             "Mean ± sample SD over three classification seeds (not independent regression pretraining seeds).", "",
             "| Model | Macro AP | AUROC | Brier | Breadth R² | Breadth MAE | Close-context accuracy | Group hit |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, values in aggregates.items():
        lines.append("| "+name+" | "+" | ".join(f"{v['mean']:.4f} ± {v['sd']:.4f}" for v in values.values())+" |")
    lines += ["", "Breadth means expected catalog-active context count, not continuous signal intensity.",
              "Per-context/group contributions, reliability, thresholds and matched seed differences are in metrics.json.",
              "Block intervals cover count/probability-error diagnostics, not AP or pretraining uncertainty.", ""]
    (output/"summary.md").write_text("\n".join(lines))
    from .report import generate
    generate([root], output/"comprehensive")
    event("classifier_comparison_complete", test_n=len(test["labels"]), models=len(results))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True); p.add_argument("--config",type=Path,required=True)
    a=p.parse_args(); run(a.root,json.loads(a.config.read_text()))
