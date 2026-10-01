"""Portable orchestration; original model, objective and optimizers are reused."""
import json
import os
from pathlib import Path
import sys

import numpy as np

from .common import classifiers, digest, require_slurm, run_id, selected_run, write_json
from .data import regression_config


def verify_preprocessing(work):
    root = work / "regression_data"
    receipt = json.loads((root / "prepared.json").read_text())
    if receipt["status"] != "complete":
        raise ValueError("Preprocessing is not complete")
    for name, sha in receipt["output_hashes"].items():
        if digest(root / name) != sha:
            raise ValueError("Prepared regression data changed: "+name)


def train_regressor(cfg, task, resume):
    if not 0 <= task < len(cfg["architectures"]):
        raise ValueError("Invalid regressor task index")
    architecture = cfg["architectures"][task]
    work = Path(cfg["paths"]["work"])
    verify_preprocessing(work)
    output = work / "regressors" / architecture
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        (output / "data").symlink_to(os.path.relpath(work / "regression_data/data", output), target_is_directory=True)
    config = regression_config(cfg, output)
    if architecture == "cnn":
        from classifier_transfer.regression import install_cnn_adapter
        from classifier_transfer.models import CNN_NAME, DilatedBlock
        from classifier_transfer.fast_convolution import interleaved_block_forward
        config["model"]["encoder_variant"] = CNN_NAME
        training = install_cnn_adapter()
        DilatedBlock.forward = interleaved_block_forward
    elif architecture == "dense":
        from classifier_transfer.dense_regression import install_dense_adapter
        from classifier_transfer.models import DENSE_NAME
        config["model"]["encoder_variant"] = DENSE_NAME
        training = install_dense_adapter()
    else:
        from enhancer_pleiotropy_model import training
    config_path = output / "run_config.json"
    if resume and json.loads(config_path.read_text()) != config:
        raise ValueError("Regressor configuration changed")
    if not resume:
        write_json(config_path, config)
    # Replace only the site-specific operational guard, not scientific code.
    training.require_training_node = lambda device: require_slurm(gpu=device != "cpu")
    sys.argv = ["enhancer-train", "--config", str(config_path), "--stop-file", str(output / "STOP")]
    if resume:
        sys.argv.append("--resume")
    training.main()
    if not (output / "model/metrics.json").is_file():
        raise SystemExit(75)
    metrics = json.loads((output / "model/metrics.json").read_text())
    if len(metrics["history"]) != config["training"]["epochs"]:
        raise ValueError("Regressor did not finish the configured schedule")
    write_json(output / "complete.json", dict(status="complete", epochs=len(metrics["history"]),
        best_epoch=metrics["best_epoch"], checkpoint_sha256=digest(output / "model/best_model.pt")))


def train_classifier(cfg, task, resume):
    from classifier_transfer import train
    from classifier_transfer.models import DilatedBlock
    from classifier_transfer.fast_convolution import interleaved_block_forward
    tasks = list(classifiers(cfg))
    if not 0 <= task < len(tasks):
        raise ValueError("Invalid classifier task index")
    entry = tasks[task]
    work = Path(cfg["paths"]["work"])
    settings = dict(cfg["classifier"], readout=entry["readout"])
    if entry["population"] == "background":
        settings["background"] = dict(enhancers_per_background=3, selection="enhancer_only")
    data = work / ("enhancers_background" if entry["population"] == "background" else "enhancers")
    checkpoint = work / "regressors" / entry["architecture"] / "model/best_model.pt"
    if entry["mode"] == "scratch":
        checkpoint = None
    else:
        done = json.loads(checkpoint.parent.parent.joinpath("complete.json").read_text())
        if done["status"] != "complete" or done["checkpoint_sha256"] != digest(checkpoint):
            raise ValueError("Pretrained parent must finish before classifier fine-tuning")
    train.require_cuda_allocation = lambda: require_slurm(gpu=True)
    if entry["architecture"] == "cnn":
        DilatedBlock.forward = interleaved_block_forward
    complete = train.run(data, work / "classifiers" / run_id(**entry), entry["architecture"],
                         entry["seed"], settings, checkpoint, resume=resume)
    if not complete:
        raise SystemExit(75)  # Do not release dependent evaluation after a checkpointed pause.


def load_classifier(path, device="cuda"):
    import torch
    from classifier_transfer.models import initialize_classifier, DilatedBlock
    from classifier_transfer.fast_convolution import interleaved_block_forward
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("kind") != "v4_context_classifier":
        raise ValueError("Expected a trusted selected classifier checkpoint")
    settings = saved["settings"]
    model = initialize_classifier(settings["architecture"], settings["seed"], np.full(8, .5),
                                  readout=settings["training"].get("readout", "legacy"))
    model.load_state_dict(saved["state_dict"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite classifier")
    if settings["architecture"] == "cnn":
        DilatedBlock.forward = interleaved_block_forward
    return model.eval().requires_grad_(False).to(device)


def evaluate(cfg):
    import torch
    from classifier_transfer.data import load_split
    from classifier_transfer.metrics import summarize
    from classifier_transfer.background import load_background, background_metrics
    from classifier_transfer.train import predict
    from classifier_transfer.data import write_json as write_metrics
    work = Path(cfg["paths"]["work"])
    entries = list(classifiers(cfg))
    locks = {}
    # Lock EVERY validation selection before opening the test split.
    for entry in entries:
        name = run_id(**entry); root = work / "classifiers" / name
        done = json.loads((root / "complete.json").read_text())
        if (done["status"] != "complete" or done["epochs"] != cfg["classifier"]["epochs"]
                or done["checkpoint_sha256"] != digest(root / "best_model.pt")):
            raise ValueError("All planned classifiers must finish before test evaluation")
        locks[name] = dict(entry, checkpoint_sha256=done["checkpoint_sha256"], epoch=done["best_epoch"])
    output = work / "evaluation"
    output.mkdir(exist_ok=False)
    write_json(output / "selection_lock.json", locks)
    test = load_split(work / "enhancers", "test")
    background = (load_background(work / "enhancers_background/background", "test", test)
                  if "background" in cfg["populations"] else None)
    device = torch.device("cuda")
    metrics = {}
    for name in locks:
        root = work / "classifiers" / name
        model = load_classifier(root / "best_model.pt")
        p = predict(model, test, device, cfg["classifier"]["batch_size"])
        arrays = dict(ids=test["ids"], probabilities=p, labels=test["labels"])
        result = dict(enhancer_only=summarize(test["labels"], p))
        if background is not None:
            bg = predict(model, background, device, cfg["classifier"]["batch_size"])
            result.update(background_metrics(test["labels"], p, bg, background["enhancer_index"]))
            arrays.update(background_ids=background["ids"], background_probabilities=bg)
        np.savez_compressed(output / (name + ".npz"), **arrays)
        metrics[name] = result
        write_metrics(output / "metrics.json", metrics)
        del model
    write_json(output / "complete.json", dict(status="complete", models=len(metrics),
        caveat="Historical chr3R, not a pristine new holdout; AP depends on the evaluation population"))


def calibrate(cfg):
    from classifier_transfer.calibrate_breadth import fit, CONTEXTS
    from classifier_transfer.data import load_split
    root = selected_run(cfg)
    work = Path(cfg["paths"]["work"])
    complete = json.loads((root / "complete.json").read_text())
    checkpoint = root / "best_model.pt"
    if complete["checkpoint_sha256"] != digest(checkpoint):
        raise ValueError("Selected checkpoint changed")
    data = load_split(work / "enhancers", "validation")
    with np.load(root / "best_validation_predictions.npz", allow_pickle=False) as f:
        np.testing.assert_array_equal(f["ids"], data["ids"])
        calibration = fit(f["probabilities"], data["labels"], "sigmoid")
    # This is the deployment fit. No in-sample performance is presented as CV/test accuracy.
    calibration.update(checkpoint_sha256=digest(checkpoint), contexts=list(CONTEXTS),
        fitting_population="enhancers_only", validation_sha256=digest(work / "enhancers/validation.npz"),
        interpretation="Deployment fit on all validation enhancers; not a held-out calibration assessment")
    path = work / "calibrators.json"
    if path.exists():
        raise FileExistsError(path)
    write_json(path, calibration)
