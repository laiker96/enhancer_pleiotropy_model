"""Bounded joint-loss fine-tuning on compute nodes or explicitly enabled local CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .browser_report import finite_or_none
from .constants import CONTEXTS
from .continuous_breadth import evaluate_activity
from .data import JointProfileCollator, JointProfileDataset, load_profiles, make_loader, read_windows
from .inference import load_model, resolve_device
from .io import atomic_write_json, sha256_file
from .master_element_calibration import calibrate
from .multitask_loss import ASSAYS, JointBreadthLoss, build_target_pools, mixed_batch_plan, summarize_numpy
from .training import (
    WarmupPlateauScheduler, atomic_torch_save, autocast_context, load_training_state,
    move_batch, reverse_complement_batch, save_training_state, seed_everything,
)


def event(name: str, **values) -> None:
    print(json.dumps(finite_or_none(dict(event=name, **values)), sort_keys=True), flush=True)


def require_compute_node(device: str, *, allow_local_cuda: bool = False) -> None:
    if socket.gethostname().split(".")[0] == "neocranex":
        raise RuntimeError("Refusing model work on the login node")
    if device == "cuda" and not os.environ.get("SLURM_JOB_ID") and not allow_local_cuda:
        raise RuntimeError("CUDA requires a Slurm compute allocation or explicit --allow-local-cuda")


def acceptable(metrics: dict, baseline: dict, tolerance: float) -> bool:
    """Guard each assay's regulatory signal, exact breadth and broad-cohort magnitude."""
    for assay in ASSAYS:
        checks = [(metrics[assay]["regulatory_profile_log_mse"], baseline[assay]["regulatory_profile_log_mse"]),
                  (metrics[assay]["regulatory"]["total"]["rmse"], baseline[assay]["regulatory"]["total"]["rmse"])]
        if baseline[assay]["broad_n"]:
            checks.append((metrics[assay]["broad_log_summary_rmse"], baseline[assay]["broad_log_summary_rmse"]))
        if any(not np.isfinite(value) or value > (1 + tolerance) * reference + 1e-12 for value, reference in checks):
            return False
    return True


@torch.no_grad()
def evaluate(model, loader, criterion, device, precision, backgrounds, peak_mask, maximum_batches=None):
    model.eval()
    observed, predicted = {a: [] for a in ASSAYS}, {a: [] for a in ASSAYS}
    sums, count, profile_errors = {}, 0, {a: [] for a in ASSAYS}
    for batch_i, batch in enumerate(loader):
        inputs = move_batch(batch, device)
        with autocast_context(device, precision):
            forward = model(*inputs[:4])
            reverse = model(*reverse_complement_batch(*inputs[:4]))
            predictions = tuple(0.5 * (x.float() + y.float().flip(1)) for x, y in zip(forward, reverse, strict=True))
            losses = criterion(predictions, inputs[4:])
        batch_n = len(inputs[0])
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + value.item() * batch_n
        for a, truth, prediction in zip(ASSAYS, inputs[4:], predictions, strict=True):
            observed[a].append(truth.float().cpu().numpy())
            predicted[a].append(prediction.float().cpu().numpy())
            profile_errors[a].append((torch.log1p(prediction) - torch.log1p(truth)).square().mean(dim=(1, 2)).cpu().numpy())
        count += batch_n
        if batch_i % 100 == 0:
            event("validation_progress", batch=batch_i + 1, elements=count)
        if maximum_batches and batch_i + 1 >= maximum_batches:
            break
    if not count or not peak_mask[:count].any():
        raise ValueError("Validation requires examples including regulatory windows")
    observed = {a: np.concatenate(v) for a, v in observed.items()}
    predicted = {a: np.concatenate(v) for a, v in predicted.items()}
    truth_summary = summarize_numpy(observed["atac"], observed["h3k27ac"])
    pred_summary = summarize_numpy(predicted["atac"], predicted["h3k27ac"])
    result = dict(elements=count, losses={k: v / count for k, v in sums.items()})
    for a, assay in enumerate(ASSAYS):
        truth, prediction = calibrate(truth_summary[a], backgrounds[assay])[1], calibrate(pred_summary[a], backgrounds[assay])[1]
        regulatory = peak_mask[:count]
        broad = regulatory & (truth.sum(axis=1) >= 7)
        log_error = np.log1p(pred_summary[a]) - np.log1p(truth_summary[a])
        errors = np.concatenate(profile_errors[assay])
        result[assay] = dict(profile_log_mse=float(errors.mean()), regulatory_profile_log_mse=float(errors[regulatory].mean()),
                             all=evaluate_activity(truth, prediction), regulatory=evaluate_activity(truth[regulatory], prediction[regulatory]),
                             regulatory_log_signal=evaluate_activity(np.log1p(truth_summary[a, regulatory]), np.log1p(pred_summary[a, regulatory])),
                             broad_n=int(broad.sum()), broad_log_summary_rmse=float(np.sqrt(np.mean(log_error[broad]**2))) if broad.any() else None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-local-cuda", action="store_true",
                        help="Explicitly permit local CUDA outside Slurm; never permit the login node")
    args = parser.parse_args()
    if args.allow_local_cuda and args.device != "cuda":
        parser.error("--allow-local-cuda requires --device cuda")
    require_compute_node(args.device, allow_local_cuda=args.allow_local_cuda)
    config = yaml.safe_load(args.config.read_text())
    training, loss_config = config["training"], config["loss"]
    for key in ("epochs", "steps_per_epoch", "natural_batch", "targeted_batch", "validation_batch", "checkpoint_every", "contrast_ramp_steps"):
        if int(training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if not 0 <= training["guard_tolerance"] < 1:
        raise ValueError("Guard tolerance must be in [0,1)")
    output = Path(config["output_directory"] + ("_smoke" if args.smoke_test else ""))
    if output.exists() and not args.resume:
        raise FileExistsError("Use --resume for a matching run or select a new output path")
    device = resolve_device(args.device)
    precision = "fp16" if device.type == "cuda" else "no"
    seed = int(config["seed"])
    seed_everything(seed, device)
    torch.set_num_threads(int(training["cpu_threads"]))
    event("preflight_start", hostname=socket.gethostname(), job_id=os.environ.get("SLURM_JOB_ID"), device=str(device),
          gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None, smoke_test=args.smoke_test)
    data_directory = Path(config["data_directory"])
    dataset_path, background_path = data_directory / "windows.tsv.gz", Path(config["background"])
    input_hashes = {"windows": sha256_file(dataset_path), "background": sha256_file(background_path),
                    "checkpoint": sha256_file(Path(config["initialization_checkpoint"]))}
    if input_hashes != config["expected_hashes"]:
        raise ValueError("Input hashes differ from the pinned experiment inputs")
    background_metadata = json.loads(background_path.with_suffix(".metadata.json").read_text())
    allowed = {"chr2R", "chr3L", "chr4", "chrY", "chrUn_CP007081v1", "chrUn_CP007120v1"}
    if not set(background_metadata["selection"]["background_chromosomes"]) or not set(background_metadata["selection"]["background_chromosomes"]).issubset(allowed):
        raise ValueError("Normalization reference must contain only retained training chromosomes")
    with np.load(background_path, allow_pickle=False) as loaded:
        if tuple(loaded["contexts"]) != CONTEXTS:
            raise ValueError("Background context order differs")
        backgrounds = {a: loaded[f"{a}_sorted"].astype(float) for a in ASSAYS}
    records = read_windows(dataset_path)
    if any(r.chrom == "chr3R" for r in records["train"]) or any(r.chrom != "chr2L" for r in records["validation"]):
        raise ValueError("Expected final training/validation chromosome split")
    counts = {s: len(r) for s, r in records.items()}
    profiles, metadata = {}, {}
    for a in ASSAYS:
        profiles[a], metadata[a] = load_profiles(data_directory / "profiles" / a, dataset_path, counts)
        if tuple(metadata[a]["contexts"]) != CONTEXTS:
            raise ValueError("Prepared profile context order differs")
        event("profile_integrity_verified", assay=a, shape=list(profiles[a]["train"].shape))
    # Test profiles are integrity-checked by the shared loader, never fitted or evaluated.
    criterion = JointBreadthLoss(backgrounds, **loss_config).to(device)
    approximation = {a: criterion.activity[a].max_reference_error for a in ASSAYS}
    if max(approximation.values()) > 0.01:
        raise ValueError("Background interpolation differs from reference by more than 0.01 activity units")
    summary = np.empty((2, counts["train"], 8), dtype=np.float64)
    for start in range(0, counts["train"], 4096):
        end = min(start + 4096, counts["train"])
        summary[:, start:end] = summarize_numpy(profiles["atac"]["train"][start:end], profiles["h3k27ac"]["train"][start:end])
    activity = np.stack([calibrate(summary[i], backgrounds[a])[1] for i, a in enumerate(ASSAYS)])
    peak = np.asarray([r.source != "genomic_background" for r in records["train"]])
    pools, pool_metadata = build_target_pools(summary, activity, peak)
    with torch.no_grad():
        approximation["training_max_error"] = max(float(np.max(np.abs(criterion.activity[a](torch.tensor(summary[i], dtype=torch.float32, device=device)).cpu().numpy() - activity[i]))) for i, a in enumerate(ASSAYS))
    if approximation["training_max_error"] > 0.01:
        raise ValueError("Training activity interpolation error exceeds 0.01")
    del summary, activity
    model, _ = load_model(config["initialization_checkpoint"], device)
    initial = torch.load(config["initialization_checkpoint"], map_location="cpu", weights_only=False)
    if initial["dataset_sha256"] != input_hashes["windows"]:
        raise ValueError("Initialization checkpoint was trained on a different prepared dataset")
    datasets = {s: JointProfileDataset(records[s], profiles["atac"][s], profiles["h3k27ac"][s], 4,
                                       training=s == "train", rc_probability=0.5 if s == "train" else 0, seed=seed)
                for s in ("train", "validation")}
    val_loader = make_loader(datasets["validation"], batch_size=int(training["validation_batch"]), workers=0, epoch=0, seed=seed, training=False, pin_memory=device.type == "cuda")
    val_peak = np.asarray([r.source != "genomic_background" for r in records["validation"]])
    epochs = 1 if args.smoke_test else int(training["epochs"])
    steps = 2 if args.smoke_test else int(training["steps_per_epoch"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=0.01)
    scheduler = WarmupPlateauScheduler(optimizer, maximum_learning_rate=training["learning_rate"], post_warmup_learning_rate=training["learning_rate"],
                                       warmup_steps=min(100, steps), decay_steps=0, plateau_factor=0.5, plateau_patience=1,
                                       plateau_threshold=1e-4, minimum_learning_rate=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    source_hashes = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    signature = dict(config=config, input_hashes=input_hashes, source_hashes=source_hashes, smoke_test=args.smoke_test,
                     allow_local_cuda=args.allow_local_cuda,
                     device=str(device), precision=precision, interpolation=approximation,
                     background_metadata_hash=sha256_file(background_path.with_suffix(".metadata.json")),
                     profiles={a: metadata[a]["outputs"] for a in ASSAYS})
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["signature"] != signature:
        raise ValueError("Existing run configuration, code or inputs differ; refusing overwrite")
    output.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        atomic_write_json(manifest_path, dict(signature=signature, host=socket.gethostname(), job_id=os.environ.get("SLURM_JOB_ID"),
                                             versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)))
    atomic_write_json(output / "sampling.json", pool_metadata)
    atomic_torch_save({a: criterion.activity[a].state_dict() for a in ASSAYS}, output / "activity_interpolation.pt")
    progress = dict(next_epoch=0, next_batch=0, history=[], best_score=-float("inf"), best_objective=-float("inf"), best_epoch=0)
    last = output / "last_checkpoint.pt"
    if args.resume and last.exists():
        progress = load_training_state(last, model, optimizer, scheduler, scaler, signature, device)
    baseline_path = output / "initialization_metrics.json"
    if args.resume and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
    else:
        baseline = evaluate(model, val_loader, criterion, device, precision, backgrounds, val_peak, 2 if args.smoke_test else None)
        atomic_write_json(baseline_path, finite_or_none(baseline))
    def save_model(path, epoch, score, passed):
        checkpoint = {k: v for k, v in initial.items() if k != "state_dict"}
        checkpoint.update(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, epoch=epoch,
                          training_stage="joint_breadth_close_context", initialization=dict(path=config["initialization_checkpoint"], sha256=input_hashes["checkpoint"]),
                          checkpoint_selection=dict(metric="negative_joint_validation_loss", mode="max", score=score, guards_passed=passed),
                          loss=dict(name="signal_breadth_group_close_context_v1", **loss_config), run_signature=signature)
        atomic_torch_save(checkpoint, path)
    if not last.exists():
        progress["best_score"] = progress["best_objective"] = -baseline["losses"]["total"]
        save_model(output / "best_model.pt", 0, progress["best_score"], True)
        save_model(output / "best_objective_model.pt", 0, progress["best_score"], True)
        save_training_state(last, model, optimizer, scheduler, scaler, signature, progress, device)
    event("joint_training_start", epochs=epochs, steps_per_epoch=steps, natural_batch=training["natural_batch"], targeted_batch=training["targeted_batch"],
          baseline_loss=baseline["losses"]["total"], approximation=approximation, resume_epoch=progress["next_epoch"], resume_batch=progress["next_batch"])
    start_epoch, start_batch = progress["next_epoch"], progress["next_batch"]
    for epoch in range(start_epoch, epochs):
        plan, target_assays, target_pairs = mixed_batch_plan(counts["train"], pools, steps=steps, natural_batch=training["natural_batch"],
                                                           targeted_batch=training["targeted_batch"], seed=seed, epoch=epoch)
        offset = start_batch if epoch == start_epoch else 0
        datasets["train"].set_epoch(epoch)
        loader = DataLoader(datasets["train"], batch_sampler=plan[offset:].tolist(), collate_fn=JointProfileCollator(), num_workers=0,
                            pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        for batch_index, batch in enumerate(loader, start=offset):
            step_started = time.monotonic()
            inputs = move_batch(batch, device)
            global_step = epoch * steps + batch_index + 1
            ramp = min(1.0, global_step / training["contrast_ramp_steps"])
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                predictions = model(*inputs[:4])
                losses = criterion(predictions, inputs[4:], natural_n=training["natural_batch"],
                                   targeted_assay=torch.as_tensor(target_assays[batch_index], device=device),
                                   targeted_pair=torch.as_tensor(target_pairs[batch_index], device=device), contrast_ramp=ramp)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError("Nonfinite joint loss; no optimizer update performed")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm) and device.type != "cuda":
                raise FloatingPointError("Nonfinite gradients")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            if not skipped:
                scheduler.step()
            if batch_index < 3 or (batch_index + 1) % 100 == 0 or batch_index + 1 == steps:
                event("joint_training_progress", epoch=epoch + 1, batch=batch_index + 1, batches=steps, contrast_ramp=ramp,
                      learning_rate=optimizer.param_groups[0]["lr"], losses={k: v.item() for k, v in losses.items()},
                      gradient_norm=gradient_norm.item(), optimizer_step_skipped=skipped, elapsed_step_seconds=time.monotonic() - step_started)
            progress.update(next_epoch=epoch, next_batch=batch_index + 1)
            if (batch_index + 1) % training["checkpoint_every"] == 0:
                save_training_state(last, model, optimizer, scheduler, scaler, signature, progress, device)
        metrics = evaluate(model, val_loader, criterion, device, precision, backgrounds, val_peak, 2 if args.smoke_test else None)
        score = -metrics["losses"]["total"]
        passed = acceptable(metrics, baseline, training["guard_tolerance"])
        if score > progress["best_objective"]:
            progress["best_objective"] = score
            save_model(output / "best_objective_model.pt", epoch + 1, score, passed)
        if passed and score > progress["best_score"]:
            progress.update(best_score=score, best_epoch=epoch + 1)
            save_model(output / "best_model.pt", epoch + 1, score, True)
        scheduler.step_validation(score)
        row = dict(epoch=epoch + 1, validation=metrics, guards_passed=passed, score=score, selected_epoch=progress["best_epoch"])
        progress["history"].append(row)
        progress.update(next_epoch=epoch + 1, next_batch=0)
        atomic_write_json(output / "history.json", finite_or_none(progress["history"]))
        save_training_state(last, model, optimizer, scheduler, scaler, signature, progress, device)
        event("joint_epoch_complete", epoch=epoch + 1, loss=-score, guards_passed=passed, selected_epoch=progress["best_epoch"])
    if args.smoke_test and scheduler.optimizer_steps == 0:
        raise FloatingPointError("Smoke test completed without a successful optimizer update")
    event("joint_training_complete", selected_epoch=progress["best_epoch"], best_validation_loss=-progress["best_score"],
          optimizer_steps=scheduler.optimizer_steps, output=str(output))


if __name__ == "__main__":
    main()
