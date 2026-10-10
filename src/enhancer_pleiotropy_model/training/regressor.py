"""Train an ATAC/H3K27ac profile regressor (the pretraining stage)."""

from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ..config import without_private
from ..constants import ASSAYS, ATAC_BINS, CONTEXTS, H3K27AC_BINS, H3K27AC_OUTPUT_POOL_SIZE
from ..data.regression import ProfileDataset, load_profiles, make_loader, profile_means, read_windows
from ..io import sha256_file, write_json
from ..models import make_regressor
from .common import (atomic_torch_save, capture_rng_state, resolve_device, restore_rng_state,
                     seed_everything)
from .losses import AlphaGenomeProfileLoss, fit_nonzero_means
from .regression_metrics import assay_metrics, scientific_composite


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


class WarmupPlateauScheduler:
    """Linear warm-up to the maximum rate, cosine decay to a plateau rate, then
    halve the rate when the validation score stops improving."""

    def __init__(self, optimizer, maximum_learning_rate: float, post_warmup_learning_rate: float,
                 warmup_steps: int, decay_steps: int, plateau_factor: float, plateau_patience: int,
                 plateau_threshold: float, minimum_learning_rate: float) -> None:
        self.optimizer = optimizer
        self.maximum_learning_rate = maximum_learning_rate
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps, self.decay_steps = warmup_steps, decay_steps
        self.scheduled_steps = warmup_steps + decay_steps
        self.optimizer_steps = 0
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=plateau_factor, patience=plateau_patience,
            threshold=plateau_threshold, threshold_mode="rel", min_lr=minimum_learning_rate)
        self._set(self._rate(1))

    def _set(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def _rate(self, step: int) -> float:
        if step <= self.warmup_steps:
            return self.maximum_learning_rate * step / self.warmup_steps
        if self.decay_steps and step <= self.scheduled_steps:
            progress = (step - self.warmup_steps) / self.decay_steps
            return self.post_warmup_learning_rate + 0.5 * (
                self.maximum_learning_rate - self.post_warmup_learning_rate) * (1.0 + math.cos(math.pi * progress))
        return self.post_warmup_learning_rate

    def step(self) -> None:
        self.optimizer_steps += 1
        if self.optimizer_steps + 1 <= self.scheduled_steps:
            self._set(self._rate(self.optimizer_steps + 1))
        elif self.optimizer_steps == self.scheduled_steps:
            self._set(self.post_warmup_learning_rate)

    def step_validation(self, score: float) -> None:
        if self.optimizer_steps >= self.scheduled_steps:
            self.plateau.step(score)

    def state_dict(self) -> dict:
        return {"optimizer_steps": self.optimizer_steps, "plateau": self.plateau.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.optimizer_steps = int(state["optimizer_steps"])
        self.plateau.load_state_dict(state["plateau"])


def autocast(device: torch.device, precision: str):
    if precision == "no":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def batch_tensors(batch: dict, device: torch.device):
    names = ("one_hot", "attention_mask", "atac_target_mask", "h3k27ac_target_mask", "atac_labels", "h3k27ac_labels")
    return tuple(batch[name].to(device, non_blocking=True) for name in names)


def total_loss(predictions, labels, criteria) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_assay = {assay: criterion.components(prediction, target)["total"]
                 for assay, prediction, target, criterion in zip(ASSAYS, predictions, labels, criteria, strict=True)}
    return sum(per_assay.values()), per_assay


def optimizer_step(loss, model, optimizer, scheduler, scaler, clip_norm: float) -> bool:
    """One AMP step. Returns True if fp16 overflow made the scaler skip the update;
    skipped updates do not advance the learning-rate schedule."""
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    except RuntimeError:
        if not scaler.is_enabled() or not any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < before
    if not skipped:
        scheduler.step()
    return skipped


@torch.no_grad()
def evaluate(model, loader, criteria, device, precision: str, max_batches: int | None = None):
    """Forward/reverse-complement-averaged predictions and mean losses on a split."""
    model.eval()
    sums, count = {"atac": 0.0, "h3k27ac": 0.0, "total": 0.0}, 0
    labels = {assay: [] for assay in ASSAYS}
    predictions = {assay: [] for assay in ASSAYS}
    for index, batch in enumerate(loader, start=1):
        x, mask, atac_mask, h3_mask, atac_y, h3_y = batch_tensors(batch, device)
        with autocast(device, precision):
            forward = model(x, mask, atac_mask, h3_mask)
            reverse = model(x.flip((1, 2)), mask.flip(1), atac_mask.flip(1), h3_mask.flip(1))
            output = tuple(0.5 * (f + r.flip(1)) for f, r in zip(forward, reverse, strict=True))
            loss, per_assay = total_loss(output, (atac_y, h3_y), criteria)
        n = len(atac_y)
        count += n
        sums["total"] += float(loss.item()) * n
        for assay in ASSAYS:
            sums[assay] += float(per_assay[assay].item()) * n
        for assay, target, prediction in zip(ASSAYS, (atac_y, h3_y), output, strict=True):
            labels[assay].append(target.cpu().numpy())
            predictions[assay].append(prediction.float().cpu().numpy())
        if max_batches and index >= max_batches:
            break
    return ({k: v / count for k, v in sums.items()},
            {a: np.concatenate(v) for a, v in labels.items()},
            {a: np.concatenate(v) for a, v in predictions.items()})


def train_regressor(config: dict[str, Any], *, resume: bool = False) -> Path:
    architecture = config["architecture"]
    seed = int(config["seed"])
    t = config["training"]
    precision = str(t["mixed_precision"]).lower()
    precision = "no" if precision in ("no", "none", "false", "off") else precision
    if precision not in ("no", "fp16", "bf16"):
        raise ValueError("training.mixed_precision must be fp16, bf16 or none")
    device = resolve_device(config["device"])
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training needs a GPU; use --set training.mixed_precision=none on CPU")
    seed_everything(seed, device)
    data = Path(config["data"])
    output = Path(config["output"]) / architecture
    last_path, best_path = output / "last_checkpoint.pt", output / "best_model.pt"
    if not resume and best_path.exists():
        raise FileExistsError(f"{output} already contains a trained model; use --resume or another output")

    windows_path = data / "windows.tsv.gz"
    windows = read_windows(windows_path)
    counts = {split: len(values) for split, values in windows.items()}
    atac, _ = load_profiles(data / "profiles" / "atac", windows_path, counts)
    h3, _ = load_profiles(data / "profiles" / "h3k27ac", windows_path, counts)
    pool = H3K27AC_OUTPUT_POOL_SIZE
    train_set = ProfileDataset(windows["train"], atac["train"], h3["train"], pool, training=True,
                               rc_probability=float(t["rc_probability"]), seed=seed)
    validation_set = ProfileDataset(windows["validation"], atac["validation"], h3["validation"], pool,
                                    training=False, rc_probability=0, seed=seed)
    validation_loader = make_loader(validation_set, batch_size=int(t["eval_batch_size"]), epoch=0, seed=seed,
                                    training=False, pin_memory=device.type == "cuda")
    regulatory = np.asarray([w.source != "genomic_background" for w in windows["validation"]])
    target_means = {"atac": profile_means(atac["train"]), "h3k27ac": profile_means(h3["train"], pool)}
    track_means = {"atac": fit_nonzero_means(atac["train"]), "h3k27ac": fit_nonzero_means(h3["train"], pool)}
    output_scaling = {assay: values.tolist() for assay, values in track_means.items()}

    model = make_regressor(architecture, output_scaling, float(config["model"]["dropout"]),
                           float(config["model"]["head_dropout"])).to(device)
    model.initialize_output_means(target_means["atac"], target_means["h3k27ac"])
    loss = config["loss"]
    weights = {k: float(loss[k]) for k in ("positional_weight", "cross_context_weight", "auxiliary_weight")}
    criteria = tuple(AlphaGenomeProfileLoss(track_means[assay], bins=bins, **weights).to(device)
                     for assay, bins in (("atac", ATAC_BINS), ("h3k27ac", H3K27AC_BINS)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t["max_learning_rate"]),
                                  weight_decay=float(t["weight_decay"]))
    batch_size, epochs = int(t["batch_size"]), int(t["epochs"])
    batches_per_epoch = math.ceil(len(train_set) / batch_size)
    scheduler = WarmupPlateauScheduler(
        optimizer, float(t["max_learning_rate"]), float(t["post_warmup_learning_rate"]),
        warmup_steps=max(1, round(epochs * batches_per_epoch * float(t["warmup_fraction"]))),
        decay_steps=max(0, round(batches_per_epoch * float(t["post_warmup_decay_epochs"]))),
        plateau_factor=float(t["plateau_factor"]), plateau_patience=int(t["plateau_patience"]),
        plateau_threshold=float(t["plateau_threshold"]), minimum_learning_rate=float(t["minimum_learning_rate"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    signature = {"config": without_private(config), "windows_sha256": sha256_file(windows_path)}
    progress = {"next_epoch": 0, "next_batch": 0, "history": [], "best_score": -math.inf, "best_epoch": 0}

    def save_state() -> None:
        atomic_torch_save({"kind": "regressor_training_state", "signature": signature,
                           "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                           "rng": capture_rng_state(device), "progress": progress}, last_path)

    if resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Configuration or data changed since this run started; cannot resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng_state(state["rng"], device)
        progress = state["progress"]
        log(f"resuming {architecture} at epoch {progress['next_epoch'] + 1}, batch {progress['next_batch']}")
    output.mkdir(parents=True, exist_ok=True)
    log(f"training {architecture} regressor: {sum(p.numel() for p in model.parameters()):,} parameters, "
        f"{counts['train']:,} training / {counts['validation']:,} validation windows, device {device}")

    max_batches = t.get("max_batches_per_epoch")
    checkpoint_every = int(t["checkpoint_every_batches"])
    for epoch in range(progress["next_epoch"], epochs):
        start_batch = progress["next_batch"] if epoch == progress["next_epoch"] else 0
        if not start_batch:
            progress["running"] = {"loss": 0.0, "n": 0}
        loader = make_loader(train_set, batch_size=batch_size, epoch=epoch, seed=seed, training=True,
                             start_batch=start_batch, pin_memory=device.type == "cuda")
        model.train()
        # Creating a partially consumed loader draws from the torch RNG; restore it on resume.
        rng_state = torch.get_rng_state() if start_batch else None
        iterator = iter(loader)
        if rng_state is not None:
            torch.set_rng_state(rng_state)
        absolute = start_batch
        for relative, batch in enumerate(iterator, start=1):
            absolute = start_batch + relative
            x, mask, atac_mask, h3_mask, atac_y, h3_y = batch_tensors(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, precision):
                loss_value, _ = total_loss(model(x, mask, atac_mask, h3_mask), (atac_y, h3_y), criteria)
            optimizer_step(loss_value, model, optimizer, scheduler, scaler, float(t["gradient_clip_norm"]))
            progress["running"]["loss"] += float(loss_value.item()) * len(atac_y)
            progress["running"]["n"] += len(atac_y)
            if absolute % 100 == 0 or absolute == batches_per_epoch:
                log(f"epoch {epoch + 1}/{epochs} batch {absolute}/{batches_per_epoch} "
                    f"loss {progress['running']['loss'] / progress['running']['n']:.4f} "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}")
            if checkpoint_every and absolute % checkpoint_every == 0:
                progress.update(next_epoch=epoch, next_batch=absolute)
                save_state()
            if max_batches and relative >= max_batches:
                break

        losses, labels, predictions = evaluate(model, validation_loader, criteria, device, precision,
                                               t.get("max_validation_batches"))
        metrics = {assay: assay_metrics(labels[assay], predictions[assay], regulatory[:len(labels[assay])])
                   for assay in ASSAYS}
        score = scientific_composite(metrics)
        scheduler.step_validation(score)
        improved = score > progress["best_score"]
        progress["history"].append({
            "epoch": epoch + 1, "training_loss": progress["running"]["loss"] / max(1, progress["running"]["n"]),
            "validation_loss": losses, "validation_score": score, "selected": improved,
            "learning_rate": optimizer.param_groups[0]["lr"], "optimizer_steps": scheduler.optimizer_steps,
            "validation": metrics})
        if improved:
            progress.update(best_score=score, best_epoch=epoch + 1)
            atomic_torch_save({
                "kind": "regressor", "architecture": architecture, "contexts": list(CONTEXTS),
                "state_dict": model.state_dict(), "output_scaling": output_scaling, "epoch": epoch + 1,
                "validation_score": score, "training_target_means": {a: v.tolist() for a, v in target_means.items()},
                "config": without_private(config), "windows_sha256": signature["windows_sha256"]}, best_path)
        write_json(output / "history.json", progress["history"])
        progress.update(next_epoch=epoch + 1, next_batch=0)
        save_state()
        log(f"epoch {epoch + 1} validation score {score:.4f} (ATAC r {metrics['atac']['window_mean']['macro']['pearson']:.3f}, "
            f"H3K27ac r {metrics['h3k27ac']['window_mean']['macro']['pearson']:.3f})"
            + ("  * best" if improved else ""))

    write_json(output / "metrics.json", {"architecture": architecture, "best_epoch": progress["best_epoch"],
                                         "best_validation_score": progress["best_score"],
                                         "checkpoint_sha256": sha256_file(best_path),
                                         "history": progress["history"], "config": without_private(config)})
    log(f"done: best epoch {progress['best_epoch']} (score {progress['best_score']:.4f}) -> {best_path}")
    return best_path
