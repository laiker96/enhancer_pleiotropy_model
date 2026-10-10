"""Train an eight-context enhancer classifier, from scratch or from a pretrained regressor.

The test split is never read during training. Checkpoints are selected by the
validation macro average precision over enhancers (BCE breaks ties).
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..config import without_private
from ..constants import CONTEXTS
from ..data.background import background_epoch, background_slice, load_background
from ..data.enhancers import load_split
from ..io import sha256_file, write_json
from ..models import MANUSCRIPT_READOUT, make_classifier
from .classification_metrics import background_metrics, summarize
from .common import atomic_torch_save, capture_rng_state, resolve_device, restore_rng_state


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def one_hot(codes: np.ndarray, device: torch.device) -> torch.Tensor:
    return F.one_hot(torch.as_tensor(codes.astype(np.int64), device=device), 4).permute(0, 2, 1).float()


@torch.no_grad()
def predict_probabilities(model, sequences: np.ndarray, device: torch.device, batch_size: int = 64,
                          half: bool | None = None) -> np.ndarray:
    """Mean of forward and reverse-complement sigmoid probabilities, ``[n, 8]``."""
    model.eval()
    half = device.type == "cuda" if half is None else half
    values = []
    for start in range(0, len(sequences), batch_size):
        x = one_hot(sequences[start:start + batch_size], device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=half):
            forward, reverse = model(x), model(x.flip((1, 2)))
        values.append(((forward.float().sigmoid() + reverse.float().sigmoid()) * .5).cpu().numpy())
    return np.concatenate(values)


def lr_multiplier(updates: int, batches: int, epochs: int) -> float:
    """Linear warm-up over the first epoch, then cosine decay to 10% at the last epoch."""
    if updates < batches:
        return (updates + 1) / batches
    progress = min(1., (updates - batches) / max(1, (epochs - 1) * batches))
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))


def run_name(architecture: str, init: str, seed: int) -> str:
    return f"{architecture}_{init}_{seed}"


def train_classifier(config: dict[str, Any], *, resume: bool = False) -> Path:
    architecture, init, seed = config["architecture"], config["init"], int(config["seed"])
    if init not in ("scratch", "pretrained"):
        raise ValueError("init must be 'scratch' or 'pretrained'")
    readout = config.get("readout", "auto")
    readout = MANUSCRIPT_READOUT[architecture] if readout == "auto" else readout
    t = config["training"]
    device = resolve_device(config["device"])
    half = device.type == "cuda"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4 if device.type == "cuda" else 1)
    data = Path(config["data"])
    output = Path(config["output"]) / run_name(architecture, init, seed)
    parent = Path(config["regressors"]) / architecture / "best_model.pt" if init == "pretrained" else None
    if parent is not None and not parent.is_file():
        raise FileNotFoundError(f"Pretrained regressor not found: {parent} (train it first, or set regressors=...)")
    last_path, best_path = output / "last_checkpoint.pt", output / "best_model.pt"
    if not resume and best_path.exists():
        raise FileExistsError(f"{output} already contains a trained model; use --resume or another output")

    train, validation = load_split(data, "train"), load_split(data, "validation")
    use_background = bool(t.get("background", True))
    if use_background:
        background = {s: load_background(data, s, part) for s, part in (("train", train), ("validation", validation))}
        background_epoch(len(train["ids"]), len(background["train"]["ids"]), seed, 0)
    settings = {"architecture": architecture, "init": init, "readout": readout, "seed": seed,
                "training": t, "data_audit_sha256": sha256_file(data / "audit.json"),
                "parent_sha256": sha256_file(parent) if parent else None}
    model = make_classifier(architecture, readout, seed, train["labels"].mean(0), parent).to(device)
    head = list(model.head.parameters())
    encoder = [p for name, p in model.named_parameters() if not name.startswith("head.")]
    lr = float(t["max_learning_rate"])
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": lr * float(t["finetune_encoder_factor"]) if parent else lr, "name": "encoder"},
        {"params": head, "lr": lr, "name": "head"}], weight_decay=float(t["weight_decay"]))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=half)
    batch_size, epochs = int(t["batch_size"]), int(t["epochs"])
    batches = math.ceil(len(train["labels"]) / batch_size)
    state = {"next_epoch": 0, "next_batch": 0, "updates": 0, "history": [], "best_ap": -1.,
             "best_bce": float("inf"), "best_epoch": None}
    if resume and last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["settings"] != settings:
            raise ValueError("Configuration or data changed since this run started; cannot resume")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng_state(saved["rng"], device)
        state = saved["progress"]
        log(f"resuming at epoch {state['next_epoch'] + 1}, batch {state['next_batch']}")
    output.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        atomic_torch_save({"kind": "classifier_training_state", "settings": settings, "state_dict": model.state_dict(),
                           "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                           "rng": capture_rng_state(device), "progress": state}, last_path)

    log(f"training {architecture} classifier ({init}, readout {readout}, seed {seed}): "
        f"{sum(p.numel() for p in model.parameters()):,} parameters, {len(train['labels']):,} training enhancers"
        + (f" + {len(train['labels']) // 3:,} backgrounds per epoch" if use_background else ""))
    max_batches = t.get("max_batches_per_epoch")
    for epoch in range(state["next_epoch"], epochs):
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(train["labels"]))
        reverse = rng.random(len(order)) < .5
        if use_background:
            bg_order, bg_reverse = background_epoch(len(order), len(background["train"]["ids"]), seed, epoch)
        model.train()
        first_batch = state["next_batch"]
        for b in range(first_batch, batches):
            begin = b * batch_size
            index = order[begin:begin + batch_size]
            codes = train["sequence"][index].copy()
            flip = reverse[begin:begin + len(index)]
            codes[flip] = 3 - codes[flip, ::-1]
            labels = train["labels"][index]
            if use_background:
                part = background_slice(begin, begin + len(index))
                bg_codes = background["train"]["sequence"][bg_order[part]].copy()
                bg_flip = bg_reverse[part]
                bg_codes[bg_flip] = 3 - bg_codes[bg_flip, ::-1]
                codes = np.concatenate((codes, bg_codes))
                labels = np.concatenate((labels, np.zeros((len(bg_codes), len(CONTEXTS)), dtype=labels.dtype)))
            x = one_hot(codes, device)
            y = torch.as_tensor(labels, dtype=torch.float32, device=device)
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * lr_multiplier(state["updates"], batches, epochs)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=half):
                loss = F.binary_cross_entropy_with_logits(model(x).float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite classifier loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(t["gradient_clip_norm"]))
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < before
            if not skipped and not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient")
            state["updates"] += int(not skipped)
            state.update(next_epoch=epoch, next_batch=b + 1)
            if (b + 1) % 100 == 0 or b + 1 == batches:
                log(f"epoch {epoch + 1}/{epochs} batch {b + 1}/{batches} loss {float(loss.detach()):.4f}")
                save()
            if max_batches and b + 1 - first_batch >= max_batches:
                break

        probabilities = predict_probabilities(model, validation["sequence"], device, batch_size, half)
        metrics = summarize(validation["labels"], probabilities)
        row = {"epoch": epoch + 1, "metrics": metrics, "updates": state["updates"],
               "learning_rates": [g["lr"] for g in optimizer.param_groups]}
        if use_background:
            bg_probabilities = predict_probabilities(model, background["validation"]["sequence"], device,
                                                     batch_size, half)
            row["background_metrics"] = background_metrics(validation["labels"], probabilities, bg_probabilities,
                                                           background["validation"]["enhancer_index"])
        ap, bce = metrics["macro_average_precision"], metrics["bce"]
        improved = ap > state["best_ap"] or (ap == state["best_ap"] and bce < state["best_bce"])
        row["selected"] = improved
        if improved:
            state.update(best_ap=ap, best_bce=bce, best_epoch=epoch + 1)
            atomic_torch_save({"kind": "classifier", "settings": settings, "epoch": epoch + 1,
                               "state_dict": model.state_dict(), "macro_ap": ap, "bce": bce}, best_path)
            np.savez_compressed(output / "best_validation_predictions.npz", ids=validation["ids"],
                                labels=validation["labels"], probabilities=probabilities)
        state["history"].append(row)
        state.update(next_epoch=epoch + 1, next_batch=0)
        write_json(output / "history.json", state["history"])
        save()
        extra = (f", enhancer-vs-background AP {row['background_metrics']['macro_active_vs_background']:.3f}"
                 if use_background else "")
        log(f"epoch {epoch + 1}: validation macro AP {ap:.4f}{extra}" + ("  * best" if improved else ""))

    write_json(output / "metrics.json", {"best_epoch": state["best_epoch"], "best_macro_ap": state["best_ap"],
                                         "checkpoint_sha256": sha256_file(best_path), "settings": settings,
                                         "config": without_private(config)})
    log(f"done: best epoch {state['best_epoch']} (macro AP {state['best_ap']:.4f}) -> {best_path}")
    return best_path
