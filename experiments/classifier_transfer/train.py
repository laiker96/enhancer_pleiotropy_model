"""Matched, checkpointable binary classification with no test-set access."""

import argparse
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from torch.nn import functional as F

from enhancer_pleiotropy_model.training import (
    atomic_torch_save, checkpoint_tensors_to_cpu, capture_rng_state, restore_rng_state,
)
from .data import load_split, digest, write_json
from .metrics import summarize
from .models import initialize_classifier, ENHANCERNET_MODELS


def event(event_type, **values):
    print(json.dumps(dict(event=event_type, **values), allow_nan=False), flush=True)


def require_cuda_allocation():
    if (not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_NODELIST")
            or socket.gethostname().split(".")[0] != "a100"):
        raise RuntimeError("CUDA training requires a CECAR A100 Slurm allocation, never login")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("Expected one allocated A100")


def batch_input(codes, device):
    return F.one_hot(torch.as_tensor(codes.astype(np.int64), device=device), 4).permute(0, 2, 1).float()


@torch.no_grad()
def predict(model, data, device, batch_size=64):
    model.eval()
    values = []
    for start in range(0, len(data["sequence"]), batch_size):
        x = batch_input(data["sequence"][start:start+batch_size], device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type=="cuda"):
            logits = model(x)
            rc_logits = model(x.flip((1, 2)))
        p = (logits.float().sigmoid()+rc_logits.float().sigmoid())*.5
        if not torch.isfinite(p).all(): raise ValueError("Nonfinite validation probabilities")
        values.append(p.cpu().numpy())
    return np.concatenate(values)


def lr_multiplier(updates, batches, epochs):
    if updates < batches: return (updates+1)/batches
    progress = min(1., (updates-batches)/max(1, (epochs-1)*batches))
    return .1 + .9*.5*(1+math.cos(math.pi*progress))


def optimizer_for(model, pretrained, config):
    head = list(model.head.parameters())
    encoder = [p for n,p in model.named_parameters() if not n.startswith("head.")]
    lr = config["max_learning_rate"]
    return torch.optim.AdamW([
        dict(params=encoder, lr=lr*config["finetune_encoder_factor"] if pretrained else lr, name="encoder"),
        dict(params=head, lr=lr, name="head")], weight_decay=config["weight_decay"])


def run(data_root, output, architecture, seed, config, checkpoint=None, device="cuda", resume=False, stop=None):
    device = torch.device(device)
    if device.type == "cuda": require_cuda_allocation()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4 if device.type == "cuda" else 1)
    output = Path(output)
    stop = Path(stop) if stop else output/"STOP"
    if stop.exists(): raise FileExistsError("Inspect and clear the run STOP marker before resuming")
    if resume:
        if not (output/"last_checkpoint.pt").is_file(): raise FileNotFoundError("Missing explicit restart")
    else: output.mkdir(parents=True, exist_ok=False)
    # Training never opens test.npz. The final comparison is a separate operation.
    train, validation = (load_split(data_root, s) for s in ("train", "validation"))
    background = None
    if config.get("background") is not None:
        from .background import load_background, background_epoch, background_slice, background_metrics
        if config["background"] != {"enhancers_per_background": 3, "selection": "enhancer_only"}:
            raise ValueError("Unsupported background training contract")
        background_root = Path(data_root)/"background"
        background = {s:load_background(background_root,s,d) for s,d in (("train",train),("validation",validation))}
        background_epoch(len(train["ids"]),len(background["train"]["ids"]),seed,0)
    settings = dict(architecture=architecture, seed=seed, training=config,
                    data_audit_sha256=digest(Path(data_root)/"audit.json"),
                    initialization_sha256=digest(checkpoint) if checkpoint else None)
    if background is not None:
        settings["background_audit_sha256"] = digest(background_root/"audit.json")
    model = initialize_classifier(architecture, seed, train["labels"].mean(0), checkpoint,
                                  config.get("readout", "legacy")).to(device)
    optimizer = optimizer_for(model, checkpoint is not None, config)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    batches = math.ceil(len(train["labels"])/config["batch_size"])
    state = dict(next_epoch=0, next_batch=0, updates=0, history=[], best_ap=-1., best_bce=float("inf"), best_epoch=None)
    if resume:
        saved = torch.load(output/"last_checkpoint.pt", map_location="cpu", weights_only=False)
        if saved["settings"] != settings: raise ValueError("Classifier restart signature changed")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng_state(saved["rng"], device)
        state = saved["progress"]
    else:
        write_json(output/"launch.json", dict(settings, parameters=sum(p.numel() for p in model.parameters()),
                    train_n=len(train["labels"]), validation_n=len(validation["labels"]),
                    head_initialization="paired seed + 100000; train prevalence logit bias", base_lrs=base_lrs))
    def save():
        saved = dict(kind="v4_context_classifier_restart", settings=settings, state_dict=model.state_dict(),
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=capture_rng_state(device), progress=state)
        atomic_torch_save(checkpoint_tensors_to_cpu(saved), output/"last_checkpoint.pt")
    event("classifier_started", **settings, base_lrs=base_lrs, batch_count=batches,
          start_epoch=state["next_epoch"]+1, start_batch=state["next_batch"])
    for epoch in range(state["next_epoch"], config["epochs"]):
        started = time.monotonic()
        rng = np.random.default_rng(seed+epoch)
        order = rng.permutation(len(train["labels"]))
        rc = rng.random(len(order)) < .5
        if background is not None:
            bg_order, bg_rc = background_epoch(len(order),len(background["train"]["ids"]),seed,epoch)
        model.train()
        for b in range(state["next_batch"], batches):
            if stop.exists():
                state.update(next_epoch=epoch, next_batch=b)
                save()
                event("classifier_paused", epoch=epoch+1, batch=b)
                return False
            begin = b*config["batch_size"]
            idx = order[begin:begin+config["batch_size"]]
            codes = train["sequence"][idx].copy()
            selected_rc = rc[begin:begin+len(idx)]
            codes[selected_rc] = 3-codes[selected_rc, ::-1]
            labels = train["labels"][idx]
            if background is not None:
                bg_slice = background_slice(begin,begin+len(idx))
                bg_codes = background["train"]["sequence"][bg_order[bg_slice]].copy()
                flip = bg_rc[bg_slice]
                bg_codes[flip] = 3-bg_codes[flip,::-1]
                codes = np.concatenate((codes,bg_codes))
                labels = np.concatenate((labels,np.zeros((len(bg_codes),8),dtype=labels.dtype)))
            x = batch_input(codes, device)
            y = torch.as_tensor(labels, dtype=torch.float32, device=device)
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base*lr_multiplier(state["updates"], batches, config["epochs"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            if not torch.isfinite(loss): raise ValueError("Nonfinite classifier loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < before
            if not skipped and not torch.isfinite(norm): raise ValueError("Nonfinite unskipped gradient")
            state["updates"] += int(not skipped)
            state.update(next_epoch=epoch, next_batch=b+1)
            if (b+1) % 100 == 0 or b+1 == batches:
                event("classifier_progress", epoch=epoch+1, batch=b+1, batches=batches, loss=float(loss.detach()),
                      learning_rates=[g["lr"] for g in optimizer.param_groups], updates=state["updates"], skipped=skipped)
                save()
        probabilities = predict(model, validation, device, config["batch_size"])
        metrics = summarize(validation["labels"], probabilities)
        bg_metrics = None
        if background is not None:
            bg_probabilities = predict(model,background["validation"],device,config["batch_size"])
            bg_metrics = background_metrics(validation["labels"],probabilities,bg_probabilities,
                                            background["validation"]["enhancer_index"])
        ap, bce = metrics["macro_average_precision"], metrics["bce"]
        if not np.isfinite(ap+bce): raise ValueError("Undefined checkpoint selection metric")
        improved = ap > state["best_ap"] or (ap == state["best_ap"] and bce < state["best_bce"])
        if improved:
            state.update(best_ap=ap, best_bce=bce, best_epoch=epoch+1)
            atomic_torch_save(checkpoint_tensors_to_cpu(dict(kind="v4_context_classifier", settings=settings,
                              epoch=epoch+1, state_dict=model.state_dict(), macro_ap=ap, bce=bce)), output/"best_model.pt")
            np.savez_compressed(output/"best_validation_predictions.npz", ids=validation["ids"], probabilities=probabilities)
            if background is not None:
                np.savez_compressed(output/"best_validation_background_predictions.npz",
                                    ids=background["validation"]["ids"],probabilities=bg_probabilities)
        row = dict(epoch=epoch+1, learning_rates=[g["lr"] for g in optimizer.param_groups], metrics=metrics,
                   updates=state["updates"], seconds=time.monotonic()-started, selected=improved)
        if background is not None:
            row["background_metrics"] = bg_metrics
            row["background_examples"] = len(bg_order)
        state["history"].append(row)
        state.update(next_epoch=epoch+1, next_batch=0)
        write_json(output/"history.json", state["history"])
        save()
        event("classifier_epoch_complete", epoch=epoch+1, macro_ap=ap, bce=bce, breadth=metrics["breadth"],
              learning_rates=row["learning_rates"], best_epoch=state["best_epoch"])
        if bg_metrics is not None:
            event("classifier_background_validation",epoch=epoch+1,
                  active_vs_background_ap=bg_metrics["active_vs_background"]["macro_average_precision"],
                  combined_ap=bg_metrics["enhancers_plus_background"]["macro_average_precision"],
                  background_fpr_at_80_recall=bg_metrics["background_rejection"]["macro_fpr_at_80_recall"])
    write_json(output/"complete.json", dict(status="complete", epochs=len(state["history"]),
               best_epoch=state["best_epoch"], best_macro_ap=state["best_ap"],
               checkpoint_sha256=digest(output/"best_model.pt"), settings=settings))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--architecture", choices=("attention", "cnn", "dense", *ENHANCERNET_MODELS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-file", type=Path)
    args = p.parse_args()
    run(args.data, args.output, args.architecture, args.seed, json.loads(args.config.read_text())["classifier"],
        args.checkpoint, resume=args.resume, stop=args.stop_file)


if __name__ == "__main__": main()
