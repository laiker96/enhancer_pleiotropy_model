"""One explicit stage per invocation; planning never trains or submits jobs."""
import argparse
import json
import os
from pathlib import Path
import signal

from .common import classifiers, config, require_slurm, run_id, pin_training_source

STAGES = ("plan", "stage-inputs", "prepare-regression", "prepare-classifiers",
          "train-regressor", "train-classifier", "evaluate", "calibrate",
          "attribute-pilot", "attribute", "modisco", "figure", "inventory")
GPU_STAGES = {"train-regressor", "train-classifier", "evaluate", "attribute-pilot", "attribute"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--config", default="reproduce/config.yaml")
    p.add_argument("--recipe", help="Named matrix in the config; uses its own work subdirectory")
    p.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--local-cpu", action="store_true", help="Explicitly allow CPU stages off Slurm")
    p.add_argument("--output", help="New PDF path for figure replay")
    a = p.parse_args(argv)
    cfg = config(a.config, recipe=a.recipe)
    if a.stage == "plan":
        print(json.dumps(dict(recipe=cfg.get("recipe"), regressors=cfg["architectures"], classifiers=list(classifiers(cfg)),
                              attribution_shards=cfg["attribution"]["shards"], paths=cfg["paths"]), indent=2))
        return
    if a.local_cpu and a.stage in GPU_STAGES:
        p.error("--local-cpu cannot authorize CUDA or bypass compute-node checks")
    if a.stage in GPU_STAGES or not a.local_cpu:
        require_slurm(gpu=a.stage in GPU_STAGES)
    # Concurrent duplicate tasks must not overwrite a checkpoint or reference accumulator.
    import fcntl
    work = Path(cfg["paths"]["work"])
    lock_directory = work / ".locks"
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock = (lock_directory / f"{a.stage}_{a.task}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if a.stage in {"train-regressor", "train-classifier"}:
        entries = cfg["architectures"] if a.stage == "train-regressor" else list(classifiers(cfg))
        if not 0 <= a.task < len(entries):
            p.error("Invalid training task")
        pin_training_source(cfg, a.stage, a.task)
        stop = (work / "regressors" / entries[a.task] / "STOP" if a.stage == "train-regressor"
                else work / "classifiers" / run_id(**entries[a.task]) / "STOP")
        def pause(*_):
            stop.parent.mkdir(parents=True, exist_ok=True)
            stop.write_text("Slurm signal: checkpoint, inspect before explicit resume.\n")
        signal.signal(signal.SIGUSR1, pause)
        signal.signal(signal.SIGTERM, pause)
    if a.stage in {"stage-inputs", "prepare-regression", "prepare-classifiers"}:
        from . import data
        getattr(data, a.stage.replace("-", "_"))(cfg)
    elif a.stage in {"train-regressor", "train-classifier", "evaluate", "calibrate"}:
        from . import training
        if a.stage.startswith("train-"):
            getattr(training, a.stage.replace("-", "_"))(cfg, a.task, a.resume)
        else:
            getattr(training, a.stage)(cfg)
    elif a.stage in {"attribute-pilot", "attribute"}:
        from .attribution import run
        run(cfg, a.task, pilot=a.stage == "attribute-pilot")
    elif a.stage == "modisco":
        from .discovery import run
        run(cfg, a.task)
    elif a.stage == "figure":
        from .figure import replay
        replay(cfg, a.output)
    else:
        from .inventory import refresh
        refresh(cfg)
