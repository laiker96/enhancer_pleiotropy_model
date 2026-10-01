"""Small path, provenance and compute-allocation helpers; no CUDA at import."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = ("cnn", "attention", "dense")
READOUTS = ("legacy", "retained_assay_hidden_v1")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def config(path, recipe=None):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if recipe is not None:
        recipes = value.get("recipes", {})
        if recipe not in recipes or not recipe.replace("-", "").isalnum():
            raise ValueError("Unknown or unsafe experiment recipe: " + str(recipe))
        overrides = recipes[recipe]
        allowed = {"architectures", "readouts", "populations", "seeds", "analysis_classifier"}
        if not isinstance(overrides, dict) or set(overrides) - allowed:
            raise ValueError("Recipes may only select the experiment matrix and analysis classifier")
        value.update(overrides)
        value["paths"]["work"] = str(Path(value["paths"]["work"]) / recipe)
        value["recipe"] = recipe
    base = (path.parent / value.get("project_root", "..")).resolve()
    value["paths"] = {k: str((base / v).resolve()) for k, v in value["paths"].items()}
    if value.get("version") != 1 or not set(value["architectures"]) <= set(ARCHITECTURES):
        raise ValueError("Unsupported reproduction version/architecture")
    if not value["architectures"] or len(set(value["architectures"])) != len(value["architectures"]):
        raise ValueError("Architectures must be nonempty and unique")
    if set(value["readouts"]) - set(READOUTS):
        raise ValueError("Unsupported transfer readout")
    if set(value["populations"]) - {"enhancer_only", "background"}:
        raise ValueError("Unsupported classifier population")
    for key in ("seeds", "readouts", "populations"):
        if not value[key] or len(value[key]) != len(set(value[key])):
            raise ValueError(key + " must be nonempty and unique")
    if any(type(s) is not int or s < 0 for s in value["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    value["config_path"] = str(path)
    selected_run(value)
    return value


def require_slurm(gpu=False):
    """Check actual node-list membership, not a site-specific hostname prefix."""
    nodes = os.environ.get("SLURM_JOB_NODELIST")
    if not os.environ.get("SLURM_JOB_ID") or not nodes:
        raise RuntimeError("Run inside a Slurm compute allocation, never on a login node")
    hosts = subprocess.check_output(["scontrol", "show", "hostnames", nodes], text=True).split()
    if socket.gethostname().split(".")[0] not in {h.split(".")[0] for h in hosts}:
        raise RuntimeError("This host is not part of the Slurm allocation")
    if gpu:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Request exactly one visible CUDA GPU per task")


def run_id(architecture, readout, population, mode, seed):
    return "__".join(map(str, (architecture, readout, population, mode, seed)))


def classifiers(cfg):
    for architecture in cfg["architectures"]:
        for readout in cfg["readouts"] if architecture != "dense" else ["legacy"]:
            for population in cfg["populations"]:
                for mode in ("scratch", "finetune"):
                    for seed in cfg["seeds"]:
                        yield dict(architecture=architecture, readout=readout,
                                   population=population, mode=mode, seed=seed)


def selected_run(cfg):
    choice = cfg["analysis_classifier"]
    if choice not in list(classifiers(cfg)):
        raise ValueError("analysis_classifier must be a configured classifier fit")
    return Path(cfg["paths"]["work"]) / "classifiers" / run_id(**choice)


def provenance(paths, settings):
    source = {str(p.relative_to(ROOT)): digest(p)
              for folder in ("src", "experiments/reproduction", "experiments/classifier_transfer",
                             "experiments/classifier_motifs", "experiments/classifier_modisco")
              for p in sorted((ROOT / folder).rglob("*.py"))}
    for name in ("reproduce.py", "stage_v4_inputs.py", "prepare_v4_dataset.py", "prepare_v4_motifs.py"):
        path = ROOT / "scripts" / name
        source[str(path.relative_to(ROOT))] = digest(path)
    record = dict(inputs={str(Path(p).resolve()): digest(p) for p in paths},
                  settings=settings, source_sha256=source)
    record["signature"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def pin_training_source(cfg, stage, task):
    """A restart cannot silently pick up a changed checkout or experiment file."""
    record = provenance([cfg["config_path"], cfg["paths"]["regression_config"]],
                        dict(stage=stage, task=task, recipe=cfg.get("recipe")))
    path = Path(cfg["paths"]["work"]) / "provenance" / f"{stage}_{task}.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError("Training source/config changed; restore the pinned code or start a new run")
    else:
        write_json(path, record)
