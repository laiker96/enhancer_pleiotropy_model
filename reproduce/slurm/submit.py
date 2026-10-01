#!/usr/bin/env python3
"""Print a Slurm plan by default. --execute is the only submission switch."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src"), str(ROOT / "scripts")]
import yaml
from reproduction.cli import GPU_STAGES, STAGES
from reproduction.common import classifiers, config, digest, write_json

PIPELINES = {
    "preprocessing": ["stage-inputs", "prepare-regression", "prepare-classifiers"],
    "training": ["train-regressor", "train-classifier", "evaluate", "calibrate"],
    "analysis": ["attribute-pilot", "attribute", "modisco"],
}


def count(cfg, stage):
    if stage == "train-regressor": return len(cfg["architectures"])
    if stage == "train-classifier": return len(list(classifiers(cfg)))
    if stage == "attribute": return cfg["attribution"]["shards"]
    if stage == "modisco": return 3 if cfg["discovery"]["grouping"] == "nonoverlap" else 8
    return 1


def command(cfg, site, stage, dependency=None, tasks=None, resume=False):
    resource = site["resources"][stage]
    gpu = stage in GPU_STAGES
    partition = site["gpu_partition" if gpu else "cpu_partition"]
    cmd = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--chdir="+str(ROOT),
           "--job-name=epm_"+stage, "--partition="+partition,
           "--cpus-per-task="+str(resource["cpus"]), "--mem="+resource["mem"], "--time="+resource["time"],
           "--output="+str(ROOT / "logs/reproduce/%x-%A_%a.log"), "--open-mode=append",
           "--signal=B:USR1@180"]
    if site.get("account"): cmd.append("--account="+site["account"])
    if gpu: cmd.append("--gres="+site["gpu_request"])
    if dependency: cmd.append("--dependency=afterok:"+dependency)
    n = count(cfg, stage)
    if tasks is not None:
        indices = [int(t) for t in tasks.split(",")]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= n for i in indices):
            raise ValueError("Task indices must be unique and within the configured stage")
        array = ",".join(map(str, indices))
    else:
        array = f"0-{n-1}"
    if n > 1 or tasks is not None:
        cmd.append("--array="+array+"%"+str(site["max_parallel"]))
    extras = site.get("extra_sbatch_args", [])
    if any(not isinstance(v, str) or not v.startswith(("--constraint=", "--qos=", "--reservation=")) for v in extras):
        raise ValueError("extra_sbatch_args supports constraint, qos and reservation only")
    cmd.extend(extras)
    python = site["modisco_python" if stage == "modisco" else "python"]
    cmd.extend([str(ROOT / "reproduce/slurm/task.sbatch"), str(ROOT), str((ROOT / python).absolute()),
                str((ROOT / site["environment_script"]).absolute()), stage, cfg["config_path"]])
    if cfg.get("recipe"): cmd.extend(["--recipe", cfg["recipe"]])
    if resume: cmd.append("--resume")
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=list(PIPELINES)+list(STAGES[1:]))
    parser.add_argument("--config", default=str(ROOT / "reproduce/config.yaml"))
    parser.add_argument("--recipe", help="Named experiment matrix; pass the same recipe at every stage")
    parser.add_argument("--site", default=str(ROOT / "reproduce/slurm/site.example.yaml"))
    parser.add_argument("--after", help="Existing numeric Slurm dependency job ID")
    parser.add_argument("--tasks", help="Explicit comma-separated task indices; one stage only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = config(args.config, recipe=args.recipe)
    site = yaml.safe_load(Path(args.site).read_text())
    stages = PIPELINES.get(args.stage, [args.stage])
    if args.tasks and len(stages) != 1: parser.error("--tasks needs a single stage")
    if args.resume and any(s not in {"train-regressor", "train-classifier", "attribute", "attribute-pilot"} for s in stages):
        parser.error("Resume only a single checkpointable stage")
    if args.after and not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", args.after):
        parser.error("Invalid dependency ID")
    if type(site["max_parallel"]) is not int or site["max_parallel"] < 1:
        parser.error("max_parallel must be positive")
    if args.execute:
        if any(site["gpu_partition" if s in GPU_STAGES else "cpu_partition"] == "CHANGE_ME" for s in stages):
            parser.error("Configure your real partition before submission")
        for p in [site["environment_script"]] + [site["modisco_python" if s == "modisco" else "python"] for s in stages]:
            if not (ROOT / p).is_file(): parser.error("Missing site runtime file: "+p)
        (ROOT / "logs/reproduce").mkdir(parents=True, exist_ok=True)
    receipt = dict(config_sha256=digest(args.config), site_sha256=digest(args.site),
                   recipe=cfg.get("recipe"), jobs=[])
    path = Path(cfg["paths"]["work"]) / "submissions" / (str(time.time_ns())+".json")
    dependency = args.after
    for stage in stages:
        cmd = command(cfg, site, stage, dependency, args.tasks, args.resume)
        print(shlex.join(cmd), flush=True)
        if args.execute:
            # Save intent before submission so a partial DAG is auditable.
            receipt["jobs"].append(dict(stage=stage, command=cmd, status="submitting"))
            write_json(path, receipt)
            response = subprocess.check_output(cmd, text=True).strip()
            dependency = response.split(";", 1)[0]
            if not re.fullmatch(r"[0-9]+", dependency): raise RuntimeError("Unexpected sbatch response: "+response)
            receipt["jobs"][-1].update(status="submitted", job_id=dependency)
            write_json(path, receipt)
        else:
            dependency = "<"+stage+"_job_id>"
    if not args.execute:
        print("DRY RUN: no files changed and no jobs submitted. Inspect resources before using --execute.")


if __name__ == "__main__": main()
