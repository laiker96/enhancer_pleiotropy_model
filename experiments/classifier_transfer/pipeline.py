"""Bounded CECAR smoke and serial experiment stages, with checkpointed pauses."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from .data import digest, load_split, write_json
from .models import CNN_NAME, DENSE_NAME, ENHANCERNET_NAMES, load_regressor
from .train import event, require_cuda_allocation


def execute(command, work, log, stop, deadline):
    """Save on advance warning; do not silently restart failed or partial stages."""
    def request_stop(signum=None, frame=None):
        stop.touch(exist_ok=True)
    old = {s:signal.signal(s, request_stop) for s in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT)}
    child = None
    started = time.monotonic()
    try:
        with log.open("x") as output, log.open() as reader:
            child=subprocess.Popen(command,cwd=work,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                for line in reader: print(line,end="",flush=True)
                if time.monotonic() >= deadline-900: request_stop()
                if time.monotonic() >= deadline: raise TimeoutError("Allocation deadline")
                time.sleep(.2)
            for line in reader: print(line,end="",flush=True)
        if child.returncode: raise RuntimeError(f"Stage exited {child.returncode}; see {log}")
        if stop.exists(): raise RuntimeError("Checkpointed pause requested; no subsequent stage launched")
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
        for s,handler in old.items():signal.signal(s,handler)
        write_json(log.with_suffix(".stage.json"), dict(command=command, working_directory=str(work),
                   elapsed_seconds=time.monotonic()-started, exit_code=child.returncode if child else None,
                   stop_requested=stop.exists()))


def check_parent(project, config):
    parent=project/config["parent_run"]
    complete=json.loads((parent/"completion.json").read_text())
    if complete["status"]!="complete" or complete["completed_epochs"]!=40 or complete["job_id"]!=config["parent_job"]:
        raise ValueError("Original regressor has not completed its approved 40 epochs")
    model_dir=parent/config["parent_model"]
    history=json.loads((model_dir/"history.json").read_text())
    if len(history)!=40 or not all(e["training_epoch_complete"] and e["epoch"]==i
                                  for i,e in enumerate(history,1)):
        raise ValueError("Parent history is incomplete")
    best=torch.load(model_dir/"best_model.pt",map_location="cpu",weights_only=False)
    selected=max(history,key=lambda e:e["scientific_composite"])
    if best["epoch"]!=selected["epoch"] or best["epoch"]!=complete["best_epoch"]:
        raise ValueError("Parent best checkpoint selection mismatch")
    if not np.isclose(best["checkpoint_selection"]["score"],selected["scientific_composite"],rtol=0,atol=1e-12):
        raise ValueError("Parent checkpoint score mismatch")
    return model_dir/"best_model.pt",best


def regression_config(project, root, name, architecture="cnn"):
    config=yaml.safe_load((project/"frozen/config/run_config.yaml").read_text())
    config=copy.deepcopy(config)
    config["output_directory"]=str(root/name)
    config["model"]["encoder_variant"]={"cnn": CNN_NAME, "dense": DENSE_NAME, **ENHANCERNET_NAMES}[architecture]
    target=root/name
    target.mkdir(exist_ok=False)
    (target/"data").symlink_to(project/"inputs/v4_4x_crested_20260908/data",target_is_directory=True)
    path=root/f"{name}.json"
    write_json(path,config)
    return path


def regression_module(architecture):
    if architecture in ENHANCERNET_NAMES: return "classifier_transfer.enhancernet_regression"
    return "classifier_transfer.dense_regression" if architecture == "dense" else "classifier_transfer.regression"


def check_prerequisite_suite(project, config):
    """Slurm success is necessary but not sufficient: require finished comparison."""
    prerequisite = config.get("prerequisite_suite")
    if prerequisite is None: return
    root = (project / prerequisite).resolve()
    if not root.is_relative_to(project.resolve() / "experiments"):
        raise ValueError("Prerequisite must be an experiment in this project")
    if digest(root / "MANIFEST.sha256") != config["prerequisite_manifest_sha256"]:
        raise ValueError("Prerequisite manifest mismatch")
    complete = json.loads((root / "complete.json").read_text())
    if (complete.get("status") != "complete" or complete.get("classifiers") != 6
            or complete.get("architectures") != ["dense"] or complete.get("regression_epochs") != 40):
        raise ValueError("Prerequisite dense suite is incomplete")
    if complete["comparison_sha256"] != digest(root / "comparison/metrics.json"):
        raise ValueError("Prerequisite comparison hash mismatch")


def reused_regressor(project, root, architecture, entry, dataset_sha256, create):
    """Lock an already completed regressor; never retrain it for a readout correction."""
    source = (project / entry["checkpoint"]).resolve()
    history_path = (project / entry["history"]).resolve()
    suite = (project / entry["suite"]).resolve()
    if not all(p.is_relative_to(project.resolve() / "experiments") for p in (source, history_path, suite)):
        raise ValueError("Reused regressor must be inside the project's experiments")
    complete = json.loads((suite / "complete.json").read_text())
    if (complete.get("status") != "complete" or complete.get("classifiers") != 12
            or complete.get("cnn_regression_epochs") != 40
            or complete["comparison_sha256"] != digest(suite / "comparison/metrics.json")
            or digest(suite / "MANIFEST.sha256") != entry["suite_manifest_sha256"]):
        raise ValueError("Reused regression suite is incomplete or changed")
    if digest(source) != entry["checkpoint_sha256"] or digest(history_path) != entry["history_sha256"]:
        raise ValueError("Reused regressor checkpoint/history hash mismatch")
    history = json.loads(history_path.read_text())
    if len(history) != 40 or not all(e["epoch"] == i and e["training_epoch_complete"]
                                     for i, e in enumerate(history, 1)):
        raise ValueError("Reused regression history is incomplete")
    model, metadata = load_regressor(source, architecture)
    del model
    selected = max(history, key=lambda e: e["scientific_composite"])
    if (metadata["dataset_sha256"] != dataset_sha256 or metadata["epoch"] != selected["epoch"]
            or not np.isclose(metadata["checkpoint_selection"]["score"], selected["scientific_composite"],
                              rtol=0, atol=1e-12)):
        raise ValueError("Reused regressor data/selection mismatch")
    target = root / "parents" / f"{architecture}_best.pt"
    history_target = root / f"{architecture}_regression/model/history.json"
    if create:
        if target.exists(): raise FileExistsError(target)
        history_target.parent.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source, target)
        shutil.copy2(history_path, history_target)
        write_json(root / "parents" / f"{architecture}.json", dict(
            checkpoint_sha256=entry["checkpoint_sha256"], epoch=metadata["epoch"],
            selection=metadata["checkpoint_selection"], reused_from=str(source),
            dataset_sha256=dataset_sha256, history_sha256=entry["history_sha256"]))
    if digest(target) != entry["checkpoint_sha256"] or digest(history_target) != entry["history_sha256"]:
        raise ValueError("Locked reused regressor changed")
    return target, metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,required=True)
    p.add_argument("--experiment",type=Path,required=True)
    p.add_argument("--mode",choices=("smoke","full"),required=True)
    a=p.parse_args()
    require_cuda_allocation()
    project=a.project.resolve(); root=a.experiment.resolve()
    if not root.is_relative_to(project/"experiments"): raise ValueError("Work must stay in experiment directory")
    config=json.loads((root/"config.json").read_text())
    architectures=config.get("architectures", ["attention", "cnn"])
    if (not architectures or len(set(architectures)) != len(architectures)
            or set(architectures)-{"attention", "cnn", "dense", *ENHANCERNET_NAMES}):
        raise ValueError("Invalid experiment architectures")
    if digest(project/"frozen/SOURCE_MANIFEST.sha256")!=config["frozen_manifest_sha256"]:
        raise ValueError("Frozen source manifest changed")
    subprocess.run(["sha256sum","-c","SOURCE_MANIFEST.sha256"],cwd=project/"frozen",check=True,stdout=subprocess.DEVNULL)
    subprocess.run(["sha256sum","-c","MANIFEST.sha256"],cwd=root,check=True,stdout=subprocess.DEVNULL)
    check_prerequisite_suite(project,config)
    parent,best=check_parent(project,config)
    if config.get("expected_attention_parent_sha256", digest(parent)) != digest(parent):
        raise ValueError("Attention parent differs from the pinned correction baseline")
    reuse = config.get("reuse_regressors", {})
    if set(reuse) - {"cnn"} or not set(reuse).issubset(architectures):
        raise ValueError("Only the completed dilated CNN can be reused in this correction")
    audit=json.loads((root/"data/audit.json").read_text())
    parent_config=yaml.safe_load((project/config["frozen_config"]).read_text())
    if audit["regression_splits"]!={k:parent_config[k] for k in ("chromosome_splits","region_splits")}:
        raise ValueError("Classifier and regression splits differ")
    deadline=time.monotonic()+(3600 if a.mode=="smoke" else 48*3600)-60
    stop=root/f"STOP_{a.mode}"
    if stop.exists(): raise FileExistsError(stop)
    locks=root/"parents"
    if a.mode=="smoke":
        locks.mkdir(exist_ok=False)
        shutil.copy2(parent,locks/"attention_best.pt")
        shutil.copy2(parent.parent/"history.json",locks/"attention_history.json")
        if digest(parent)!=digest(locks/"attention_best.pt"): raise ValueError("Parent copy changed")
        write_json(locks/"attention.json",dict(checkpoint_sha256=digest(parent),epoch=best["epoch"],
                   dataset_sha256=best["dataset_sha256"],selection=best["checkpoint_selection"]))
        smoke=root/"smoke"; smoke.mkdir()
        # Fresh regressors use the unchanged AG loss and real-profile two-update smoke gate.
        checkpoints={"attention": locks/"attention_best.pt"}
        regression_metadata={}
        for architecture in architectures:
            if architecture == "attention": continue
            if architecture in reuse:
                checkpoints[architecture], meta = reused_regressor(
                    project, root, architecture, reuse[architecture], best["dataset_sha256"], create=True)
                regression_metadata[architecture] = meta["architecture"]
                continue
            name=f"{architecture}_regression_smoke"
            regconfig=regression_config(project,root,name,architecture)
            module=regression_module(architecture)
            execute([sys.executable,"-u","-m",module,"--config",str(regconfig),"--smoke-test"],
                    project,smoke/f"{architecture}_regression.log",stop,deadline)
            checkpoints[architecture]=root/name/"model/best_model.pt"
            loaded,meta=load_regressor(checkpoints[architecture],architecture)
            regression_metadata[architecture]=meta["architecture"]
            del loaded
        small=smoke/"data"; small.mkdir()
        for split,n in (("train",32),("validation",128)):
            data=load_split(root/"data",split)
            np.savez_compressed(small/f"{split}.npz",**{k:v[:n] for k,v in data.items()})
        write_json(small/"audit.json",dict(outputs={f"{s}.npz":digest(small/f"{s}.npz") for s in ("train","validation")}))
        small_config=copy.deepcopy(config); small_config["classifier"].update(epochs=1,batch_size=8)
        write_json(smoke/"config.json",small_config)
        reports=[]
        for architecture in architectures:
            for mode in ("scratch","finetune"):
                name=f"{architecture}_{mode}"; output=smoke/name
                command=[sys.executable,"-u","-m","classifier_transfer.train","--data",str(small),"--output",str(output),
                         "--config",str(smoke/"config.json"),"--architecture",architecture,"--seed",str(config["seeds"][0]),"--stop-file",str(stop)]
                checkpoint=checkpoints[architecture]
                if mode=="finetune":command += ["--checkpoint",str(checkpoint)]
                execute(command,project,smoke/f"{name}.log",stop,deadline)
                saved=torch.load(output/"last_checkpoint.pt",map_location="cpu",weights_only=False)
                if saved["progress"]["updates"]<2 or not all(torch.isfinite(v).all() for v in saved["state_dict"].values()):
                    raise ValueError("Classifier CUDA smoke did not pass")
                reports.append(dict(name=name,updates=saved["progress"]["updates"]))
        write_json(root/"smoke_passed.json",dict(status="passed",parent_sha256=digest(locks/"attention_best.pt"),
                   source_manifest_sha256=digest(root/"MANIFEST.sha256"),classifiers=reports,
                   regressor_architectures=regression_metadata,gpu=torch.cuda.get_device_name()))
        event("classifier_suite_smoke_passed",classifiers=reports)
        return
    passed=json.loads((root/"smoke_passed.json").read_text())
    if (passed["status"]!="passed" or passed["parent_sha256"]!=digest(locks/"attention_best.pt")
            or passed["source_manifest_sha256"]!=digest(root/"MANIFEST.sha256")):
        raise ValueError("Successful matching CUDA smoke required")
    (root/"runs").mkdir(exist_ok=False)
    event("classifier_suite_started",job_id=os.environ["SLURM_JOB_ID"],parent_epoch=best["epoch"],config=config)
    for architecture in architectures:
        if architecture in reuse:
            reused_regressor(project, root, architecture, reuse[architecture], best["dataset_sha256"], create=False)
        elif architecture!="attention":
            name=f"{architecture}_regression"
            regconfig=regression_config(project,root,name,architecture)
            module=regression_module(architecture)
            execute([sys.executable,"-u","-m",module,"--config",str(regconfig),"--stop-file",str(stop)],
                    project,root/"logs"/f"{name}.log",stop,deadline)
            history=json.loads((root/name/"model/history.json").read_text())
            if len(history)!=40 or not all(e["training_epoch_complete"] for e in history):
                raise ValueError("Regression did not complete 40 epochs")
            source=root/name/"model/best_model.pt"
            model,metadata=load_regressor(source,architecture)
            if metadata["dataset_sha256"]!=best["dataset_sha256"]: raise ValueError("Regressors used different datasets")
            if metadata["epoch"]!=max(history,key=lambda e:e["scientific_composite"])["epoch"]:
                raise ValueError("Regressor checkpoint does not match the best validation composite")
            del model
            shutil.copy2(source,locks/f"{architecture}_best.pt")
            if digest(source)!=digest(locks/f"{architecture}_best.pt"): raise ValueError("Regressor parent copy changed")
            write_json(locks/f"{architecture}.json",dict(checkpoint_sha256=digest(source),epoch=metadata["epoch"],selection=metadata["checkpoint_selection"]))
        for seed in config["seeds"]:
            for mode in ("scratch","finetune"):
                name=f"{architecture}_{mode}_{seed}"
                command=[sys.executable,"-u","-m","classifier_transfer.train","--data",str(root/"data"),"--output",str(root/"runs"/name),
                         "--config",str(root/"config.json"),"--architecture",architecture,"--seed",str(seed),"--stop-file",str(stop)]
                if mode=="finetune":command += ["--checkpoint",str(locks/f"{architecture}_best.pt")]
                execute(command,project,root/"logs"/f"{name}.log",stop,deadline)
                if not (root/"runs"/name/"complete.json").is_file(): raise ValueError("Classifier did not finish")
    execute([sys.executable,"-u","-m","classifier_transfer.evaluate","--root",str(root),"--config",str(root/"config.json")],
            project,root/"logs/comparison.log",stop,deadline)
    count=len(architectures)*2*len(config["seeds"])
    write_json(root/"complete.json",dict(status="complete",classifiers=count,regression_epochs=40,
               architectures=architectures,
               **({"cnn_regression_epochs": 40} if "cnn" in architectures else {}),
               comparison_sha256=digest(root/"comparison/metrics.json")))
    event("classifier_suite_complete",classifiers=count)


if __name__ == "__main__": main()
