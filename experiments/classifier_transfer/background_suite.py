"""Thirty background-augmented classifiers using locked, completed regressors."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from .background import background_metrics, load_background
from .data import digest, load_split, write_json
from .metrics import summarize
from .models import initialize_classifier
from .pipeline import execute
from .report import cached_predictions
from .train import event, predict, require_cuda_allocation, run


def training_config(config, architecture):
    return dict(config["classifier"],readout=config["readouts"][architecture])


def baseline_locks(project, config, data_root):
    """Check every baseline and parent before launching a new training stage."""
    locks={}
    for architecture in config["architectures"]:
        root=project/config["baseline_suites"][architecture]
        complete=json.loads((root/"complete.json").read_text())
        metrics_path=root/"comparison/metrics.json"
        if complete["status"]!="complete" or complete["comparison_sha256"]!=digest(metrics_path):
            raise ValueError("Baseline comparison must finish before background training")
        metrics=json.loads(metrics_path.read_text())
        old_config=json.loads((root/"config.json").read_text())["classifier"]
        old_config=dict(old_config,readout=old_config.get("readout","legacy"))
        new_config=training_config(config,architecture)
        del new_config["background"]
        if old_config!=new_config:raise ValueError("Baseline architecture/training protocol differs")
        parent=root/"parents"/(architecture+"_best.pt")
        if digest(parent)!=config["parent_checkpoint_sha256"][architecture]:raise ValueError("Parent checkpoint changed")
        for split in ("train","validation","test"):
            # Preparation audit metadata suffices; training need not open test arrays.
            old=json.loads((root/"data/audit.json").read_text())["outputs"][split+".npz"]
            new=json.loads((data_root/"audit.json").read_text())["outputs"][split+".npz"]
            if old!=new:raise ValueError("Enhancer baseline data differ")
        for mode in ("scratch","finetune"):
            for seed in config["seeds"]:
                name=f"{architecture}_{mode}_{seed}"
                lock=metrics["selection"][name]
                if (lock["architecture"]!=architecture or lock["seed"]!=seed or lock["mode"]!=mode):
                    raise ValueError("Baseline model identity mismatch")
                if digest(root/"runs"/name/"best_model.pt")!=lock["checkpoint_sha256"]:
                    raise ValueError("Baseline classifier checkpoint changed")
                locks[name]=dict(lock,baseline_root=str(root),parent_checkpoint=str(parent),
                                parent_sha256=config["parent_checkpoint_sha256"][architecture])
    return locks


def expected_settings(config, data, architecture, seed, mode):
    return dict(architecture=architecture,seed=seed,training=training_config(config,architecture),
        data_audit_sha256=digest(data/"audit.json"),
        background_audit_sha256=digest(data/"background/audit.json"),
        initialization_sha256=config["parent_checkpoint_sha256"][architecture] if mode=="finetune" else None)


def prepare_smoke_data(data, output, limit=192):
    """Small aligned train/validation pools; never open either test array."""
    output.mkdir(exist_ok=False)
    (output/"background").mkdir()
    for split in ("train","validation"):
        enh=load_split(data,split);bg=load_background(data/"background",split,enh)
        n=min(limit,len(bg["ids"]))
        indices=bg["enhancer_index"][:n]
        subset={k:v[indices] for k,v in enh.items()}
        positives=subset["labels"].sum(0)
        if n<3 or (positives==0).any() or (positives==n).any():
            raise ValueError("Smoke subset requires both classes in every context")
        np.savez_compressed(output/(split+".npz"),**subset)
        subset_bg={k:v[:n] for k,v in bg.items()}
        subset_bg["enhancer_index"]=np.arange(n)
        np.savez_compressed(output/"background"/(split+".npz"),**subset_bg)
    for target,source in ((output,data),(output/"background",data/"background")):
        write_json(target/"audit.json",dict(scope="CUDA smoke train/validation only",
            source_audit_sha256=digest(source/"audit.json"),
            outputs={s+".npz":digest(target/(s+".npz")) for s in ("train","validation")}))


def smoke_signature(root, config):
    return dict(package_manifest_sha256=digest(root/"MANIFEST.sha256"),
        data_audit_sha256=digest(root/"data/audit.json"),
        background_audit_sha256=digest(root/"data/background/audit.json"),
        parent_checkpoint_sha256=config["parent_checkpoint_sha256"])


def check_smoke(root, config, locks):
    passed=json.loads((root/"smoke_passed.json").read_text())
    if (passed["status"]!="passed" or passed["signature"]!=smoke_signature(root,config)
            or json.loads((root/"smoke/baseline_lock.json").read_text())!=locks
            or len(passed["classifiers"])!=2*len(config["architectures"])):
        raise ValueError("Successful matching CUDA smoke required")


def smoke_suite(project, root, config):
    require_cuda_allocation()
    if digest(project/"frozen/SOURCE_MANIFEST.sha256")!=config["frozen_manifest_sha256"]:
        raise ValueError("Frozen model source manifest changed")
    locks=baseline_locks(project,config,root/"data")
    smoke=root/"smoke";smoke.mkdir(exist_ok=False)
    write_json(smoke/"baseline_lock.json",locks)
    prepare_smoke_data(root/"data",smoke/"data")
    n=len(load_split(smoke/"data","train")["ids"])
    reports=[]
    for architecture in config["architectures"]:
        small=dict(training_config(config,architecture),epochs=1)
        for mode in ("scratch","finetune"):
            seed=config["seeds"][0];name=f"{architecture}_{mode}_{seed}"
            parent=Path(locks[name]["parent_checkpoint"]) if mode=="finetune" else None
            torch.cuda.reset_peak_memory_stats()
            if not run(smoke/"data",smoke/name,architecture,seed,small,parent,stop=root/"STOP_smoke"):
                raise RuntimeError("CUDA smoke paused before completion")
            saved=torch.load(smoke/name/"last_checkpoint.pt",map_location="cpu",weights_only=False)
            expected=math.ceil(n/small["batch_size"])
            if saved["progress"]["updates"]!=expected or not all(torch.isfinite(v).all() for v in saved["state_dict"].values()):
                raise ValueError("CUDA smoke requires finite weights and no skipped updates")
            reports.append(dict(name=name,updates=expected,peak_memory_bytes=torch.cuda.max_memory_allocated()))
            del saved
    write_json(root/"smoke_passed.json",dict(status="passed",signature=smoke_signature(root,config),
        classifiers=reports,gpu=torch.cuda.get_device_name()))
    event("background_smoke_passed",classifiers=reports)


def train_suite(project, root, config, resume=False):
    require_cuda_allocation()
    if not root.is_relative_to(project/"experiments"):raise ValueError("Experiment must stay in project")
    if digest(project/"frozen/SOURCE_MANIFEST.sha256")!=config["frozen_manifest_sha256"]:
        raise ValueError("Frozen model source manifest changed")
    data=root/"data"
    locks=baseline_locks(project,config,data)
    check_smoke(root,config,locks)
    if resume:
        if json.loads((root/"baseline_lock.json").read_text())!=locks:raise ValueError("Baseline locks changed")
    else:
        if (root/"baseline_lock.json").exists():raise FileExistsError("Use explicit --resume")
        write_json(root/"baseline_lock.json",locks)
        (root/"parents").mkdir()
        for architecture in config["architectures"]:
            source=project/config["baseline_suites"][architecture]/"parents"/(architecture+"_best.pt")
            shutil.copy2(source,root/"parents"/source.name)
    for architecture in config["architectures"]:
        if digest(root/"parents"/(architecture+"_best.pt"))!=config["parent_checkpoint_sha256"][architecture]:
            raise ValueError("Local parent lock changed")
    (root/"runs").mkdir(exist_ok=True);(root/"logs").mkdir(exist_ok=True)
    deadline=time.monotonic()+48*3600-60;stop=root/"STOP_full"
    if stop.exists():raise FileExistsError("Inspect and clear STOP_full before resuming")
    for architecture in config["architectures"]:
        cfg=root/(architecture+"_training.json")
        write_json(cfg,dict(classifier=training_config(config,architecture)))
        for mode in ("scratch","finetune"):
            for seed in config["seeds"]:
                name=f"{architecture}_{mode}_{seed}";out=root/"runs"/name
                done=out/"complete.json"
                if done.exists():
                    c=json.loads(done.read_text())
                    if (c["status"]!="complete" or c["epochs"]!=config["classifier"]["epochs"]
                            or c["settings"]!=expected_settings(config,data,architecture,seed,mode)
                            or c["checkpoint_sha256"]!=digest(out/"best_model.pt")):
                        raise ValueError("Completed classifier signature changed")
                    continue
                cmd=[sys.executable,"-u","-m","classifier_transfer.train","--data",str(data),
                     "--output",str(out),"--architecture",architecture,"--seed",str(seed),
                     "--config",str(cfg),"--stop-file",str(stop)]
                if mode=="finetune":cmd += ["--checkpoint",str(root/"parents"/(architecture+"_best.pt"))]
                if out.exists():
                    if not resume:raise FileExistsError(out)
                    cmd += ["--resume"]
                log=root/"logs"/(name+("_resume_"+str(time.time_ns()) if resume else "")+".log")
                execute(cmd,project,log,stop,deadline)


def evaluate_suite(project, root, config):
    require_cuda_allocation()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    baseline=json.loads((root/"baseline_lock.json").read_text())
    if baseline!=baseline_locks(project,config,root/"data"):raise ValueError("Baselines changed")
    locks={}
    # Do not open test arrays until all 30 models and their validation selections are locked.
    for name,entry in baseline.items():
        directory=root/"runs"/name
        done=json.loads((directory/"complete.json").read_text())
        if (done["status"]!="complete" or done["epochs"]!=config["classifier"]["epochs"]
                or done["checkpoint_sha256"]!=digest(directory/"best_model.pt")
                or done["settings"]!=expected_settings(config,root/"data",entry["architecture"],entry["seed"],entry["mode"])):
            raise ValueError("All background classifiers must be complete and unchanged")
        locks[name]=dict(entry,new_checkpoint_sha256=done["checkpoint_sha256"],new_epoch=done["best_epoch"])
    output=root/"comparison";output.mkdir(exist_ok=False)
    write_json(output/"selection_lock.json",locks)
    test=load_split(root/"data","test");background=load_background(root/"data/background","test",test)
    results={}
    for name,lock in locks.items():
        condition={}
        for arm in ("enhancer_only_training","background_training"):
            base=Path(lock["baseline_root"]) if arm=="enhancer_only_training" else root
            saved=torch.load(base/"runs"/name/"best_model.pt",map_location="cpu",weights_only=False)
            model=initialize_classifier(lock["architecture"],lock["seed"],np.full(8,.5),
                                        readout=config["readouts"][lock["architecture"]]).cuda()
            model.load_state_dict(saved["state_dict"],strict=True)
            p=cached_predictions(base/"comparison"/(name+".npz"),test) if arm=="enhancer_only_training" else predict(model,test,torch.device("cuda"))
            b=predict(model,background,torch.device("cuda"))
            condition[arm]=dict(enhancer_only=summarize(test["labels"],p),
                **background_metrics(test["labels"],p,b,background["enhancer_index"]))
            np.savez_compressed(output/(name+"__"+arm+".npz"),ids=test["ids"],probabilities=p,
                                background_ids=background["ids"],background_probabilities=b)
            del model,saved
        condition["paired_ap_difference"]={view:condition["background_training"][view]["macro_average_precision"]-
            condition["enhancer_only_training"][view]["macro_average_precision"]
            for view in ("enhancer_only","active_vs_background","enhancers_plus_background")}
        results[name]=condition
        event("background_test_evaluated",model=name,paired_ap_difference=condition["paired_ap_difference"])
    write_json(output/"metrics.json",dict(status="complete",test_n=len(test["ids"]),selection=locks,
        results=results,scope="Historical chr3R; matched existing backgrounds; AP and FPR diagnostics do not alter thresholds/models",
        background_audit_sha256=digest(root/"data/background/audit.json")))
    write_json(root/"complete.json",dict(status="complete",classifiers=len(results),
        comparison_sha256=digest(output/"metrics.json")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,required=True);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--stage",choices=["smoke","train","evaluate"],required=True);p.add_argument("--resume",action="store_true")
    a=p.parse_args();project=a.project.resolve();root=a.root.resolve()
    config=json.loads((root/"config.json").read_text())
    if a.stage=="smoke":smoke_suite(project,root,config)
    elif a.stage=="train":train_suite(project,root,config,a.resume)
    else:evaluate_suite(project,root,config)
