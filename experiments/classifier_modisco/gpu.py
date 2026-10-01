"""Resume frozen attribution on an allocated CECAR RTX GPU, never a login node."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from classifier_motifs.attribution import load_model, one_hot, ensemble, score_batch
from .common import digest, event, load_inputs, projected_contributions, require_allocation, write_json


def check_result(result, codes):
    if any(not np.isfinite(v).all() for v in result.values()): raise ValueError("Nonfinite attribution")
    projected = projected_contributions(codes[:,768:1280], result["hypothetical_central512"])
    np.testing.assert_allclose(projected,result["actual"][:,768:1280],atol=1e-7,rtol=1e-5)


def smoke(root, data, attribution_config, model, provenance):
    # First verify compatibility with already-saved local FP32 scores.
    source = sorted((root/"seed_chunks").glob("chunk_*.npz"))[0]
    with np.load(source,allow_pickle=False) as f: previous = dict(f)
    idx = previous["indices"]
    fresh = score_batch(model,data,idx,attribution_config,"cuda")
    check_result(fresh,data["sequence"][idx])
    for key in ("actual","hypothetical_central512","logits","probabilities"):
        np.testing.assert_allclose(fresh[key],previous[key],atol=3e-5,rtol=2e-3)
    if not fresh["quality_pass"].all(): raise ValueError("Cross-device attribution failed convergence")
    breadth = data["labels"].sum(1)
    rng = np.random.default_rng(attribution_config["seed"])
    idx = np.concatenate([rng.choice(np.flatnonzero((data["split"]=="train")&(breadth==k)),4,replace=False) for k in range(1,9)])
    start = time.monotonic()
    fresh = score_batch(model,data,idx,attribution_config,"cuda")
    seconds = time.monotonic()-start
    check_result(fresh,data["sequence"][idx])
    if fresh["quality_pass"].mean() < attribution_config["minimum_pass_fraction"]: raise ValueError("Breadth-stratified pilot failed")
    val = np.flatnonzero(data["split"]=="validation")[:64]
    with torch.no_grad(): predicted = ensemble(model,one_hot(data["sequence"][val],"cuda"))[1].cpu().numpy()
    with np.load(root/"cached_validation.npz",allow_pickle=False) as f:
        if not np.array_equal(data["ids"][val],f["ids"][:64]): raise ValueError("Cached validation IDs differ")
        difference = float(np.max(np.abs(predicted-f["probabilities"][:64])))
    if difference >= .003: raise ValueError("Frozen classifier differs from archived predictions")
    report = dict(status="passed",job_id=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,seconds_per_32=seconds,
        projected_hours=len(data["ids"])*seconds/32/3600,validation_max_difference=difference,
        cross_device_chunk=source.name,convergence_pass_fraction=float(fresh["quality_pass"].mean()),
        input_provenance_sha256=digest(root/"input_provenance.json"))
    write_json(root/"cuda_smoke.json",report); event("cecar_attribution_smoke",**report)


def run(root):
    require_allocation("gpu")
    if not torch.cuda.is_available() or not any(name in torch.cuda.get_device_name() for name in ("RTX 2080","RTX 4070")):
        raise RuntimeError("Expected an allocated CECAR RTX 2080/4070")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK",4)))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, provenance, data = load_inputs(root)
    original_config = json.loads((root/"attribution_config.json").read_text())
    signature = digest(root/"attribution_config.json")
    model = load_model(root/"best_model.pt","cuda")
    smoke(root,data,original_config,model,provenance)
    chunks = root/"chunks"; chunks.mkdir(exist_ok=True)
    started, failed = time.monotonic(),0
    for begin in range(0,len(data["ids"]),original_config["batch_size"]):
        if (root/"STOP").exists(): raise RuntimeError("Paused after saved attribution chunk")
        idx = np.arange(begin,min(begin+original_config["batch_size"],len(data["ids"])))
        path = chunks/("chunk_%06d.npz"%begin)
        seed = root/"seed_chunks"/path.name
        if not path.exists() and seed.exists(): shutil.copy2(seed,path)
        if path.exists():
            with np.load(path,allow_pickle=False) as f:
                if str(f["signature"]) != signature or not np.array_equal(f["indices"],idx): raise ValueError("Resume chunk differs")
                result = {k:f[k] for k in f.files if k not in ("indices","signature")}
        else:
            result = score_batch(model,data,idx,original_config,"cuda")
            temporary = path.with_suffix(".partial.npz")
            check_result(result,data["sequence"][idx])
            np.savez_compressed(temporary,indices=idx,signature=signature,**result)
            temporary.replace(path)
        check_result(result,data["sequence"][idx])
        failed += int((~result["quality_pass"]).sum())
        done = int(idx[-1])+1
        event("cecar_attribution_progress",completed=done,total=len(data["ids"]),quality_failures=failed,seconds=round(time.monotonic()-started,1))
        if done >=512 and 1-failed/done < original_config["minimum_pass_fraction"]:
            raise RuntimeError("Too many unconverged attribution scores")
    write_json(root/"attribution_complete.json",dict(status="complete",elements=len(data["ids"]),quality_failures=failed,
        signature=signature,input_provenance_sha256=digest(root/"input_provenance.json"),
        chunks={p.name:digest(p) for p in sorted(chunks.glob("chunk_*.npz"))}))
    event("cecar_attribution_complete",elements=len(data["ids"]),quality_failures=failed)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",type=Path,required=True)
    run(p.parse_args().root)
