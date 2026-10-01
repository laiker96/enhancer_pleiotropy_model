"""Resumable, wall-limited CECAR pilot; never launches a production analysis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from classifier_motifs.attribution import dinucleotide_shuffle, load_model, one_hot, seed_for
from classifier_motifs.context_attribution import (CONTEXTS, REGIONS,
    context_integrated_gradients, exact_ism, ism_positions, map_agreement,
    pilot_indices, region_masks)
from .common import digest, event, require_allocation, write_json
from .original_intervals import load_intervals


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def check_config(config):
    if (config["contexts"] != list(CONTEXTS) or config["input_bp"] != 2048
            or config["references_per_block"] != [50, 100]
            or config["integration_steps"] != [64, 128, 256, 512]
            or config["reference_blocks"] != 2 or not 1 <= config["per_degree"] <= 16
            or not 60 <= config["max_seconds"] <= 13200
            or config["internal_batch"] < 1
            or config["absolute_tolerance"] != .01 or config["relative_tolerance"] != .01):
        raise ValueError("Unexpected bounded-pilot contract; revise deliberately, not by silent override")


def refine_reference(model, x, baseline, masks, config, anchor=False):
    comparisons = []
    previous = None
    for steps in config["integration_steps"]:
        # Old configs retain their sequential numerical implementation. New
        # optimized experiments must explicitly record the context batch size.
        result = context_integrated_gradients(model, x, baseline, steps, config["internal_batch"],
                                             context_batch=config.get("context_batch", 1))
        if previous is not None:
            first, second = previous["actual"][0].cpu().numpy(), result["actual"][0].cpu().numpy()
            comparisons.append(dict(from_steps=last_steps, to_steps=steps,
                full=map_agreement(first, second),
                regions={name: map_agreement(first[:, mask], second[:, mask])
                         for name, mask in zip(REGIONS, masks)}))
        tolerance = config["absolute_tolerance"]+config["relative_tolerance"]*result["target_difference"].abs()
        passed = result["delta"].abs() <= tolerance
        if passed.all() and (not anchor or steps >= 128):
            break
        previous, last_steps = result, steps
    return result, int(steps), passed, comparisons


def validate_state(state, signature, maximum):
    count = int(state["count"])
    if (str(state["signature"]) != signature or not 0 <= count <= maximum
            or state["sum_hypothetical"].shape != (8, 4, 2048)
            or not np.isfinite(state["sum_hypothetical"]).all()
            or state["delta"].shape != (count, 8)
            or state["difference"].shape != (count, 8)
            or state["steps"].shape != (count,)
            or state["passed"].shape != (count, 8)
            or len(state["reference_hashes"]) != count
            or not np.isfinite(state["delta"]).all() or not np.isfinite(state["difference"]).all()):
        raise ValueError("Resume state belongs to a different contract or is malformed")


def run(project, root):
    require_allocation("gpu")
    began = time.monotonic()
    config = json.loads((root/"config.json").read_text())
    check_config(config)
    for relative, expected in config["source_hashes"].items():
        if digest(project/relative) != expected:
            raise ValueError("Frozen source/input changed: "+relative)
    parent = project/config["parent"]
    original = project/config["original"]
    with np.load(parent/"cohort.npz", allow_pickle=False) as f:
        data = dict(f)
    intervals = load_intervals(original, data)
    indices = pilot_indices(data, intervals, config["per_degree"], config["seed"])
    signature = hashlib.sha256((digest(root/"config.json")+digest(root/"MANIFEST.sha256")).encode()).hexdigest()
    out = root/"pilot"
    out.mkdir(exist_ok=True)
    selection = dict(signature=signature, train_only=True, numerical_pilot_not_prevalence=True,
        contexts=list(CONTEXTS), regions=list(REGIONS), indices=indices.tolist(),
        ids=data["ids"][indices].tolist(), labels=data["labels"][indices].tolist(),
        chrom=data["chrom"][indices].tolist(), summit=data["summit"][indices].tolist(),
        native_start=intervals["start"][indices].tolist(), native_end=intervals["end"][indices].tolist(),
        native_offsets=intervals["offset"][indices].tolist(), lengths=intervals["length"][indices].tolist())
    selection_path = out/"selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Refusing to mix pilot selections or changed code")
    write_json(selection_path, selection)
    if (out/"complete.json").exists():
        raise ValueError("Pilot already complete; inspect results instead of rerunning")
    stopped = [False]
    for signum in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.__setitem__(0, True))
    def stop():
        return stopped[0] or (root/"STOP").exists() or time.monotonic()-began >= config["max_seconds"]
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated CUDA GPU not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = load_model(parent/"best_model.pt", "cuda")
    event("context_pilot_started", examples=len(indices), targets=8,
          gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, dtype="float32",
          max_seconds=config["max_seconds"], signature=signature)
    completed = 0
    maximum = config["references_per_block"][-1]
    for index in indices:
        i = int(index)
        folder = out/f"enhancer_{i:06d}"
        folder.mkdir(exist_ok=True)
        if (folder/"complete.json").exists():
            receipt = json.loads((folder/"complete.json").read_text())
            if receipt["signature"] != signature:
                raise ValueError("Completed enhancer contract mismatch")
            for name, expected in receipt["outputs"].items():
                if digest(folder/name) != expected:
                    raise ValueError("Completed enhancer output changed: "+name)
            completed += 1
            continue
        x = one_hot(data["sequence"][[i]], "cuda")
        codes = data["sequence"][i]
        masks = region_masks(int(intervals["offset"][i]), int(intervals["length"][i]))
        for block in range(2):
            path = folder/f"block_{block}_state.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as f:
                    state = dict(f)
                validate_state(state, signature, maximum)
            else:
                state = dict(signature=np.asarray(signature), count=np.asarray(0),
                    sum_hypothetical=np.zeros((8, 4, 2048), np.float64),
                    delta=np.empty((0, 8)), difference=np.empty((0, 8)),
                    steps=np.empty(0, np.int32), passed=np.empty((0, 8), bool),
                    reference_hashes=np.empty(0, dtype="U64"), diagnostics=np.asarray("[]"))
            diagnostics = json.loads(str(state["diagnostics"]))
            for reference in range(int(state["count"]), maximum):
                if stop():
                    save_npz(path, **state)
                    write_json(out/"progress.json", dict(status="paused", signature=signature,
                        completed=completed, total=len(indices), index=i, block=block,
                        references=int(state["count"]), elapsed_seconds=time.monotonic()-began,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        reason="wall_budget_signal_or_STOP", job_id=os.environ["SLURM_JOB_ID"]))
                    event("context_pilot_paused", completed=completed, index=i, block=block, references=int(state["count"]))
                    return
                reference_seed = seed_for(config["seed"], str(data["ids"][i]), "context_pilot", block, reference)
                shuffled = dinucleotide_shuffle(codes, reference_seed)
                result, steps, passed, comparisons = refine_reference(model, x,
                    one_hot(shuffled[None], "cuda"), masks, config, anchor=reference == 0)
                hyp = result["hypothetical"][0].cpu().numpy()
                actual = result["actual"][0].cpu().numpy()
                np.testing.assert_allclose(hyp[:, codes, np.arange(2048)], actual, atol=1e-7, rtol=1e-5)
                state["sum_hypothetical"] += hyp
                state["count"] = np.asarray(reference+1)
                state["delta"] = np.vstack([state["delta"], result["delta"].cpu().numpy()])
                state["difference"] = np.vstack([state["difference"], result["target_difference"].cpu().numpy()])
                state["steps"] = np.append(state["steps"], np.int32(steps))
                state["passed"] = np.vstack([state["passed"], passed.cpu().numpy()])
                state["reference_hashes"] = np.append(state["reference_hashes"], hashlib.sha256(shuffled.tobytes()).hexdigest())
                diagnostics.append(dict(reference=reference, seed=reference_seed,
                    unchanged=bool(np.array_equal(shuffled, codes)), integration_comparisons=comparisons))
                state["diagnostics"] = np.asarray(json.dumps(diagnostics, allow_nan=False))
                if reference+1 in config["references_per_block"]:
                    mean = (state["sum_hypothetical"]/(reference+1)).astype(np.float32)
                    save_npz(folder/f"block_{block}_n{reference+1}.npz", hypothetical=mean,
                        actual=mean[:, codes, np.arange(2048)], sequence=codes, region_masks=masks,
                        logits=result["logits"][0].cpu().numpy(), probabilities=result["probabilities"][0].cpu().numpy(),
                        labels=data["labels"][i], signature=np.asarray(signature))
                # Commit state only after its reference-count snapshot exists.
                if (reference+1) % 10 == 0:
                    save_npz(path, **state)
                    event("context_reference_progress", index=i, block=block, references=reference+1,
                          completed=completed, elapsed_seconds=time.monotonic()-began,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            save_npz(path, **state)
        agreement = {}
        for count in config["references_per_block"]:
            with np.load(folder/f"block_0_n{count}.npz") as a, np.load(folder/f"block_1_n{count}.npz") as b:
                agreement[str(count)] = dict(full=map_agreement(a["actual"], b["actual"]),
                    regions={name: map_agreement(a["actual"][:, mask], b["actual"][:, mask])
                             for name, mask in zip(REGIONS, masks)})
                if count == maximum:
                    combined = (a["hypothetical"]+b["hypothetical"])*.5
        actual = combined[:, codes, np.arange(2048)]
        positions = ism_positions(actual, masks, seed_for(config["seed"], i, "ism"))
        mutations = exact_ism(model, x, positions, batch_size=8)
        mutations["ig_hypothetical_difference"] = np.stack([
            combined[:, base, p]-combined[:, int(codes[p]), p]
            for p, base in zip(mutations["positions"], mutations["alternate"])])
        save_npz(folder/"ism.npz", **mutations)
        write_json(folder/"agreement.json", dict(independent_reference_blocks=True,
            comparisons=agreement, contexts=list(CONTEXTS),
            ism_note="Exact mutant-WT logit deltas; hypothetical IG differences are approximations, not exact mutation predictions"))
        outputs = {p.name: digest(p) for p in sorted(folder.iterdir()) if p.suffix in (".json", ".npz")}
        write_json(folder/"complete.json", dict(signature=signature, index=i,
            outputs=outputs, reference_blocks=2, references_per_block=maximum))
        completed += 1
        event("context_enhancer_complete", completed=completed, total=len(indices), index=i)
        write_json(out/"progress.json", dict(status="in_progress", signature=signature,
            completed=completed, total=len(indices), elapsed_seconds=time.monotonic()-began))
    write_json(out/"complete.json", dict(status="complete", signature=signature,
        examples=completed, job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds_this_allocation=time.monotonic()-began,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        interpretation="Numerical pilot only. Review completeness, independent-reference stability and ISM before production."))
    event("context_pilot_complete", examples=completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.project.resolve(), args.root.resolve())


if __name__ == "__main__":
    main()
