"""Reference pilot and original-interval hypothetical scores on CECAR CUDA."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from classifier_motifs.attribution import (active_weights, dinucleotide_shuffle,
    ensemble, integrated_gradients, load_model, one_hot, seed_for)
from .common import digest, event, require_allocation, write_json
from .original_intervals import contract, load_intervals


def per_reference(model, data, indices, config, device, reference):
    x = one_hot(data["sequence"][indices], device)
    weights = active_weights(torch.as_tensor(data["labels"][indices], device=device))
    codes = np.stack([dinucleotide_shuffle(data["sequence"][i],
        seed_for(config["seed"], data["ids"][i], reference)) for i in indices])
    baseline = one_hot(codes, device)
    result = integrated_gradients(model, x, baseline, weights, config["steps"], config["internal_batch"])
    steps_used = torch.full((len(indices),), config["steps"], device=device, dtype=torch.int32)
    for steps in (64, 128):
        if steps <= config["steps"]:
            continue
        tolerance = config["absolute_tolerance"]+config["relative_tolerance"]*result["target_difference"].abs()
        bad = torch.where(result["delta"].abs() > tolerance)[0]
        if not len(bad):
            break
        refined = integrated_gradients(model, x[bad], baseline[bad], weights[bad], steps, config["internal_batch"])
        for key, value in refined.items():
            result[key][bad] = value
        steps_used[bad] = steps
    tolerance = config["absolute_tolerance"]+config["relative_tolerance"]*result["target_difference"].abs()
    result["quality_pass"] = result["delta"].abs() <= tolerance
    result["steps"] = steps_used
    return {key: value.detach().cpu().numpy() for key, value in result.items()}


def score_original(model, data, intervals, indices, config, reference_counts):
    """Keep full integration/context, store only original enhancer hypotheses."""
    maximum = int(intervals["length"][indices].max())
    hyp = np.zeros((len(indices), 4, maximum), np.float32)
    actual = np.zeros((len(indices), 2048), np.float32)
    deltas, differences, steps, quality = [], [], [], []
    snapshots = {}
    for reference in range(max(reference_counts)):
        result = per_reference(model, data, indices, config, "cuda", reference)
        actual += result["actual"]
        for row, i in enumerate(indices):
            start, length = int(intervals["offset"][i]), int(intervals["length"][i])
            hyp[row, :, :length] += result["hypothetical"][row, :, start:start+length]
        deltas.append(result["delta"]); differences.append(result["target_difference"])
        steps.append(result["steps"]); quality.append(result["quality_pass"])
        if reference+1 in reference_counts:
            count = reference+1
            snapshots[count] = dict(hypothetical=hyp.copy()/count, actual=actual.copy()/count,
                delta=np.stack(deltas, 1), target_difference=np.stack(differences, 1),
                steps=np.stack(steps, 1), quality_pass=np.stack(quality, 1).all(1))
    return snapshots


def validate_projection(result, data, intervals, indices):
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite attribution")
    for row, index in enumerate(indices):
        start, length = int(intervals["offset"][index]), int(intervals["length"][index])
        codes = data["sequence"][index, start:start+length]
        projected = result["hypothetical"][row, codes, np.arange(length)]
        np.testing.assert_allclose(projected, result["actual"][row, start:start+length], atol=1e-7, rtol=1e-5)


def stability(first, second, lengths):
    rows = []
    for a, b, length in zip(first, second, lengths):
        a, b = a[:length].astype(float), b[:length].astype(float)
        norm = np.linalg.norm(a)*np.linalg.norm(b)
        cosine = float(a@b/norm) if norm else None
        top = max(5, int(np.ceil(.1*length)))
        ai, bi = np.argsort(np.abs(a))[-top:], np.argsort(np.abs(b))[-top:]
        intersection, union = np.intersect1d(ai, bi), np.union1d(ai, bi)
        weight = np.maximum(np.abs(a), np.abs(b))
        agreement = float(weight[np.sign(a) == np.sign(b)].sum()/weight.sum()) if weight.sum() else None
        rows.append(dict(cosine=cosine, top10pct_jaccard=float(len(intersection)/len(union)),
                         magnitude_weighted_sign_agreement=agreement))
    return rows


def observed_original(result, data, intervals, indices):
    values = np.zeros((len(indices), result["hypothetical"].shape[2]), np.float32)
    for row, i in enumerate(indices):
        start, length = int(intervals["offset"][i]), int(intervals["length"][i])
        values[row, :length] = result["actual"][row, start:start+length]
    return values


def pilot(root, parent, data, intervals, config, model, extended=False):
    breadth = data["labels"].sum(1)
    rng = np.random.default_rng(config["seed"])
    # Stratification is for this numerical pilot only, never for production discovery.
    indices = np.concatenate([rng.choice(np.flatnonzero((data["split"] == "train") & (breadth == k)),
        4, replace=False) for k in range(1, 9)])
    started = time.monotonic()
    counts = [2, 10, 12, 20, 40, 50, 100] if extended else [2, 10, 12, 20]
    pilot_output = root/"reference_extended" if extended else root
    pilot_output.mkdir(exist_ok=True)
    snapshots = score_original(model, data, intervals, indices, config, counts)
    for count, result in snapshots.items():
        validate_projection(result, data, intervals, indices)
        np.savez_compressed(pilot_output/f"pilot_references_{count}.npz", indices=indices, **result)
    values = {k: observed_original(v, data, intervals, indices) for k, v in snapshots.items()}
    # Compare independent halves, not nested estimates only (part/whole inflates agreement).
    independent_two = (values[12]*12-values[10]*10)/2
    independent_ten = values[20]*2-values[10]
    comparisons = {
        "independent_2_vs_2": stability(values[2], independent_two, intervals["length"][indices]),
        "independent_10_vs_10": stability(values[10], independent_ten, intervals["length"][indices]),
        "nested_2_vs_20": stability(values[2], values[20], intervals["length"][indices]),
        "nested_10_vs_20": stability(values[10], values[20], intervals["length"][indices]),
    }
    if extended:
        comparisons.update({
            "independent_20_vs_20": stability(values[20], values[40]*2-values[20], intervals["length"][indices]),
            "independent_50_vs_50": stability(values[50], values[100]*2-values[50], intervals["length"][indices]),
            "nested_20_vs_100": stability(values[20], values[100], intervals["length"][indices]),
            "nested_50_vs_100": stability(values[50], values[100], intervals["length"][indices]),
        })
    summaries = {name: {key: dict(median=float(np.median([r[key] for r in rows])),
        p10=float(np.quantile([r[key] for r in rows], .1))) for key in rows[0]}
        for name, rows in comparisons.items()}
    # Regression check against saved parent chunks with the exact same two references.
    complete = json.loads((parent/"attribution_complete.json").read_text())
    for row, index in enumerate(indices):
        name = "chunk_%06d.npz" % (int(index)//config["batch_size"]*config["batch_size"])
        path = parent/"chunks"/name
        if digest(path) != complete["chunks"][name]:
            raise ValueError("Parent attribution chunk changed")
        with np.load(path, allow_pickle=False) as old:
            position = int(np.flatnonzero(old["indices"] == index)[0])
            np.testing.assert_allclose(snapshots[2]["actual"][row], old["actual"][position], atol=3e-5, rtol=2e-3)
    elapsed = time.monotonic()-started
    report = dict(status="complete_awaiting_reference_choice", indices=indices.tolist(),
        summaries=summaries, comparisons=comparisons, seconds=elapsed,
        quality_pass_fraction={str(k): float(v["quality_pass"].mean()) for k, v in snapshots.items()},
        projected_hours_full_10_references=len(data["ids"])/len(indices)*elapsed*10/counts[-1]/3600,
        old_two_reference_reproduction="passed", job_id=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, config_sha256=digest(root/"config.json"))
    write_json(root/("reference_pilot_extended.json" if extended else "reference_pilot.json"), report)
    event("original_reference_pilot_complete", **{k: v for k, v in report.items() if k not in ("comparisons", "indices")})


def attribute(root, parent, data, intervals, config, model, shard_index=0, shards=1):
    decision = json.loads((root/"reference_choice.json").read_text())
    if decision["pilot_sha256"] != digest(root/decision.get("pilot_file", "reference_pilot.json")):
        raise ValueError("Reference choice does not match reviewed pilot")
    count = int(decision["references"])
    if count not in (2, 10, 20, 50, 100):
        raise ValueError("Unsupported reference choice")
    if not 0 <= shard_index < shards or shards != decision.get("shards", 1):
        raise ValueError("Unexpected shard allocation")
    config = dict(config, references=count)
    signature = digest(root/"reference_choice.json")
    parent_complete = json.loads((parent/"attribution_complete.json").read_text())
    chunks = root/"chunks"; chunks.mkdir(exist_ok=True)
    failed, processed = 0, 0
    saved_paths = []
    for batch, begin in enumerate(range(0, len(data["ids"]), config["batch_size"])):
        if batch % shards != shard_index:
            continue
        indices = np.arange(begin, min(begin+config["batch_size"], len(data["ids"])))
        path = chunks/("chunk_%06d.npz" % begin)
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                if str(f["signature"]) != signature or not np.array_equal(f["indices"], indices):
                    raise ValueError("Incompatible resume chunk")
                result = {key: f[key] for key in ("hypothetical", "actual", "delta", "target_difference", "steps", "quality_pass")}
        else:
            # Reuse old central scores only when the original two-reference protocol is retained.
            covered = (intervals["offset"][indices] >= 768) & (
                intervals["offset"][indices]+intervals["length"][indices] <= 1280)
            if count == 2:
                old_path = parent/"chunks"/path.name
                if digest(old_path) != parent_complete["chunks"][path.name]:
                    raise ValueError("Parent chunk changed")
                with np.load(old_path, allow_pickle=False) as f:
                    np.testing.assert_array_equal(f["indices"], indices)
                    result = {key: f[key].copy() for key in ("actual", "delta", "target_difference", "steps", "quality_pass")}
                    result["hypothetical"] = np.zeros((len(indices), 4, int(intervals["length"][indices].max())), np.float32)
                    for row in np.flatnonzero(covered):
                        i = indices[row]; offset = int(intervals["offset"][i])-768
                        length = int(intervals["length"][i])
                        result["hypothetical"][row, :, :length] = f["hypothetical_central512"][row, :, offset:offset+length]
                missing = np.flatnonzero(~covered)
                if len(missing):
                    fresh = score_original(model, data, intervals, indices[missing], config, [count])[count]
                    for key in result:
                        if key == "hypothetical":
                            result[key][missing, :, :fresh[key].shape[2]] = fresh[key]
                        else:
                            result[key][missing] = fresh[key]
            else:
                result = score_original(model, data, intervals, indices, config, [count])[count]
            validate_projection(result, data, intervals, indices)
            tmp = path.with_suffix(".partial.npz")
            np.savez_compressed(tmp, indices=indices, signature=signature, **result)
            tmp.replace(path)
        validate_projection(result, data, intervals, indices)
        failed += int((~result["quality_pass"]).sum())
        saved_paths.append(path)
        processed += len(indices)
        if processed % 256 == 0 or indices[-1]+1 == len(data["ids"]):
            event("original_attribution_progress", shard=shard_index, shards=shards,
                completed_in_shard=processed, total_cohort=len(data["ids"]), references=count, quality_failures=failed)
        if processed >= 512 and 1-failed/processed < config["minimum_pass_fraction"]:
            raise ValueError("Attribution convergence below required pass fraction")
        if (root/"STOP").exists():
            raise RuntimeError("Stopped after a checkpointed chunk")
    destination = "attribution_complete.json" if shards == 1 else f"attribution_shard_{shard_index}.json"
    write_json(root/destination, dict(status="complete", references=count, shard_index=shard_index, shards=shards,
        elements=processed, quality_failures=failed, signature=signature,
        intervals_sha256=digest(root/"intervals.npz"),
        chunks={p.name: digest(p) for p in saved_paths}))
    event("original_attribution_shard_complete", shard=shard_index, elements=processed, references=count)


def run(project, root, stage, shard_index=0, shards=1):
    require_allocation("gpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation required")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, parent, data = contract(project, root)
    intervals = load_intervals(root, data)
    config = json.loads((parent/"attribution_config.json").read_text())
    model = load_model(parent/"best_model.pt", "cuda")
    if stage.startswith("pilot"):
        pilot(root, parent, data, intervals, config, model, extended=stage == "pilot_extended")
    else:
        attribute(root, parent, data, intervals, config, model, shard_index, shards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "pilot_extended", "attribute"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    run(args.project, args.root, args.stage, args.shard_index, args.shards)
