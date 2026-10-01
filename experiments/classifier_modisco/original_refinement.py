"""CPU-only recovery from saved raw discoveries; never rerun attribution/discovery."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .common import digest, event, require_allocation, write_json
from .original_discovery import refine
from .original_intervals import GROUPS, contract, load_intervals


INPUTS = ("motifs.h5", "motifs.audit.json", "selection.json", "examples.npz", "discovery_inputs.npz")


def verify_inputs(root, source, config, data, intervals, group_index):
    name, low, high = GROUPS[group_index]
    selection = json.loads((source/"selection.json").read_text())
    parameters = dict(config["discovery_parameters"])
    parameters["seed"] += group_index
    attribution = json.loads((root/"attribution_complete.json").read_text())
    if (attribution["status"] != "complete" or attribution["quality_failures"] != 0
            or selection["group"] != name or not selection["train_only"]
            or selection["quality_excluded"] != 0 or selection["enhancer_downsampling"]
            or selection["parameters"] != parameters
            or selection["interval"] != "original_catalog_bounds"
            or selection["intervals_sha256"] != digest(root/"intervals.npz")
            or selection["attribution_sha256"] != digest(root/"attribution_complete.json")):
        raise ValueError("Recovery input contract differs from completed production discovery")
    breadth = data["labels"].sum(1)
    eligible = (data["split"] == "train") & (breadth >= low) & (breadth <= high)
    indices = np.random.default_rng(parameters["seed"]).permutation(np.flatnonzero(eligible))
    if len(indices) != selection["elements"] or len(indices) != selection["eligible_before_quality"]:
        raise ValueError("Discovery cohort size changed")
    lengths = intervals["length"][indices]
    with np.load(source/"examples.npz", allow_pickle=False) as f:
        for key, value in dict(indices=indices, ids=data["ids"][indices], lengths=lengths,
                chrom=data["chrom"][indices], start=intervals["start"][indices],
                end=intervals["end"][indices]).items():
            np.testing.assert_array_equal(f[key], value)
    with np.load(source/"discovery_inputs.npz", allow_pickle=False) as f:
        sequence, hypothetical = f["sequence"], f["hypothetical"]
        np.testing.assert_array_equal(f["lengths"], lengths)
    if sequence.shape != (len(indices), int(lengths.max()), 4) or hypothetical.shape != sequence.shape:
        raise ValueError("Discovery arrays have incompatible dimensions")
    for row, i in enumerate(indices):
        start, length = int(intervals["offset"][i]), int(lengths[row])
        np.testing.assert_array_equal(sequence[row, :length], np.eye(4)[data["sequence"][i, start:start+length]])
        if np.any(sequence[row, length:] != 0) or not np.isfinite(hypothetical[row, :length]).all():
            raise ValueError("Invalid padding or hypothetical scores")
    return sequence, hypothetical, lengths, parameters


def recover(project, root, output_root, group_index):
    require_allocation("cpu")
    config, _, data = contract(project, root)
    intervals = load_intervals(root, data)
    name = GROUPS[group_index][0]
    source, output = root/"groups"/name, output_root/"groups"/name
    before = {filename: digest(source/filename) for filename in INPUTS}
    sequence, hyp, lengths, parameters = verify_inputs(root, source, config, data, intervals, group_index)
    audit = json.loads((source/"motifs.audit.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"inputs.json", dict(source=str(source.relative_to(project)), sha256=before,
        selection_sha256=before["selection.json"], config_sha256=digest(root/"config.json")))
    event("original_refinement_started", group=name, raw_patterns={s:audit[s]["patterns"] for s in ("positive", "negative")})
    np.random.seed(parameters["seed"])
    counts = refine(output, sequence, hyp, lengths, parameters, raw_path=source/"motifs.h5")
    if before != {filename: digest(source/filename) for filename in INPUTS}:
        raise ValueError("Raw discovery inputs changed during refinement")
    write_json(output/"complete.json", dict(status="complete", group=name, refinement_only=True,
        attribution_rerun=False, discovery_rerun=False, raw_patterns={s:audit[s]["patterns"] for s in ("positive", "negative")},
        refined_patterns=counts, input_sha256=before, job_id=os.environ["SLURM_JOB_ID"],
        package_sha256=digest(output_root/"MANIFEST.sha256"),
        files={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    event("original_refinement_complete", group=name, refined_patterns=counts)


def summarize(root, output_root):
    require_allocation("cpu")
    groups = []
    for name, _, _ in GROUPS:
        output = output_root/"groups"/name
        record = json.loads((output/"complete.json").read_text())
        if record["status"] != "complete" or record["package_sha256"] != digest(output_root/"MANIFEST.sha256"):
            raise ValueError("Incomplete or incompatible refinement")
        for filename, expected in record["input_sha256"].items():
            if digest(root/"groups"/name/filename) != expected:
                raise ValueError("Raw discovery changed: "+name+"/"+filename)
        for filename, expected in record["files"].items():
            if digest(output/filename) != expected:
                raise ValueError("Refinement output changed: "+name+"/"+filename)
        groups.append(record)
    write_json(output_root/"discovery_complete.json", dict(status="complete", groups=groups,
        refinement_only=True, attribution_rerun=False, discovery_rerun=False,
        sequence_scan="pending", predictive_scan="pending", annotation="pending", figures="pending"))
    event("original_refinement_summary_complete", groups=len(groups))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("recover", "summarize"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-index", type=int, choices=range(len(GROUPS)))
    args = parser.parse_args()
    if args.stage == "summarize":
        summarize(args.root, args.output)
    else:
        if args.group_index is None:
            parser.error("--group-index is required for recover")
        recover(args.project, args.root, args.output, args.group_index)
