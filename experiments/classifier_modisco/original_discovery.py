"""Independent signed fits and native pattern refinement in original enhancers."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .common import digest, event, require_allocation, write_json
from .original_intervals import (GROUPS, bounded_native_aggregator, contract,
    discover_original, load_intervals, make_track_set)


def collect(root, data, intervals):
    complete = json.loads((root/"attribution_complete.json").read_text())
    if complete["intervals_sha256"] != digest(root/"intervals.npz"):
        raise ValueError("Attribution/interval mismatch")
    n, width = len(data["ids"]), int(intervals["length"].max())
    hyp = np.zeros((n, width, 4), np.float32)
    quality, seen = np.zeros(n, bool), np.zeros(n, bool)
    for name, expected in complete["chunks"].items():
        path = root/"chunks"/name
        if digest(path) != expected:
            raise ValueError("Changed attribution chunk")
        with np.load(path, allow_pickle=False) as f:
            indices = f["indices"]
            if seen[indices].any() or str(f["signature"]) != complete["signature"]:
                raise ValueError("Duplicate or incompatible attribution chunk")
            hyp[indices, :f["hypothetical"].shape[2]] = f["hypothetical"].transpose(0, 2, 1)
            for row, i in enumerate(indices):
                offset, length = int(intervals["offset"][i]), int(intervals["length"][i])
                codes = data["sequence"][i, offset:offset+length]
                np.testing.assert_allclose(hyp[i, np.arange(length), codes],
                    f["actual"][row, offset:offset+length], atol=1e-7, rtol=1e-5)
            quality[indices] = f["quality_pass"]; seen[indices] = True
    if not seen.all():
        raise ValueError("Incomplete attribution cohort")
    return hyp, quality


def trimmed_pattern(pattern, fraction=.3):
    """HDMA-style contribution-magnitude terminal trim, safe for either sign."""
    weight = np.abs(pattern.contrib_scores).sum(1)
    keep = np.flatnonzero(weight >= fraction*weight.max()) if weight.max() > 0 else []
    if not len(keep):
        return None, None
    left, right = int(keep[0]), int(keep[-1])+1
    return pattern.trim_to_idx(left, right), [left, right]


def sequence_qc(pattern):
    ppm = pattern.sequence
    ppm = (ppm+.001)/(ppm.sum(1, keepdims=True)+.004)
    information = np.sum(ppm*np.log2(ppm/.25), axis=1)
    passing = information > .2
    longest = current = 0
    for value in passing:
        current = current+1 if value else 0
        longest = max(longest, current)
    return dict(longest_run_above_point2_bits=int(longest), total_information_bits=float(information.sum()),
                sequence_qc_pass=longest >= 5)


def load_patterns(path, track_set, sign):
    import h5py
    from modiscolite.core import Seqlet, SeqletSet
    patterns = []
    with h5py.File(path, "r") as handle:
        for name, group in handle.get(sign, {}).items():
            s = group["seqlets"]
            coords = [Seqlet(int(i), int(start), int(end), bool(rc)) for i, start, end, rc in
                zip(s["example_idx"][:], s["start"][:], s["end"][:], s["is_revcomp"][:])]
            pattern = SeqletSet(track_set.create_seqlets(coords))
            np.testing.assert_allclose(pattern.sequence, group["sequence"][:], atol=1e-7)
            np.testing.assert_allclose(pattern.contrib_scores, group["contrib_scores"][:], atol=1e-7)
            patterns.append((name, pattern))
    return patterns


def refine(output, sequences, hyp, lengths, parameters, *, raw_path=None):
    from modiscolite import io
    track = make_track_set(sequences, hyp, lengths)
    collections, rows, mappings = {}, [], []
    with bounded_native_aggregator() as (native, _):
        for sign, numeric in (("pos_patterns", 1), ("neg_patterns", -1)):
            original = load_patterns(raw_path or output/"motifs.h5", track, sign)
            candidates, lookup = [], {}
            for name, pattern in original:
                core, bounds = trimmed_pattern(pattern)
                record = dict(source=sign+"/"+name, raw_width=len(pattern), trim=bounds)
                if core is None:
                    rows.append(dict(record, status="zero_contribution")); continue
                qc = sequence_qc(core)
                signed_sum = float(core.contrib_scores.sum())
                magnitude = float(np.abs(core.contrib_scores).sum())
                concordance = numeric*signed_sum/magnitude if magnitude else 0.
                record.update(qc, signed_contribution=signed_sum, sign_purity=concordance)
                if concordance <= 0:
                    rows.append(dict(record, status="discordant_final_sign")); continue
                if not qc["sequence_qc_pass"]:
                    rows.append(dict(record, status="low_sequence_information")); continue
                rows.append(dict(record, status="candidate"))
                # Native cross-contamination compares flattened, aligned seqlets
                # of equal width. Trimmed cores are for QC and final display,
                # not inputs to that fixed-width comparison.
                signature = tuple(sorted(s.string for s in pattern.seqlets))
                lookup.setdefault(signature, []).append(record["source"])
                candidates.append(pattern)
            if len(candidates) > 1:
                if len({len(p) for p in candidates}) != 1:
                    raise ValueError("Native collapse requires equal-width raw patterns; do not pre-trim")
                merged, hierarchy = native.SimilarPatternsCollapser(candidates, track,
                    min_overlap=.7,
                    prob_and_pertrack_sim_merge_thresholds=[(.8, .8), (.5, .85), (.2, .9)],
                    prob_and_pertrack_sim_dealbreaker_thresholds=[(.4, .75), (.2, .8), (.1, .85), (0., .9)],
                    min_frac=.2, min_num=30, flank_to_add=0,
                    window_size=parameters["trim_to_window_size"],
                    bg_freq=np.mean(np.concatenate([s.sequence for p in candidates for s in p.seqlets]), axis=0),
                    max_seqlets_subsample=1000)

                def members(node):
                    if node.child_nodes:
                        return sorted(set(x for child in node.child_nodes for x in members(child)))
                    return lookup[tuple(sorted(s.string for s in node.pattern.seqlets))]

                member_map = {id(node.pattern): members(node) for node in hierarchy.root_nodes}
                sources = [member_map[id(pattern)] for pattern in merged]
            else:
                merged = candidates
                sources = [lookup[tuple(sorted(s.string for s in p.seqlets))] for p in merged]
            retained = []
            for pattern, source in zip(merged, sources):
                core, bounds = trimmed_pattern(pattern)
                if core is None:
                    mappings.append(dict(sources=source, output=None, status="merged_zero_contribution")); continue
                qc = sequence_qc(core)
                if np.sign(core.contrib_scores.sum()) != numeric or not qc["sequence_qc_pass"]:
                    mappings.append(dict(sources=source, output=None, status="merged_sign_or_sequence_qc_failed", **qc)); continue
                target = sign+"/pattern_"+str(len(retained))
                mappings.append(dict(sources=source, output=target, status="retained", final_trim=bounds, **qc))
                retained.append(core)
            collections[sign] = retained
    io.save_hdf5(output/"nonredundant.h5", collections["pos_patterns"], collections["neg_patterns"], sequences.shape[1])
    write_json(output/"refinement.json", dict(method="Native equal-width SimilarPatternsCollapser, then CWM-magnitude 30% trim, separately per group/sign",
        trim_policy="Trimmed cores determine candidate QC; untrimmed native patterns and seqlets enter collapse; outputs are trimmed again",
        broad_candidate_policy="All within-sign patterns are candidates; native CWM/seqlet criteria decide merges",
        qc_policy="Retained prior user criterion: >=5 consecutive sequence-information columns >0.2 bits",
        source_patterns=rows, mapping=mappings,
        remaining_variant_review="Not yet visually curated; do not equate nonredundant patterns with unique TFs"))
    return {key: len(value) for key, value in collections.items()}


def smoke(root):
    require_allocation("cpu")
    parameters = json.loads((root/"config.json").read_text())["discovery_parameters"]
    rng = np.random.default_rng(471)
    lengths = rng.integers(100, 161, size=512)
    codes = rng.integers(0, 4, (512, 160))
    hyp = rng.normal(0, .01, (512, 160, 4)).astype(np.float32)
    motif = np.array([0, 1, 2, 0, 3, 1, 1, 2, 3, 0, 2, 2])
    for i, length in enumerate(lengths):
        start = int(rng.integers(25, length-40))
        codes[i, start:start+len(motif)] = motif
        hyp[i, np.arange(start, start+len(motif)), motif] += 1 if i < 256 else -1
        hyp[i, length:] = 1e6  # Must never affect threshold estimation or seqlets.
    sequences = np.eye(4, dtype=np.float32)[codes]
    output = root/"synthetic_smoke"; output.mkdir(exist_ok=True)
    audit = discover_original(sequences, hyp, lengths, parameters, output/"motifs.h5", 1000)
    if audit["positive"]["patterns"] < 1 or audit["negative"]["patterns"] < 1:
        raise RuntimeError("Failed to recover both signs of the planted synthetic motif")
    counts = refine(output, sequences, hyp, lengths, parameters)
    if min(counts.values()) < 1:
        raise RuntimeError("Native refinement lost a planted sign")
    write_json(root/"synthetic_passed.json", dict(status="passed", audit=audit, refined_patterns=counts,
        job_id=os.environ["SLURM_JOB_ID"], config_sha256=digest(root/"config.json")))
    event("original_modisco_synthetic_passed", patterns=counts)


def assemble(project, root):
    require_allocation("cpu")
    _, _, data = contract(project, root)
    decision = json.loads((root/"reference_choice.json").read_text())
    shards = decision["shards"]
    chunks, seen = {}, np.zeros(len(data["ids"]), bool)
    failed = 0
    for shard in range(shards):
        report = json.loads((root/f"attribution_shard_{shard}.json").read_text())
        if (report["shard_index"] != shard or report["shards"] != shards
                or report["signature"] != digest(root/"reference_choice.json")
                or report["intervals_sha256"] != digest(root/"intervals.npz")):
            raise ValueError("Incompatible attribution shard")
        failed += report["quality_failures"]
        for name, expected in report["chunks"].items():
            if name in chunks or digest(root/"chunks"/name) != expected:
                raise ValueError("Repeated or changed attribution chunk")
            with np.load(root/"chunks"/name, allow_pickle=False) as f:
                indices = f["indices"]
                if seen[indices].any():
                    raise ValueError("Repeated attributed enhancer")
                seen[indices] = True
            chunks[name] = expected
    if not seen.all():
        raise ValueError("Incomplete sharded attribution cohort")
    write_json(root/"attribution_complete.json", dict(status="complete", elements=len(seen),
        references=decision["references"], quality_failures=failed, shards=shards,
        signature=digest(root/"reference_choice.json"), intervals_sha256=digest(root/"intervals.npz"), chunks=chunks))
    event("original_attribution_assembled", elements=len(seen), quality_failures=failed, shards=shards)


def run(project, root, group_index):
    require_allocation("cpu")
    config, _, data = contract(project, root)
    ready = json.loads((root/"synthetic_passed.json").read_text())
    if ready["status"] != "passed" or ready["config_sha256"] != digest(root/"config.json"):
        raise ValueError("Successful boundary-aware synthetic smoke required")
    intervals = load_intervals(root, data)
    hyp, quality = collect(root, data, intervals)
    name, low, high = GROUPS[group_index]
    parameters = dict(config["discovery_parameters"])
    parameters["seed"] += group_index
    breadth = data["labels"].sum(1)
    eligible = (data["split"] == "train") & (breadth >= low) & (breadth <= high)
    indices = np.random.default_rng(parameters["seed"]).permutation(np.flatnonzero(eligible & quality))
    output = root/"groups"/name; output.mkdir(parents=True, exist_ok=True)
    if (output/"complete.json").exists():
        raise FileExistsError("Group already complete")
    lengths = intervals["length"][indices]
    selected_hyp = hyp[indices, :int(lengths.max())].copy()
    sequences = np.zeros_like(selected_hyp)
    for row, i in enumerate(indices):
        start, length = int(intervals["offset"][i]), int(intervals["length"][i])
        sequences[row, :length] = np.eye(4)[data["sequence"][i, start:start+length]]
    np.savez_compressed(output/"examples.npz", indices=indices, ids=data["ids"][indices], lengths=lengths,
        chrom=data["chrom"][indices], start=intervals["start"][indices], end=intervals["end"][indices])
    np.savez_compressed(output/"discovery_inputs.npz", sequence=sequences, hypothetical=selected_hyp, lengths=lengths)
    selection = dict(group=name, elements=len(indices), eligible_before_quality=int(eligible.sum()),
        quality_excluded=int(np.sum(eligible & ~quality)), train_only=True,
        enhancer_downsampling=False, interval="original_catalog_bounds", seed=parameters["seed"],
        parameters=parameters, intervals_sha256=digest(root/"intervals.npz"),
        attribution_sha256=digest(root/"attribution_complete.json"))
    write_json(output/"selection.json", selection)
    event("original_modisco_group_start", **selection)
    audit = discover_original(sequences, selected_hyp, lengths, parameters,
        output/"motifs.h5", parameters["max_seqlets_per_metacluster"])
    counts = refine(output, sequences, selected_hyp, lengths, parameters)
    write_json(output/"complete.json", dict(status="complete", group=name, audit=audit,
        refined_patterns=counts, job_id=os.environ["SLURM_JOB_ID"],
        config_sha256=digest(root/"config.json"),
        files={p.name: digest(p) for p in output.iterdir() if p.is_file()}))
    event("original_modisco_group_complete", group=name, refined_patterns=counts)


def summarize(root):
    require_allocation("cpu")
    groups = []
    for name, _, _ in GROUPS:
        output = root/"groups"/name
        complete = json.loads((output/"complete.json").read_text())
        for key, expected in complete["files"].items():
            if digest(output/key) != expected:
                raise ValueError("Changed completed group output")
        groups.append(complete)
    write_json(root/"discovery_complete.json", dict(status="complete", groups=groups,
        sequence_scan="pending", predictive_scan="pending", annotation="pending", figures="pending"))
    event("original_discovery_complete", groups=len(groups))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "assemble", "discover", "summarize"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--group-index", type=int)
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke(args.root)
    elif args.stage == "assemble":
        assemble(args.project, args.root)
    elif args.stage == "summarize":
        summarize(args.root)
    else:
        if args.group_index is None or not 0 <= args.group_index < len(GROUPS):
            parser.error("--group-index 0..7 required")
        run(args.project, args.root, args.group_index)
