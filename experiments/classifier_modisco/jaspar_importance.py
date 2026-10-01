"""Fixed-reference site importance from existing 50-reference IG; CPU allocation only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .common import digest, event, require_allocation, write_json
from .tomtom_atlas import read_meme

GROUPS = ("exact_1",) + tuple(f"ge_{k}" for k in range(2, 9))
EXACT_GROUPS = tuple(f"exact_{k}" for k in range(1, 9))
SITE_P = 1e-4


def analysis_settings(config):
    key = config.get("database_key", "jaspar")
    counts = {"jaspar": 296, "flyfactorsurvey": 656}
    grouping = config.get("grouping", "cumulative")
    if key not in counts or grouping not in ("cumulative", "exact"):
        raise ValueError("Unsupported reference database or degree grouping")
    groups = EXACT_GROUPS if grouping == "exact" else GROUPS
    labels = [str(k) for k in range(1, 9)] if grouping == "exact" else ["1"] + [f"[{k}-8]" for k in range(2, 9)]
    return key, counts[key], grouping, groups, labels


def union_importance(actual, intervals):
    """Signed IG/bp on the union of hit bases: overlaps/strands count only once."""
    values = np.asarray(actual)
    mask = np.zeros(len(values), bool)
    for start, end in intervals:
        if not 0 <= start < end <= len(values):
            raise ValueError("Motif hit outside original enhancer")
        mask[start:end] = True
    if not mask.any():
        return None, 0
    if not np.isfinite(values[mask]).all():
        raise ValueError("Nonfinite attribution")
    return float(values[mask].mean(dtype=np.float64)), int(mask.sum())


def summarize(scores, metadata, profiles, groups=GROUPS):
    if scores.shape != (len(profiles), len(metadata["ids"])):
        raise ValueError("Motif/element score alignment mismatch")
    rows = []
    breadth, split = metadata["breadth"], metadata["split"]
    for i, profile in enumerate(profiles):
        means, carriers, denominators, deviations = [], [], [], []
        for group in groups:
            kind, degree = group.split("_")
            if kind not in ("exact", "ge") or not 1 <= int(degree) <= 8:
                raise ValueError("Unknown degree group: " + group)
            mask = (split == "train") & ((breadth == int(degree)) if kind == "exact" else (breadth >= int(degree)))
            selected = scores[i, mask]
            selected = selected[np.isfinite(selected)]
            means.append(float(selected.mean()) if len(selected) else None)
            deviations.append(float(selected.std(ddof=1)) if len(selected) > 1 else None)
            carriers.append(len(selected)); denominators.append(int(mask.sum()))
        rows.append(dict(id=profile["id"], name=profile["name"], means=means,
                         motif_containing_enhancers=carriers, group_enhancers=denominators,
                         standard_deviation=deviations))
    return rows


def read_actual(source, metadata):
    complete = json.loads((source / "attribution_complete.json").read_text())
    if complete["status"] != "complete" or complete["references"] != 50:
        raise ValueError("Completed 50-reference attribution required")
    if digest(source / "intervals.npz") != complete["intervals_sha256"]:
        raise ValueError("Changed original intervals")
    with np.load(source / "intervals.npz", allow_pickle=False) as saved:
        intervals = dict(saved)
    n, width = len(metadata["ids"]), int(metadata["length"].max())
    np.testing.assert_array_equal(intervals["length"], metadata["length"])
    actual = np.full((n, width), np.nan, dtype=np.float32)
    seen = np.zeros(n, bool)
    for number, (name, expected) in enumerate(complete["chunks"].items(), 1):
        path = source / "chunks" / name
        if digest(path) != expected:
            raise ValueError("Changed attribution chunk: " + name)
        with np.load(path, allow_pickle=False) as saved:
            indices = saved["indices"]
            if seen[indices].any() or str(saved["signature"]) != complete["signature"]:
                raise ValueError("Duplicate or incompatible attribution chunk")
            if not saved["quality_pass"].all():
                raise ValueError("A failed attribution needs an explicit exclusion policy")
            values = saved["actual"]
            for row, index in enumerate(indices):
                offset, length = int(intervals["offset"][index]), int(intervals["length"][index])
                actual[index, :length] = values[row, offset:offset + length]
            seen[indices] = True
        if number % 100 == 0:
            event("attribution_loaded", chunks=number, total=len(complete["chunks"]))
    if not seen.all():
        raise ValueError("Missing attribution rows")
    return actual


def training_fasta(source, destination, metadata):
    seen = set()
    with source.open() as handle, destination.open("x") as out:
        keep = False
        for line in handle:
            if line.startswith(">"):
                ident = line[1:].strip()
                if not ident.startswith("e") or not ident[1:].isdigit():
                    raise ValueError("Unexpected FASTA sequence ID")
                index = int(ident[1:])
                if index in seen or index >= len(metadata["ids"]):
                    raise ValueError("Invalid or duplicate FASTA row")
                seen.add(index)
                keep = metadata["split"][index] == "train"
            if keep:
                out.write(line)
    if len(seen) != len(metadata["ids"]):
        raise ValueError("Incomplete enhancer FASTA")


def scan_shard(number, ids, binary, database, background, fasta, metadata, actual, root, database_key="jaspar"):
    command = [str(binary), "--text", "--thresh", str(SITE_P), "--motif-pseudo", "0.1",
               "--bgfile", str(background)]
    for ident in ids:
        command += ["--motif", ident]
    command += [str(database), str(fasta)]
    sites = {ident: defaultdict(list) for ident in ids}
    counts = dict.fromkeys(ids, 0)
    event(database_key + "_scan_started", shard=number, profiles=len(ids))
    with (root / f"scan_{number}.stderr.log").open("x") as errors, gzip.open(root / f"hits_{number}.tsv.gz", "wt") as archive:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            header = next(process.stdout)
            if not header.startswith("motif_id\tmotif_alt_id\tsequence_name"):
                raise ValueError("Unexpected FIMO header")
            archive.write(header)
            for line in process.stdout:
                archive.write(line)
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip().split("\t")
                ident, seqid = fields[0], fields[2]
                if ident not in sites or not seqid.startswith("e") or not seqid[1:].isdigit():
                    raise ValueError("Unknown FIMO motif or sequence")
                index = int(seqid[1:])
                if metadata["split"][index] != "train" or float(fields[7]) > SITE_P:
                    raise ValueError("Unexpected FIMO split or p value")
                sites[ident][index].append((int(fields[3]) - 1, int(fields[4])))
                counts[ident] += 1
            if process.wait():
                raise RuntimeError("FIMO failed: shard " + str(number))
        finally:
            if process.poll() is None:
                process.terminate(); process.wait()
    result = np.full((len(ids), len(metadata["ids"])), np.nan, dtype=np.float32)
    covered = np.zeros(result.shape, dtype=np.uint16)
    for j, ident in enumerate(ids):
        for index, intervals in sites[ident].items():
            length = int(metadata["length"][index])
            result[j, index], covered[j, index] = union_importance(actual[index, :length], intervals)
    event(database_key + "_scan_complete", shard=number, sites=sum(counts.values()), profiles_with_hits=sum(bool(v) for v in counts.values()))
    return ids, result, covered, counts, command


def main(project, root):
    require_allocation("cpu")
    config = json.loads((root / "config.json").read_text())
    database_key, expected_count, grouping, groups, labels = analysis_settings(config)
    for name, expected in config["inputs"].items():
        if digest(project / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    source = project / config["attribution"]
    sequence = project / config["sequence_report"]
    database, binary = project / config["database"], project / config["fimo"]
    profiles = list(read_meme(database).values())
    if len(profiles) != expected_count or subprocess.check_output([str(binary), "--version"], text=True).strip() != "5.5.9":
        raise ValueError(f"Expected all {expected_count} {database_key} profiles and FIMO 5.5.9")
    with np.load(sequence / "cohort_metadata.npz", allow_pickle=False) as saved:
        metadata = dict(saved)
    if len(metadata["ids"]) != 40338 or len(np.unique(metadata["ids"])) != 40338:
        raise ValueError("Unexpected cohort")
    event(database_key + "_importance_started", profiles=len(profiles), grouping=grouping,
          training_enhancers=int((metadata["split"] == "train").sum()))
    actual = read_actual(source, metadata)
    training_fasta(sequence / "enhancers.fa", root / "training_enhancers.fa", metadata)
    workers = min(8, int(os.environ["SLURM_CPUS_PER_TASK"]))
    ids = [p["id"] for p in profiles]
    scores = np.full((len(ids), len(metadata["ids"])), np.nan, dtype=np.float32)
    covered = np.zeros(scores.shape, dtype=np.uint16)
    commands, hit_counts = {}, {}
    lookup = {ident: i for i, ident in enumerate(ids)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(scan_shard, i, ids[i::workers], binary, database,
            sequence / "training_background.txt", root / "training_enhancers.fa", metadata, actual, root, database_key)
            for i in range(workers)]
        for future in as_completed(futures):
            shard_ids, values, bases, counts, command = future.result()
            index = [lookup[ident] for ident in shard_ids]
            scores[index], covered[index] = values, bases
            hit_counts.update(counts); commands[shard_ids[0]] = command
    trajectories = summarize(scores, metadata, profiles, groups=groups)
    summary = dict(status="complete", profiles=len(profiles), groups=groups,
        group_labels=labels, grouping=grouping, database_key=database_key, split="train",
        metric="Mean signed integrated-gradient attribution per matched base; mean over motif-containing enhancers",
        overlap_policy="Union of matched bases within enhancer/motif; reverse-strand and overlapping hits count once",
        missing_policy="No hit means missing, not zero; only finite motif-carrier scores enter group means",
        site_p=SITE_P, references=50, interval="original enhancer", trajectories=trajectories,
        hit_counts=hit_counts, commands=commands, input_hashes=config["inputs"],
        caveats=["Exact degree groups are disjoint" if grouping == "exact" else "Nested groups overlap",
                 "Attribution target averages observed-active-context logits",
                 "Conditional mean importance is not motif prevalence or TF occupancy",
                 "All profiles included; no selection by match q value or importance",
                 "Different motifs can cover the same bases; their importance is not additive"])
    write_json(root / "summary.json", summary)
    np.savez_compressed(root / "importance.npz", motif_ids=np.array(ids), scores=scores,
                        covered_bases=covered, **metadata)
    outputs = [root / "summary.json", root / "importance.npz"]
    write_json(root / "complete.json", dict(status="complete", job=os.environ["SLURM_JOB_ID"],
        files={p.name: digest(p) for p in outputs}, profiles=len(profiles), inference_rerun=False, discovery_rerun=False))
    event(database_key + "_importance_complete", profiles=len(profiles), profiles_with_hits=sum(v > 0 for v in hit_counts.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); main(args.project.resolve(), args.root.resolve())
