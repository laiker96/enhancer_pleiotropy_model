"""Original-enhancer inputs and length-aware TF-MoDISco 2.5.2 adapter.

The classifier still sees its genomic 2048 bp input. No zero padding enters
the null distribution or a seqlet. Native clustering/collapsing is unchanged
except for per-example bounds and a short-pattern trim-window guard.
"""
import argparse
import contextlib
import csv
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import types

import numpy as np

from .common import digest, event, require_allocation, write_json


GROUPS = [("exact_1", 1, 1)] + [("ge_"+str(k), k, 8) for k in range(2, 9)]


def contract(project, root):
    config = json.loads((root/"config.json").read_text())
    parent = project/config["parent"]
    if digest(parent/"MANIFEST.sha256") != config["parent_manifest_sha256"]:
        raise ValueError("Parent package changed")
    provenance = json.loads((parent/"input_provenance.json").read_text())
    for name in ("cohort.npz", "best_model.pt", "attribution_config.json"):
        if digest(parent/name) != provenance["inputs"][name]:
            raise ValueError("Parent input changed: "+name)
    with np.load(parent/"cohort.npz", allow_pickle=False) as f:
        data = dict(f)
    return config, parent, data


def join_intervals(data, rows):
    lookup = {}
    for row in rows:
        key = row["master_dhs_id"]
        if key in lookup:
            raise ValueError("Duplicate catalog ID: "+key)
        lookup[key] = row
    if len(np.unique(data["ids"])) != len(data["ids"]):
        raise ValueError("Duplicate cohort IDs")
    selected = [lookup[str(key)] for key in data["ids"]]
    start = np.asarray([int(r["start"]) for r in selected], dtype=np.int64)
    end = np.asarray([int(r["end"]) for r in selected], dtype=np.int64)
    summit = np.asarray([int(r["summit"]) for r in selected], dtype=np.int64)
    width = np.asarray([int(r["width_bp"]) for r in selected], dtype=np.int64)
    if not (np.array_equal(summit, data["summit"])
            and np.array_equal([r["chrom"] for r in selected], data["chrom"])
            and np.array_equal(end-start, width)
            and (width > 0).all() and (start <= summit).all() and (summit < end).all()):
        raise ValueError("Catalog/cohort coordinate disagreement")
    offset = start-(summit-1024)
    if (offset < 0).any() or (offset+width > 2048).any():
        raise ValueError("An enhancer extends outside the classifier input")
    return dict(ids=data["ids"], start=start, end=end, offset=offset, length=width)


def prepare(project, root):
    require_allocation("cpu")
    config, parent, data = contract(project, root)
    catalog = root/"original_catalog.tsv.gz"
    if digest(catalog) != config["catalog_sha256"]:
        raise ValueError("Catalog checksum changed")
    with gzip.open(catalog, "rt") as handle:
        intervals = join_intervals(data, csv.DictReader(handle, delimiter="\t"))
    np.savez_compressed(root/"intervals.npz", **intervals)
    lengths, offsets = intervals["length"], intervals["offset"]
    breadth = data["labels"].sum(1)
    outside = (offsets < 768) | (offsets+lengths > 1280)
    groups = [dict(name=name, split=split, n=int(np.sum(
        (breadth >= low) & (breadth <= high) & (data["split"] == split))))
        for name, low, high in GROUPS for split in ("train", "validation", "test")]
    report = dict(elements=len(lengths), minimum_length=int(lengths.min()),
        maximum_length=int(lengths.max()), median_length=float(np.median(lengths)),
        outside_old_central512=int(outside.sum()), groups=groups,
        intervals_sha256=digest(root/"intervals.npz"),
        catalog_sha256=digest(catalog), parent_manifest_sha256=digest(parent/"MANIFEST.sha256"),
        config_sha256=digest(root/"config.json"), job_id=os.environ["SLURM_JOB_ID"])
    write_json(root/"prepared.json", report)
    event("original_intervals_prepared", **report)


def load_intervals(root, data):
    prepared = json.loads((root/"prepared.json").read_text())
    if digest(root/"intervals.npz") != prepared["intervals_sha256"]:
        raise ValueError("Prepared interval checksum changed")
    with np.load(root/"intervals.npz", allow_pickle=False) as f:
        result = dict(f)
    np.testing.assert_array_equal(result["ids"], data["ids"])
    return result


def smooth_valid(scores, lengths, window):
    """Ragged rolling sums: no padding, no windows crossing an enhancer end."""
    if len(scores) != len(lengths) or window < 1:
        raise ValueError("Invalid rolling-window input")
    tracks = []
    for row, length in zip(scores, lengths):
        if not window <= length <= len(row) or not np.isfinite(row[:length]).all():
            raise ValueError("Invalid/short enhancer scores")
        cumulative = np.r_[0., np.cumsum(row[:length], dtype=np.float64)]
        tracks.append(cumulative[window:]-cumulative[:-window])
    return tracks


def extract_original_seqlets(scores, lengths, config):
    """Native threshold functions applied exclusively to valid enhancer windows."""
    from modiscolite import core, extract_seqlets as native
    window, flank = config["sliding_window_size"], config["flank_size"]
    tracks = smooth_valid(scores, lengths, window)
    values = np.concatenate(tracks)
    sample = values
    if len(sample) > 1000000:
        sample = np.random.RandomState(1234).choice(sample, 1000000, replace=False)
    positive = np.sort(sample[sample >= 0])
    negative = np.sort(sample[sample < 0])[::-1]
    if not len(positive) or not len(negative) or np.ptp(values) == 0:
        raise ValueError("Two-sided non-degenerate attribution required for native null fitting")
    pnull, nnull = native._laplacian_null(tracks, window, 10000)
    pos = native._isotonic_thresholds(positive, pnull, True, config["target_seqlet_fdr"])
    neg = native._isotonic_thresholds(negative, nnull, False, config["target_seqlet_fdr"])
    pos, neg = native._refine_thresholds(np.r_[positive, negative], pos, neg, .03, .2)
    distribution = np.sort(np.abs(values))
    ptrans, ntrans = np.searchsorted(distribution, [abs(pos), abs(neg)])/len(distribution)
    weak = min(min(ptrans, ntrans)-.0001, .8)
    # Clamp only the numerical lower limit, never use Python's negative indexing.
    threshold = distribution[max(0, int(weak*len(distribution)))]
    seqlets = []
    suppress = int(.5*window)+flank
    for index, track in enumerate(tracks):
        selected = (track >= pos) | (track <= neg)
        candidate = np.where(selected, np.abs(track), -np.inf)
        starts = np.arange(len(candidate))
        valid = (starts >= flank) & (starts+window+flank <= lengths[index])
        candidate[~valid] = -np.inf
        while np.isfinite(candidate).any():
            maximum = int(np.argmax(candidate))
            seqlets.append(core.Seqlet(index, maximum-flank, maximum+window+flank, False))
            lo = max(int(np.floor(maximum+.5-suppress)), 0)
            hi = min(int(np.ceil(maximum+.5+suppress)), len(candidate))
            candidate[lo:hi] = -np.inf
    return seqlets, float(threshold), dict(valid_windows=len(values),
        positive_threshold=float(pos), negative_threshold=float(neg),
        weak_sign_threshold=float(threshold), extracted_seqlets=len(seqlets))


@contextlib.contextmanager
def bounded_native_aggregator():
    """Patch a private module copy, not installed files or native algorithms.

    Expected source snippets/version are checked before changing boundary guards.
    A TrackSet assertion independently catches any missed out-of-bounds request.
    """
    from modiscolite import aggregator, tfmodisco
    if importlib.metadata.version("modisco") != "2.5.2":
        raise ValueError("The boundary adapter is pinned to TF-MoDISco 2.5.2")
    source = inspect.getsource(aggregator)
    replacements = {
        "start >= 0 and end <= track_set.length": (
            "start >= 0 and end <= track_set.lengths[seqlet.example_idx]", 2),
        "seqlet.start >= 0 and seqlet.end < track_set.length": (
            "seqlet.start >= 0 and seqlet.end <= track_set.lengths[seqlet.example_idx]", 1),
    }
    patched = source
    for old, (new, count) in replacements.items():
        if patched.count(old) != count:
            raise ValueError("Unexpected native boundary-check implementation")
        patched = patched.replace(old, new)
    # Contribution-trimmed cores can be shorter than the native 30-bp polish
    # window. Preserve the whole shorter core instead of an invalid window.
    marker = "\t# Trim by IC\n\tppm = pattern.sequence"
    if patched.count(marker) != 1:
        raise ValueError("Unexpected native polishing implementation")
    patched = patched.replace(marker, "\twindow_size = min(window_size, len(pattern))\n"+marker)
    module = types.ModuleType("modiscolite.original_interval_aggregator")
    module.__package__ = "modiscolite"
    exec(compile(patched, "<modisco-2.5.2-original-interval-boundaries>", "exec"), module.__dict__)
    previous = tfmodisco.aggregator
    tfmodisco.aggregator = module
    try:
        yield module, hashlib.sha256(source.encode()).hexdigest()
    finally:
        tfmodisco.aggregator = previous


def make_track_set(sequences, hypothetical, lengths):
    from modiscolite.core import TrackSet

    class BoundedTrackSet(TrackSet):
        def create_seqlets(self, seqlets):
            for seqlet in seqlets:
                if not (0 <= seqlet.example_idx < len(self.lengths)
                        and 0 <= seqlet.start < seqlet.end <= self.lengths[seqlet.example_idx]):
                    raise ValueError("Native operation attempted to cross an original enhancer boundary")
            return super().create_seqlets(seqlets)

    if sequences.shape != hypothetical.shape or sequences.ndim != 3 or sequences.shape[2] != 4:
        raise ValueError("Expected matching [enhancers,max_length,ACGT] arrays")
    for s, h, length in zip(sequences, hypothetical, lengths):
        if (not (s[:length].sum(1) == 1).all() or not np.isin(s[:length], [0, 1]).all()
                or not np.isfinite(h[:length]).all()):
            raise ValueError("Invalid valid-region sequence/attribution")
    track_set = BoundedTrackSet(sequences, sequences*hypothetical, hypothetical)
    track_set.lengths = np.asarray(lengths)
    return track_set


def discover_original(sequences, hypothetical, lengths, config, output, cap):
    from modiscolite import io, tfmodisco
    np.random.seed(config["seed"])
    track_set = make_track_set(sequences, hypothetical, lengths)
    coords, threshold, audit = extract_original_seqlets(
        track_set.contrib_scores.sum(2), lengths, config)
    seqlets = track_set.create_seqlets(coords)
    by_sign = {1: [], -1: []}
    flank, window = config["flank_size"], config["sliding_window_size"]
    for seqlet in seqlets:
        score = seqlet.contrib_scores[flank:flank+window].sum()
        if abs(score) > threshold:
            by_sign[1 if score > 0 else -1].append(seqlet)
    output.parent.mkdir(parents=True, exist_ok=True)
    patterns = {}
    with bounded_native_aggregator() as (_, native_sha):
        for sign in (1, -1):
            available = by_sign[sign]
            chosen = available[:cap]
            label = "positive" if sign == 1 else "negative"
            audit[label] = dict(available_seqlets=len(available), used_seqlets=len(chosen),
                               cap=cap, status="below_min_metacluster_size")
            if len(available) > 100:
                event("original_modisco_sign", sign=label, **audit[label])
                keys = ("n_leiden_runs", "trim_to_window_size", "initial_flank_to_add", "final_flank_to_add")
                patterns[sign] = tfmodisco.seqlets_to_patterns(chosen, track_set, track_signs=sign,
                    **{key: config[key] for key in keys}) or []
                audit[label]["status"] = "fitted" if patterns[sign] else "fitted_no_patterns"
            else:
                patterns[sign] = []
            for pattern in patterns[sign]:
                for seqlet in pattern.seqlets:
                    if not 0 <= seqlet.start < seqlet.end <= lengths[seqlet.example_idx]:
                        raise ValueError("Final seqlet violates original bounds")
            audit[label]["patterns"] = len(patterns[sign])
    temporary = output.with_suffix(".partial.h5")
    io.save_hdf5(temporary, patterns[1], patterns[-1], window_size=sequences.shape[1])
    temporary.replace(output)
    audit["native_aggregator_sha256"] = native_sha
    audit["padding_in_threshold_distribution"] = False
    audit["model_input_bp"] = 2048
    audit["discovery_interval"] = "original_enhancer"
    write_json(output.with_suffix(".audit.json"), audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.project, args.root)
