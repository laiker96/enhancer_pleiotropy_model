"""Rebuild the published-data boundary: normalized v4 tracks and its catalog."""
import csv
import gzip
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import yaml

from classifier_transfer.data import CONTEXTS, SPLITS, digest, load_split, prepare, write_json
from classifier_transfer.background_matching import gc_features, match_background, sequence_key


def stage_inputs(cfg):
    from stage_v4_inputs import (ARCHIVE_ROOT, stage_archive, parse_manifest, copy_verified_member)
    archive, root = (Path(cfg["paths"][k]) for k in ("archive", "inputs"))
    # The original staging verifies the archive hash and every extracted member.
    stage_archive(archive, root)
    with tarfile.open(archive, "r:") as tar:
        manifest = parse_manifest(tar.extractfile(ARCHIVE_ROOT + "/FILE_MANIFEST.sha256").read().decode())
        name = "catalog/noncontributing_dhs_p60/active_enhancers_wide.tsv.gz"
        copy_verified_member(tar, name, root / name, manifest[name])
    write_json(root / "catalog_provenance.json", {name: manifest[name]})


def regression_config(cfg, output):
    root = Path(cfg["paths"]["inputs"])
    value = yaml.safe_load(Path(cfg["paths"]["regression_config"]).read_text())
    value["output_directory"] = str(output)
    value["inputs"] = {key: str(root / name) for key, name in dict(
        reference_fasta="reference/dm6.fa", blacklist_bed="reference/dm6.blacklist.bed",
        master_dhs_bed="regulatory/master_dhs/master_dhs.bed",
        master_dhs_summits_bed="regulatory/master_dhs/master_dhs_summits.bed",
        h3k27ac_peak_directory="model_h3k27ac_peaks",
        bigwig_directory="regulatory/normalized_mean_bigwig").items()}
    return value


def prepare_regression(cfg):
    import prepare_v4_dataset as original
    work = Path(cfg["paths"]["work"])
    work.mkdir(parents=True, exist_ok=True)
    path = work / "preprocessing.json"
    if path.exists():
        raise FileExistsError("Inspect existing preprocessing before retrying")
    write_json(path, regression_config(cfg, work / "regression_data"))
    # CLI already enforces a real generic allocation, or explicit local CPU use.
    original.require_compute_node = lambda: None
    sys.argv = ["prepare_v4_dataset", "--config", str(path)]
    original.main()


def prepare_test_background(enhancer_root, windows, output):
    """Historical seed-20260915 test matching, without cached-run prerequisites."""
    enhancer_root, windows, output = map(Path, (enhancer_root, windows, output))
    if output.exists():
        raise FileExistsError(output)
    metadata_path = windows.with_name("windows.metadata.json")
    metadata = json.loads(metadata_path.read_text())
    if digest(windows) != metadata["output"]["sha256"]:
        raise ValueError("Regression windows changed")
    test = load_split(enhancer_root, "test")
    if not np.all(test["chrom"] == "chr3R"):
        raise ValueError("Historical matched test requires chr3R")
    forbidden = {sequence_key(row) for split in SPLITS
                 for row in load_split(enhancer_root, split)["sequence"]}
    lookup = np.full(256, 255, np.uint8)
    lookup[np.frombuffer(b"ACGT", np.uint8)] = np.arange(4)
    codes, ids, starts, ends, features = [], [], [], [], []
    seen_ids, seen_coordinates, seen_sequences = set(), set(), set()
    candidate_n = duplicate_n = 0
    with gzip.open(windows, "rt") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["split"] != "test" or row["source"] != "genomic_background":
                continue
            candidate_n += 1
            start, end = int(row["start"]), int(row["end"])
            if (row["chrom"] != "chr3R" or end-start != 2048 or start < 0
                    or int(row["target_start"]) != start+768 or int(row["target_end"]) != start+1280):
                raise ValueError("Invalid test background geometry")
            sequence = lookup[np.frombuffer(row["sequence"].encode("ascii"), np.uint8)]
            if len(sequence) != 2048 or (sequence > 3).any():
                raise ValueError("Invalid background DNA")
            if row["record_id"] in seen_ids or start in seen_coordinates:
                raise ValueError("Duplicate background ID/coordinate")
            seen_ids.add(row["record_id"]); seen_coordinates.add(start)
            key = sequence_key(sequence)
            if key in forbidden or key in seen_sequences:
                duplicate_n += 1
                continue
            seen_sequences.add(key)
            codes.append(sequence); ids.append(row["record_id"])
            starts.append(start); ends.append(end); features.append(gc_features(sequence[None])[0])
    if candidate_n != metadata["candidate_and_sampling_counts"]["test"]["selected_background"]:
        raise ValueError("Wrong background count")
    features = np.asarray(features)
    positive = gc_features(test["sequence"])
    matched = match_background(positive, features, seed=20260915)
    selected = np.flatnonzero(matched >= 0)
    paired = matched[selected]
    if not len(selected) or len(np.unique(paired)) != len(paired):
        raise ValueError("No valid unique test matches")
    if np.max(np.abs(features[paired] - positive[selected])) > .02 + 1e-12:
        raise ValueError("Background GC caliper violated")
    output.mkdir(parents=True)
    np.savez_compressed(output / "matched_background.npz", sequence=np.stack(codes)[paired],
        ids=np.asarray(ids)[paired], chrom=np.full(len(paired), "chr3R"),
        start=np.asarray(starts)[paired], end=np.asarray(ends)[paired], enhancer_index=selected,
        enhancer_ids=test["ids"][selected], enhancer_gc=positive[selected], background_gc=features[paired])
    write_json(output / "matching.json", dict(status="complete", seed=20260915,
        background_definition="Central 512 bp avoids master DHS and H3K27ac consensus; flanks can be regulatory",
        matching="Unique same-chromosome full-input and central512 GC matches, each within 0.02",
        candidates=candidate_n, exact_or_rc_sequence_exclusions=duplicate_n,
        unmatched_ids=test["ids"][matched < 0].tolist(),
        input_sha256={str((enhancer_root / (s + ".npz")).resolve()): digest(enhancer_root / (s + ".npz")) for s in SPLITS},
        background_sha256=digest(output / "matched_background.npz")))


def prepare_classifiers(cfg):
    from prepare_v4_motifs import load_cohort
    from enhancer_pleiotropy_model.io import read_fasta
    from classifier_transfer.background_data import prepare as prepare_background
    from classifier_modisco.original_intervals import join_intervals
    inputs, work = (Path(cfg["paths"][k]) for k in ("inputs", "work"))
    root = work / "catalog"
    root.mkdir(parents=True, exist_ok=False)
    catalog = inputs / "catalog/noncontributing_dhs_p60/active_enhancers_wide.tsv.gz"
    fasta, blacklist = inputs / "reference/dm6.fa", inputs / "reference/dm6.blacklist.bed"
    frame, sequences = load_cohort(catalog, read_fasta(fasta)[0])
    frame.to_csv(root / "cohort.tsv.gz", sep="\t", index=False, compression="gzip")
    np.savez_compressed(root / "sequences.npz", ids=frame.master_dhs_id.to_numpy(dtype=str),
                        sequences=np.asarray(sequences, dtype="S2048"), valid=frame.sequence_valid.to_numpy())
    write_json(root / "prepared.json", dict(inputs={str(p): digest(p) for p in (catalog, fasta, blacklist)},
        outputs={name: digest(root / name) for name in ("cohort.tsv.gz", "sequences.npz")}))
    scientific = regression_config(cfg, work / "regression_data")
    enhancer_root = work / "enhancers"
    report = prepare(root / "cohort.tsv.gz", root / "sequences.npz", scientific,
                     enhancer_root, root / "prepared.json", blacklist)
    expected = {"train": 26930, "validation": 4062, "test": 9346}
    if {s: report["splits"][s]["n"] for s in SPLITS} != expected:
        raise ValueError("v4 reference split counts changed; inspect rather than silently continuing")
    # Keep original catalog ordering for motif discovery and ID-based shuffle seeds.
    parts = {s: load_split(enhancer_root, s) for s in SPLITS}
    rows = {str(ident): (s, i) for s, part in parts.items() for i, ident in enumerate(part["ids"])}
    order = [rows[str(ident)] for ident in frame.master_dhs_id if str(ident) in rows]
    cohort = {k: np.asarray([parts[s][k][i] for s, i in order])
              for k in ("ids", "sequence", "labels", "chrom", "summit")}
    cohort["split"] = np.asarray([s for s, _ in order])
    intervals = join_intervals(cohort, frame.to_dict("records"))
    np.savez_compressed(root / "cohort.npz", **cohort)
    np.savez_compressed(root / "intervals.npz", **intervals)
    windows = work / "regression_data/data/windows.tsv.gz"
    if "background" in cfg["populations"]:
        prepare_test_background(enhancer_root, windows, work / "test_background")
        prepare_background(enhancer_root, windows, work / "test_background", work / "enhancers_background")
    write_json(root / "complete.json", dict(status="complete", splits=expected,
        files={name: digest(root / name) for name in ("cohort.npz", "intervals.npz")}))
