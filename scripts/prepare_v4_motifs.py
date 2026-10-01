"""Local-only, verified v4 motif inputs and exact training-background calibration."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import socket
import tarfile

import numpy as np
import pandas as pd
import pyBigWig

from stage_v4_inputs import ARCHIVE_SHA256, ARCHIVE_ROOT, copy_verified_member, parse_manifest
from enhancer_pleiotropy_model.constants import CONTEXTS
from enhancer_pleiotropy_model.io import atomic_write_json, read_fasta, sha256_file
from enhancer_pleiotropy_model.master_element_calibration import calibrate, fit_empirical_background
from enhancer_pleiotropy_model.multitask_loss import summarize_numpy


CHECKPOINT_SHA256 = "15e99ee4eb46b1e00a03872ec1782c1752819510735aeefb0330a86b897b5f2c"
PREPARED_SHA256 = "dbd9858ff5357b0fa4c44ab7a3b4fa41e03eda423486cf54d47cb702fca98c73"
MOTIFS_SHA256 = "1bd7f6a63a9a20cdd01711f643461d1eec1304a71f7a6b635b3305eb3ee32ad7"
ASSAYS = ("atac", "h3k27ac")
CATALOG = "catalog/noncontributing_dhs_p60/active_enhancers_wide.tsv.gz"


def event(name, **values):
    print(json.dumps(dict(event=name, **values), sort_keys=True), flush=True)


def require_local():
    if socket.gethostname().split(".")[0] == "neocranex" or socket.gethostname().startswith("nodo"):
        raise RuntimeError("This analysis is local only; do not run on cluster hosts")


def checked(path, expected):
    digest = sha256_file(path)
    if digest != expected:
        raise ValueError(f"Checksum differs: {path}")
    event("checksum_ok", path=str(path), sha256=digest)


def stage(archive, destination):
    checked(archive, ARCHIVE_SHA256)
    names = ["README.md", CATALOG, "catalog/noncontributing_dhs_p60/metrics.json",
             "reference/dm6.fa", "reference/dm6.blacklist.bed"]
    names += [f"regulatory/normalized_mean_bigwig/{c}.{a}.mean.background_tmm.bw"
              for a in ASSAYS for c in CONTEXTS]
    with tarfile.open(archive, "r:") as tar:
        members = tar.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError("Duplicate archive members")
        manifest = parse_manifest(tar.extractfile(f"{ARCHIVE_ROOT}/FILE_MANIFEST.sha256").read().decode())
        for name in names:
            target = destination / name
            if target.exists():
                checked(target, manifest[name])
            else:
                copy_verified_member(tar, name, target, manifest[name])
            event("motif_input_staged", path=name)
    return {name: manifest[name] for name in names}


def training_background(training_root):
    checked(training_root / "prepared.json", PREPARED_SHA256)
    prepared = json.loads((training_root / "prepared.json").read_text())
    names = ["data/windows.tsv.gz"] + [f"data/profiles/{a}/train_profiles.npy" for a in ASSAYS]
    for name in names:
        checked(training_root / name, prepared["output_hashes"][name])
    mask = []
    counts = {"train": 0, "validation": 0, "test": 0}
    with gzip.open(training_root / names[0], "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            counts[row["split"]] += 1
            if row["split"] == "train":
                mask.append(row["source"] == "genomic_background")
    if counts != prepared["window_counts"] or sum(mask) != 357391:
        raise ValueError("Unexpected v4 training/background cohort")
    arrays = [np.load(training_root / name, mmap_mode="r", allow_pickle=False) for name in names[1:]]
    if any(x.dtype != np.float32 or x.shape != (len(mask), bins, 8)
           for x, bins in zip(arrays, (32, 96))):
        raise ValueError("Prepared profile geometry differs")
    indices = np.flatnonzero(mask)
    summaries = np.empty((2, len(indices), 8), dtype=np.float64)
    for start in range(0, len(indices), 4096):
        selected = indices[start:start + 4096]
        summaries[:, start:start + len(selected)] = summarize_numpy(*(x[selected] for x in arrays))
    return {a: fit_empirical_background(summaries[i]) for i, a in enumerate(ASSAYS)}


def assign_partition(chrom, summit, seed=20260910):
    start, end = summit - 1024, summit + 1024
    if chrom == "chr3R":
        return "test", "descriptive_only"
    if chrom == "chr2L":
        split = "validation" if end <= 11751856 else "train" if start >= 11761856 else "buffer"
    else:
        split = "train" if chrom in {"chrX", "chr2R", "chr3L", "chr4", "chrY", "chrUn_CP007081v1", "chrUn_CP007120v1"} else "unassigned"
    if split != "train":
        return split, "descriptive_only"
    if start // 1000000 != (end - 1) // 1000000:
        return split, "block_boundary"
    block = f"{chrom}:{summit // 1000000}"
    fraction = int.from_bytes(hashlib.sha256(f"{seed}:{block}".encode()).digest()[:8], "big") / 2**64
    return split, "discovery" if fraction < .6 else "confirmation"


def load_cohort(catalog, genome):
    frame = pd.read_csv(catalog, sep="\t")
    if len(frame) != 40455 or frame.master_dhs_id.duplicated().any():
        raise ValueError("Expected 40,455 unique v4 active enhancer IDs")
    membership = frame[[f"{c}__context_membership" for c in CONTEXTS]].to_numpy()
    percentiles = frame[[f"{c}__h3k27ac_max_500_background_percentile" for c in CONTEXTS]].to_numpy(float)
    calls = frame[[f"{c}__h3k27ac_active" for c in CONTEXTS]].to_numpy()
    if (not np.isfinite(percentiles).all() or not np.isin(membership, (0, 1)).all()
            or not np.isin(calls, (0, 1)).all() or np.any((percentiles < 0) | (percentiles > 1))
            or not np.array_equal((membership == 1) & (percentiles > .6), calls == 1)
            or not np.array_equal(calls.sum(axis=1), frame.active_context_count.to_numpy())
            or not frame.active_context_count.between(1, 8).all() or frame.blacklist_overlap.any()
            or not frame.regulatory_class.isin(("distal_enhancer_like", "proximal_enhancer_like")).all()):
        raise ValueError("Invalid or inconsistent v4 catalog activity")
    sequences, reasons = [], []
    for row in frame.itertuples():
        sequence = genome.get(row.chrom, "")[row.summit - 1024:row.summit + 1024] if row.summit >= 1024 else ""
        reason = "boundary_or_missing_chromosome" if len(sequence) != 2048 else "ambiguous_sequence" if set(sequence) - set("ACGT") else ""
        sequences.append(sequence if not reason else "")
        reasons.append(reason)
    frame["exclusion_reason"] = reasons
    frame["sequence_valid"] = np.asarray(reasons) == ""
    partitions = [assign_partition(r.chrom, int(r.summit)) for r in frame.itertuples()]
    frame["model_split"] = [p[0] for p in partitions]
    frame["motif_partition"] = [p[1] for p in partitions]
    frame["genomic_block"] = frame.chrom + ":" + (frame.summit // 1000000).astype(str)
    frame["gc"] = [(s.count("G") + s.count("C")) / len(s) if s else np.nan for s in sequences]
    return frame, sequences


def quantify(frame, v4):
    values = np.full((2, len(frame), 8), np.nan, np.float64)
    missing = np.zeros((2, len(frame), 8), np.int16)
    for ai, assay in enumerate(ASSAYS):
        radius = 256 if assay == "atac" else 768
        for ci, context in enumerate(CONTEXTS):
            path = v4 / f"regulatory/normalized_mean_bigwig/{context}.{assay}.mean.background_tmm.bw"
            with pyBigWig.open(str(path)) as bw:
                for i, row in enumerate(frame.itertuples()):
                    if not row.sequence_valid:
                        continue
                    signal = bw.values(row.chrom, row.summit - radius, row.summit + radius, numpy=True)
                    if np.isinf(signal).any() or np.any(signal[np.isfinite(signal)] < 0):
                        raise ValueError(f"Invalid BigWig signal: {path}, {row.master_dhs_id}")
                    missing[ai, i, ci] = np.isnan(signal).sum()
                    signal = np.nan_to_num(signal, nan=0.).astype(np.float64)
                    values[ai, i, ci] = signal.mean() if assay == "atac" else signal.reshape(3, 512).mean(axis=1).max()
            event("observed_track_quantified", assay=assay, context=context, elements=int(frame.sequence_valid.sum()))
    return values, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motifs", type=Path, required=True)
    args = parser.parse_args()
    require_local()
    root = args.output
    if (root / "prepared.json").exists():
        raise FileExistsError("Completed motif preparation already exists; do not overwrite")
    root.mkdir(parents=True, exist_ok=True)
    training = root / "inputs/training"
    checked(training / "model/best_model.pt", CHECKPOINT_SHA256)
    checked(args.motifs, MOTIFS_SHA256)
    inputs = stage(args.archive, root / "inputs/v4")
    backgrounds = training_background(training)
    np.savez_compressed(root / "backgrounds.npz", **backgrounds)
    genome, _ = read_fasta(root / "inputs/v4/reference/dm6.fa")
    frame, sequences = load_cohort(root / "inputs/v4" / CATALOG, genome)
    del genome
    observed, missing = quantify(frame, root / "inputs/v4")
    valid = frame.sequence_valid.to_numpy()
    for ai, assay in enumerate(ASSAYS):
        activity = np.full((len(frame), 8), np.nan)
        activity[valid] = calibrate(observed[ai, valid], backgrounds[assay])[1]
        frame[f"observed_{assay}_breadth"] = activity.sum(axis=1)
        frame[f"observed_{assay}_peak"] = np.max(observed[ai], axis=1)
        frame[f"observed_{assay}_dominant"] = [CONTEXTS[j] for j in np.argmax(np.nan_to_num(observed[ai]), axis=1)]
        for ci, context in enumerate(CONTEXTS):
            frame[f"observed_{assay}_{context}_signal"] = observed[ai, :, ci]
            frame[f"observed_{assay}_{context}_activity"] = activity[:, ci]
    frame.to_csv(root / "cohort.tsv.gz", sep="\t", index=False)
    np.savez_compressed(root / "sequences.npz", ids=frame.master_dhs_id.to_numpy(dtype="S32"),
                        sequences=np.asarray(sequences, dtype="S2048"), valid=valid)
    np.savez_compressed(root / "observed.npz", signal=observed, missing_bases=missing)
    outputs = {name: sha256_file(root / name) for name in ("backgrounds.npz", "cohort.tsv.gz", "sequences.npz", "observed.npz")}
    report = dict(status="complete", catalog_elements=len(frame), valid_sequences=int(valid.sum()),
                  excluded=frame.loc[~valid, "exclusion_reason"].value_counts().to_dict(),
                  partitions=frame.loc[valid, "motif_partition"].value_counts().to_dict(),
                  counts_by_catalog_breadth=frame.active_context_count.value_counts().sort_index().to_dict(),
                  checkpoint_sha256=CHECKPOINT_SHA256, checkpoint_epoch=16, checkpoint_learning_rate=.00005,
                  archive_sha256=ARCHIVE_SHA256, prepared_training_sha256=PREPARED_SHA256,
                  motifs=str(args.motifs.resolve()), motifs_sha256=MOTIFS_SHA256, inputs=inputs, outputs=outputs,
                  contexts=CONTEXTS, background_n=len(backgrounds["atac"]),
                  reference_sha256={a: hashlib.sha256(r.tobytes()).hexdigest() for a, r in backgrounds.items()},
                  breadth_definition="sum of eight max(0, 2 * empirical training-background midrank percentile - 1)",
                  observed_summaries="v4 BigWigs: ATAC central512 mean; H3 max of three512 means; missing coverage zero, recorded",
                  catalog_counts="v4 membership AND H3 noncontributing-DHS percentile > 0.6; distinct from continuous breadth",
                  inference="not yet run", scientific_scope="Known-motif analysis; no de novo discovery or causal binding claims",
                  confirmation="Genomic-block-disjoint within model-training regions, not independent of model training; validation/test descriptive only")
    atomic_write_json(root / "prepared.json", report)
    event("motif_preparation_complete", **report)


if __name__ == "__main__":
    main()
