"""Read-only catalog audit and hashed, split-separated classification arrays."""

import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

CONTEXTS = ("ab", "e13", "e5", "ead", "hid", "lb", "o", "wid")
SPLITS = ("train", "validation", "test")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def assign_split(chrom, start, end, config):
    choices = []
    for split in SPLITS:
        if chrom in config["chromosome_splits"][split]: choices.append(split)
        for region in config["region_splits"][split]:
            name, interval = region.split(":")
            lo, hi = map(int, interval.split("-"))
            if name == chrom and start >= lo and end <= hi: choices.append(split)
    if len(choices) > 1: raise ValueError("Ambiguous split configuration")
    return choices[0] if choices and start >= 0 else "excluded"


def prepare(cohort, sequences, config, output, prepared, blacklist):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    original = json.loads(Path(prepared).read_text())
    for path in (cohort, sequences):
        if digest(path) != original["outputs"][Path(path).name]:
            raise ValueError("Catalog/sequence preparation hash mismatch")
    with gzip.open(cohort, "rt") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    cached = np.load(sequences, allow_pickle=False)
    ids = np.asarray([r["master_dhs_id"] for r in rows])
    if len(rows) != 40455 or len(set(ids)) != len(ids) or not np.array_equal(ids, cached["ids"].astype(str)):
        raise ValueError("Expected aligned 40,455 unique catalog IDs")
    labels = np.asarray([[int(r[c+"__h3k27ac_active"]) for c in CONTEXTS] for r in rows], dtype=np.int64)
    membership = np.asarray([[int(r[c+"__context_membership"]) for c in CONTEXTS] for r in rows])
    percentiles = np.asarray([[float(r[c+"__h3k27ac_max_500_background_percentile"]) for c in CONTEXTS] for r in rows])
    if (not np.isfinite(percentiles).all() or not ((percentiles >= 0) & (percentiles <= 1)).all()
            or not np.isin(labels, (0, 1)).all() or not np.isin(membership, (0, 1)).all()
            or not np.array_equal(labels, (membership == 1) & (percentiles > .6))
            or not labels.any(axis=1).all()
            or not np.array_equal(labels.sum(1), [int(r["active_context_count"]) for r in rows])
            or any(int(r["blacklist_overlap"]) != 0 for r in rows)):
        raise ValueError("Invalid/missing catalog activity labels")
    blacklist_intervals = collections.defaultdict(list)
    with Path(blacklist).open() as handle:
        for line in handle:
            if line.startswith(("#", "track", "browser")) or not line.strip(): continue
            chrom, start, end = line.split()[:3]
            blacklist_intervals[chrom].append((int(start), int(end)))
    split, reasons, canonical = [], [], collections.defaultdict(list)
    dna = np.zeros((len(rows), 2048), dtype=np.uint8)
    lookup = np.full(256, 255, dtype=np.uint8)
    lookup[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4)
    for i, (row, sequence) in enumerate(zip(rows, cached["sequences"], strict=True)):
        seq = bytes(sequence)
        start, end = int(row["summit"])-1024, int(row["summit"])+1024
        part = assign_split(row["chrom"], start, end, config)
        reason = ""
        if len(seq) != 2048 or any(b not in b"ACGT" for b in seq): reason = "invalid_sequence"
        elif part == "excluded": reason = "split_boundary"
        elif any(start < hi and end > lo for lo, hi in blacklist_intervals[row["chrom"]]): reason = "input_blacklist_overlap"
        if bool(cached["valid"][i]) != (len(seq) == 2048 and all(b in b"ACGT" for b in seq)):
            raise ValueError("Cached sequence validity mismatch")
        split.append(part)
        reasons.append(reason)
        if not reason:
            dna[i] = lookup[np.frombuffer(seq, dtype=np.uint8)]
            rc = seq.translate(bytes.maketrans(b"ACGT", b"TGCA"))[::-1]
            canonical[hashlib.sha256(min(seq, rc)).digest()].append(i)
    for indices in canonical.values():
        if len({split[i] for i in indices}) > 1:
            for i in indices: reasons[i] = "cross_split_identical_sequence_or_rc"
    split = np.asarray(split)
    keep = np.asarray(reasons) == ""
    # Complete input containment is checked above, not merely center membership.
    output.mkdir(parents=True)
    summary = {}
    for name in SPLITS:
        selected = keep & (split == name)
        y = labels[selected].astype(np.uint8)
        if not len(y) or np.any(y.sum(0) == 0) or np.any(y.sum(0) == len(y)):
            raise ValueError(f"Both label classes required in {name}")
        np.savez_compressed(output / f"{name}.npz", ids=ids[selected], sequence=dna[selected], labels=y,
                            chrom=np.asarray([r["chrom"] for r in rows])[selected],
                            summit=np.asarray([int(r["summit"]) for r in rows])[selected])
        summary[name] = dict(n=len(y), positives=dict(zip(CONTEXTS, y.sum(0).tolist())),
                             breadth_counts=dict(zip(*[v.tolist() for v in np.unique(y.sum(1), return_counts=True)])))
    report = dict(status="complete", catalog_n=len(rows), retained_n=int(keep.sum()), contexts=CONTEXTS,
                  splits=summary, exclusions=dict(collections.Counter(r for r in reasons if r)),
                  excluded_ids={str(ids[i]): r for i, r in enumerate(reasons) if r},
                  inputs={str(p): digest(p) for p in (cohort, sequences, prepared, blacklist)},
                  regression_splits={k: config[k] for k in ("chromosome_splits", "region_splits")},
                  outputs={f"{s}.npz": digest(output/f"{s}.npz") for s in SPLITS},
                  label_definition="DHS membership AND atlas H3K27ac percentile > 0.60",
                  scope="Known active enhancers only; zeros are catalog-negative, not proven inactive",
                  test_caveat="chr3R was examined in previous development; not a pristine holdout")
    write_json(output/"audit.json", report)
    print(json.dumps({k: report[k] for k in ("status", "retained_n", "splits", "exclusions")}), flush=True)
    return report


def load_split(root, split):
    if split not in SPLITS: raise ValueError(split)
    audit = json.loads((Path(root)/"audit.json").read_text())
    path = Path(root)/f"{split}.npz"
    if digest(path) != audit["outputs"][path.name]: raise ValueError("Classification data hash mismatch")
    with np.load(path, allow_pickle=False) as handle:
        return {k: handle[k] for k in handle.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motifs", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import yaml
    prepare(args.motifs/"cohort.tsv.gz", args.motifs/"sequences.npz", yaml.safe_load(args.config.read_text()),
            args.output, args.motifs/"prepared.json", args.motifs/"inputs/v4/reference/dm6.blacklist.bed")


if __name__ == "__main__": main()
