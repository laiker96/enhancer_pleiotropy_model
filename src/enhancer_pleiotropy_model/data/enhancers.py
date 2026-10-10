"""Enhancer classification data from the drosophila-cCREs regulatory-element catalog.

An enhancer is a master DHS whose regulatory class is enhancer-like, that does
not overlap the blacklist and that is active in at least one context. It is
active in a context when it belongs to that context's DHS set AND its H3K27ac
signal exceeds the given percentile of the context's genomic background.
Each enhancer becomes a 2,048-bp input centred on its DHS summit. Enhancers are
removed when that input contains non-ACGT bases, crosses a split boundary or
touches the blacklist, or when its sequence (or reverse complement) also
occurs in another split.
"""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import CONTEXTS, INPUT_BP, SPLITS
from ..io import read_bed_intervals, read_fasta, read_json, sha256_file, write_json

LOOKUP = np.full(256, 255, dtype=np.uint8)
LOOKUP[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4)


def select_enhancers(catalog: pd.DataFrame, classes: list[str], percentile: float) -> pd.DataFrame:
    """Rows of ``master_elements_wide`` that are enhancers active in >= 1 context."""
    membership = catalog[[f"{c}__context_membership" for c in CONTEXTS]].to_numpy() == 1
    background = catalog[[f"{c}__h3k27ac_max_500_background_percentile" for c in CONTEXTS]].to_numpy(float)
    active = membership & (np.nan_to_num(background, nan=0.0) > percentile)
    keep = (catalog.regulatory_class.isin(classes).to_numpy() & active.any(axis=1)
            & (catalog.blacklist_overlap.to_numpy() == 0))
    table = catalog.loc[keep].reset_index(drop=True).copy()
    for index, context in enumerate(CONTEXTS):
        table[f"{context}__active"] = active[keep, index].astype(np.uint8)
    table["active_context_count"] = active[keep].sum(axis=1)
    if table.master_dhs_id.duplicated().any():
        raise ValueError("Duplicate enhancer IDs in the catalog")
    if f"{CONTEXTS[0]}__h3k27ac_active" in table:  # cross-check against the catalog's own call
        catalog_calls = table[[f"{c}__h3k27ac_active" for c in CONTEXTS]].to_numpy().astype(bool)
        ours = table[[f"{c}__active" for c in CONTEXTS]].to_numpy().astype(bool)
        if not np.array_equal(catalog_calls, ours):
            raise ValueError("Activity labels differ from the catalog's h3k27ac_active calls")
    return table


def split_of(chrom: str, start: int, end: int, splits: dict) -> str:
    """The split whose chromosome or region fully contains [start, end), else 'excluded'."""
    choices = []
    for split in SPLITS:
        if chrom in splits[split].get("chromosomes", []):
            choices.append(split)
        for region in splits[split].get("regions", []):
            name, interval = region.split(":")
            lo, hi = map(int, interval.split("-"))
            if name == chrom and start >= lo and end <= hi:
                choices.append(split)
    if len(choices) > 1:
        raise ValueError("Ambiguous split configuration")
    return choices[0] if choices and start >= 0 else "excluded"


def prepare_enhancers(*, catalog: Path, reference_fasta: Path, blacklist: Path, splits: dict,
                      output: Path, classes: list[str], percentile: float) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    table = select_enhancers(pd.read_csv(catalog, sep="\t"), classes, percentile)
    genome, _ = read_fasta(reference_fasta)
    blacklisted = read_bed_intervals(blacklist)
    labels = table[[f"{c}__active" for c in CONTEXTS]].to_numpy(np.uint8)
    half = INPUT_BP // 2
    sequences = np.zeros((len(table), INPUT_BP), dtype=np.uint8)
    split, reasons, canonical = [], [], collections.defaultdict(list)
    for i, row in enumerate(table.itertuples()):
        start, end = int(row.summit) - half, int(row.summit) + half
        sequence = genome.get(row.chrom, "")[start:end].encode() if start >= 0 else b""
        part = split_of(row.chrom, start, end, splits)
        reason = ""
        if len(sequence) != INPUT_BP or set(sequence) - set(b"ACGT"):
            reason = "invalid_sequence"
        elif part == "excluded":
            reason = "split_boundary"
        elif any(start < hi and end > lo for lo, hi in blacklisted.get(row.chrom, [])):
            reason = "input_blacklist_overlap"
        split.append(part)
        reasons.append(reason)
        if not reason:
            sequences[i] = LOOKUP[np.frombuffer(sequence, dtype=np.uint8)]
            rc = sequence.translate(bytes.maketrans(b"ACGT", b"TGCA"))[::-1]
            canonical[hashlib.sha256(min(sequence, rc)).digest()].append(i)
    for indices in canonical.values():  # identical DNA must not span two splits
        if len({split[i] for i in indices}) > 1:
            for i in indices:
                reasons[i] = "cross_split_identical_sequence_or_rc"
    split, reasons = np.asarray(split), np.asarray(reasons)
    keep = reasons == ""
    ids = table.master_dhs_id.to_numpy(dtype=str)
    summary = {}
    for name in SPLITS:
        selected = keep & (split == name)
        y = labels[selected]
        if not len(y) or np.any(y.sum(0) == 0) or np.any(y.sum(0) == len(y)):
            raise ValueError(f"Both label classes are required in every context of {name}")
        np.savez_compressed(output / f"{name}.npz", ids=ids[selected], sequence=sequences[selected],
                            labels=y, chrom=table.chrom.to_numpy(dtype=str)[selected],
                            summit=table.summit.to_numpy(np.int64)[selected])
        summary[name] = {"n": int(len(y)), "positives": dict(zip(CONTEXTS, y.sum(0).tolist()))}
    listing = table[["master_dhs_id", "chrom", "start", "end", "summit", "regulatory_class",
                     *[f"{c}__active" for c in CONTEXTS], "active_context_count"]].copy()
    listing.insert(5, "split", np.where(keep, split, "excluded"))
    listing.insert(6, "exclusion_reason", reasons)
    listing.to_csv(output / "enhancers.tsv.gz", sep="\t", index=False)
    audit = {
        "catalog_enhancers": int(len(table)), "retained": int(keep.sum()), "splits": summary,
        "exclusions": {str(k): v for k, v in collections.Counter(r for r in reasons if r).items()},
        "label_definition": f"context DHS member AND H3K27ac background percentile > {percentile}",
        "enhancer_classes": classes, "split_definition": splits,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)}
                   for name, path in (("catalog", catalog), ("reference_fasta", reference_fasta),
                                      ("blacklist", blacklist))},
        "outputs": {f"{s}.npz": sha256_file(output / f"{s}.npz") for s in SPLITS},
    }
    write_json(output / "audit.json", audit)
    return audit


def load_split(root: Path, split: str) -> dict[str, np.ndarray]:
    audit = read_json(Path(root) / "audit.json")
    path = Path(root) / f"{split}.npz"
    if sha256_file(path) != audit["outputs"][path.name]:
        raise ValueError(f"{path} does not match its audit.json")
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}
