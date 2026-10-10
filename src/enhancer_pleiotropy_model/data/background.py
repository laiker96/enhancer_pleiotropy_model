"""GC-matched genomic background for classifier training and evaluation.

Background candidates are the regression windows whose 512-bp centre overlaps
no master DHS and no H3K27ac consensus peak (flanks may still be regulatory).
Each enhancer is paired with at most one background from the same chromosome
and split whose whole-input and central-512-bp GC fractions both lie within
``caliper``. Sequences that duplicate an enhancer (or one another), in either
orientation, are never used. Backgrounds carry all-zero labels.

During training, one background is appended for every three enhancers in a
batch (see :func:`background_slice`).
"""

from __future__ import annotations

import collections
import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..constants import CONTEXTS, INPUT_BP, SPLITS
from ..io import open_text, read_json, sha256_file, write_json
from .enhancers import LOOKUP, load_split

ENHANCERS_PER_BACKGROUND = 3


def gc_features(codes: np.ndarray) -> np.ndarray:
    """Whole-input and central-512-bp GC fraction of integer-encoded 2,048-bp DNA."""
    codes = np.asarray(codes)
    gc = (codes == 1) | (codes == 2)
    return np.stack((gc.mean(1), gc[:, 768:1280].mean(1)), axis=1)


def sequence_key(codes: np.ndarray) -> bytes:
    """Orientation-independent identity of a sequence."""
    return hashlib.sha256(min(codes.tobytes(), (3 - codes[::-1]).tobytes())).digest()


def match_background(positive: np.ndarray, background: np.ndarray, seed: int, caliper: float = .02):
    """Greedy one-to-one GC matching, hardest-to-match enhancers first.

    Returns, for every positive, the index of its background or -1.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(background))
    tree = cKDTree(background[permutation])
    closest = tree.query(positive, k=1, p=np.inf)[0]
    shuffled = rng.permutation(len(positive))
    order = shuffled[np.argsort(-closest[shuffled], kind="stable")]
    used = np.zeros(len(background), dtype=bool)
    match = np.full(len(positive), -1, dtype=np.int64)
    for i in order:
        candidates = np.asarray(tree.query_ball_point(positive[i], caliper, p=np.inf), dtype=int)
        candidates = candidates[~used[candidates]]
        if not len(candidates):
            continue
        distances = np.square(background[permutation[candidates]] - positive[i]).sum(1)
        selected = candidates[np.lexsort((candidates, distances))[0]]
        used[selected] = True
        match[i] = permutation[selected]
    return match


def _background_rows(windows: Path, splits: tuple[str, ...]):
    with open_text(windows) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["split"] in splits and row["source"] == "genomic_background":
                codes = LOOKUP[np.frombuffer(row["sequence"].encode("ascii"), dtype=np.uint8)]
                if len(codes) != INPUT_BP or (codes > 3).any():
                    raise ValueError(f"Invalid background DNA in {row['record_id']}")
                yield row, codes


def _save(path: Path, enhancers: dict, rows: dict, positive_gc: np.ndarray, background_gc: np.ndarray,
          matched: np.ndarray) -> dict:
    index = np.flatnonzero(matched >= 0)
    paired = matched[index]
    delta = background_gc[paired] - positive_gc[index]
    if not len(paired) or len(np.unique(paired)) != len(paired) or np.abs(delta).max() > .02 + 1e-12:
        raise ValueError("Invalid background matches")
    np.savez_compressed(
        path, ids=np.asarray(rows["ids"])[paired], chrom=np.asarray(rows["chrom"])[paired],
        start=np.asarray(rows["start"])[paired], end=np.asarray(rows["end"])[paired],
        sequence=np.stack([rows["sequence"][i] for i in paired]),
        labels=np.zeros((len(paired), len(CONTEXTS)), dtype=np.uint8),
        enhancer_index=index, enhancer_ids=enhancers["ids"][index],
        enhancer_gc=positive_gc[index], background_gc=background_gc[paired])
    return {"enhancers": int(len(enhancers["ids"])), "backgrounds": int(len(paired)),
            "unmatched_enhancer_ids": enhancers["ids"][matched < 0].tolist(),
            "gc_mean_absolute_difference": np.abs(delta).mean(0).tolist()}


def prepare_backgrounds(*, enhancer_dir: Path, windows: Path, seed: int, test_seed: int,
                        caliper: float) -> dict:
    """Write ``<enhancer_dir>/background/{train,validation,test}.npz`` and an audit."""
    output = Path(enhancer_dir) / "background"
    output.mkdir(parents=True, exist_ok=True)
    enhancers = {s: load_split(enhancer_dir, s) for s in SPLITS}
    forbidden = {sequence_key(row) for s in SPLITS for row in enhancers[s]["sequence"]}
    summary = {}

    # Test (chr3R): drop duplicates of enhancers and of earlier test candidates.
    rows = collections.defaultdict(list)
    seen = set()
    for row, codes in _background_rows(windows, ("test",)):
        key = sequence_key(codes)
        if key in forbidden or key in seen:
            continue
        seen.add(key)
        for field, value in (("ids", row["record_id"]), ("chrom", row["chrom"]), ("start", int(row["start"])),
                             ("end", int(row["end"])), ("sequence", codes)):
            rows[field].append(value)
    test_gc = gc_features(np.stack(rows["sequence"]))
    positive = gc_features(enhancers["test"]["sequence"])
    summary["test"] = _save(output / "test.npz", enhancers["test"], rows, positive, test_gc,
                            match_background(positive, test_gc, test_seed, caliper))
    with np.load(output / "test.npz") as handle:
        forbidden.update(sequence_key(row) for row in handle["sequence"])

    # Train/validation: a sequence seen in both pools is removed from both.
    pools = {s: collections.defaultdict(list) for s in ("train", "validation")}
    seen = {}
    for row, codes in _background_rows(windows, ("train", "validation")):
        split = row["split"]
        key = sequence_key(codes)
        if key in forbidden:
            continue
        if key in seen:
            previous_split, previous_index = seen[key]
            pools[previous_split]["keep"][previous_index] = False
            continue
        pool = pools[split]
        seen[key] = (split, len(pool["ids"]))
        for field, value in (("ids", row["record_id"]), ("chrom", row["chrom"]), ("start", int(row["start"])),
                             ("end", int(row["end"])), ("sequence", codes), ("keep", True)):
            pool[field].append(value)
    for split, pool in pools.items():
        features = gc_features(np.stack(pool["sequence"]))
        chrom, keep = np.asarray(pool["chrom"]), np.asarray(pool["keep"])
        part = enhancers[split]
        positive = gc_features(part["sequence"])
        matched = np.full(len(part["ids"]), -1, dtype=np.int64)
        for chromosome in sorted(set(part["chrom"])):
            positives = np.flatnonzero(part["chrom"] == chromosome)
            candidates = np.flatnonzero((chrom == chromosome) & keep)
            if len(candidates):
                local = match_background(positive[positives], features[candidates], seed, caliper)
                ok = local >= 0
                matched[positives[ok]] = candidates[local[ok]]
        if split == "train" and (matched >= 0).sum() < len(part["ids"]) // ENHANCERS_PER_BACKGROUND:
            raise ValueError("Too few training backgrounds for the 1:3 batch mixing")
        summary[split] = _save(output / f"{split}.npz", part, pool, positive, features, matched)
    write_json(output / "audit.json", {
        "definition": "regression background windows (central 512 bp free of DHS and H3K27ac peaks), "
                      "same-chromosome one-to-one GC matching of whole input and central 512 bp",
        "seed": seed, "test_seed": test_seed, "gc_caliper": caliper, "splits": summary,
        "windows": {"path": str(windows), "sha256": sha256_file(windows)},
        "outputs": {f"{s}.npz": sha256_file(output / f"{s}.npz") for s in SPLITS},
    })
    return summary


def load_background(root: Path, split: str, enhancers: dict) -> dict[str, np.ndarray]:
    audit = read_json(Path(root) / "background" / "audit.json")
    path = Path(root) / "background" / f"{split}.npz"
    if sha256_file(path) != audit["outputs"][path.name]:
        raise ValueError(f"{path} does not match its audit.json")
    with np.load(path, allow_pickle=False) as handle:
        data = dict(handle)
    if not np.array_equal(data["enhancer_ids"], enhancers["ids"][data["enhancer_index"]]):
        raise ValueError(f"{path} is not aligned with the enhancer split")
    return data


def background_epoch(n_enhancers: int, n_background: int, seed: int, epoch: int):
    """Backgrounds used in one epoch and their reverse-complement flags.

    An independent RNG leaves the enhancer order and augmentation untouched.
    """
    count = n_enhancers // ENHANCERS_PER_BACKGROUND
    if n_background < count:
        raise ValueError("Not enough training backgrounds")
    rng = np.random.default_rng(seed + 10_000_000 + epoch)
    return rng.permutation(n_background)[:count], rng.random(count) < .5


def background_slice(enhancer_begin: int, enhancer_end: int) -> slice:
    """Backgrounds that accompany enhancers [begin, end) of the epoch order."""
    return slice(enhancer_begin // ENHANCERS_PER_BACKGROUND, enhancer_end // ENHANCERS_PER_BACKGROUND)
