"""Known insect motif recurrence and controlled frozen-model perturbations.

Motif occurrence is not binding; associations and model perturbations are not
experimental causal effects. No new model training or de novo motif discovery.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import socket
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from .constants import CONTEXTS
from .io import atomic_write_json, sha256_file


ASSAYS = ("atac", "h3k27ac")
ASSOCIATION_EXPOSURES = ("observed_atac_breadth", "observed_h3k27ac_breadth", "active_context_count")
MIN_CHROMOSOME_ELEMENTS = 25


def event(name, **values):
    print(json.dumps(dict(event=name, **values), sort_keys=True), flush=True)


@dataclass
class Motif:
    identifier: str
    name: str
    pwm: np.ndarray

    @property
    def width(self):
        return len(self.pwm)

    def matrix(self, reverse=False):
        matrix = np.log2((self.pwm + 1e-4) / .25)
        return matrix[::-1, ::-1] if reverse else matrix

    def relative(self, sequence, reverse=False):
        if len(sequence) != self.width or set(sequence) - set("ACGT"):
            raise ValueError("Motif scoring needs length-matched ACGT sequence")
        matrix = self.matrix(reverse)
        indices = np.asarray(["ACGT".index(c) for c in sequence])
        low, high = matrix.min(axis=1).sum(), matrix.max(axis=1).sum()
        return float((matrix[np.arange(self.width), indices].sum() - low) / (high - low))


def read_motifs(path):
    lines = path.read_text().splitlines()
    motifs = []
    for i, line in enumerate(lines):
        if not line.startswith("MOTIF "):
            continue
        fields = line.split(maxsplit=2)
        j = i + 1
        while j < len(lines) and "letter-probability matrix:" not in lines[j]:
            j += 1
        match = re.search(r"\bw=\s*(\d+)", lines[j]) if j < len(lines) else None
        if match is None:
            raise ValueError("Missing motif width")
        width = int(match.group(1))
        pwm = np.asarray([[float(x) for x in row.split()] for row in lines[j + 1:j + 1 + width]])
        if (pwm.shape != (width, 4) or not np.isfinite(pwm).all() or np.any(pwm < 0)
                or np.any(pwm.sum(axis=1) <= 0)):
            raise ValueError("Invalid motif matrix")
        pwm = pwm / pwm.sum(axis=1, keepdims=True)
        if np.allclose(pwm, .25):
            raise ValueError("Uninformative motif")
        motifs.append(Motif(fields[1], fields[2] if len(fields) > 2 else fields[1], pwm))
    if not motifs or len({m.identifier for m in motifs}) != len(motifs):
        raise ValueError("Empty or duplicate motif collection")
    return motifs


def encode(sequences):
    if not sequences or any(len(s) != len(sequences[0]) or set(s) - set("ACGT") for s in sequences):
        raise ValueError("Aligned ACGT sequences required")
    lookup = np.full(256, 255, np.uint8)
    lookup[np.frombuffer(b"ACGT", np.uint8)] = np.arange(4)
    return lookup[np.frombuffer("".join(sequences).encode(), np.uint8)].reshape(len(sequences), -1)


def scan_one(encoded, motif):
    windows = np.lib.stride_tricks.sliding_window_view(encoded, motif.width, axis=1)
    matrix = motif.matrix()
    low, high = matrix.min(axis=1).sum(), matrix.max(axis=1).sum()
    scores = []
    for reverse in (False, True):
        raw = motif.matrix(reverse)[np.arange(motif.width), windows].sum(axis=-1)
        scores.append((raw - low) / (high - low))
    forward, reverse = scores
    fpos, rpos = forward.argmax(axis=1), reverse.argmax(axis=1)
    rows = np.arange(len(encoded))
    use_reverse = reverse[rows, rpos] > forward[rows, fpos]
    return (np.where(use_reverse, reverse[rows, rpos], forward[rows, fpos]).astype(np.float32),
            np.where(use_reverse, rpos, fpos).astype(np.int16), use_reverse)


def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    result = np.full(len(pvalues), np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    order = valid[np.argsort(pvalues[valid])]
    if len(order):
        adjusted = pvalues[order] * len(order) / np.arange(1, len(order) + 1)
        result[order] = np.minimum(1., np.minimum.accumulate(adjusted[::-1])[::-1])
    return result


def design_matrix(frame, exposure):
    columns = [np.ones(len(frame)), frame[exposure].to_numpy(float)]
    for name, transform in (("gc", False), ("observed_atac_peak", True),
                            ("observed_h3k27ac_peak", True), ("width_bp", True),
                            ("nearest_tss_distance_bp", True)):
        values = frame[name].to_numpy(float)
        if transform:
            values = np.log1p(np.abs(values))
        scale = values.std()
        if scale > 0:
            columns.append((values - values.mean()) / scale)
    for name in ("chrom", "observed_h3k27ac_dominant"):
        columns.extend(pd.get_dummies(frame[name], drop_first=True, dtype=float).to_numpy().T)
    selected = []
    for column in columns:
        candidate = np.column_stack(selected + [column])
        if np.linalg.matrix_rank(candidate) > len(selected):
            selected.append(column)
        elif len(selected) == 1:
            raise ValueError("Exposure has no independent variation")
    return np.column_stack(selected)


def association_mask(frame, partition):
    """Outcome-independent chromosome support filter within each partition.

    Sparse chromosome dummy variables caused separation for every motif in v1.
    Keep these elements in the scan/inventory, but not adjusted inference.
    """
    base = frame.sequence_valid & frame.motif_partition.eq(partition)
    counts = frame.loc[base, "chrom"].value_counts()
    return base & frame.chrom.isin(counts.index[counts >= MIN_CHROMOSOME_ELEMENTS])


def association_fit_health(table):
    health = {}
    for partition in ("discovery", "confirmation"):
        for exposure in ASSOCIATION_EXPOSURES:
            subset = table[table.partition.eq(partition) & table.exposure.eq(exposure)]
            valid_primary = (np.isclose(subset.threshold, .9) & subset.status.eq("ok")
                             & np.isfinite(subset.p_value) & np.isfinite(subset.q_value))
            health[f"{partition}/{exposure}"] = dict(
                status_counts={str(k): int(v) for k, v in subset.status.value_counts().items()},
                valid_primary_fits=int(valid_primary.sum()))
    return health, all(item["valid_primary_fits"] > 0 for item in health.values())


def logistic_association(design, outcome, blocks):
    y = np.asarray(outcome, float)
    block_ids, codes = np.unique(blocks, return_inverse=True)
    base = dict(n=len(y), motif_positive_n=int(y.sum()), blocks=len(block_ids))
    if min(y.sum(), len(y) - y.sum()) < 25 or len(block_ids) < 20:
        return dict(base, status="insufficient_support", coefficient=np.nan, p_value=np.nan)
    beta = np.zeros(design.shape[1])
    converged = False
    for _ in range(100):
        p = expit(design @ beta)
        weight = np.maximum(p * (1 - p), 1e-8)
        hessian = design.T @ (weight[:, None] * design) + np.eye(len(beta)) * 1e-8
        step = np.linalg.solve(hessian, design.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    if not converged or np.max(np.abs(beta)) > 30:
        return dict(base, status="nonconverged_or_separated", coefficient=np.nan, p_value=np.nan)
    p = expit(design @ beta)
    bread = np.linalg.inv(design.T @ ((p * (1 - p))[:, None] * design) + np.eye(len(beta)) * 1e-8)
    cluster_scores = np.zeros((len(block_ids), len(beta)))
    np.add.at(cluster_scores, codes, design * (y - p)[:, None])
    correction = len(block_ids) / (len(block_ids) - 1) * (len(y) - 1) / (len(y) - len(beta))
    covariance = bread @ (cluster_scores.T @ cluster_scores) @ bread * correction
    se = float(np.sqrt(max(covariance[1, 1], 0)))
    if se <= 0 or not np.isfinite(se):
        return dict(base, status="invalid_uncertainty", coefficient=np.nan, p_value=np.nan)
    return dict(base, status="ok", coefficient=float(beta[1]), robust_se=se,
                odds_ratio=float(np.exp(beta[1])), ci_lower=float(np.exp(beta[1] - 1.96 * se)),
                ci_upper=float(np.exp(beta[1] + 1.96 * se)), p_value=float(2 * norm.sf(abs(beta[1] / se))))


def load_prepared(root):
    report = json.loads((root / "prepared.json").read_text())
    if report["status"] != "complete" or tuple(report["contexts"]) != CONTEXTS:
        raise ValueError("Incomplete or incompatible motif inputs")
    for name, digest in report["outputs"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Prepared motif file changed: {name}")
    path = Path(report["motifs"])
    if sha256_file(path) != report["motifs_sha256"]:
        raise ValueError("Motif database changed")
    frame = pd.read_csv(root / "cohort.tsv.gz", sep="\t", low_memory=False)
    with np.load(root / "sequences.npz", allow_pickle=False) as data:
        if not np.array_equal(data["ids"].astype(str), frame.master_dhs_id.to_numpy(str)):
            raise ValueError("Sequence/catalog ID mismatch")
        sequences = data["sequences"].astype(str).tolist()
    return report, frame, sequences, read_motifs(path)


def scan_signature(root, report):
    return hashlib.sha256(json.dumps(dict(prepared=sha256_file(root / "prepared.json"),
                                         motifs=report["motifs_sha256"], region=[768, 1280]), sort_keys=True).encode()).hexdigest()


def load_scan(root, report, frame, motifs):
    metadata = json.loads((root / "scan.json").read_text())
    signature = scan_signature(root, report)
    if (metadata["status"] != "complete" or metadata["signature"] != signature
            or metadata["output_sha256"] != sha256_file(root / "motif_scan.npz")):
        raise ValueError("Motif scan provenance mismatch")
    with np.load(root / "motif_scan.npz", allow_pickle=False) as data:
        if str(data["signature"]) != signature or list(data["motif_ids"]) != [m.identifier for m in motifs]:
            raise ValueError("Motif scan order/signature changed")
        arrays = {key: data[key] for key in ("score", "position", "reverse")}
    if any(a.shape != (len(frame), len(motifs)) for a in arrays.values()):
        raise ValueError("Motif scan shape mismatch")
    valid = frame.sequence_valid.to_numpy(bool)
    if not np.isfinite(arrays["score"][valid]).all():
        raise ValueError("Nonfinite motif scores for valid sequences")
    return arrays


def scan(root, workers=2, chunk_size=128):
    report, frame, sequences, motifs = load_prepared(root)
    signature = scan_signature(root, report)
    if (root / "scan.json").exists():
        load_scan(root, report, frame, motifs)
        event("motif_scan_already_complete")
        return
    partial = root / "scan_partial"
    partial.mkdir(exist_ok=True)
    score = np.full((len(frame), len(motifs)), np.nan, np.float32)
    position = np.full(score.shape, -1, np.int16)
    reverse = np.zeros(score.shape, bool)
    valid = np.flatnonzero(frame.sequence_valid.to_numpy(bool))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(valid), chunk_size):
            indices = valid[start:start + chunk_size]
            path = partial / f"chunk_{start:06d}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    if str(data["signature"]) != signature or not np.array_equal(data["indices"], indices):
                        raise ValueError("Motif scan resume signature changed")
                    score[indices], position[indices], reverse[indices] = data["score"], data["position"], data["reverse"]
            else:
                encoded = encode([sequences[i][768:1280] for i in indices])
                for j, (s, p, r) in enumerate(pool.map(lambda m: scan_one(encoded, m), motifs)):
                    score[indices, j], position[indices, j], reverse[indices, j] = s, p + 768, r
                temporary = path.with_suffix(".partial.npz")
                np.savez_compressed(temporary, signature=signature, indices=indices,
                                    score=score[indices], position=position[indices], reverse=reverse[indices])
                temporary.replace(path)
            event("motif_scan_progress", completed=min(start + chunk_size, len(valid)), total=len(valid), motifs=len(motifs))
    np.savez_compressed(root / "motif_scan.npz", signature=signature, score=score,
                        position=position, reverse=reverse, motif_ids=[m.identifier for m in motifs])
    atomic_write_json(root / "scan.json", dict(status="complete", signature=signature, motifs=len(motifs),
                                               elements=len(valid), score="relative PWM log-odds, max over both strands/central512; not a binding p-value",
                                               output_sha256=sha256_file(root / "motif_scan.npz")))


def associations(root):
    report, frame, _, motifs = load_prepared(root)
    score = load_scan(root, report, frame, motifs)["score"]
    if (root / "associations.json").exists():
        raise FileExistsError("Association analysis already completed")
    rows, prevalence, exclusions, cohorts = [], [], [], {}
    for partition in ("discovery", "confirmation"):
        base = frame.sequence_valid & frame.motif_partition.eq(partition)
        selected = association_mask(frame, partition)
        excluded = frame.loc[base & ~selected, ["master_dhs_id", "chrom", "motif_partition"]].copy()
        excluded["reason"] = f"chromosome_has_fewer_than_{MIN_CHROMOSOME_ELEMENTS}_valid_partition_elements"
        exclusions.append(excluded)
        subset = frame.loc[selected].copy()
        cohorts[partition] = dict(input_elements=int(base.sum()), eligible_elements=len(subset),
                                 excluded_elements=len(excluded),
                                 excluded_chromosome_counts={str(k): int(v) for k, v in excluded.chrom.value_counts().items()})
        event("association_cohort", partition=partition, **cohorts[partition])
        for exposure in ASSOCIATION_EXPOSURES:
            design = design_matrix(subset, exposure)
            if not np.isfinite(design).all():
                raise ValueError("Nonfinite association covariates")
            group = []
            for j, motif in enumerate(motifs):
                for threshold in (.85, .90, .95):
                    stats = logistic_association(design, score[selected, j] >= threshold,
                                                  subset.genomic_block.to_numpy())
                    group.append(dict(motif_id=motif.identifier, motif_name=motif.name,
                                      partition=partition, exposure=exposure, threshold=threshold, **stats))
            adjusted = bh_adjust([r["p_value"] for r in group])
            for row, q in zip(group, adjusted):
                row["q_value"] = q
            rows.extend(group)
            event("motif_associations_progress", partition=partition, exposure=exposure, comparisons=len(group),
                  valid_fits=sum(row["status"] == "ok" for row in group))
    for j, motif in enumerate(motifs):
        for count in range(1, 9):
            for partition in ("discovery", "confirmation", "descriptive_only"):
                selected = frame.sequence_valid & (frame.active_context_count == count) & (frame.motif_partition == partition)
                if selected.any():
                    prevalence.append(dict(motif_id=motif.identifier, motif_name=motif.name, active_context_count=count,
                                           partition=partition, n=int(selected.sum()), motif_positive_n=int((score[selected, j] >= .9).sum()),
                                           prevalence=float((score[selected, j] >= .9).mean())))
    table = pd.DataFrame(rows)
    health, usable = association_fit_health(table)
    table.to_csv(root / "motif_associations.tsv", sep="\t", index=False)
    pd.DataFrame(prevalence).to_csv(root / "motif_prevalence.tsv", sep="\t", index=False)
    pd.concat(exclusions).to_csv(root / "association_exclusions.tsv", sep="\t", index=False)
    atomic_write_json(root / "associations.json", dict(status="complete" if usable else "failed_no_valid_fits", primary_threshold=.9,
                      sensitivity_thresholds=[.85, .95], exposures=list(ASSOCIATION_EXPOSURES),
                      minimum_chromosome_elements=MIN_CHROMOSOME_ELEMENTS, cohorts=cohorts, fit_health=health,
                      source_sha256=sha256_file(Path(__file__)),
                      covariates=["GC", "log1p ATAC/H3 peak", "log1p DHS width", "log1p absolute TSS distance", "chromosome", "dominant H3 context"],
                      multiple_testing="BH over all motifs and three thresholds, separately per exposure and partition",
                      method="motif-presence logistic slope with 1-Mb genomic-block sandwich uncertainty",
                      interpretation="observational motif recurrence; model perturbation pending",
                      scan_sha256=sha256_file(root / "motif_scan.npz"),
                      outputs={name: sha256_file(root / name) for name in ("motif_associations.tsv", "motif_prevalence.tsv", "association_exclusions.tsv")}))
    if not usable:
        raise RuntimeError("Association fitting failed: a partition/exposure has no valid primary-threshold fits; perturbations must not proceed")


def scramble_site(sequence, start, motif, reverse, seed, minimum_drop=.2):
    original = sequence[start:start + motif.width]
    baseline = motif.relative(original, reverse)
    rng = np.random.default_rng(seed)
    for _ in range(200):
        shuffled = "".join(rng.permutation(list(original)))
        if max(motif.relative(shuffled, False), motif.relative(shuffled, True)) <= baseline - minimum_drop:
            return sequence[:start] + shuffled + sequence[start + motif.width:]
    return None


def matched_control(sequence, start, width, seed):
    target = sequence[start:start + width]
    counts = np.asarray([target.count(c) for c in "ACGT"])
    choices = [p for p in range(768, 1280 - width + 1)
               if p + width <= start or p >= start + width]
    if not choices:
        return None
    distances = [np.abs(np.asarray([sequence[p:p + width].count(c) for c in "ACGT"]) - counts).sum() for p in choices]
    rng = np.random.default_rng(seed)
    chosen = int(rng.choice(np.asarray(choices)[np.asarray(distances) == min(distances)]))
    segment = sequence[chosen:chosen + width]
    for _ in range(100):
        shuffled = "".join(rng.permutation(list(segment)))
        if shuffled != segment:
            return sequence[:chosen] + shuffled + sequence[chosen + width:], chosen, min(distances)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("scan", "associate"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] == "neocranex" or socket.gethostname().startswith("nodo"):
        raise RuntimeError("Motif analysis is local only")
    if args.workers not in (1, 2):
        raise ValueError("Use one or two local CPU workers")
    started = time.monotonic()
    if args.stage == "scan":
        scan(args.output, args.workers)
    else:
        associations(args.output)
    event("motif_stage_complete", stage=args.stage, seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
