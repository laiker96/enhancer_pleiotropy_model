"""Frozen v4 CREsted motif-disruption experiment; CUDA requires explicit approval.

All elements contribute to the motif scan. Model perturbations use a deterministic,
count-stratified subset of confirmation elements, never validation/test selection.
"""

import argparse
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd

from .breadth_metrics import GROUPS
from .constants import CONTEXTS
from .inference import load_model, predict_sequences
from .io import atomic_write_json, sha256_file
from .master_element_calibration import calibrate
from .motif_analysis import (ASSAYS, MIN_CHROMOSOME_ELEMENTS, association_mask, event, load_prepared, load_scan,
                             matched_control, scramble_site)
from .multitask_loss import summarize_numpy


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def score_sequences(model, sequences, backgrounds, device, batch_size):
    profiles = predict_sequences(model, sequences, device=device, batch_size=batch_size,
                                 reverse_complement_ensemble=True, mixed_precision="no")
    signal = summarize_numpy(*profiles)
    activity = np.stack([calibrate(signal[a], backgrounds[assay])[1]
                         for a, assay in enumerate(ASSAYS)])
    return signal, activity


def choose_candidates(table):
    """Fixed discovery-only selection: positive slope, q<.05, top eight/assay."""
    selected = []
    for assay in ASSAYS:
        subset = table[(table.partition == "discovery")
                       & (table.exposure == f"observed_{assay}_breadth")
                       & np.isclose(table.threshold, .9) & (table.status == "ok")
                       & np.isfinite(table.q_value) & np.isfinite(table.coefficient)]
        if subset.empty:
            raise ValueError(f"No valid discovery fits for {assay}; this is not a no-candidate result")
        subset = subset[(subset.q_value < .05) & (subset.coefficient > 0)]
        subset = subset.sort_values(["q_value", "coefficient", "motif_id"],
                                    ascending=[True, False, True]).head(8)
        selected.extend(subset.motif_id.tolist())
    return sorted(set(selected))


def selected_sites(frame, score, motif_id, max_per_count):
    eligible = association_mask(frame, "confirmation") & (score >= .9)
    result = []
    for count in range(1, 9):
        indices = np.flatnonzero(eligible & (frame.active_context_count == count))
        ordered = sorted(indices, key=lambda i: seed_for(20260910, motif_id, frame.iloc[i].master_dhs_id))
        result.extend(ordered[:max_per_count])
    return result


def make_variants(sequence, start, motif, reverse, identifier):
    variants, details = [sequence], []
    for replicate in range(3):
        seed = seed_for(20260910, identifier, motif.identifier, replicate)
        mutant = scramble_site(sequence, start, motif, reverse, seed)
        if mutant is None:
            return None, "motif_cannot_be_composition_scrambled"
        chosen = None
        for attempt in range(12):
            control = matched_control(sequence, start, motif.width, seed_for(seed, "control", attempt))
            if control is None:
                continue
            changed, position, mismatch = control
            control_score = max(motif.relative(sequence[position:position + motif.width], r) for r in (False, True))
            if mismatch <= 2 and control_score < .85:
                chosen = (changed, position, mismatch, control_score)
                break
        if chosen is None:
            return None, "no_low_pwm_composition_matched_control"
        variants.extend([mutant, chosen[0]])
        details.append(dict(replicate=replicate, control_start=chosen[1],
                            control_composition_l1=int(chosen[2]), control_pwm=chosen[3],
                            mutant_site=mutant[start:start + motif.width],
                            mutant_pwm=max(motif.relative(mutant[start:start + motif.width], r) for r in (False, True))))
    return (variants, details), None


def effect_record(signal, activity):
    if signal.shape != (2, 7, 8) or activity.shape != signal.shape:
        raise ValueError("Need reference plus three paired mutant/control predictions")
    if not np.isfinite(signal).all() or not np.isfinite(activity).all():
        raise ValueError("Nonfinite perturbation predictions")
    result = {}
    for a, assay in enumerate(ASSAYS):
        for label, values in (("signal", signal[a]), ("log1p_signal", np.log1p(signal[a])), ("activity", activity[a])):
            reference = values[0]
            disruption = values[[1, 3, 5]].mean(axis=0) - reference
            control = values[[2, 4, 6]].mean(axis=0) - reference
            for c, context in enumerate(CONTEXTS):
                for name, vector in (("reference", reference), ("disruption_delta", disruption),
                                     ("control_delta", control), ("excess_delta", disruption - control)):
                    result[f"{assay}_{context}_{label}_{name}"] = float(vector[c])
            if label == "activity":
                for name, vector in (("reference", reference), ("disruption_delta", disruption),
                                     ("control_delta", control), ("excess_delta", disruption - control)):
                    result[f"{assay}_breadth_{name}"] = float(vector.sum())
                    for group, contexts in GROUPS.items():
                        result[f"{assay}_{group}_{name}"] = float(sum(vector[CONTEXTS.index(c)] for c in contexts))
    return result


def smoke(root, model, backgrounds, sequences, frame, device, batch_size):
    chosen = [sequences[i] for i in np.flatnonzero(frame.sequence_valid)[:2]]
    reverse = [s.translate(str.maketrans("ACGT", "TGCA"))[::-1] for s in chosen]
    started = time.monotonic()
    first = score_sequences(model, chosen, backgrounds, device, batch_size)
    repeated = score_sequences(model, chosen, backgrounds, device, batch_size)
    rc = score_sequences(model, reverse, backgrounds, device, batch_size)
    for actual in (repeated, rc):
        for a, b in zip(first, actual):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    report = dict(status="passed", device=device, sequences=2, seconds=time.monotonic() - started,
                  checks=["finite nonnegative signals", "repeat determinism", "forward/RC summary agreement"],
                  maximum_repeat_error=float(np.max(np.abs(first[0] - repeated[0]))),
                  maximum_rc_error=float(np.max(np.abs(first[0] - rc[0]))))
    atomic_write_json(root / f"model_smoke_{device}.json", report)
    event("model_smoke_passed", **report)


def perturb(root, model, backgrounds, report, frame, sequences, motifs, device, batch_size, max_per_count):
    if (root / "perturbations.json").exists():
        raise FileExistsError("Perturbation analysis already completed")
    scan = load_scan(root, report, frame, motifs)
    metadata = json.loads((root / "associations.json").read_text())
    if metadata["status"] != "complete" or metadata["scan_sha256"] != sha256_file(root / "motif_scan.npz"):
        raise ValueError("Association inputs changed")
    if metadata.get("minimum_chromosome_elements") != MIN_CHROMOSOME_ELEMENTS:
        raise ValueError("Association and perturbation chromosome eligibility rules differ")
    for name, digest in metadata["outputs"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("Association output changed")
    table = pd.read_csv(root / "motif_associations.tsv", sep="\t")
    candidates = choose_candidates(table)
    event("perturbation_candidates_selected", count=len(candidates), motifs=candidates)
    contract = dict(checkpoint=report["checkpoint_sha256"], prepared=sha256_file(root / "prepared.json"),
                    associations=sha256_file(root / "associations.json"), candidates=candidates,
                    max_per_count=max_per_count, replicates=3, seed=20260910, device=device,
                    batch_size=batch_size, source=sha256_file(Path(__file__)),
                    minimum_chromosome_elements=MIN_CHROMOSOME_ELEMENTS,
                    selection="top eight positive q<.05 discovery associations per continuous assay, PWM .9",
                    scoring="FP32 forward/RC profile ensemble; exact training-background calibration")
    contract_path = root / "perturbation_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Perturbation resume contract changed")
    atomic_write_json(contract_path, contract)
    partial = root / "perturbation_partial"
    partial.mkdir(exist_ok=True)
    results, failures = [], []
    for j, motif in enumerate(motifs):
        if motif.identifier not in candidates:
            continue
        indices = selected_sites(frame, scan["score"][:, j], motif.identifier, max_per_count)
        for n, i in enumerate(indices):
            row = frame.iloc[i]
            identity = dict(master_dhs_id=row.master_dhs_id, motif_id=motif.identifier, motif_name=motif.name,
                            genomic_block=row.genomic_block, active_context_count=int(row.active_context_count),
                            observed_atac_breadth=float(row.observed_atac_breadth),
                            observed_h3k27ac_breadth=float(row.observed_h3k27ac_breadth),
                            site_start=int(scan["position"][i, j]), reverse=bool(scan["reverse"][i, j]),
                            original_pwm=float(scan["score"][i, j]))
            path = partial / f"{motif.identifier}_{i}.json"
            if path.exists():
                saved = json.loads(path.read_text())
            else:
                variants, failure = make_variants(sequences[i], identity["site_start"], motif,
                                                  identity["reverse"], row.master_dhs_id)
                if failure:
                    saved = dict(identity, status="excluded", reason=failure)
                else:
                    strings, details = variants
                    signal, activity = score_sequences(model, strings, backgrounds, device, batch_size)
                    saved = dict(identity, status="ok", controls=details,
                                 predictions=dict(signal=signal.tolist(), activity=activity.tolist(),
                                                  sequence_order=["reference", "mutant0", "control0", "mutant1", "control1", "mutant2", "control2"]),
                                 **effect_record(signal, activity))
                atomic_write_json(path, saved)
            (results if saved["status"] == "ok" else failures).append(saved)
            if (n + 1) % 16 == 0 or n + 1 == len(indices):
                event("perturbation_progress", motif=motif.identifier, completed=n + 1, total=len(indices),
                      successful_sites=len(results), excluded_sites=len(failures))
    effects = pd.DataFrame([{k: v for k, v in row.items() if k not in ("controls", "predictions")} for row in results])
    effects.to_csv(root / "motif_perturbation_effects.tsv", sep="\t", index=False)
    pd.DataFrame(failures).to_csv(root / "motif_perturbation_exclusions.tsv", sep="\t", index=False)
    if len(effects):
        columns = [f"{assay}_breadth_{kind}_delta" for assay in ASSAYS for kind in ("disruption", "control", "excess")]
        grouped = effects.groupby(["motif_id", "motif_name", "active_context_count"])[columns].agg(["count", "mean", "median"])
        grouped.columns = ["_".join(parts) for parts in grouped.columns]
        grouped.to_csv(root / "motif_perturbation_by_breadth.tsv", sep="\t")
    status = "complete" if results else "no_candidates" if not candidates else "no_scorable_sites"
    atomic_write_json(root / "perturbations.json", dict(status=status, **contract,
                      successful_sites=len(results), excluded_sites=len(failures),
                      interpretation="Negative disruption delta means the intact site supported predicted breadth. Excess subtracts the matched-control change. Model effects, not binding or biological causality.",
                      limitations=["known motifs only", "one strongest central512 site per enhancer/motif", "mononucleotide, not dinucleotide, matched disruptions", "up to one-base composition mismatch at control sites", "count-stratified subset, not population-weighted", "confirmation is disjoint from motif discovery, not model training", "group summaries descriptive, no perturbation significance test"]))
    event("perturbation_stage_complete", status=status, candidates=len(candidates),
          successful_sites=len(results), excluded_sites=len(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "perturb"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-approved", action="store_true", help="Only after the user has approved local CUDA computation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-per-count", type=int, default=32)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] == "neocranex" or socket.gethostname().startswith("nodo"):
        raise RuntimeError("Motif analysis is local only")
    if args.device == "cuda" and not args.cuda_approved:
        raise RuntimeError("Obtain user permission before local CUDA computation")
    if args.batch_size < 1 or args.max_per_count < 1:
        raise ValueError("Positive batch and sampling sizes required")
    report, frame, sequences, motifs = load_prepared(args.output)
    checkpoint = args.output / "inputs/training/model/best_model.pt"
    if sha256_file(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint checksum mismatch")
    model, metadata = load_model(checkpoint, device=args.device)
    if metadata.epoch != 16:
        raise ValueError("Expected selected v4 CREsted epoch 16")
    with np.load(args.output / "backgrounds.npz", allow_pickle=False) as data:
        backgrounds = {assay: data[assay] for assay in ASSAYS}
    smoke(args.output, model, backgrounds, sequences, frame, args.device, args.batch_size)
    if args.stage == "perturb":
        perturb(args.output, model, backgrounds, report, frame, sequences, motifs,
                args.device, args.batch_size, args.max_per_count)


if __name__ == "__main__":
    main()
