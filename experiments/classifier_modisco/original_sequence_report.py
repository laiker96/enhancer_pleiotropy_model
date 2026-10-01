"""Frozen original-enhancer motif scans and annotation; CECAR CPU only.

Sequence occurrence is not predictive occurrence. Discovery signs and seqlet
support are retained as separate attributes, never inferred from PWM hits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .common import digest, event, require_allocation, write_json
from .original_intervals import GROUPS, contract, load_intervals
from .tomtom_atlas import DATABASES, meme_queries, query_id, run as annotate

THRESHOLDS = (1e-5, 1e-4, 1e-3)
SPLITS = ("train", "validation", "test")


def shuffle(sequence, seed):
    """Same randomized Euler-trail null as attribution.py, without Torch.

    Exact mono/dinucleotide counts and endpoints; not uniform sampling of all
    possible shuffles. One paired null per enhancer, used descriptively only.
    """
    sequence = np.asarray(sequence, dtype=np.uint8)
    if sequence.ndim != 1 or not len(sequence) or not np.isin(sequence, range(4)).all():
        raise ValueError("Expected nonempty ACGT codes")
    rng = np.random.default_rng(seed)
    adjacency = [sequence[1:][sequence[:-1] == i].copy() for i in range(4)]
    for values in adjacency:
        rng.shuffle(values)
    remaining = [len(v) for v in adjacency]
    stack, trail = [int(sequence[0])], []
    while stack:
        v = stack[-1]
        if remaining[v]:
            remaining[v] -= 1
            stack.append(int(adjacency[v][remaining[v]]))
        else:
            trail.append(stack.pop())
    result = np.asarray(trail[::-1], np.uint8)
    if len(result) != len(sequence) or result[-1] != sequence[-1]:
        raise ValueError("Dinucleotide shuffle failed")
    return result


def frequency(pvalues, mask, lengths, width, threshold):
    """Denominator includes every group enhancer, including ones too short.

    A short enhancer has no possible complete match; separately record the
    number long enough, rather than silently changing the biological cohort.
    """
    n = int(mask.sum())
    hits = int(np.sum((pvalues <= threshold) & mask))
    return dict(n=n, hits=hits, fraction=hits/n if n else None,
                long_enough=int(np.sum(mask & (lengths >= width))))


def prepare(project, source, root):
    import h5py
    from .report_breadth import BASES, COLORS, GLYPHS
    _, parent, data = contract(project, source)
    intervals = load_intervals(source, data)
    refined = source/"refinement_v3"
    complete = json.loads((refined/"discovery_complete.json").read_text())
    if complete["status"] != "complete":
        raise ValueError("Discovery has not completed")
    groups, inputs = [], {}
    breadth = data["labels"].sum(1)
    for name, low, high in GROUPS:
        directory = refined/"groups"/name
        mapping = json.loads((directory/"refinement.json").read_text())
        retained = {r["output"]: r for r in mapping["mapping"] if r["status"] == "retained"}
        rows = []
        with h5py.File(directory/"nonredundant.h5", "r") as h5:
            for pattern, metadata in retained.items():
                node = h5[pattern]
                pwm = np.asarray(node["sequence"], dtype=float)
                cwm = np.asarray(node["contrib_scores"], dtype=float)
                if (pwm.ndim != 2 or pwm.shape[1] != 4 or cwm.shape != pwm.shape
                        or not np.isfinite(pwm).all() or not np.isfinite(cwm).all()
                        or (pwm < 0).any() or not np.allclose(pwm.sum(1), 1, atol=1e-6)):
                    raise ValueError("Invalid saved motif: "+name+"/"+pattern)
                pwm /= pwm.sum(1, keepdims=True)  # only floating-point normalization
                sign = "positive" if pattern.startswith("pos_") else "negative"
                if float(cwm.sum()) * (1 if sign == "positive" else -1) <= 0:
                    raise ValueError("Final CWM sign disagrees")
                rows.append(dict(id=name+"/"+pattern, pattern=pattern, sign=sign,
                    trimmed_pwm=pwm.tolist(), cwm=cwm.tolist(),
                    seqlets=int(len(node["seqlets"]["example_idx"])),
                    quality=dict(passed=True), refinement=metadata))
        groups.append(dict(name=name, minimum_breadth=low, maximum_breadth=high,
            n=int(np.sum((breadth >= low) & (breadth <= high) & (data["split"] == "train"))), rows=rows))
        for filename in ("nonredundant.h5", "refinement.json"):
            inputs[str((directory/filename).relative_to(project))] = digest(directory/filename)
    if sum(len(g["rows"]) for g in groups) != 95:
        raise ValueError("Unexpected corrected motif collection; review before changing")
    for path in (parent/"cohort.npz", parent/"best_model.pt", source/"intervals.npz",
                 refined/"discovery_complete.json"):
        inputs[str(path.relative_to(project))] = digest(path)
    audit = dict(groups=groups, colors=COLORS, glyphs=GLYPHS, inputs=inputs,
        rules=dict(interval="original enhancer start:end, not central 512",
            ranking="training sequence occurrence at site p <= 1e-4, separately per group/sign; ID breaks ties",
            denominator="all enhancers in group and split; too-short elements have zero complete hits",
            primary_site_p=1e-4, sensitivity_site_p=list(THRESHOLDS),
            null="one deterministic exact mono/dinucleotide-preserving Euler shuffle per enhancer; descriptive, not a significance test",
            strand="both; one enhancer counted once regardless of strand/hit count",
            discovery_sign="sign of attribution pattern, not a sign assigned to sequence hits",
            predictive_occurrence="not performed; user requested sequence-scan report first",
            matrices="unchanged final native-collapse/CWM-trim outputs; no extra reclustering or logo-only trimming"))
    (root/"scans").mkdir()
    counts = np.zeros(4, dtype=np.int64)
    unchanged = 0
    with (root/"enhancers.fa").open("x") as real, (root/"paired_null.fa").open("x") as null:
        for i, (offset, length) in enumerate(zip(intervals["offset"], intervals["length"])):
            codes = data["sequence"][i, offset:offset+length]
            if not np.isin(codes, range(4)).all():
                raise ValueError("Non-ACGT enhancer")
            seed = int.from_bytes(hashlib.sha256(("scan20260918|"+str(data["ids"][i])).encode()).digest()[:8], "little")
            shuffled = shuffle(codes, seed)
            unchanged += int(np.array_equal(codes, shuffled))
            real.write(f">e{i}\n"+"".join(BASES[b] for b in codes)+"\n")
            null.write(f">e{i}\n"+"".join(BASES[b] for b in shuffled)+"\n")
            if data["split"][i] == "train":
                counts += np.bincount(codes, minlength=4)
    background = (counts+counts[::-1])/(2*counts.sum())
    (root/"training_background.txt").write_text("# Zero-order training enhancer background; RC-symmetric\n"+
        "\n".join(f"{b} {p:.12g}" for b, p in zip(BASES, background))+"\n")
    np.savez_compressed(root/"cohort_metadata.npz", ids=data["ids"], breadth=breadth,
        split=data["split"], length=intervals["length"], start=intervals["start"],
        end=intervals["end"], chrom=data["chrom"])
    audit["background"] = dict(probabilities=background.tolist(), train_base_counts=counts.tolist(), unchanged_null_sequences=unchanged)
    for name in ("enhancers.fa", "paired_null.fa", "cohort_metadata.npz", "training_background.txt"):
        audit.setdefault("prepared_files", {})[name] = digest(root/name)
    write_json(root/"motif_audit.json", audit)
    event("sequence_inputs_ready", motifs=95, enhancers=len(breadth), unchanged_nulls=unchanged)
    return audit


def parse_fimo_line(line, lookup, best, lengths, widths):
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 10:
        raise ValueError("Unexpected FIMO --text columns: "+line[:100])
    motif, _, sequence, start, end, strand, _, p, _, _ = fields
    if not sequence.startswith("e") or motif not in lookup or strand not in ("+", "-"):
        raise ValueError("Unknown FIMO ID/strand")
    i, j = int(sequence[1:]), lookup[motif]
    start, end, p = int(start), int(end), float(p)
    if not (0 <= i < best.shape[1] and 1 <= start <= end <= lengths[i]
            and end-start+1 == widths[j] and 0 <= p <= 1):
        raise ValueError("FIMO hit outside original enhancer or invalid p")
    best[j, i] = min(best[j, i], p)


def scan_group(root, group, metadata):
    name, rows = group["name"], group["rows"]
    destination = root/"scans"/name
    destination.mkdir()
    query = destination/"motifs.meme"
    query.write_text(meme_queries(dict(groups=[group])))
    lookup = {query_id(row): j for j, row in enumerate(rows)}
    widths = [len(r["trimmed_pwm"]) for r in rows]
    # Probe each PWM's maximum-scoring possible sequence using FIMO itself.
    # This makes the unattainable site-p cutoff of very short motifs explicit.
    background = np.array(json.loads((root/"motif_audit.json").read_text())["background"]["probabilities"])
    with (destination/"best_possible.fa").open("x") as handle:
        for j, row in enumerate(rows):
            codes = np.argmax(np.asarray(row["trimmed_pwm"])/background, axis=1)
            handle.write(f">e{j}\n"+"".join("ACGT"[b] for b in codes)+"\n")
    probe_command = [str(root/"bin/fimo"), "--text", "--thresh", "1", "--motif-pseudo", "0.1",
        "--bgfile", str(root/"training_background.txt"), str(query), str(destination/"best_possible.fa")]
    probe = subprocess.run(probe_command, text=True, capture_output=True, check=True)
    (destination/"best_possible.fimo.tsv").write_text(probe.stdout)
    attainable = np.ones((len(rows), len(rows)))
    for line in probe.stdout.splitlines()[1:]:
        if line and not line.startswith("#"):
            parse_fimo_line(line, lookup, attainable, widths, widths)
    minimum_possible = np.diag(attainable).tolist()
    saved, commands = {}, []
    for kind, fasta in (("real", "enhancers.fa"), ("null", "paired_null.fa")):
        best = np.ones((len(rows), len(metadata["ids"])), dtype=np.float64)
        command = [str(root/"bin/fimo"), "--text", "--thresh", "0.001", "--motif-pseudo", "0.1",
                   "--bgfile", str(root/"training_background.txt"), str(query), str(root/fasta)]
        commands.append(command)
        event("scan_start", group=name, kind=kind)
        with (destination/f"{kind}.stderr.log").open("x") as stderr, gzip.open(destination/f"{kind}.fimo.tsv.gz", "wt") as archive:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, text=True)
            try:
                header = next(process.stdout).rstrip("\n")
                if not header.startswith("motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\tscore\tp-value\tq-value\tmatched_sequence"):
                    raise ValueError("Unexpected FIMO header: "+header)
                archive.write(header+"\n")
                for line in process.stdout:
                    archive.write(line)
                    if line.strip() and not line.startswith("#"):
                        parse_fimo_line(line, lookup, best, metadata["length"], widths)
                if process.wait() != 0:
                    raise RuntimeError("FIMO failed; see stderr")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        saved[kind] = best
        event("scan_done", group=name, kind=kind)
    np.savez_compressed(destination/"minimum_p.npz", query_ids=np.array(list(lookup)), **saved)
    summaries = {}
    for j, row in enumerate(rows):
        summaries[row["id"]] = {}
        for threshold in THRESHOLDS:
            values = summaries[row["id"]][str(threshold)] = {}
            for split in SPLITS:
                mask = ((metadata["split"] == split) & (metadata["breadth"] >= group["minimum_breadth"])
                        & (metadata["breadth"] <= group["maximum_breadth"]))
                values[split] = {kind: frequency(saved[kind][j], mask, metadata["length"], widths[j], threshold)
                                 for kind in ("real", "null")}
    result = dict(group=name, summaries=summaries, commands=commands,
                  binary_sha256=digest(root/"bin/fimo"), motifs_sha256=digest(query),
                  minimum_possible_p=dict(zip((row["id"] for row in rows), minimum_possible)),
                  attainable_probe_command=probe_command)
    write_json(destination/"summary.json", result)
    return result


def run(project, source, root):
    require_allocation("cpu")
    for binary in ("fimo", "tomtom"):
        if subprocess.check_output([str(root/"bin"/binary), "--version" if binary == "fimo" else "-version"], text=True).strip() != "5.5.9":
            raise ValueError("Expected MEME 5.5.9")
    audit = prepare(project, source, root)
    with np.load(root/"cohort_metadata.npz", allow_pickle=False) as data:
        metadata = dict(data)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda group: scan_group(root, group, metadata), audit["groups"]))
    for group, result in zip(audit["groups"], results):
        for row in group["rows"]:
            row["occurrence"] = result["summaries"][row["id"]]
            row["minimum_possible_p"] = result["minimum_possible_p"][row["id"]]
        for sign in ("positive", "negative"):
            selected = sorted((r for r in group["rows"] if r["sign"] == sign),
                key=lambda r: (-r["occurrence"][str(1e-4)]["train"]["real"]["fraction"], r["id"]))
            for rank, row in enumerate(selected, 1):
                row["rank"] = rank
    write_json(root/"scan_audit.json", audit)
    with (root/"occurrence_metrics.tsv").open("x") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["motif", "group", "sign", "training_rank", "site_p_threshold", "split", "cohort", "hits", "n", "fraction", "long_enough", "minimum_possible_site_p", "threshold_attainable"])
        for group in audit["groups"]:
            for row in group["rows"]:
                for threshold, splits in row["occurrence"].items():
                    for split, cohorts in splits.items():
                        for kind, value in cohorts.items():
                            writer.writerow([row["id"], group["name"], row["sign"], row["rank"], threshold, split, kind,
                                value["hits"], value["n"], value["fraction"], value["long_enough"],
                                row["minimum_possible_p"], row["minimum_possible_p"] <= float(threshold)])
    annotation = root/"annotation"
    annotation.mkdir()
    (annotation/"references").symlink_to(root/"references", target_is_directory=True)
    annotate(annotation, root/"scan_audit.json", root/"bin/tomtom",
        query_rule="All 95 original-enhancer refinement_v3 positive and negative matrices, unchanged; no further trimming")
    event("analysis_ready_for_rendering", motifs=95)
    from .original_sequence_pdf import render
    render(root, project/"experiments/repeat_relationship_20260918")
    write_json(root/"complete.json", dict(status="complete", job=os.environ["SLURM_JOB_ID"],
        scan_audit_sha256=digest(root/"scan_audit.json"), figures="generated; local visual QA pending",
        predictive_occurrence="not requested", files={str(p.relative_to(root)): digest(p)
            for p in sorted((root/"output/pdf").glob("*.pdf"))}))
    event("sequence_report_complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    run(args.project.resolve(), args.source.resolve(), args.root.resolve())
