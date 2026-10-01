"""Audited, post-hoc motif families; no seqlet reassignment or model execution."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .paper_motifs import run_core
from .report_pdf import sha
from .tomtom_atlas import run, save_json


RULES = dict(information_bits=.2, consecutive=5, sign_consistency=.75,
    bootstrap_replicates=2000, seed=20260917, pwm_similarity=.85, contribution_similarity=.70,
    information_coverage=.75, shorter_length_coverage=.70, longer_length_coverage=.50,
    min_joint_informative_positions=5, min_overlap=6, strong_base_probability=.60,
    strong_base_information=.40, linkage="complete; every cross-cluster pair must pass",
    sensitivity=[[.80, .60], [.85, .70], [.90, .80]])


def event(name, **values):
    print(json.dumps(dict(event=name, **values)), flush=True)


def enhancer_means(example_idx, scores):
    """One vote per enhancer, not one vote per seqlet."""
    example_idx, scores = np.asarray(example_idx), np.asarray(scores, dtype=float)
    if (scores.ndim != 3 or scores.shape[-1] != 4 or len(scores) != len(example_idx)
            or not len(scores) or not np.isfinite(scores).all()):
        raise ValueError("Invalid aligned seqlet scores")
    unique, inverse, counts = np.unique(example_idx, return_inverse=True, return_counts=True)
    per = np.zeros((len(unique), *scores.shape[1:]), dtype=float)
    np.add.at(per, inverse, scores)
    per /= counts[:, None, None]
    return unique, per


def signed_summary(per, ident, replicates=2000):
    values = per.sum((1, 2))/per.shape[1]
    mean = float(values.mean())
    positive_fraction = float((values > 0).mean())
    negative_fraction = float((values < 0).mean())
    seed = RULES["seed"]+int(hashlib.sha256(ident.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates)
    for start in range(0, replicates, 100):
        n = min(100, replicates-start)
        bootstrap[start:start+n] = values[rng.integers(0, len(values), (n, len(values)))].mean(1)
    lo, hi = np.quantile(bootstrap, [.025, .975])
    return dict(mean=mean, median=float(np.median(values)), ci95=[float(lo), float(hi)],
        positive_fraction=positive_fraction, negative_fraction=negative_fraction,
        sign="positive" if mean > 0 else "negative" if mean < 0 else "zero")


def classify_sign(source_sign, summary):
    if summary["sign"] != source_sign:
        return "discordant"
    if summary[source_sign+"_fraction"] < RULES["sign_consistency"]:
        return "mixed"
    return source_sign


def information(pwm):
    pwm = np.asarray(pwm, dtype=float)
    return np.maximum(0., 2+(pwm*np.log2(np.maximum(pwm, 1e-300))).sum(1))


def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.clip(np.dot(a, b)/denominator, -1, 1)) if denominator else 0.


def alignments(first, second):
    """Admissible alignments, plus explicit rejections of diagnostic conflicts.

    Offset places the start of the oriented second motif relative to first.
    Native TF annotations are never used in this procedure.
    """
    a, ac = np.asarray(first["trimmed_pwm"]), np.asarray(first["core_contributions"])
    b0, bc0 = np.asarray(second["trimmed_pwm"]), np.asarray(second["core_contributions"])
    ai = information(a)
    output, conflicts = [], 0
    for orientation in ("+", "-"):
        b, bc = (b0, bc0) if orientation == "+" else (b0[::-1, ::-1], bc0[::-1, ::-1])
        bi = information(b)
        for offset in range(-len(b)+1, len(a)):
            astart, bstart = max(0, offset), max(0, -offset)
            length = min(len(a)-astart, len(b)-bstart)
            if (length < min(RULES["min_overlap"], len(a), len(b))
                    or length/min(len(a), len(b)) < RULES["shorter_length_coverage"]
                    or length/max(len(a), len(b)) < RULES["longer_length_coverage"]):
                continue
            ap, bp = a[astart:astart+length], b[bstart:bstart+length]
            aic, bic = ai[astart:astart+length], bi[bstart:bstart+length]
            coverage = [float(aic.sum()/ai.sum()), float(bic.sum()/bi.sum())]
            if min(coverage) < RULES["information_coverage"]:
                continue
            joint = int(((aic > .2) & (bic > .2)).sum())
            if joint < RULES["min_joint_informative_positions"]:
                continue
            confident = ((ap.max(1) >= RULES["strong_base_probability"])
                & (bp.max(1) >= RULES["strong_base_probability"])
                & (aic >= RULES["strong_base_information"])
                & (bic >= RULES["strong_base_information"]))
            if (confident & (ap.argmax(1) != bp.argmax(1))).any():
                conflicts += 1
                continue
            pwm_r = cosine(ap-.25, bp-.25)
            contribution_r = cosine(ac[astart:astart+length], bc[bstart:bstart+length])
            output.append(dict(offset=offset, orientation=orientation, overlap=length,
                information_coverage=coverage, joint_informative=joint,
                pwm=pwm_r, contribution=contribution_r))
    return dict(alignments=output, strong_conflict_alignments=conflicts)


def choose_alignment(pair, pwm_threshold, contribution_threshold):
    valid = [a for a in pair["alignments"]
             if a["pwm"] >= pwm_threshold and a["contribution"] >= contribution_threshold]
    if not valid:
        return None
    return max(valid, key=lambda a: (min(a["pwm"], a["contribution"]), a["pwm"],
        a["contribution"], a["overlap"], a["orientation"] == "+", -abs(a["offset"]), -a["offset"]))


def pair_key(a, b):
    return "|".join(sorted((a, b)))


def complete_families(ids, pairs, pwm_threshold=.85, contribution_threshold=.70):
    """Deterministic complete-link agglomeration; no single-link chaining."""
    ids = sorted(ids)
    similarities = {}
    for a, b in itertools.combinations(ids, 2):
        match = choose_alignment(pairs[pair_key(a, b)], pwm_threshold, contribution_threshold)
        if match:
            similarities[pair_key(a, b)] = (match["pwm"]+match["contribution"])/2
    clusters = [(ident,) for ident in ids]
    while True:
        candidates = []
        for i, a in enumerate(clusters):
            for j in range(i+1, len(clusters)):
                b = clusters[j]
                keys = [pair_key(x, y) for x in a for y in b]
                if all(k in similarities for k in keys):
                    candidates.append((-min(similarities[k] for k in keys), tuple(sorted(a+b)), i, j))
        if not candidates:
            return clusters, similarities
        _, merged, i, j = min(candidates)
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]+[merged]
        clusters.sort()


def union_support(members, by_id, groups):
    result = {}
    for group in groups:
        relevant = [by_id[i] for i in members if by_id[i]["group"] == group["name"]]
        # Local example indices are comparable ONLY inside one discovery group.
        indices = sorted({i for r in relevant for i in r["support_indices"]})
        result[group["name"]] = dict(count=len(indices), total=group["n"],
            fraction=len(indices)/group["n"], summed_pattern_support=sum(r["supporting_enhancers"] for r in relevant),
            patterns=len(relevant), indices=indices)
    return result


def analyze(source, source_audit, previous, root):
    import h5py
    from .report_breadth import digest, pwm_quality
    from .breadth import importance_summary
    started = time.monotonic()
    root.mkdir()
    (root/"references").symlink_to(os.path.relpath(previous/"references", root), target_is_directory=True)
    source_data = json.loads(source_audit.read_text())
    config_path = source/"package/config.json"
    config = json.loads(config_path.read_text())
    data = dict(groups=[], rules=RULES, sources={str(config_path):sha(config_path), str(source_audit):sha(source_audit)},
        source_root=str(source), bases=source_data["bases"], colors=source_data["colors"],
        glyphs=source_data["glyphs"], model=source_data["model"], generator_sha256=sha(__file__))
    for definition in config["groups"]:
        name = definition["name"]
        directory = source/"full_consensus_inputs"/name
        complete = json.loads((directory/"complete.json").read_text())
        assert complete["status"] == "complete" and complete["group"] == definition
        assert complete["analysis_config_sha256"] == sha(config_path)
        for filename in ("motifs.h5", "motif_importance.json", "complete.json"):
            path = directory/filename
            data["sources"][str(path)] = digest(path)
            if filename != "complete.json":
                assert data["sources"][str(path)] == complete["outputs"][filename]
        original = json.loads((directory/"motif_importance.json").read_text())
        group = dict(**definition, n=complete["discovery_elements"], rows=[])
        with h5py.File(directory/"motifs.h5", "r") as handle:
            assert {r["pattern"] for r in original} == {sign+"/"+key for sign in ("pos_patterns", "neg_patterns") for key in handle.get(sign, {})}
            for saved in sorted(original, key=lambda r:r["pattern"]):
                p = handle[saved["pattern"]]
                pwm = p["sequence"][:]
                qc = pwm_quality(pwm)
                qc.pop("trimmed_sequence")
                seqlets = p["seqlets"]
                scores = seqlets["contrib_scores"][:]
                seq = seqlets["sequence"][:]
                assert scores.shape == seq.shape and scores.shape[1:] == pwm.shape
                assert np.isin(seq, [0, 1]).all() and (seq.sum(2) == 1).all()
                assert np.allclose(seq.mean(0), pwm, atol=1e-7)
                assert np.allclose(scores.mean(0), p["contrib_scores"][:], atol=1e-6)
                indices = seqlets["example_idx"][:]
                assert indices.dtype.kind in "iu" and (indices >= 0).all() and (indices < group["n"]).all()
                old_stats = importance_summary(indices, scores, group["n"])
                assert old_stats["supporting_enhancers"] == saved["supporting_enhancers"]
                assert np.isclose(old_stats["mean_contribution_per_base"], saved["mean_contribution_per_base"])
                row = dict(id=f"{name}/{saved['pattern']}", group=name, pattern=saved["pattern"],
                    source_sign=saved["ranking_direction"], supporting_enhancers=saved["supporting_enhancers"],
                    representation=saved["assigned_enhancer_fraction"], seqlets=saved["seqlets"],
                    original_full_mean=saved["mean_contribution_per_base"], quality=qc, full_pwm=pwm.tolist())
                if qc["passed"]:
                    start, end, core = run_core(pwm.tolist())
                    unique, per = enhancer_means(indices, scores[:, start:end, :])
                    row.update(trimmed_pwm=core, core_start0=start, core_end0=end,
                        support_indices=unique.tolist(), core_contributions=per.mean(0).tolist(),
                        per_enhancer_core_means=(per.sum((1, 2))/(end-start)).tolist())
                    row["signed"] = signed_summary(per, row["id"], RULES["bootstrap_replicates"])
                    row["polarity"] = classify_sign(row["source_sign"], row["signed"])
                else:
                    row["polarity"] = "excluded"
                group["rows"].append(row)
        data["groups"].append(group)
        event("sign_audit_group", group=name, counts=dict(Counter(r["polarity"] for r in group["rows"])))
    by_id = {r["id"]:r for g in data["groups"] for r in g["rows"]}
    primary = [r for r in by_id.values() if r["polarity"] in ("positive", "negative")]
    pairs = {}
    for sign in ("positive", "negative"):
        rows = sorted([r for r in primary if r["polarity"] == sign], key=lambda r:r["id"])
        for i, (a, b) in enumerate(itertools.combinations(rows, 2)):
            pairs[pair_key(a["id"], b["id"])] = alignments(a, b)
            if i % 500 == 0:
                event("pairwise_progress", sign=sign, compared=i)
    sensitivity = []
    families = []
    for thresholds in RULES["sensitivity"]:
        entry = dict(pwm=thresholds[0], contribution=thresholds[1], signs={})
        for sign in ("positive", "negative"):
            ids = [r["id"] for r in primary if r["polarity"] == sign]
            clusters, similarities = complete_families(ids, pairs, *thresholds)
            entry["signs"][sign] = dict(families=len(clusters), non_singleton=sum(len(c)>1 for c in clusters),
                clusters=clusters, eligible_pairs=len(similarities))
            if thresholds != [RULES["pwm_similarity"], RULES["contribution_similarity"]]:
                continue
            candidates = []
            for members in clusters:
                def centrality(ident):
                    others = [similarities[pair_key(ident, x)] for x in members if x != ident]
                    return sum(others)/len(others) if others else 1.
                medoid = min(members, key=lambda i:(-centrality(i), -by_id[i]["supporting_enhancers"], i))
                support = union_support(members, by_id, data["groups"])
                candidates.append(dict(polarity=sign, members=list(members), representative=medoid,
                    group_support=support, max_fraction=max(g["fraction"] for g in support.values()),
                    minimum_pair_similarity=min((similarities[pair_key(a,b)] for a,b in itertools.combinations(members,2)), default=1.)))
            candidates.sort(key=lambda f:(-f["max_fraction"], f["members"]))
            for i, family in enumerate(candidates, 1):
                family["id"] = ("P" if sign == "positive" else "N")+f"{i:02d}"
                for ident in family["members"]:
                    by_id[ident]["family"] = family["id"]
            families.extend(candidates)
        sensitivity.append(entry)
    data.update(families=families, sensitivity=sensitivity,
        summary=dict(patterns=len(by_id), source_signs=dict(Counter(r["source_sign"] for r in by_id.values())),
            polarity=dict(Counter(r["polarity"] for r in by_id.values())),
            families=dict(Counter(f["polarity"] for f in families)), pairwise_comparisons=len(pairs),
            seconds=time.monotonic()-started))
    save_json(root/"pairs.json", pairs)
    data["pairs_sha256"] = sha(root/"pairs.json")
    save_json(root/"analysis.json", data)
    event("family_analysis_complete", **data["summary"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["analyze", "search"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--source-audit", type=Path)
    p.add_argument("--previous", type=Path)
    p.add_argument("--tomtom", type=Path)
    a = p.parse_args()
    if a.stage == "analyze":
        analyze(a.source, a.source_audit, a.previous, a.root)
    else:
        run(a.root, a.root/"analysis.json", a.tomtom,
            query_rule="All 131 quality-passing positive/negative-source run-anchored cores; original signs and sign-audit flags preserved")


if __name__ == "__main__":
    main()
