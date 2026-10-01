"""Known-motif summaries by observed breadth, with train-only candidate selection."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
import torch

from classifier_transfer.data import CONTEXTS, digest, write_json
from enhancer_pleiotropy_model.motif_analysis import bh_adjust, read_motifs
from enhancer_pleiotropy_model.motif_perturbation import make_variants
from .attribution import ensemble, load_model, one_hot, seed_for


def groups(breadth):
    return [("exact_"+str(k), breadth == k) for k in range(1,9)]+[("ge_"+str(k), breadth >= k) for k in range(2,9)]


def simple(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                positive_fraction=float((values > 0).mean()) if len(values) else None)


def adjusted_breadth_slope(values, mask, data, probabilities):
    """OLS among motif-present enhancers, genomic-block cluster-robust uncertainty.

    Control GC, observed signal strength, model confidence and active-context
    proportions, not all eight binary labels (their sum is the exposure).
    """
    idx = np.flatnonzero(mask & np.isfinite(values))
    if len(idx) < 100: return dict(status="insufficient_sites", n=len(idx))
    labels = data["labels"][idx].astype(float)
    breadth = labels.sum(1)
    block = np.asarray([str(data["chrom"][i])+":"+str(int(data["summit"][i])//1000000) for i in idx])
    unique = np.unique(block)
    if len(unique) < 10 or np.std(breadth) == 0:
        return dict(status="insufficient_blocks_or_breadth", n=len(idx), blocks=len(unique))
    gc = np.isin(data["sequence"][idx, 768:1280], [1,2]).mean(1)
    confidence = (probabilities[idx]*labels).sum(1)/breadth
    candidates = [gc, np.log1p(data["observed_atac_peak"][idx]), np.log1p(data["observed_h3_peak"][idx]), confidence]
    candidates += [labels[:, j]/breadth for j in range(7)]
    chromosomes = data["chrom"][idx]
    candidates += [(chromosomes == c).astype(float) for c in np.unique(chromosomes)[1:]]
    columns = [np.ones(len(idx)), breadth]
    for v in candidates:
        if v.std() < 1e-10: continue
        v = (v-v.mean())/v.std()
        candidate = np.column_stack(columns+[v])
        if np.linalg.matrix_rank(candidate) == len(columns)+1: columns.append(v)
    x = np.column_stack(columns); y = values[idx]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y-x@beta
    bread = np.linalg.inv(x.T@x)
    scores = np.stack([x[block == b].T@residual[block == b] for b in unique])
    covariance = bread@(scores.T@scores)@bread
    covariance *= len(unique)/(len(unique)-1)*(len(y)-1)/(len(y)-x.shape[1])
    se = float(np.sqrt(max(0., covariance[1,1])))
    if not se > 0 or not np.isfinite(se): return dict(status="undefined_uncertainty", n=len(idx))
    return dict(status="ok", n=len(idx), blocks=len(unique), coefficient=float(beta[1]), standard_error=se,
                p_value=float(2*student_t.sf(abs(beta[1]/se), df=len(unique)-1)),
                ci95=[float(beta[1]-student_t.ppf(.975,len(unique)-1)*se), float(beta[1]+student_t.ppf(.975,len(unique)-1)*se)])


def assemble(root, data, prepared):
    complete = json.loads((root/"attribution_complete.json").read_text())
    if complete["config_sha256"] != prepared["config_sha256"]: raise ValueError("Wrong attribution contract")
    n = len(data["ids"])
    central = np.empty((n, 512), np.float32)
    probability, logits = np.empty((n,8), np.float32), np.empty((n,8), np.float32)
    valid, seen = np.zeros(n, bool), np.zeros(n, bool)
    for name, expected in complete["chunks"].items():
        path = root/"chunks"/name
        if digest(path) != expected: raise ValueError("Attribution chunk checksum mismatch")
        with np.load(path, allow_pickle=False) as saved:
            idx = saved["indices"]
            if seen[idx].any() or str(saved["signature"]) != prepared["config_sha256"]: raise ValueError("Repeated/wrong chunk")
            central[idx] = saved["actual"][:, 768:1280]
            probability[idx], logits[idx], valid[idx], seen[idx] = saved["probabilities"], saved["logits"], saved["quality_pass"], True
    if not seen.all(): raise ValueError("Missing attributed elements")
    return central, probability, logits, valid


def analyze(root):
    from .run import load, event
    config, data, prepared = load(root)
    central, probabilities, logits, valid = assemble(root, data, prepared)
    breadth = data["labels"].sum(1)
    prefixes = np.pad(np.cumsum(central, axis=1), ((0,0),(1,0)))
    values = np.full(data["motif_score"].shape, np.nan, np.float32)
    rows, slopes = [], []
    for j,(identifier,name,width) in enumerate(zip(data["motif_ids"], data["motif_names"], data["motif_widths"])):
        present = (data["motif_score"][:,j] >= config["motif_threshold"]) & valid
        idx = np.flatnonzero(present)
        start = data["motif_position"][idx,j]-768
        if np.any(start < 0) or np.any(start+width > 512): raise ValueError("Motif outside attribution crop")
        values[idx,j] = (prefixes[idx,start+width]-prefixes[idx,start])/width-central[idx].mean(1)
        for split in ("train", "validation", "test"):
            split_mask = (data["split"] == split) & valid
            for label, group_mask in groups(breadth):
                mask = split_mask & group_mask
                rows.append(dict(motif_id=str(identifier), motif_name=str(name), width=int(width), split=split, breadth_group=label,
                    cohort_n=int(mask.sum()), motif_present_n=int((mask & present).sum()),
                    prevalence=float((mask & present).sum()/mask.sum()) if mask.any() else None,
                    importance=simple(values[mask,j])))
            slope = adjusted_breadth_slope(values[:,j], split_mask, data, probabilities)
            slopes.append(dict(motif_id=str(identifier), motif_name=str(name), split=split, **slope))
        event("motif_summary_progress", completed=j+1, total=len(data["motif_ids"]))
    for split in ("train", "validation", "test"):
        selected = [r for r in slopes if r["split"] == split and r["status"] == "ok"]
        for r,q in zip(selected, bh_adjust(np.array([r["p_value"] for r in selected]))): r["q_value"] = float(q)
    # Candidate list is frozen from TRAINING IMPORTANCE, not held-out trends.
    eligible = [r for r in rows if r["split"] == "train" and r["breadth_group"] == "ge_2"
                and r["motif_present_n"] >= 100 and r["importance"]["mean"] > 0]
    eligible.sort(key=lambda r:(-r["importance"]["mean"], r["motif_id"]))
    candidates = eligible[:config["max_candidates"]]
    write_json(root/"motif_by_breadth.json", rows)
    write_json(root/"motif_breadth_slopes.json", slopes)
    write_json(root/"candidates.json", dict(selection="Top positive train ge2 mean site IG per base minus same-element central512 mean; minimum100 motif-positive sites; exploratory, not a significance filter", motifs=candidates))
    np.savez_compressed(root/"motif_importance.npz", ids=data["ids"], motif_ids=data["motif_ids"],
                        importance=values, quality_pass=valid, probabilities=probabilities, logits=logits)
    summary = dict(status="complete", elements=len(data["ids"]), quality_excluded=int((~valid).sum()),
        known_motifs=len(data["motif_ids"]), candidates=[r["motif_id"] for r in candidates],
        score="Mean per-base active-context IG within strongest PWM site minus that enhancer's central512 per-base mean",
        groups="Exact breadth1..8 and overlapping >=2..8, split-separated",
        regression="Motif-present elements only; adjusted for centralGC, observed ATAC/H3 peak, mean-active prediction, context proportions, chromosome; 1Mb cluster-robust uncertainty; BH within split",
        caveats=config["limitation"], outputs={p:digest(root/p) for p in ("motif_by_breadth.json","motif_breadth_slopes.json","candidates.json","motif_importance.npz")})
    write_json(root/"analysis_complete.json", summary)
    event("known_motif_analysis_complete", candidates=summary["candidates"], quality_excluded=summary["quality_excluded"])


def perturbation_effects(logits, probabilities, labels):
    if logits.shape != (7,8) or probabilities.shape != (7,8): raise ValueError("Reference plus three paired motif/control variants required")
    active = np.asarray(labels, bool)
    if not active.any(): raise ValueError("At least one observed active context required")
    result = {}
    for key, values in (("logit",logits),("probability",probabilities)):
        drop = values[0]-values[[1,3,5]].mean(0)
        control = values[0]-values[[2,4,6]].mean(0)
        adjusted = drop-control
        result[key] = dict(reference=values[0].tolist(), drop=drop.tolist(), control_drop=control.tolist(),
            adjusted_drop=adjusted.tolist(), mean_active_drop=float(drop[active].mean()),
            adjusted_mean_active_drop=float(adjusted[active].mean()),
            adjusted_mean_inactive_drop=float(adjusted[~active].mean()) if (~active).any() else None,
            positive_active_contexts=int((adjusted[active] > 0).sum()))
    result["expected_breadth_drop"] = float(np.sum(result["probability"]["drop"]))
    result["adjusted_expected_breadth_drop"] = float(np.sum(result["probability"]["adjusted_drop"]))
    return result


def ism(root):
    from .run import load, require_local_cuda, event
    require_local_cuda()
    config, data, prepared = load(root)
    summary = json.loads((root/"analysis_complete.json").read_text())
    if digest(root/"candidates.json") != summary["outputs"]["candidates.json"]: raise ValueError("Candidate list changed")
    candidates = json.loads((root/"candidates.json").read_text())["motifs"]
    if digest(Path(prepared["motif_library"])) != prepared["motif_library_sha256"]: raise ValueError("Motif library changed")
    motifs = {m.identifier:m for m in read_motifs(Path(prepared["motif_library"]))}
    model = load_model(Path(config["checkpoint"]), "cuda")
    if digest(root/"motif_importance.npz") != summary["outputs"]["motif_importance.npz"]: raise ValueError("Motif scores changed")
    with np.load(root/"motif_importance.npz", allow_pickle=False) as f: valid = f["quality_pass"]
    breadth = data["labels"].sum(1)
    partial = root/"ism_partial"; partial.mkdir(exist_ok=True)
    alphabet = np.asarray(list("ACGT")); completed = 0
    for candidate in candidates:
        motif = motifs[candidate["motif_id"]]
        j = list(data["motif_ids"]).index(motif.identifier)
        for split in config["ism_splits"]:
            for k in range(1,9):
                choices = np.flatnonzero((data["split"] == split) & (breadth == k) & valid & (data["motif_score"][:,j] >= config["motif_threshold"]))
                rng = np.random.default_rng(seed_for(config["seed"],motif.identifier,split,k))
                choices = rng.permutation(choices)[:config["ism_per_exact_breadth_per_split"]]
                for i in choices:
                    if (root/"STOP").exists(): raise RuntimeError("ISM paused at checkpointed element")
                    path = partial/(motif.identifier+"_"+str(data["ids"][i])+".json")
                    if path.exists():
                        saved=json.loads(path.read_text())
                        if saved["config_sha256"] != prepared["config_sha256"]: raise ValueError("ISM partial contract changed")
                        continue
                    sequence = "".join(alphabet[data["sequence"][i]])
                    start = int(data["motif_position"][i,j])
                    variants, failure = make_variants(sequence, start, motif, bool(data["motif_reverse"][i,j]), data["ids"][i])
                    record = dict(config_sha256=prepared["config_sha256"], motif_id=motif.identifier, motif_name=motif.name,
                        id=str(data["ids"][i]), split=split, breadth=int(k), labels=data["labels"][i].tolist(),
                        block=str(data["chrom"][i])+":"+str(int(data["summit"][i])//1000000),
                        site_start=start, status="excluded" if failure else "ok", failure=failure)
                    if not failure:
                        changed, details = variants
                        if len(changed) != 7 or changed[0] != sequence: raise ValueError("Reference/variant ordering changed")
                        codes = np.stack([np.asarray(["ACGT".index(c) for c in s], np.uint8) for s in changed])
                        with torch.no_grad(): logits, probabilities = ensemble(model, one_hot(codes, "cuda"))
                        if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all(): raise ValueError("Nonfinite ISM predictions")
                        record.update(variants=details, logits=logits.cpu().tolist(), probabilities=probabilities.cpu().tolist(),
                                      effects=perturbation_effects(logits.cpu().numpy(), probabilities.cpu().numpy(), data["labels"][i]))
                    write_json(path, record)
                    completed += 1
                    if completed % 32 == 0: event("motif_ism_progress", newly_completed=completed, motif=motif.identifier)
    files = sorted(partial.glob("*.json"))
    write_json(root/"ism_complete.json", dict(status="complete" if candidates else "no_candidates", records=len(files),
        config_sha256=prepared["config_sha256"], outputs={p.name:digest(p) for p in files}))


def block_interval(values, blocks, seed, iterations=500):
    unique, inverse = np.unique(blocks, return_inverse=True)
    if len(unique) < 5: return None
    totals = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0,len(unique),(iterations,len(unique)))
    boot = totals[selected].sum(1)/counts[selected].sum(1)
    return np.quantile(boot,[.025,.975]).tolist()


def summarize_ism(root):
    from .run import event
    completion = json.loads((root/"ism_complete.json").read_text())
    records = []
    for name, sha in completion["outputs"].items():
        path = root/"ism_partial"/name
        if digest(path) != sha: raise ValueError("ISM record changed")
        records.append(json.loads(path.read_text()))
    ok = [r for r in records if r["status"] == "ok"]
    summary = []
    for motif in sorted({r["motif_id"] for r in records}):
        for split in ("train","test"):
            rows = [r for r in ok if r["motif_id"] == motif and r["split"] == split]
            breadth = np.asarray([r["breadth"] for r in rows])
            for name, mask in groups(breadth):
                subset = [r for r, keep in zip(rows,mask) if keep]
                effects = np.asarray([r["effects"]["logit"]["adjusted_mean_active_drop"] for r in subset])
                blocks = np.asarray([r["block"] for r in subset])
                summary.append(dict(motif_id=motif, split=split, breadth_group=name, adjusted_active_logit_drop=simple(effects),
                    block_bootstrap_ci95=block_interval(effects,blocks,seed_for(motif,split,name)) if len(effects) else None,
                    adjusted_expected_breadth_drop=simple([r["effects"]["adjusted_expected_breadth_drop"] for r in subset]),
                    mean_positive_active_context_fraction=float(np.mean([r["effects"]["logit"]["positive_active_contexts"]/r["breadth"] for r in subset])) if subset else None,
                    context_adjusted_logit_drops={c:simple([r["effects"]["logit"]["adjusted_drop"][j] for r in subset if r["labels"][j]]) for j,c in enumerate(CONTEXTS)}))
    write_json(root/"ism_by_breadth.json", summary)
    write_json(root/"complete.json", dict(status="complete", scored_sites=len(ok), exclusions=len(records)-len(ok),
        known_motif_screen=True, de_novo_discovery=False, expected_breadth="sum of classifier probabilities, not signal breadth",
        caveat="Model sensitivity, not causal biology. ISM is breadth-stratified and not population-weighted; nested thresholds overlap; CI descriptive across selected sites.",
        outputs={"ism_by_breadth.json":digest(root/"ism_by_breadth.json")}))
    event("classifier_motif_analysis_complete", scored_sites=len(ok), exclusions=len(records)-len(ok))
