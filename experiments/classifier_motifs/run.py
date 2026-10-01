"""Local, resumable attribution, followed by known-motif analysis and direct ISM."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from classifier_transfer.data import CONTEXTS, digest, load_split, write_json
from .attribution import (CHECKPOINT_SHA, active_weights, dinucleotide_shuffle, ensemble,
                          integrated_gradients, load_model, one_hot, score_batch, seed_for)


def event(name, **values):
    print(json.dumps(dict(event=name, **values), allow_nan=False), flush=True)


def dependency_sources():
    names = ("classifier_transfer.models", "classifier_transfer.fast_convolution", "classifier_transfer.data",
             "enhancer_pleiotropy_model.model", "enhancer_pleiotropy_model.motif_analysis",
             "enhancer_pleiotropy_model.motif_perturbation")
    return {name:Path(importlib.import_module(name).__file__).resolve() for name in names}


def prepare(config_path):
    config = json.loads(config_path.read_text())
    output = Path(config["output"])
    if output.exists(): raise FileExistsError(output)
    source = Path(config["data"])
    if digest(Path(config["checkpoint"])) != CHECKPOINT_SHA: raise ValueError("Wrong checkpoint")
    audit = json.loads((source/"audit.json").read_text())
    parts = [load_split(source, s) for s in ("train", "validation", "test")]
    data = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["split"] = np.concatenate([np.repeat(s, len(p["ids"])) for s,p in zip(("train", "validation", "test"), parts)])
    if (len(data["ids"]) != 40338 or len(np.unique(data["ids"])) != len(data["ids"])
            or data["sequence"].shape != (40338, 2048) or not np.isin(data["sequence"], range(4)).all()
            or not np.isin(data["labels"], [0, 1]).all() or not data["labels"].any(1).all()):
        raise ValueError("Unexpected classifier cohort")
    # Reuse only the sequence-motif scan, NOT old regressor attributions/results.
    from enhancer_pleiotropy_model.motif_analysis import load_prepared, load_scan
    motif_root = Path(config["motif_root"])
    original, frame, sequences, motifs = load_prepared(motif_root)
    scan = load_scan(motif_root, original, frame, motifs)
    mapping = {v:i for i,v in enumerate(frame.master_dhs_id.astype(str))}
    indices = np.asarray([mapping[v] for v in data["ids"]])
    selected = frame.iloc[indices]
    if not np.array_equal(selected.active_context_count.to_numpy(), data["labels"].sum(1)):
        raise ValueError("Motif cohort / classifier labels differ")
    for j,c in enumerate(CONTEXTS):
        if not np.array_equal(selected[c+"__h3k27ac_active"].to_numpy(), data["labels"][:, j]):
            raise ValueError("Context-label mismatch")
    alphabet = np.asarray(list("ACGT"))
    for start in range(0, len(indices), 128):
        for i in range(start, min(start+128, len(indices))):
            if "".join(alphabet[data["sequence"][i]]) != sequences[indices[i]]:
                raise ValueError("Motif coordinates / classifier sequence mismatch")
    data["motif_score"] = scan["score"][indices]
    data["motif_position"] = scan["position"][indices]
    data["motif_reverse"] = scan["reverse"][indices]
    data["motif_ids"] = np.asarray([m.identifier for m in motifs])
    data["motif_names"] = np.asarray([m.name for m in motifs])
    data["motif_widths"] = np.asarray([m.width for m in motifs])
    data["observed_atac_peak"] = selected.observed_atac_peak.to_numpy(float)
    data["observed_h3_peak"] = selected.observed_h3k27ac_peak.to_numpy(float)
    for name in ("observed_atac_peak", "observed_h3_peak"):
        if not np.isfinite(data[name]).all() or (data[name] < 0).any():
            raise ValueError("Invalid observed signal covariates")
    output.mkdir(parents=True)
    (output/"chunks").mkdir(); (output/"source").mkdir()
    np.savez_compressed(output/"records.npz", **data)
    shutil.copy2(config_path, output/"config.json")
    for path in Path(__file__).parent.glob("*.py"): shutil.copy2(path, output/"source"/path.name)
    # Pin the executed classifier and convolution implementation as well.
    for name,path in dependency_sources().items():
        shutil.copy2(path, output/"source"/(name+".py"))
    provenance = dict(status="prepared", checkpoint_sha256=CHECKPOINT_SHA, config_sha256=digest(config_path),
        data_audit_sha256=digest(source/"audit.json"), input_split_hashes=audit["outputs"],
        motif_scan_sha256=digest(motif_root/"motif_scan.npz"), motif_library_sha256=original["motifs_sha256"],
        motif_library=original["motifs"], records_sha256=digest(output/"records.npz"),
        source_hashes={p.name:digest(p) for p in (output/"source").iterdir()}, contexts=CONTEXTS,
        counts={s:{str(k):int(((data["split"] == s)&(data["labels"].sum(1) == k)).sum()) for k in range(1,9)}
                for s in ("train", "validation", "test")})
    write_json(output/"prepared.json", provenance)
    event("classifier_motifs_prepared", elements=len(data["ids"]), counts=provenance["counts"])


def load(root):
    config = json.loads((root/"config.json").read_text())
    prepared = json.loads((root/"prepared.json").read_text())
    if digest(root/"config.json") != prepared["config_sha256"] or digest(root/"records.npz") != prepared["records_sha256"]:
        raise ValueError("Prepared analysis changed")
    for name, expected in prepared["source_hashes"].items():
        if digest(root/"source"/name) != expected: raise ValueError("Frozen source changed")
    for path in Path(__file__).parent.glob("*.py"):
        if digest(path) != prepared["source_hashes"].get(path.name): raise ValueError("Executed analysis differs from frozen source")
    for name,path in dependency_sources().items():
        if digest(path) != prepared["source_hashes"].get(name+".py"): raise ValueError("Executed dependency differs from frozen source")
    with np.load(root/"records.npz", allow_pickle=False) as f: data = dict(f)
    return config, data, prepared


def require_local_cuda():
    if os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0] in {"a100", "cecar", "neocranex"}:
        raise RuntimeError("Attribution must stay on the authorized local workstation")
    if not torch.cuda.is_available() or "GTX 1060" not in torch.cuda.get_device_name():
        raise RuntimeError("Expected the authorized local GTX 1060")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def pilot(root):
    require_local_cuda()
    config, data, prepared = load(root)
    model = load_model(Path(config["checkpoint"]), "cuda")
    rng = np.random.default_rng(config["seed"])
    breadth = data["labels"].sum(1)
    indices = np.concatenate([rng.choice(np.flatnonzero((data["split"] == "train")&(breadth == k)), 4, replace=False) for k in range(1,9)])
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    result = score_batch(model, data, indices, config, "cuda")
    seconds = time.monotonic()-start
    # Validate the primary aggregation and compare 32/refined with fixed 64 steps.
    comparison = indices[::4]  # One enhancer from each exact breadth, 1..8.
    x = one_hot(data["sequence"][comparison], "cuda")
    weights = active_weights(torch.as_tensor(data["labels"][comparison], device="cuda"))
    reference = one_hot(np.stack([dinucleotide_shuffle(data["sequence"][i], seed_for(config["seed"], data["ids"][i], 0)) for i in comparison]), "cuda")
    low = integrated_gradients(model, x, reference, weights, 32, config["internal_batch"])
    high = integrated_gradients(model, x, reference, weights, 64, config["internal_batch"])
    from scipy.stats import spearmanr
    agreements = [float(spearmanr(a[768:1280], b[768:1280]).statistic) for a,b in zip(low["actual"].cpu().numpy(), high["actual"].cpu().numpy())]
    with torch.no_grad():
        a = ensemble(model, x)[1]; b = ensemble(model, x.flip((1,2)))[1]
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    # Check against previously archived validation predictions (trained in FP16).
    val = np.flatnonzero(data["split"] == "validation")[:64]
    with torch.no_grad(): p = ensemble(model, one_hot(data["sequence"][val], "cuda"))[1].cpu().numpy()
    cached = Path(config["checkpoint"]).parent/"best_validation_predictions.npz"
    with np.load(cached, allow_pickle=False) as f:
        if not np.array_equal(f["ids"][:64], data["ids"][val]): raise ValueError("Cached validation identity mismatch")
        prediction_difference = float(np.max(np.abs(p-f["probabilities"][:64])))
    good = bool(result["quality_pass"].mean() >= config["minimum_pass_fraction"]
                and np.median(agreements) >= .95 and prediction_difference < .003)
    report = dict(status="passed" if good else "failed", config_sha256=prepared["config_sha256"],
        checkpoint_sha256=CHECKPOINT_SHA, elements=len(indices), seconds=seconds,
        projected_attribution_hours=len(data["ids"])*seconds/len(indices)/3600,
        convergence_pass_fraction=float(result["quality_pass"].mean()), max_abs_delta=float(np.abs(result["delta"]).max()),
        steps_counts={str(v):int((result["steps"] == v).sum()) for v in np.unique(result["steps"])},
        median_32_vs_64_central_spearman=float(np.median(agreements)),
        comparison_ids=data["ids"][comparison].tolist(), comparison_breadths=breadth[comparison].tolist(),
        central_spearman_32_vs_64=agreements,
        max_cached_validation_probability_difference=prediction_difference,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda)
    write_json(root/"pilot.json", report)
    event("attribution_pilot", **report)
    if not good: raise RuntimeError("Pilot quality gate failed; full analysis not started")


def attribute(root):
    require_local_cuda()
    config, data, prepared = load(root)
    pilot_report = json.loads((root/"pilot.json").read_text())
    if pilot_report["status"] != "passed" or pilot_report["config_sha256"] != prepared["config_sha256"]:
        raise ValueError("Matching successful pilot required")
    model = load_model(Path(config["checkpoint"]), "cuda")
    started, done, failed = time.monotonic(), 0, 0
    for begin in range(0, len(data["ids"]), config["batch_size"]):
        if (root/"STOP").exists():
            event("attribution_paused", next_index=begin); return False
        indices = np.arange(begin, min(begin+config["batch_size"], len(data["ids"])))
        path = root/"chunks"/("chunk_%06d.npz" % begin)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["signature"]) != prepared["config_sha256"] or not np.array_equal(saved["indices"], indices):
                    raise ValueError("Attribution chunk identity changed")
                failed += int((~saved["quality_pass"]).sum())
        else:
            result = score_batch(model, data, indices, config, "cuda")
            if any(not np.isfinite(v).all() for v in result.values()): raise ValueError("Nonfinite attribution")
            failed += int((~result["quality_pass"]).sum())
            temporary = path.with_suffix(".partial.npz")
            np.savez_compressed(temporary, indices=indices, signature=prepared["config_sha256"], **result)
            temporary.replace(path)
        done += len(indices)
        event("attribution_progress", completed=done, total=len(data["ids"]), quality_failures=failed,
              seconds=round(time.monotonic()-started, 1), last_index=int(indices[-1]))
        if done >= 512 and (1-failed/done) < config["minimum_pass_fraction"]:
            raise RuntimeError("Too many unconverged elements; stopped rather than accepting unreliable scores")
    write_json(root/"attribution_complete.json", dict(status="complete", elements=done, quality_failures=failed,
        config_sha256=prepared["config_sha256"], seconds=time.monotonic()-started,
        chunks={p.name:digest(p) for p in sorted((root/"chunks").glob("chunk_*.npz"))}))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "pilot", "attribute", "analyze", "ism", "summarize_ism"))
    p.add_argument("--config", type=Path, default=Path("config/classifier_motifs_20260916.json"))
    p.add_argument("--root", type=Path, default=Path("results/classifier_motifs_20260916"))
    args = p.parse_args()
    if args.stage == "prepare": prepare(args.config)
    elif args.stage == "pilot": pilot(args.root)
    elif args.stage == "attribute":
        if not attribute(args.root): raise SystemExit(2)
    else:
        from .analysis import analyze, ism, summarize_ism
        {"analyze":analyze, "ism":ism, "summarize_ism":summarize_ism}[args.stage](args.root)


if __name__ == "__main__": main()
