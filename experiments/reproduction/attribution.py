"""Full-length context IG, with a mandatory numerical pilot on each GPU type."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from classifier_motifs.attribution import one_hot
from classifier_motifs.calibrated_attribution import CalibratedTargets, integrate
from classifier_motifs.context_attribution import pilot_indices
from classifier_modisco.calibrated_context import score_chunk, shuffled, compare_grid
from classifier_transfer.data import load_split
from .common import digest, provenance, require_slurm, selected_run, write_json
from .training import load_classifier


def validate_settings(settings):
    if (settings["references"] != 100 or settings["steps"] != 64
            or settings["reference_block"] < 1 or settings["enhancer_batch"] < 1
            or settings["pair_batch"] < 1 or settings["internal_batch"] < settings["pair_batch"]
            or not 1 <= settings["target_batch"] <= 9 or settings["shards"] < 1
            or settings["max_seconds"] <= 0):
        raise ValueError("Expected fixed IG64/100 references and valid computational batches")


def run(cfg, shard, pilot=False):
    require_slurm(gpu=True)
    settings = cfg["attribution"]
    validate_settings(settings)
    if not 0 <= shard < settings["shards"]:
        raise ValueError("Invalid attribution shard")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    work = Path(cfg["paths"]["work"])
    checkpoint = selected_run(cfg) / "best_model.pt"
    paths = [checkpoint, work / "calibrators.json", work / "catalog/cohort.npz", work / "catalog/intervals.npz"]
    record = provenance(paths, settings)
    signature = record["signature"]
    root = work / "attribution"
    if pilot:
        root.mkdir(exist_ok=True)
        if (root / "manifest.json").exists():
            if json.loads((root / "manifest.json").read_text()) != record:
                raise ValueError("Attribution inputs/code/settings changed; use a new output root")
        else:
            write_json(root / "manifest.json", record)
    elif json.loads((root / "manifest.json").read_text()) != record:
        raise ValueError("Attribution pilot signature changed")
    with np.load(paths[2], allow_pickle=False) as f:
        data = dict(f)
    with np.load(paths[3], allow_pickle=False) as f:
        intervals = dict(f)
    np.testing.assert_array_equal(data["ids"], intervals["ids"])
    model = load_classifier(checkpoint)
    calibration = json.loads(paths[1].read_text())
    target = CalibratedTargets(model, calibration, expected_checkpoint_sha256=digest(checkpoint)).cuda().eval()
    batches = {k: settings[k] for k in ("pair_batch", "internal_batch", "target_batch")}
    hardware = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda)
    import hashlib
    hardware_key = hashlib.sha256(json.dumps(hardware, sort_keys=True).encode()).hexdigest()[:16]
    pilot_dir = root / ("pilot_" + hardware_key)
    interrupted = False
    def stop_signal(*_):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGUSR1, stop_signal)
    deadline = time.monotonic() + settings["max_seconds"]
    def stop():
        return interrupted or time.monotonic() >= deadline or (root / "STOP").exists()
    workers = min(4, max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        if pilot:
            pilot_dir.mkdir(exist_ok=True)
            indices = pilot_indices(data, intervals, 2, settings["seed"])
            x = one_hot(data["sequence"][indices], "cuda")
            b = one_hot(np.stack([shuffled((data["sequence"][i], str(data["ids"][i]), 0, settings["seed"]))[0]
                                 for i in indices]), "cuda")
            labels = torch.as_tensor(data["labels"][indices], device="cuda")
            baseline = integrate(target, x, b, labels, 64, len(x), 1, stop)
            candidate = integrate(target, x, b, labels, 64, max(len(x), batches["internal_batch"]), batches["target_batch"], stop)
            for key in baseline:
                torch.testing.assert_close(baseline[key], candidate[key], atol=3e-5, rtol=2e-3)
            high = integrate(target, x, b, labels, 128, max(len(x), batches["internal_batch"]), batches["target_batch"], stop)
            agreement = compare_grid(candidate["actual"].cpu().numpy(), high["actual"].cpu().numpy(), data, intervals, indices)
            eligible = [r for r in agreement if r["region"] == "native" and not r["tiny"]]
            grid_fraction = sum(r["cosine"] is not None and r["cosine"] >= .99 for r in eligible) / max(1, len(eligible))
            validation = load_split(work / "enhancers", "validation")
            with np.load(selected_run(cfg) / "best_validation_predictions.npz", allow_pickle=False) as f:
                np.testing.assert_array_equal(f["ids"], validation["ids"])
                rows = np.linspace(0, len(validation["ids"])-1, min(64, len(validation["ids"])), dtype=int)
                with torch.no_grad():
                    replay = target.endpoints(one_hot(validation["sequence"][rows], "cuda"))["probabilities"].cpu().numpy()
                endpoint_error = float(np.max(np.abs(replay - f["probabilities"][rows])))
            del baseline, candidate, high, x, b, labels
            started = time.monotonic()
            result = score_chunk(target, data, intervals, indices, settings, batches,
                                 pilot_dir / "reference_maps.npz", signature, pool, stop)
            fractions = result["quality_pass"].mean(0)
            breadth_fraction = float(result["breadth_quality_pass"].mean())
            passed = grid_fraction >= .95 and (fractions >= .95).all() and breadth_fraction >= .95 and endpoint_error <= .005
            write_json(pilot_dir / "report.json", dict(status="passed" if passed else "failed", signature=signature,
                hardware=hardware, settings=batches, integration_native_fraction_cosine_ge_099=grid_fraction,
                quality_pass_fraction=fractions.tolist(), breadth_pass_fraction=breadth_fraction,
                endpoint_max_error=endpoint_error, seconds=time.monotonic()-started,
                maps_sha256=digest(pilot_dir / "reference_maps.npz"), agreement=agreement))
            if not passed:
                raise ValueError("Numerical pilot failed; full attribution is blocked")
            return
        gate = json.loads((pilot_dir / "report.json").read_text())
        if (gate["status"] != "passed" or gate["signature"] != signature or gate["hardware"] != hardware
                or gate["maps_sha256"] != digest(pilot_dir / "reference_maps.npz")):
            raise ValueError("Need a passing pilot on this GPU/software combination")
        directory = root / "chunks"
        directory.mkdir(exist_ok=True)
        hashes, failures, count = {}, np.zeros(9, int), 0
        progress = dict(shard=shard, elements=0, quality_failures=failures.tolist(), signature=signature)
        for batch, start in enumerate(range(0, len(data["ids"]), settings["enhancer_batch"])):
            if batch % settings["shards"] != shard:
                continue
            if stop():
                raise TimeoutError("Checkpointed stop requested")
            indices = np.arange(start, min(start + settings["enhancer_batch"], len(data["ids"])))
            path = directory / ("chunk_%06d.npz" % start)
            result = score_chunk(target, data, intervals, indices, settings, batches, path, signature, pool, stop)
            count += len(indices); failures += (~result["quality_pass"]).sum(0)
            hashes[path.name] = digest(path)
            progress = dict(shard=shard, elements=count, quality_failures=failures.tolist(), signature=signature)
            write_json(root / f"progress_{shard}.json", progress)
            print(json.dumps(progress), flush=True)
            if count >= 256 and (1-failures/count < .95).any():
                raise ValueError("Numerical quality below 95%; maps retained for inspection")
        write_json(root / f"shard_{shard}.json", dict(status="complete", **progress, chunks=hashes))
