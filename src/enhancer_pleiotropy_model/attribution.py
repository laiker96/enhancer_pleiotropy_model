"""Integrated-gradients attribution maps for the eight calibrated context probabilities.

For each enhancer and each context, integrated gradients (IG) explain the
calibrated probability of activity ``q_c(x)`` relative to a reference sequence:

* **References** - ``references`` (100) dinucleotide shuffles of the enhancer's
  own 2,048 bp (randomised Euler trail; mono- and dinucleotide counts are kept
  exactly). Shuffle ``r`` of enhancer ``id`` is seeded by ``sha256(seed|id|r)``,
  so references do not depend on batching or order.
* **Integration** - Gauss-Legendre quadrature with ``steps`` (64) nodes along the
  straight path from reference to sequence; ``q_c`` uses the mean of forward and
  reverse-complement probabilities, then the context's Platt calibration.
* **Maps** - ``hypothetical[c, base, position]`` is the path-averaged gradient
  minus its value at the reference base (so it is the contribution any base would
  make at that position); ``actual[c, position]`` projects it onto the observed
  base. Both are averaged over references; per-reference projection happens first.
* **Quality** - completeness ``delta = sum(actual) - (q(x) - q(reference))`` must
  satisfy ``|delta| <= 0.002 + 0.05 |q(x) - q(reference)|`` for every reference.

Outputs are written as resumable chunks of ``enhancers_per_chunk`` enhancers and
can be split across GPUs with ``--shard i --shards n``.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .calibration import load_calibration
from .config import without_private
from .constants import CONTEXTS, INPUT_BP, SPLITS
from .data.enhancers import load_split
from .io import sha256_file, write_json

TARGETS = tuple(f"calibrated_probability_{c}" for c in CONTEXTS)


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


# ---------------------------------------------------------------- references

def seed_for(*values) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, values)).encode()).digest()[:8], "little")


def dinucleotide_shuffle(sequence: np.ndarray, seed: int) -> np.ndarray:
    """Random Euler trail through the sequence's dinucleotide graph (exact di/mononucleotide counts)."""
    sequence = np.asarray(sequence).astype(np.uint8)
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
        raise RuntimeError("Dinucleotide shuffle failed")
    return result


def references_for(codes: np.ndarray, identifier: str, count: int, seed: int):
    values = [dinucleotide_shuffle(codes, seed_for(seed, identifier, r)) for r in range(count)]
    return np.stack(values), np.asarray([hashlib.sha256(v.tobytes()).hexdigest() for v in values], dtype="U64")


def one_hot(codes, device) -> torch.Tensor:
    return F.one_hot(torch.as_tensor(np.asarray(codes).copy(), dtype=torch.long, device=device), 4) \
        .permute(0, 2, 1).float()


# ---------------------------------------------------------------- targets and IG

class CalibratedTargets(nn.Module):
    """Classifier -> eight calibrated probabilities (mean forward/RC probability, then Platt)."""

    def __init__(self, model: nn.Module, calibration: dict) -> None:
        super().__init__()
        if min(calibration["a"]) <= 0:  # raw, temperature and sigmoid all have this form with a > 0
            raise ValueError("Calibration slopes must be positive")
        self.model = model
        self.register_buffer("a", torch.tensor(calibration["a"], dtype=torch.float32))
        self.register_buffer("b", torch.tensor(calibration["b"], dtype=torch.float32))
        self.epsilon = calibration["probability_clip"]
        self.eval()

    def endpoints(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Both orientations in one batch; gradients map back through the flip.
        forward, reverse = self.model(torch.cat((x, x.flip((1, 2))), 0)).chunk(2)
        probability = (forward.sigmoid() + reverse.sigmoid()) * .5
        calibrated = torch.sigmoid(self.a * torch.logit(probability.clamp(self.epsilon, 1 - self.epsilon)) + self.b)
        return {"orientation_logits": torch.stack((forward, reverse), 1), "logits": (forward + reverse) * .5,
                "probabilities": probability, "calibrated_probabilities": calibrated}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.endpoints(x)["calibrated_probabilities"]


@lru_cache(maxsize=8)
def _quadrature(steps: int):
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    return (nodes + 1) / 2, weights / 2


def integrate(target: CalibratedTargets, x: torch.Tensor, baseline: torch.Tensor, steps: int = 64,
              internal_batch: int = 128, target_batch: int = 1) -> dict[str, torch.Tensor]:
    """IG for independent (sequence, reference) pairs ``[n, 4, L]`` and all eight targets."""
    count = len(TARGETS)
    nodes, quadrature = (torch.as_tensor(v, device=x.device, dtype=x.dtype) for v in _quadrature(steps))
    integrated = torch.zeros((len(x), count, 4, x.shape[-1]), device=x.device, dtype=x.dtype)
    directions = torch.eye(count, device=x.device, dtype=x.dtype)
    group = internal_batch // len(x)
    if group < 1:
        raise ValueError("internal_batch must be at least the number of pairs")
    for begin in range(0, steps, group):
        alpha = nodes[begin:begin + group]
        points = (baseline[None] + alpha[:, None, None, None] * (x - baseline)[None]).flatten(0, 1)
        points.requires_grad_(True)

        def outputs(values):
            scores = target(values)
            return scores.sum(0), torch.isfinite(scores).all()

        if target_batch == 1:
            scores, _ = outputs(points)
        else:
            scores, pullback, _ = torch.func.vjp(outputs, points, has_aux=True)
        for first in range(0, count, target_batch):
            end = min(first + target_batch, count)
            if target_batch == 1:
                grad = torch.autograd.grad(scores[first], points, retain_graph=end < count)[0][None]
            else:
                grad = torch.vmap(lambda d: pullback(d, retain_graph=end < count, create_graph=False)[0])(
                    directions[first:end])
            grad = grad.reshape(end - first, len(alpha), *x.shape)
            integrated[:, first:end] += (grad * quadrature[None, begin:begin + len(alpha), None, None, None]
                                         ).sum(1).transpose(0, 1)
    hypothetical = integrated - (integrated * baseline[:, None]).sum(2, keepdim=True)
    actual = (hypothetical * x[:, None]).sum(2)
    with torch.no_grad():
        difference = target(x) - target(baseline)
    result = {"hypothetical": hypothetical, "actual": actual, "delta": actual.sum(2) - difference,
              "target_difference": difference}
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("Non-finite attribution")
    return {key: value.detach() for key, value in result.items()}


def quality(delta: np.ndarray, difference: np.ndarray) -> np.ndarray:
    return np.abs(delta) <= .002 + .05 * np.abs(difference)


# ---------------------------------------------------------------- chunks

def _new_state(n: int, references: int) -> dict[str, np.ndarray]:
    t = len(TARGETS)
    return {"count": np.asarray(0),
            "sum_hyp": np.zeros((n, t, 4, INPUT_BP)), "sum_sq_hyp": np.zeros((n, t, 4, INPUT_BP)),
            "sum_half_actual": np.zeros((n, 2, t, INPUT_BP)), "sum_sq_breadth": np.zeros((n, INPUT_BP)),
            "delta": np.zeros((n, references, t), np.float32),
            "target_difference": np.zeros((n, references, t), np.float32),
            "reference_orientation_logits": np.zeros((n, references, 2, 8), np.float32),
            "reference_probabilities": np.zeros((n, references, 8), np.float32),
            "reference_calibrated_probabilities": np.zeros((n, references, 8), np.float32)}


def _save(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def attribute_chunk(target: CalibratedTargets, cohort: dict, indices: np.ndarray, config: dict, path: Path,
                    should_stop=lambda: False) -> dict:
    """All references for the enhancers ``indices``; resumes from ``<path>.state.npz``."""
    device = next(target.parameters()).device
    count, block = int(config["references"]), int(config["reference_block"])
    state_path = path.with_suffix(".state.npz")
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as saved:
            state = dict(saved)
    else:
        state = _new_state(len(indices), count)
    codes = cohort["sequence"][indices]
    made = [references_for(codes[k], str(cohort["ids"][i]), count, int(config["seed"])) for k, i in enumerate(indices)]
    references = np.stack([m[0] for m in made])
    hashes = np.stack([m[1] for m in made])
    x = one_hot(codes, device)
    for begin in range(int(state["count"]), count, block):
        end = min(begin + block, count)
        rows = np.repeat(np.arange(len(indices)), end - begin)
        refs = np.tile(np.arange(begin, end), len(indices))
        results = []
        for first in range(0, len(rows), int(config["pair_batch"])):
            if should_stop():
                _save(state_path, **state)
                raise TimeoutError("stopped; rerun to resume")
            rr, rf = rows[first:first + config["pair_batch"]], refs[first:first + config["pair_batch"]]
            baseline = one_hot(references[rr, rf], device)
            out = integrate(target, x[rr], baseline, int(config["steps"]), int(config["internal_batch"]),
                            int(config["target_batch"]))
            with torch.no_grad():
                out.update({f"reference_{k}": v for k, v in target.endpoints(baseline).items() if k != "logits"})
            results.append((rr, rf, {k: v.cpu().numpy() for k, v in out.items()}))
        # Fixed pair order and float64 sums keep results independent of batching.
        for rr, rf, values in results:
            for pair, (row, reference) in enumerate(zip(rr, rf)):
                hyp = values["hypothetical"][pair].astype(np.float64)
                actual = values["actual"][pair].astype(np.float64)
                state["sum_hyp"][row] += hyp
                state["sum_sq_hyp"][row] += hyp * hyp
                state["sum_half_actual"][row, int(reference >= count // 2)] += actual
                state["sum_sq_breadth"][row] += actual.sum(0) ** 2
                for key in ("delta", "target_difference"):
                    state[key][row, reference] = values[key][pair]
                for key in ("orientation_logits", "probabilities", "calibrated_probabilities"):
                    state[f"reference_{key}"][row, reference] = values[f"reference_{key}"][pair]
        state["count"] = np.asarray(end)
        if end < count:  # a crash loses at most one reference block
            _save(state_path, **state)

    mean = state["sum_hyp"] / count
    variance = np.maximum(0, (state["sum_sq_hyp"] - state["sum_hyp"] ** 2 / count) / (count - 1))
    hypothetical, hypothetical_se = mean.astype(np.float32), np.sqrt(variance / count).astype(np.float32)
    project = lambda v: np.take_along_axis(v, codes[:, None, None, :], axis=2)[:, :, 0]  # noqa: E731
    breadth = project(mean).sum(1)
    breadth_se = np.sqrt(np.maximum(0, (state["sum_sq_breadth"] - count * breadth ** 2) / (count - 1)) / count)
    with torch.no_grad():
        endpoints = {k: v.cpu().numpy() for k, v in target.endpoints(x).items()}
    passed = quality(state["delta"], state["target_difference"])
    breadth_passed = (np.abs(state["delta"].sum(2)) <= .02 + .05 * np.abs(state["target_difference"].sum(2)))
    result = dict(
        indices=indices, targets=np.asarray(TARGETS), references=np.asarray(count), steps=np.asarray(config["steps"]),
        hypothetical=hypothetical, actual=project(hypothetical), hypothetical_reference_se=hypothetical_se,
        actual_reference_se=project(hypothetical_se), breadth_reference_se=breadth_se.astype(np.float32),
        actual_reference_halves=(state["sum_half_actual"] / (count // 2)).astype(np.float32),
        delta=state["delta"], target_difference=state["target_difference"], reference_quality_pass=passed,
        quality_pass=passed.all(1), breadth_quality_pass=breadth_passed.all(1), reference_hashes=hashes,
        **endpoints, **{key: cohort[key][indices] for key in
                        ("ids", "labels", "split", "chrom", "summit", "sequence", "native_start",
                         "native_end", "native_offset", "native_length")},
        **{key: state[key] for key in ("reference_orientation_logits", "reference_probabilities",
                                       "reference_calibrated_probabilities")})
    _save(path, **result)
    state_path.unlink(missing_ok=True)
    return result


# ---------------------------------------------------------------- driver

def load_cohort(data: Path) -> dict[str, np.ndarray]:
    """All retained enhancers (train, then validation, then test), with native intervals."""
    parts = {split: load_split(data, split) for split in SPLITS}
    cohort = {key: np.concatenate([parts[s][key] for s in SPLITS])
              for key in ("ids", "sequence", "labels", "chrom", "summit")}
    cohort["split"] = np.concatenate([np.full(len(parts[s]["ids"]), s) for s in SPLITS])
    table = pd.read_csv(Path(data) / "enhancers.tsv.gz", sep="\t").set_index("master_dhs_id")
    native = table.loc[cohort["ids"], ["start", "end"]].to_numpy(np.int64)
    cohort["native_start"], cohort["native_end"] = native[:, 0], native[:, 1]
    cohort["native_offset"] = cohort["native_start"] - (cohort["summit"] - INPUT_BP // 2)
    cohort["native_length"] = cohort["native_end"] - cohort["native_start"]
    return cohort


def run_attribution(config: dict, shard: int = 0, shards: int = 1, limit: int | None = None) -> Path:
    from .models import load_classifier, use_interleaved_dilation
    from .training.common import resolve_device

    device = resolve_device(config["device"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(config["output"])
    (output / "chunks").mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config["classifier"])
    calibration_path = Path(config.get("calibration") or checkpoint.parent / "calibration.json")
    model, _ = load_classifier(checkpoint)
    if config.get("interleaved_dilation", True):
        use_interleaved_dilation(model)
    target = CalibratedTargets(model.requires_grad_(False), load_calibration(calibration_path)).to(device)
    cohort = load_cohort(Path(config["data"]))
    write_json(output / "config.json", {**without_private(config), "classifier_sha256": sha256_file(checkpoint),
                                        "calibration_sha256": sha256_file(calibration_path),
                                        "targets": list(TARGETS), "enhancers": int(len(cohort["ids"]))})
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGUSR1):  # e.g. Slurm time limit: finish the batch, save, exit
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    size = int(config["enhancers_per_chunk"])
    starts = [s for k, s in enumerate(range(0, len(cohort["ids"]), size)) if k % shards == shard]
    if limit is not None:
        starts = starts[:limit]
    log(f"shard {shard + 1}/{shards}: {len(starts)} chunks of {size} enhancers, device {device}")
    failures, done = np.zeros(len(TARGETS), int), 0
    for number, start in enumerate(starts, 1):
        path = output / "chunks" / f"chunk_{start:06d}.npz"
        indices = np.arange(start, min(start + size, len(cohort["ids"])))
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                failures += (~saved["quality_pass"]).sum(0)
        else:
            began = time.monotonic()
            result = attribute_chunk(target, cohort, indices, config, path, lambda: stopped[0])
            failures += (~result["quality_pass"]).sum(0)
            log(f"chunk {number}/{len(starts)} ({path.name}) in {time.monotonic() - began:.0f} s; "
                f"enhancers failing the completeness check per context: {(~result['quality_pass']).sum(0).tolist()}")
        done += len(indices)
    summary = output / f"shard_{shard}_of_{shards}.json"
    write_json(summary, {"status": "complete" if limit is None else "partial", "shard": shard, "shards": shards,
                         "enhancers": done, "quality_failures": dict(zip(CONTEXTS, failures.tolist()))})
    log(f"done: {done} enhancers -> {output / 'chunks'}")
    return summary
