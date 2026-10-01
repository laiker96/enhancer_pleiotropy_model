"""Full-input, eight-context IG and bounded-pilot utilities. No training."""
from functools import lru_cache

import numpy as np
import torch

from .attribution import ensemble


CONTEXTS = ("ab", "e13", "e5", "ead", "hid", "lb", "o", "wid")
REGIONS = ("native", "inner_flanks", "middle_flanks", "outer_flanks")


@lru_cache(maxsize=8)
def _quadrature(steps):
    """Reuse the same CPU quadrature across references; never cache model graphs."""
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    return (nodes+1)/2, weights/2


def _context_targets(model, points):
    logits, _ = ensemble(model, points)
    if logits.shape != (len(points), 8):
        raise ValueError("Expected eight finite classifier logits")
    return logits.sum(0), torch.isfinite(logits).all()


def context_integrated_gradients(model, x, baseline, steps=64, internal_batch=32,
                                *, context_batch=8):
    """Batch context vector-Jacobian products while sharing each forward graph.

    Returns [example, context, ACGT, position] hypothetical scores, and the
    observed-base projection [example, context, position]. All targets use
    identical references/nodes. Reverse-complement gradients map back through
    the differentiable input flip in ensemble(). ``internal_batch`` caps the
    number of interpolated inputs; ``context_batch`` controls the independent
    derivative directions processed together through torch.func.vjp + vmap
    (1 retains the sequential path). Higher-order graph creation is disabled.
    Larger batches trade memory for fewer autograd calls, not fewer references
    or integration nodes. Callers must record both settings in new run configs.
    """
    if (x.ndim != 3 or not len(x) or x.shape[1] != 4 or x.shape != baseline.shape
            or steps < 1 or internal_batch < len(x) or model.training):
        raise ValueError("Matching one-hot inputs, eval model and valid integration sizes required")
    if not isinstance(context_batch, int) or not 1 <= context_batch <= 8:
        raise ValueError("context_batch must be an integer from 1 to 8")
    # Reduce validation to one host synchronization instead of six per reference.
    endpoints_valid = torch.stack([
        torch.isfinite(value).all() & ((value == 0) | (value == 1)).all()
        & (value.sum(1) == 1).all() for value in (x, baseline)]).all()
    if not endpoints_valid:
        raise ValueError("Expected finite, normalized one-hot ACGT endpoints")
    nodes, quadrature = _quadrature(steps)
    alpha = torch.as_tensor(nodes, device=x.device, dtype=x.dtype)
    weights = torch.as_tensor(quadrature, device=x.device, dtype=x.dtype)
    integrated = torch.zeros((len(x), 8, 4, x.shape[2]), device=x.device, dtype=x.dtype)
    directions = torch.eye(8, device=x.device, dtype=x.dtype)
    logits_finite = torch.ones((), device=x.device, dtype=torch.bool)
    group = max(1, internal_batch//len(x))
    for start in range(0, steps, group):
        a = alpha[start:start+group]
        points = (baseline[None]+a[:, None, None, None]*(x-baseline)[None]).flatten(0, 1).requires_grad_(True)
        if context_batch == 1:
            targets, finite = _context_targets(model, points)
        else:
            targets, pullback, finite = torch.func.vjp(
                lambda values: _context_targets(model, values), points, has_aux=True)
        # Keep the check on-device until the final validation, not inside the loop.
        logits_finite &= finite
        for context in range(0, 8, context_batch):
            end = min(context+context_batch, 8)
            if context_batch == 1:
                grad = torch.autograd.grad(targets[context], points,
                                           retain_graph=end < 8)[0].unsqueeze(0)
            else:
                # The older is_grads_batched backend falls back to sequential
                # operators for this CNN. Modern vmap has native batching rules.
                # Do not build higher-order graphs or retain the final graph.
                grad = torch.vmap(lambda direction: pullback(direction,
                    retain_graph=end < 8, create_graph=False)[0])(directions[context:end])
            grad = grad.reshape(end-context, len(a), *x.shape)
            weighted = grad*weights[None, start:start+len(a), None, None, None]
            integrated[:, context:end] += weighted.sum(1).transpose(0, 1)
        if context_batch != 1:
            del pullback
    hyp = integrated-(integrated*baseline[:, None]).sum(2, keepdim=True)
    actual = (hyp*x[:, None]).sum(2)
    with torch.no_grad():
        original, probabilities = ensemble(model, x)
        reference, _ = ensemble(model, baseline)
    difference = original-reference
    result = dict(hypothetical=hyp, actual=actual, delta=actual.sum(2)-difference,
                  target_difference=difference, logits=original, probabilities=probabilities)
    if not torch.stack([logits_finite, *(torch.isfinite(value).all() for value in result.values())]).all():
        raise ValueError("Nonfinite attribution")
    return {key: value.detach() for key, value in result.items()}


def region_masks(offset, length, input_bp=2048):
    if input_bp != 2048 or not 0 <= offset < offset+length <= input_bp:
        raise ValueError("Native enhancer must lie inside the full2048 input")
    positions = np.arange(input_bp)
    native = (positions >= offset) & (positions < offset+length)
    inner = (positions >= 768) & (positions < 1280)
    middle = (positions >= 512) & (positions < 1536)
    masks = np.stack([native, inner & ~native, middle & ~inner & ~native,
                      ~middle & ~native])
    if not np.all(masks.sum(0) == 1):
        raise ValueError("Regions must partition the input without overlap")
    return masks


def pilot_indices(data, intervals, per_degree, seed):
    """Training-only balanced degrees, round-robin length/GC/label strata.

    This is deliberately a numerical stress-test sample, not prevalence data.
    Interleave degrees so a wall-limited prefix still spans degrees1..8.
    """
    labels = data["labels"]
    if (labels.shape != (len(data["ids"]), 8) or not np.isin(labels, [0, 1]).all()
            or not labels.any(1).all() or per_degree < 1
            or len(np.unique(data["ids"])) != len(labels)
            or not np.array_equal(data["ids"], intervals["ids"])):
        raise ValueError("Valid unique enhancer IDs, labels and aligned intervals required")
    sequence = data["sequence"]
    if sequence.shape != (len(labels), 2048) or not np.isin(sequence, range(4)).all():
        raise ValueError("Expected full2048 ACGT codes")
    gc = ((sequence == 1) | (sequence == 2)).mean(1)
    degree = labels.sum(1)
    rng = np.random.default_rng(seed)
    selected = []
    for k in range(1, 9):
        eligible = np.flatnonzero((degree == k) & (data["split"] == "train"))
        if len(eligible) < per_degree:
            raise ValueError("Too few training enhancers in degree "+str(k))
        length_bin = np.searchsorted(np.quantile(intervals["length"][eligible], [.25, .5, .75]), intervals["length"][eligible])
        gc_bin = np.searchsorted(np.quantile(gc[eligible], [.25, .5, .75]), gc[eligible])
        strata = {}
        for j in rng.permutation(len(eligible)):
            i = eligible[j]
            key = (int(length_bin[j]), int(gc_bin[j]), tuple(labels[i].tolist()))
            strata.setdefault(key, []).append(int(i))
        keys = list(strata)
        order = rng.permutation(len(keys))
        chosen = []
        while len(chosen) < per_degree:
            for j in order:
                if strata[keys[j]]:
                    chosen.append(strata[keys[j]].pop())
                    if len(chosen) == per_degree:
                        break
        selected.append(chosen)
    return np.asarray(selected, dtype=np.int64).T.ravel()


def map_agreement(first, second):
    """Per-context map diagnostics; undefined cosine/sign scores stay missing."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.shape != second.shape or first.ndim != 2 or not np.isfinite([first, second]).all():
        raise ValueError("Matching finite context-by-position arrays required")
    rows = []
    for a, b in zip(first, second):
        norm = np.linalg.norm(a)*np.linalg.norm(b)
        magnitude = np.maximum(np.abs(a), np.abs(b))
        count = max(1, int(np.ceil(.1*len(a)))) if len(a) else 0
        ai = np.argsort(np.abs(a), kind="stable")[-count:] if count else []
        bi = np.argsort(np.abs(b), kind="stable")[-count:] if count else []
        union = np.union1d(ai, bi)
        rows.append(dict(cosine=float(a@b/norm) if norm else None,
            weighted_sign_agreement=float(magnitude[np.sign(a) == np.sign(b)].sum()/magnitude.sum()) if magnitude.sum() else None,
            top10pct_jaccard=float(len(np.intersect1d(ai, bi))/len(union)) if norm and len(union) else None,
            mean_absolute_difference=float(np.abs(a-b).mean()) if len(a) else None))
    return rows


def ism_positions(actual, masks, seed):
    """Two largest all-context scores and two random other bases per compartment."""
    rng = np.random.default_rng(seed)
    score = np.abs(actual).mean(0)
    chosen = []
    for mask in masks:
        valid = np.flatnonzero(mask)
        top = valid[np.argsort(-score[valid], kind="stable")[:2]]
        rest = np.setdiff1d(valid, top)
        chosen.extend(top.tolist())
        chosen.extend(rng.choice(rest, min(2, len(rest)), replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def exact_ism(model, x, positions, batch_size=16):
    if len(x) != 1 or batch_size < 1 or model.training:
        raise ValueError("ISM expects one input and an eval model")
    if len(set(map(int, positions))) != len(positions) or any(p < 0 or p >= x.shape[2] for p in positions):
        raise ValueError("Invalid or duplicate mutation position")
    codes = x.argmax(1)[0]
    mutations = [(int(p), base) for p in positions for base in range(4) if base != int(codes[p])]
    values = []
    with torch.no_grad():
        wt, _ = ensemble(model, x)
        for start in range(0, len(mutations), batch_size):
            batch = mutations[start:start+batch_size]
            mutated = x.repeat(len(batch), 1, 1)
            for j, (position, base) in enumerate(batch):
                mutated[j, :, position] = 0
                mutated[j, base, position] = 1
            values.append((ensemble(model, mutated)[0]-wt).cpu().numpy())
    return dict(positions=np.asarray([p for p, _ in mutations], np.int64),
                alternate=np.asarray([b for _, b in mutations], np.uint8),
                delta_logits=np.concatenate(values) if values else np.empty((0, 8), np.float32))
