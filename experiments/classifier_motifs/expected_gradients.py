"""Reference/path sampling for eight-context expected gradients, without noise.

The target is the uniform mean IG over a FIXED reference pool, not an assertion
of exact Shapley values. Hypothetical channels retain the existing per-reference
one-hot projection; projection must precede averaging over references.
"""
import numpy as np
import torch

from .context_attribution import _context_targets, _quadrature


def sampling_plan(references, samples, seed, scheme='iid'):
    if references < 1 or samples < 1 or scheme not in ('iid', 'stratified'):
        raise ValueError('Positive counts and iid/stratified sampling required')
    rng = np.random.default_rng(seed)
    if scheme == 'iid':
        ids = rng.integers(references, size=samples)
        alpha = rng.random(samples)
        weights = np.full(samples, 1/samples)
    else:
        if samples < references:
            raise ValueError('Stratified plan must cover every reference')
        counts = np.full(references, samples//references)
        counts[rng.permutation(references)[:samples % references]] += 1
        ids = np.repeat(np.arange(references), counts)
        alpha = np.concatenate([(np.arange(n)+rng.random(n))/n for n in counts])
        # Unequal counts must NOT change the uniform reference-pool target.
        weights = np.repeat(1/(references*counts), counts)
        order = rng.permutation(samples)
        ids, alpha, weights = ids[order], alpha[order], weights[order]
    return dict(reference=ids, alpha=alpha, weights=weights)


def quadrature_plan(references, steps):
    if references < 1 or steps < 1:
        raise ValueError('Positive reference and integration counts required')
    alpha, weights = _quadrature(steps)
    return dict(reference=np.repeat(np.arange(references), steps),
                alpha=np.tile(alpha, references), weights=np.tile(weights/references, references))


def stack_plans(plans):
    return {key:np.stack([p[key] for p in plans]) for key in ('reference', 'alpha', 'weights')}


def context_sampled_gradients(model, x, references, plan, internal_batch=1024,
                              should_stop=lambda: False):
    """Return [example, context, base, position] maps from weighted path draws.

    x: [N,4,L], references: [N,R,4,L], plan arrays: [N,S]. Shared forward/RC
    computation, sequential context gradients. No smoothing, mixed precision,
    reference substitution, or cross-example averaging. Quadrature plans also
    allow direct comparison with the pre-existing reference-averaged IG kernel.
    """
    if (x.ndim != 3 or not len(x) or x.shape[1] != 4 or references.ndim != 4
            or references.shape[0] != len(x) or references.shape[2:] != x.shape[1:]
            or not references.shape[1] or internal_batch < len(x) or model.training):
        raise ValueError('Aligned one-hot examples/references, eval model and valid batch required')
    ids, alpha, weights = (np.asarray(plan[k]) for k in ('reference', 'alpha', 'weights'))
    if (ids.ndim != 2 or ids.shape[0] != len(x) or not ids.shape[1]
            or alpha.shape != ids.shape or weights.shape != ids.shape
            or not np.issubdtype(ids.dtype, np.integer) or ids.min() < 0
            or ids.max() >= references.shape[1] or not np.isfinite(alpha).all()
            or not np.isfinite(weights).all() or (alpha < 0).any() or (alpha > 1).any()
            or (weights < 0).any() or not np.allclose(weights.sum(1), 1, atol=1e-12, rtol=0)):
        raise ValueError('Invalid sampling plan')
    valid = torch.stack([torch.isfinite(v).all() & ((v == 0) | (v == 1)).all()
                         & (v.sum(-2) == 1).all() for v in (x, references)]).all()
    if not valid:
        raise ValueError('Expected normalized finite one-hot endpoints')
    ids = torch.as_tensor(ids, device=x.device)
    alpha = torch.as_tensor(alpha, device=x.device, dtype=x.dtype)
    weights = torch.as_tensor(weights, device=x.device, dtype=x.dtype)
    hyp = torch.zeros((len(x), 8, 4, x.shape[-1]), device=x.device, dtype=x.dtype)
    examples = torch.arange(len(x), device=x.device)[None]
    group = internal_batch//len(x)
    finite = torch.ones((), dtype=torch.bool, device=x.device)
    for start in range(0, ids.shape[1], group):
        if should_stop():
            raise TimeoutError('Expected-gradients sampling stopped at batch boundary')
        selected = references[examples, ids[:, start:start+group].T]
        a = alpha[:, start:start+group].T[:, :, None, None]
        w = weights[:, start:start+group].T[:, :, None, None]
        points = (selected+a*(x[None]-selected)).flatten(0, 1).requires_grad_(True)
        targets, batch_finite = _context_targets(model, points)
        finite &= batch_finite
        for context in range(8):
            grad = torch.autograd.grad(targets[context], points, retain_graph=context < 7)[0]
            grad = grad.reshape(*selected.shape)
            projected = grad-(grad*selected).sum(2, keepdim=True)
            hyp[:, context] += (projected*w).sum(0)
    actual = (hyp*x[:, None]).sum(2)
    if not torch.stack([finite, torch.isfinite(hyp).all(), torch.isfinite(actual).all()]).all():
        raise ValueError('Nonfinite expected gradients')
    return dict(hypothetical=hyp.detach(), actual=actual.detach())
