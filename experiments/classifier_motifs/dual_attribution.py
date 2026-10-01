"""Two IG targets with the original fixed-50-reference scientific protocol."""
import numpy as np
import torch

from .attribution import active_weights, ensemble
from .context_attribution import _quadrature


TARGETS = ('mean_active_logit', 'soft_breadth')


def targets(model, x, weights):
    logits, probabilities = ensemble(model, x)
    # Preserve the existing RC ensemble: mean(sigmoid(logits)), not
    # sigmoid(mean(logits)). The observed-active mask never changes on the path.
    return torch.stack(((logits*weights).sum(1), probabilities.sum(1)), 1)


def dual_integrated_gradients(model, x, baseline, labels, steps=32,
                              internal_batch=64, should_stop=lambda: False):
    if (model.training or x.ndim != 3 or not len(x) or x.shape[1] != 4
            or x.shape != baseline.shape or len(labels) != len(x)
            or internal_batch < len(x) or steps < 1):
        raise ValueError('Aligned inputs, eval model and valid integration batch required')
    weights = active_weights(labels)
    if not torch.stack([torch.isfinite(v).all() & ((v == 0) | (v == 1)).all()
                        & (v.sum(1) == 1).all() for v in (x, baseline)]).all():
        raise ValueError('Expected finite normalized one-hot endpoints')
    nodes, quadrature = _quadrature(steps)
    nodes = torch.as_tensor(nodes, device=x.device, dtype=x.dtype)
    quadrature = torch.as_tensor(quadrature, device=x.device, dtype=x.dtype)
    integrated = torch.zeros((len(x), 2, 4, x.shape[-1]), device=x.device, dtype=x.dtype)
    group = internal_batch//len(x)
    finite = torch.ones((), device=x.device, dtype=torch.bool)
    for begin in range(0, steps, group):
        if should_stop():
            raise TimeoutError('Stopped at integration batch boundary')
        a = nodes[begin:begin+group]
        points = (baseline[None]+a[:, None, None, None]*(x-baseline)[None]).flatten(0, 1).requires_grad_(True)
        output = targets(model, points, weights.repeat(len(a), 1))
        finite &= torch.isfinite(output).all()
        for target in range(2):
            grad = torch.autograd.grad(output[:, target].sum(), points, retain_graph=target == 0)[0]
            integrated[:, target] += (grad.reshape(len(a), *x.shape)
                *quadrature[begin:begin+len(a), None, None, None]).sum(0)
    hyp = integrated-(integrated*baseline[:, None]).sum(2, keepdim=True)
    actual = (hyp*x[:, None]).sum(2)
    with torch.no_grad():
        difference = targets(model, x, weights)-targets(model, baseline, weights)
    result = dict(hypothetical=hyp, actual=actual,
                  delta=actual.sum(2)-difference, target_difference=difference)
    if not torch.stack([finite, *(torch.isfinite(v).all() for v in result.values())]).all():
        raise ValueError('Nonfinite dual attribution')
    return {key:value.detach() for key,value in result.items()}


def refine_dual(model, x, baseline, labels, config, should_stop=lambda: False):
    """Original adaptive rule by default; an explicit empty grid fixes resolution."""
    refinements = config.get('refinement_steps', (64, 128))
    if 'refinement_steps' in config and any(not isinstance(s, int) or s <= config['steps'] for s in refinements):
        raise ValueError('Refinement points must exceed the starting resolution')
    if list(refinements) != sorted(set(refinements)):
        raise ValueError('Refinement grid must be strictly increasing')
    result = dual_integrated_gradients(model, x, baseline, labels, config['steps'],
                                      config['internal_batch'], should_stop)
    used = torch.full((len(x), 2), config['steps'], device=x.device, dtype=torch.int32)
    for steps in refinements:
        bad = result['delta'].abs() > (config['absolute_tolerance']
            +config['relative_tolerance']*result['target_difference'].abs())
        rows = torch.where(bad.any(1))[0]
        if not len(rows):
            break
        refined = dual_integrated_gradients(model, x[rows], baseline[rows], labels[rows],
                                           steps, config['internal_batch'], should_stop)
        # A second target needing refinement must not silently change the old
        # target's accepted grid, output, or recorded steps.
        for target in range(2):
            selected = bad[rows, target]
            for key in result:
                result[key][rows[selected], target] = refined[key][selected, target]
            used[rows[selected], target] = steps
    result['steps'] = used
    result['quality_pass'] = result['delta'].abs() <= (config['absolute_tolerance']
        +config['relative_tolerance']*result['target_difference'].abs())
    return {key:value.cpu().numpy() for key,value in result.items()}
