"""Independent adaptive integration for a batch of enhancer-reference pairs."""
import numpy as np

from classifier_motifs.context_attribution import (
    REGIONS, context_integrated_gradients, map_agreement)


def refine_pair_batch(model, x, baseline, masks, config, anchors, should_stop=lambda: False):
    """Return one old-style (result, steps, passed, comparisons) tuple per pair.

    Only unfinished pairs advance to the next integration grid. Reference
    generation and averaging remain outside this function; no pair is dropped
    or combined with another. The model must not couple examples in eval mode.
    """
    count = len(x)
    masks = np.asarray(masks, dtype=bool)
    if (not count or x.shape != baseline.shape or masks.shape != (count, 4, x.shape[-1])
            or len(anchors) != count or config['internal_batch'] < count):
        raise ValueError('Unaligned pairs, masks, anchors or too-small integration batch')
    grids = config['integration_steps']
    if not grids or any(n < 1 for n in grids) or sorted(set(grids)) != list(grids):
        raise ValueError('Integration grids must be positive and strictly increasing')
    pending = list(range(count))
    results, previous = [None]*count, [None]*count
    comparisons = [[] for _ in range(count)]
    for steps in grids:
        if should_stop():
            raise TimeoutError('Pair-batch budget/stop requested')
        values = context_integrated_gradients(model, x[pending], baseline[pending],
            steps, config['internal_batch'], context_batch=config.get('context_batch', 1))
        tolerance = config['absolute_tolerance']+config['relative_tolerance']*values['target_difference'].abs()
        passed = values['delta'].abs() <= tolerance
        acceptable = passed.all(1).cpu().tolist()
        actual = values['actual'].cpu().numpy()
        remaining = []
        for j, i in enumerate(pending):
            if previous[i] is not None:
                last_steps, first = previous[i]
                comparisons[i].append(dict(from_steps=last_steps, to_steps=steps,
                    full=map_agreement(first, actual[j]),
                    regions={name: map_agreement(first[:, mask], actual[j][:, mask])
                             for name, mask in zip(REGIONS, masks[i])}))
            results[i] = ({key:value[j:j+1] for key,value in values.items()},
                          int(steps), passed[j:j+1], comparisons[i])
            if not (acceptable[j] and (not anchors[i] or steps >= 128)):
                remaining.append(i)
                previous[i] = (steps, actual[j].copy())
        pending = remaining
        if not pending:
            break
    return results
