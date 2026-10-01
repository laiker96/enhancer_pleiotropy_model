"""Signed Gaussian-smoothed sensitivities, not reference-based contributions."""
import numpy as np
import torch

from .attribution import active_weights, seed_for


def gaussian_noise(ids, samples, length, seed=20260922, repeat=0):
    """Common random numbers across targets/sigmas; stable to example batching."""
    if samples < 1 or length < 1 or len(set(map(str, ids))) != len(ids):
        raise ValueError('Positive dimensions and unique IDs required')
    return np.stack([np.stack([np.random.default_rng(seed_for(
        'smoothgrad', seed, str(identifier), repeat, sample)).standard_normal(
            (4, length)).astype(np.float32) for identifier in ids])
        for sample in range(samples)])


def paired_gaussian_noise(ids, samples, length, seed=20260922, repeat=0):
    """Antithetic pairs; samples counts both +epsilon and -epsilon inputs."""
    if samples < 4 or samples % 2:
        raise ValueError('An even count of at least four noisy inputs is required')
    base = gaussian_noise(ids, samples//2, length, seed, repeat)
    return np.stack((base, -base), axis=1).reshape(samples, len(ids), 4, length)


def smoothgrad(target, x, labels, noise, sigma, counts=(32, 64),
               internal_batch=64, should_stop=lambda: False, paired=False, target_count=9):
    """Average signed gradients at x + sigma * noise for each of nine outputs.

    Explicit noise [sample, example, ACGT, position] permits paired comparisons.
    No clipping, absolute values, squaring, input multiplication, or baseline.
    One shared forward graph per batch, nine matrix-free backward directions.
    Save raw gradients and zero-channel-mean sensitivities separately. The latter
    removes the component normal to the DNA simplex; it is NOT IG completeness.
    With paired=True, SE uses independent pair means, not correlated inputs.
    """
    if (target.training or x.ndim != 3 or x.shape[1] != 4 or not len(x)
            or noise.ndim != 4 or tuple(noise.shape[1:]) != tuple(x.shape)
            or not np.isfinite(sigma) or sigma < 0 or internal_batch < len(x)
            or not counts or tuple(sorted(set(counts))) != tuple(counts)
            or counts[0] < 1 or counts[-1] > len(noise)
            or target_count not in (8, 9)
            or not torch.isfinite(x).all() or not ((x == 0) | (x == 1)).all()
            or not (x.sum(1) == 1).all()):
        raise ValueError('Invalid SmoothGrad inputs, counts or batching')
    if paired and (any(c < 4 or c % 2 for c in counts) or len(noise) % 2
                   or not np.array_equal(noise[::2], -noise[1::2])):
        raise ValueError('Paired SE requires exact adjacent +/- pairs and even counts>=4')
    weights = active_weights(labels).to(device=x.device, dtype=x.dtype)
    if len(weights) != len(x):
        raise ValueError('Labels and inputs must align')
    total = torch.zeros((len(x), target_count, 4, x.shape[-1]), device=x.device, dtype=torch.float64)
    squared = torch.zeros_like(total)
    pending = torch.zeros_like(total) if paired else None
    results = {}
    group = internal_batch // len(x)
    begin = 0
    finite = torch.ones((), device=x.device, dtype=torch.bool)
    for count in counts:
        while begin < count:
            if should_stop():
                raise TimeoutError('Stopped at SmoothGrad batch boundary')
            end = min(begin + group, count)
            epsilon = torch.as_tensor(noise[begin:end], device=x.device, dtype=x.dtype)
            points = (x[None] + sigma * epsilon).flatten(0, 1).detach().requires_grad_(True)
            scores = target(points, weights.repeat(end-begin, 1))
            if scores.shape != (len(points), 9):
                raise ValueError('Expected nine target readouts')
            finite &= torch.isfinite(scores).all() & torch.isfinite(points).all()
            for t in range(target_count):
                grad = torch.autograd.grad(scores[:, t].sum(), points,
                    retain_graph=t < target_count-1)[0].reshape(end-begin, *x.shape).double()
                finite &= torch.isfinite(grad).all()
                # Fixed summation order, independent of microbatch size.
                for j, value in enumerate(grad):
                    total[:, t] += value
                    if not paired:
                        squared[:, t] += value.square()
                    elif (begin+j) % 2 == 0:
                        pending[:, t] = value
                    else:
                        squared[:, t] += ((pending[:, t]+value)*.5).square()
            begin = end
        mean = total / count
        units = count//2 if paired else count
        correction = units*mean.square() if paired else total.square()/count
        se = ((squared-correction).clamp_min(0)/((units-1)*units)).sqrt() if units > 1 else torch.zeros_like(mean)
        centered = mean - mean.mean(2, keepdim=True)
        results[count] = {key: value.float().detach().cpu().numpy() for key, value in dict(
            gradient=mean, gradient_noise_se=se, centered_sensitivity=centered,
            observed_sensitivity=(centered*x[:, None]).sum(2)).items()}
    if not finite:
        raise ValueError('Nonfinite SmoothGrad calculation')
    return results


def select_mutation_positions(identifier, offset, length, input_bp, maps):
    """Random sites are primary; union of methods' top sites is secondary."""
    if not 0 <= offset < offset+length <= input_bp:
        raise ValueError('Invalid native enhancer interval')
    native = np.arange(offset, offset+length)
    flanks = np.setdiff1d(np.arange(input_bp), native)
    rng = np.random.default_rng(seed_for('smoothgrad_ism', identifier, 20260922))
    random_native = rng.choice(native, min(8, len(native)), replace=False)
    random_flank = rng.choice(flanks, min(4, len(flanks)), replace=False)
    top = []
    for values in maps:
        if values.shape != (9, input_bp) or not np.isfinite(values).all():
            raise ValueError('Invalid site-selection maps')
        importance = np.abs(values[:8].sum(0))
        top.extend(native[np.argsort(-importance[native], kind='stable')[:2]])
    positions = np.unique(np.r_[random_native, random_flank, top]).astype(np.int64)
    return positions, dict(random_native=np.isin(positions, random_native),
        random_flank=np.isin(positions, random_flank), top_union=np.isin(positions, top))
