"""Bounded frozen-model SmoothGrad/IG/ISM comparison; no production changes."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time

import numpy as np
from scipy.stats import spearmanr
import torch

from classifier_motifs.attribution import active_weights, one_hot
from classifier_motifs.calibrated_attribution import CalibratedTargets, TARGETS, load_classifier
from classifier_motifs.context_attribution import map_agreement
from classifier_motifs.smoothgrad import gaussian_noise, select_mutation_positions, smoothgrad
from .calibrated_context import require_gpu, save_npz, validate_output
from .common import digest, event, write_json


def add_breadth(values):
    return np.concatenate((values, values[:, :8].sum(1, keepdims=True)), axis=1)


def compare_maps(first, second, data):
    rows = []
    for i, identifier in enumerate(data['ids']):
        lo, size = int(data['native_offset'][i]), int(data['native_length'][i])
        for region, sl in [('native', slice(lo, lo+size)), ('full', slice(None))]:
            a, b = add_breadth(first[i:i+1])[0, :, sl], add_breadth(second[i:i+1])[0, :, sl]
            for t, row in enumerate(map_agreement(a, b)):
                rows.append(dict(id=str(identifier), degree=int(data['labels'][i].sum()),
                    target=(*TARGETS, 'expected_breadth')[t], region=region, **row))
    return rows


def mutation_metrics(exact, estimate, mask):
    a, b = add_breadth(exact)[mask], add_breadth(estimate)[mask]
    rows = []
    for t, name in enumerate((*TARGETS, 'expected_breadth')):
        varying = len(a) > 1 and np.std(a[:, t]) > 0 and np.std(b[:, t]) > 0
        nonzero = np.abs(a[:, t]) > 1e-7
        rows.append(dict(target=name, n=len(a),
            pearson=float(np.corrcoef(a[:, t], b[:, t])[0, 1]) if varying else None,
            spearman=float(spearmanr(a[:, t], b[:, t]).statistic) if varying else None,
            mae=float(np.abs(a[:, t]-b[:, t]).mean()) if len(a) else None,
            nonzero_n=int(nonzero.sum()),
            sign_agreement=float((np.sign(a[nonzero, t]) == np.sign(b[nonzero, t])).mean()) if nonzero.any() else None))
    return rows


def cpu_preflight(parent, checkpoint, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight requires CUDA hidden and uninitialized')
    if output.exists():
        raise ValueError('Refusing to overwrite a preflight report')
    torch.set_num_threads(2)
    target = CalibratedTargets(load_classifier(checkpoint, 'cpu'),
        json.loads((parent/'calibrators.json').read_text())['enhancers_only']).eval()
    with np.load(parent/'endpoint_replay.npz', allow_pickle=False) as saved:
        x = one_hot(saved['sequence'][:2], 'cpu')
        labels = torch.as_tensor(saved['labels'][:2])
        cached = saved['probabilities'][:2]
        ids = saved['ids'][:2]
    with torch.no_grad():
        replay_error = float(np.abs(target.endpoints(x)['probabilities'].numpy()-cached).max())
    if replay_error > .005:
        raise ValueError('Real checkpoint endpoint replay failed')
    noise = gaussian_noise(ids, 2, x.shape[-1])
    small = smoothgrad(target, x, labels, noise, .1, (2,), 2)[2]
    large = smoothgrad(target, x, labels, noise, .1, (2,), 4)[2]
    errors = {}
    for key in small:
        np.testing.assert_allclose(small[key], large[key], atol=3e-5, rtol=2e-3)
        errors[key] = float(np.abs(small[key]-large[key]).max())
    direct = []
    for epsilon in noise:
        points = (x+.1*torch.as_tensor(epsilon)).requires_grad_(True)
        score = target(points, active_weights(labels))[:, :8].sum()
        direct.append(torch.autograd.grad(score, points)[0].numpy())
    breadth = small['gradient'][:, :8].sum(1)
    np.testing.assert_allclose(breadth, np.mean(direct, 0), atol=3e-5, rtol=2e-3)
    result = dict(status='passed', no_cuda=not torch.cuda.is_initialized(),
        checkpoint_sha256=digest(checkpoint), calibration_sha256=digest(parent/'calibrators.json'),
        endpoint_replay_max_error=replay_error, batch_max_errors=errors,
        summed_breadth_max_error=float(np.abs(breadth-np.mean(direct, 0)).max()),
        examples=2, noise_samples=2, sigma=.1, torch=torch.__version__)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    event('smoothgrad_cpu_preflight', **result)


def require_benchmark_gpu(local_cuda, provenance):
    if not local_cuda:
        if require_gpu() != 'rtx2080':
            raise ValueError('The cluster pilot requires one RTX2080')
        return 'cecar_rtx2080'
    if (os.environ.get('SLURM_JOB_ID') or not provenance.get('local_hostname')
            or socket.gethostname().split('.')[0] != provenance['local_hostname']
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Local CUDA requires its explicit flag, packaged local host and one visible GPU')
    return 'local_cuda'


def verified_pilot_report(parent, signature):
    """Read the RTX2080 pilot without depending on newer orchestration APIs."""
    directory = parent/'pilot_rtx2080'
    report = json.loads((directory/'report.json').read_text())
    if (report['status'] != 'passed' or report['signature'] != signature
            or report['family'] != 'rtx2080'
            or set(report['files']) != {'grid_check.npz', 'references100.npz'}):
        raise ValueError('Missing or incompatible passing RTX2080 pilot')
    for name, sha in report['files'].items():
        if digest(directory/name) != sha:
            raise ValueError('Saved IG pilot result changed: '+name)
    return report


def run(parent, checkpoint, output, max_seconds=1680, local_cuda=False, internal_batch=64):
    package = Path(__file__).resolve().parents[2]
    provenance = json.loads((package/'provenance.json').read_text())
    execution = require_benchmark_gpu(local_cuda, provenance)
    if not 0 < max_seconds <= 1680 or not 16 <= internal_batch <= 64:
        raise ValueError('Pilot requires <=28 compute minutes and batches of16..64')
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def stop():
        return stopped[0] or time.monotonic()-began >= max_seconds
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    signature = digest(parent/'MANIFEST.sha256')
    if (signature != provenance['parent_manifest_sha256']
            or digest(parent/'calibrators.json') != digest(package/'calibrators.json')):
        raise ValueError('The IG parent or frozen calibration changed')
    report = verified_pilot_report(parent, signature)
    config = json.loads((parent/'config.json').read_text())
    source = parent/'pilot_rtx2080/references100.npz'
    with np.load(source, allow_pickle=False) as saved:
        data = dict(saved)
    validate_output(data, data['indices'], signature, config)
    degree = data['labels'].sum(1)
    if (len(degree) != 16 or not (data['split'] == 'train').all()
            or not np.array_equal(np.bincount(degree.astype(int), minlength=9)[1:], [2]*8)
            or config['references'] != 100 or config['steps'] != 64):
        raise ValueError('Expected the balanced sixteen-training-enhancer IG64/ref100 pilot')
    cal = parent/'calibrators.json'
    target = CalibratedTargets(load_classifier(checkpoint, 'cuda'),
        json.loads(cal.read_text())['enhancers_only']).to('cuda').eval()
    x = one_hot(data['sequence'], 'cuda')
    labels = torch.as_tensor(data['labels'], device='cuda')
    weights = active_weights(labels)
    with torch.no_grad():
        endpoints = target.endpoints(x)
        np.testing.assert_allclose(endpoints['calibrated_probabilities'].cpu(),
            data['calibrated_probabilities'], atol=3e-5, rtol=2e-3)
    metadata = dict(status='running', no_training=True, job=os.environ.get('SLURM_JOB_ID'), execution=execution, pid=os.getpid(),
        node=socket.gethostname(), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        numpy=np.__version__, seed=20260922, noise_sigmas=[.05, .1, .2], samples=[32, 64],
        repeats=2, internal_batch=internal_batch, targets=list(TARGETS), max_seconds=max_seconds,
        checkpoint_sha256=digest(checkpoint), calibration_sha256=digest(cal),
        ig_maps_sha256=digest(source), parent_manifest_sha256=signature,
        code_sha256={str(p): digest(p) for p in (Path(__file__), Path(__file__).parents[1]/'classifier_motifs/smoothgrad.py')},
        historical_ig_seconds=report['seconds_for_16_enhancers_100_references'], historical_ig_gpu=report['gpu'],
        timing_caveat='Historical IG includes shuffling/checkpoint writes and may use different hardware; not a matched speedup.',
        interpretation='Sensitivity, not a complete reference-based contribution decomposition. No motif discovery in this pilot.')
    write_json(output/'runtime.json', metadata)
    event('smoothgrad_started', **metadata)
    # Warm-up and real-device batching check before timing any method.
    epsilon = gaussian_noise(data['ids'][:2], 2, 2048)
    small = smoothgrad(target, x[:2], labels[:2], epsilon, .1, (2,), 2, stop)
    large = smoothgrad(target, x[:2], labels[:2], epsilon, .1, (2,), 4, stop)
    for key in small[2]:
        np.testing.assert_allclose(small[2][key], large[2][key], atol=3e-5, rtol=2e-3)
    del small, large
    methods, variants = {}, []
    specifications = [(0., 0, (1,))] + [(sigma, repeat, (32, 64))
        for sigma in (.05, .1, .2) for repeat in range(2)]
    for sigma, repeat, counts in specifications:
        if stop(): raise TimeoutError('SmoothGrad pilot time limit')
        noise = gaussian_noise(data['ids'], max(counts), 2048, repeat=repeat)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        maps = smoothgrad(target, x, labels, noise, sigma, counts, internal_batch, stop)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        for samples, values in maps.items():
            name = f'sigma{sigma:g}_n{samples}_repeat{repeat}'
            methods[name] = values
            save_npz(output/(name+'.npz'), **values, ids=data['ids'], targets=np.asarray(TARGETS))
            row = dict(name=name, sigma=sigma, samples=samples, repeat=repeat,
                seconds_for_full_call=elapsed, full_call_samples=max(counts),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                versus_ig=compare_maps(values['observed_sensitivity'], data['actual'], data))
            if samples == 64:
                row['prefix32_vs64'] = compare_maps(maps[32]['observed_sensitivity'], values['observed_sensitivity'], data)
            if repeat:
                row['independent_noise_agreement'] = compare_maps(
                    methods[f'sigma{sigma:g}_n{samples}_repeat0']['observed_sensitivity'], values['observed_sensitivity'], data)
            variants.append(row)
        write_json(output/'progress.json', dict(status='maps_in_progress', variants=variants))
        event('smoothgrad_variant_complete', sigma=sigma, repeat=repeat, seconds=elapsed)
    # Same mutations for every method; random native sites define primary scores.
    actual, estimates = [], {name: [] for name in ['ig64_ref100', *methods]}
    rows, masks = [], {name: [] for name in ('random_native', 'random_flank', 'top_union')}
    for i, identifier in enumerate(data['ids']):
        if stop(): raise TimeoutError('SmoothGrad mutation diagnostic time limit')
        maps = [data['actual'][i]] + [v['observed_sensitivity'][i] for name, v in methods.items() if name.endswith('repeat0')]
        positions, selection = select_mutation_positions(str(identifier),
            int(data['native_offset'][i]), int(data['native_length'][i]), 2048, maps)
        changes = [(int(p), base) for p in positions for base in range(4) if base != data['sequence'][i, p]]
        for first in range(0, len(changes), internal_batch):
            if stop(): raise TimeoutError('SmoothGrad mutation batch time limit')
            chunk = changes[first:first+internal_batch]
            mutated = x[i:i+1].repeat(len(chunk), 1, 1)
            for j, (p, base) in enumerate(chunk):
                mutated[j, :, p] = 0
                mutated[j, base, p] = 1
            with torch.no_grad():
                difference = target(mutated, weights[i:i+1].repeat(len(chunk), 1))-target(x[i:i+1], weights[i:i+1])
            actual.extend(difference.cpu().numpy())
            for p, base in chunk:
                wt = int(data['sequence'][i, p])
                rows.append((i, p, wt, base))
                for key in masks:
                    masks[key].append(bool(selection[key][np.searchsorted(positions, p)]))
                estimates['ig64_ref100'].append(data['hypothetical'][i, :, base, p]-data['hypothetical'][i, :, wt, p])
                for name, values in methods.items():
                    estimates[name].append(values['gradient'][i, :, base, p]-values['gradient'][i, :, wt, p])
    actual = np.asarray(actual)
    diagnostic = {}
    for name, values in estimates.items():
        values = np.asarray(values)
        diagnostic[name] = {key: mutation_metrics(actual, values, np.asarray(mask)) for key, mask in masks.items()}
        diagnostic[name]['per_enhancer_random_native'] = [dict(id=str(identifier),
            degree=int(degree[i]), metrics=mutation_metrics(actual, values,
                np.asarray(masks['random_native']) & (np.asarray(rows)[:, 0] == i)))
            for i, identifier in enumerate(data['ids'])]
    save_npz(output/'mutations.npz', mutations=np.asarray(rows), ids=data['ids'],
        exact_delta=actual, **{name: np.asarray(value) for name, value in estimates.items()},
        **{name: np.asarray(mask) for name, mask in masks.items()})
    write_json(output/'result.json', dict(**{**metadata, 'status': 'complete'},
        variants=variants, mutation_metrics=diagnostic,
        elapsed_seconds=time.monotonic()-began, files={p.name: digest(p) for p in output.glob('*.npz')}))
    event('smoothgrad_complete', output=str(output), elapsed_seconds=time.monotonic()-began)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-preflight', action='store_true')
    parser.add_argument('--local-cuda', action='store_true')
    parser.add_argument('--internal-batch', type=int, default=64)
    args = parser.parse_args()
    if args.cpu_preflight:
        if args.local_cuda:
            parser.error('CPU preflight cannot request local CUDA')
        cpu_preflight(args.parent.resolve(), args.checkpoint.resolve(), args.output.resolve())
    else:
        run(args.parent.resolve(), args.checkpoint.resolve(), args.output.resolve(),
            local_cuda=args.local_cuda, internal_batch=args.internal_batch)
