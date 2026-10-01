"""Local, bounded SmoothGrad tuning followed by a locked confirmation set."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from classifier_motifs.attribution import active_weights, one_hot, seed_for
from classifier_motifs.calibrated_attribution import CalibratedTargets, TARGETS, load_classifier
from classifier_motifs.smoothgrad import gaussian_noise, paired_gaussian_noise, smoothgrad
from .calibrated_context import save_npz
from .common import digest, event, write_json
from .smoothgrad_benchmark import compare_maps, mutation_metrics, require_benchmark_gpu


SEED = 20260922
SIGMAS = (.025, .05, .075, .1)


def select_examples(data, old, per_degree=4):
    """Fresh, balanced tuning/confirmation sets; no2048-window or DNA duplicates."""
    n = len(data['ids'])
    if (len(np.unique(data['ids'])) != n or data['sequence'].shape != (n, 2048)
            or not np.isin(data['sequence'], range(4)).all()
            or data['labels'].shape != (n, 8) or not np.isin(data['labels'], [0, 1]).all()
            or not data['labels'].any(1).all() or per_degree < 1):
        raise ValueError('Malformed training population')
    def dna_key(sequence):
        sequence = np.asarray(sequence, dtype=np.uint8)
        return min(hashlib.sha256(sequence.tobytes()).digest(),
                   hashlib.sha256((3-sequence[::-1]).tobytes()).digest())
    used_ids = set(map(str, old['ids']))
    used_dna = {dna_key(s) for s in old['sequence']}
    windows = list(zip(map(str, old['chrom']), map(int, old['summit'])))
    rng = np.random.default_rng(SEED)
    degree = data['labels'].sum(1)
    groups = {}
    for k in range(8, 0, -1):
        chosen = []
        for i in rng.permutation(np.flatnonzero(degree == k)):
            chrom, summit = str(data['chrom'][i]), int(data['summit'][i])
            if str(data['ids'][i]) in used_ids or any(c == chrom and abs(s-summit) < 2048 for c, s in windows):
                continue
            key = dna_key(data['sequence'][i])
            if key in used_dna:
                continue
            used_dna.add(key)
            used_ids.add(str(data['ids'][i]))
            windows.append((chrom, summit))
            chosen.append(int(i))
            if len(chosen) == 2*per_degree:
                break
        if len(chosen) != 2*per_degree:
            raise ValueError('Insufficient independent examples for degree '+str(k))
        groups[k] = chosen
    tune = [groups[k][j] for j in range(per_degree) for k in range(1, 9)]
    confirm = [groups[k][j+per_degree] for j in range(per_degree) for k in range(1, 9)]
    return np.asarray(tune+confirm), np.asarray(['tune']*len(tune)+['confirm']*len(confirm))


def make_mutants(codes, offset, length, identifier, screen):
    """Random SNVs and10bp composition-preserving block shuffles; no named TFs."""
    if not 0 <= offset < offset+length <= len(codes) or length < 10:
        raise ValueError('Invalid native interval for10bp perturbations')
    rng = np.random.default_rng(seed_for('sg_refinement_mutations', SEED, identifier))
    native = np.arange(offset, offset+length)
    flank = np.setdiff1d(np.arange(len(codes)), native)
    mutants, metadata, skipped = [], [], []
    def append(sequence, start, end, region, kind):
        mutants.append(sequence)
        metadata.append((start, end, region, kind))
    for region, positions, count in [('native', native, 8), ('flank', flank, 4)]:
        for position in rng.choice(positions, min(count, len(positions)), replace=False):
            for base in range(4):
                if base == codes[position]:
                    continue
                changed = codes.copy()
                changed[position] = base
                append(changed, int(position), int(position)+1, region, 'snv')
    starts = np.arange(offset, offset+length-9)
    flank_starts = np.r_[np.arange(max(0, offset-9)), np.arange(offset+length, len(codes)-9)]
    windows = [(int(s), 'native', 'block_random') for s in rng.choice(starts, min(2, len(starts)), replace=False)]
    if len(flank_starts):
        windows.append((int(rng.choice(flank_starts)), 'flank', 'block_random'))
    breadth = screen[:8].sum(0)
    scores = np.convolve(breadth[offset:offset+length], np.ones(10), mode='valid')
    windows.extend([(int(starts[np.argmax(scores)]), 'native', 'block_top_high'),
                    (int(starts[np.argmin(scores)]), 'native', 'block_top_low')])
    for start, region, kind in windows:
        segment = codes[start:start+10]
        if len(np.unique(segment)) == 1:
            skipped.append(dict(start=start, region=region, kind=kind, reason='homopolymer_cannot_be_composition_shuffled'))
            continue
        for replicate in range(3):
            for attempt in range(100):
                shuffled = rng.permutation(segment)
                if not np.array_equal(shuffled, segment):
                    break
            else:
                raise ValueError('Could not construct nonidentity block shuffle')
            changed = codes.copy()
            changed[start:start+10] = shuffled
            append(changed, start, start+10, region, kind)
    return np.asarray(mutants, dtype=np.uint8), dict(
        start=np.asarray([m[0] for m in metadata]), end=np.asarray([m[1] for m in metadata]),
        region=np.asarray([m[2] for m in metadata]), kind=np.asarray([m[3] for m in metadata])), skipped


def estimate_changes(gradient, data, perturbations):
    result = []
    for i, mutant in zip(perturbations['example'], perturbations['sequence']):
        codes, g = data['sequence'][i], gradient[i]
        positions = np.flatnonzero(mutant != codes)
        result.append((g[:, mutant[positions], positions]-g[:, codes[positions], positions]).sum(1))
    return np.asarray(result)


def score_changes(exact, predicted, perturbations, data):
    rows = []
    random_native = (perturbations['kind'] == 'snv') & (perturbations['region'] == 'native')
    for i, identifier in enumerate(data['ids']):
        mask = random_native & (perturbations['example'] == i)
        metrics = mutation_metrics(exact, predicted, mask)
        # Undefined predictions receive0; truly constant exact effects are excluded.
        scores = [m['pearson'] if m['pearson'] is not None else 0.
                  for t, m in enumerate(metrics[:8]) if np.std(exact[mask, t]) > 1e-10]
        rows.append(dict(id=str(identifier), degree=int(data['labels'][i].sum()),
            macro_native_pearson=float(np.mean(scores)) if scores else None, metrics=metrics))
    eligible = [r['macro_native_pearson'] for r in rows if r['macro_native_pearson'] is not None]
    grouped = {}
    for kind in ('snv', 'block_random', 'block_top_high', 'block_top_low'):
        for region in ('native', 'flank'):
            mask = (perturbations['kind'] == kind) & (perturbations['region'] == region)
            if mask.any():
                grouped[kind+'_'+region] = mutation_metrics(exact, predicted, mask)
    return dict(macro_native_pearson=float(np.mean(eligible)) if eligible else None,
                per_enhancer=rows, groups=grouped)


def choose_method(results):
    """Select only from tuning reports, before confirmation predictions exist."""
    ranking = []
    for name, runs in results.items():
        if any(r['phase'] != 'tune' for r in runs):
            raise ValueError('Confirmation results cannot select hyperparameters')
        values = [r['metrics']['macro_native_pearson'] for r in runs]
        if not values or any(v is None or not np.isfinite(v) for v in values):
            raise ValueError('Missing tuning criterion')
        ranking.append(dict(method=name, score=float(np.mean(values))))
    return sorted(ranking, key=lambda x: (-x['score'], x['method']))


def bootstrap_confirmation(candidate, baseline, degree):
    delta = np.asarray(candidate, dtype=float)-np.asarray(baseline, dtype=float)
    if len(delta) != len(degree):
        raise ValueError('Aligned enhancer-level confirmation scores required')
    valid = np.isfinite(delta)
    excluded = int((~valid).sum())
    delta, degree = delta[valid], np.asarray(degree)[valid]
    if len(delta) < 2:
        return dict(mean_change=None, ci95=None, n_enhancers=len(delta), excluded=excluded)
    rng = np.random.default_rng(seed_for('sg_confirm_bootstrap', SEED))
    groups = [np.flatnonzero(degree == k) for k in np.unique(degree)]
    resampled = np.concatenate([rng.choice(g, (2000, len(g)), replace=True) for g in groups], axis=1)
    lower, upper = np.quantile(delta[resampled].mean(1), [.025, .975])
    return dict(mean_change=float(delta.mean()), ci95=[float(lower), float(upper)], n_enhancers=len(delta), excluded=excluded,
        units='enhancers, stratified by degree; mutations are not independent replicates')


def attribute(target, data, sigma, scheme, repeat, stop):
    output = []
    torch.cuda.synchronize()
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for first in range(0, len(data['ids']), 16):
        sl = slice(first, first+16)
        counts = (1,) if scheme == 'plain' else (32, 64)
        sampler = paired_gaussian_noise if scheme == 'paired' else gaussian_noise
        noise = sampler(data['ids'][sl], max(counts), 2048, repeat=repeat)
        values = smoothgrad(target, one_hot(data['sequence'][sl], 'cuda'),
            torch.as_tensor(data['labels'][sl], device='cuda'), noise, sigma, counts,
            16, stop, paired=scheme == 'paired')
        final = values[max(counts)]
        if scheme != 'plain':
            final['observed_prefix32'] = values[32]['observed_sensitivity']
        output.append(final)
    torch.cuda.synchronize()
    return {key: np.concatenate([v[key] for v in output]) for key in output[0]}, dict(
        seconds=time.monotonic()-began, peak_allocated_bytes=torch.cuda.max_memory_allocated())


def build_perturbations(target, data, screen, stop):
    collections = {key: [] for key in ('sequence', 'example', 'start', 'end', 'region', 'kind')}
    skipped = []
    for i, identifier in enumerate(data['ids']):
        mutants, meta, omissions = make_mutants(data['sequence'][i], int(data['native_offset'][i]),
            int(data['native_length'][i]), str(identifier), screen[i])
        collections['sequence'].append(mutants)
        collections['example'].append(np.full(len(mutants), i))
        for key, value in meta.items(): collections[key].append(value)
        skipped.extend([dict(id=str(identifier), **row) for row in omissions])
    perturb = {key: np.concatenate(values) for key, values in collections.items()}
    exact, endpoints = [], []
    with torch.no_grad():
        for start in range(0, len(data['ids']), 16):
            if stop(): raise TimeoutError('Stopped before WT inference')
            sl = slice(start, start+16)
            endpoints.append(target(one_hot(data['sequence'][sl], 'cuda'),
                active_weights(torch.as_tensor(data['labels'][sl], device='cuda'))).cpu().numpy())
        wt = np.concatenate(endpoints)
        for start in range(0, len(perturb['example']), 16):
            if stop(): raise TimeoutError('Stopped during mutation inference')
            sl = slice(start, start+16)
            indices = perturb['example'][sl]
            prediction = target(one_hot(perturb['sequence'][sl], 'cuda'),
                active_weights(torch.as_tensor(data['labels'][indices], device='cuda'))).cpu().numpy()
            exact.append(prediction-wt[indices])
    exact = np.concatenate(exact)
    if not np.isfinite(exact).all(): raise ValueError('Nonfinite exact mutation effects')
    return perturb, exact, wt, skipped


def cpu_preflight(package, checkpoint, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight must hide CUDA')
    if output.exists(): raise ValueError('Refusing to overwrite preflight')
    torch.set_num_threads(2)
    with np.load(package/'examples.npz', allow_pickle=False) as saved:
        x = one_hot(saved['sequence'][:2], 'cpu')
        labels = torch.as_tensor(saved['labels'][:2])
        ids = saved['ids'][:2]
    target = CalibratedTargets(load_classifier(checkpoint, 'cpu'),
        json.loads((package/'calibrators.json').read_text())['enhancers_only']).eval()
    noise = paired_gaussian_noise(ids, 4, 2048)
    a = smoothgrad(target, x, labels, noise, .1, (4,), 2, paired=True)[4]
    b = smoothgrad(target, x, labels, noise, .1, (4,), 4, paired=True)[4]
    errors = {}
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=3e-5, rtol=2e-3)
        errors[key] = float(np.abs(a[key]-b[key]).max())
    direct = []
    for epsilon in noise:
        points = (x+.1*torch.as_tensor(epsilon)).requires_grad_(True)
        total = target(points, active_weights(labels))[:, :8].sum()
        direct.append(torch.autograd.grad(total, points)[0].numpy())
    np.testing.assert_allclose(a['gradient'][:, :8].sum(1), np.mean(direct, 0), atol=3e-5, rtol=2e-3)
    result = dict(status='passed', no_cuda=not torch.cuda.is_initialized(), batch_max_errors=errors,
        summed_breadth_max_error=float(np.abs(a['gradient'][:, :8].sum(1)-np.mean(direct, 0)).max()),
        manifest_sha256=digest(package/'MANIFEST.sha256'), checkpoint_sha256=digest(checkpoint))
    write_json(output, result)
    event('refinement_cpu_preflight', **result)


def run(package, checkpoint, output, max_seconds=1680):
    if not 0 < max_seconds <= 1680: raise ValueError('Maximum28 compute minutes')
    provenance = json.loads((package/'provenance.json').read_text())
    require_benchmark_gpu(True, provenance)
    output.mkdir(parents=True, exist_ok=False)
    began, stopped = time.monotonic(), [False]
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def stop(): return stopped[0] or time.monotonic()-began >= max_seconds
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with np.load(package/'examples.npz', allow_pickle=False) as saved: cohort = dict(saved)
    if len(cohort['ids']) != 64 or len(np.unique(cohort['ids'])) != 64:
        raise ValueError('Expected64 unique prepared examples')
    for phase in ('tune', 'confirm'):
        degree = cohort['labels'][cohort['phase'] == phase].sum(1).astype(int)
        if not np.array_equal(np.bincount(degree, minlength=9)[1:], [4]*8):
            raise ValueError('Both phases require four enhancers per degree')
    target = CalibratedTargets(load_classifier(checkpoint, 'cuda'),
        json.loads((package/'calibrators.json').read_text())['enhancers_only']).to('cuda').eval()
    # Check the new paired kernel on this GPU before spending the pilot budget.
    x = one_hot(cohort['sequence'][:2], 'cuda')
    labels = torch.as_tensor(cohort['labels'][:2], device='cuda')
    noise = paired_gaussian_noise(cohort['ids'][:2], 4, 2048)
    a = smoothgrad(target, x, labels, noise, .1, (4,), 2, stop, paired=True)[4]
    b = smoothgrad(target, x, labels, noise, .1, (4,), 4, stop, paired=True)[4]
    for key in a: np.testing.assert_allclose(a[key], b[key], atol=3e-5, rtol=2e-3)
    metadata = dict(status='running', pid=os.getpid(), no_training=True, gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, targets=list(TARGETS), max_seconds=max_seconds,
        package_manifest_sha256=digest(package/'MANIFEST.sha256'), checkpoint_sha256=digest(checkpoint),
        sampling='64 Gaussian inputs:64 IID samples or32 antithetic pairs; two repeats',
        selection='Mean enhancer-level, eight-context Pearson on random native SNVs; average two repeats',
        confirmation_scope='Disjoint attribution-method confirmation, not an independent model-generalization test')
    write_json(output/'runtime.json', metadata)
    event('smoothgrad_refinement_started', **metadata)
    specs = {'plain': (0., 'plain')}
    specs.update({f'{scheme}_{sigma:g}': (sigma, scheme) for sigma in SIGMAS for scheme in ('iid', 'paired')})
    summaries = {}
    winner = None
    for phase in ('tune', 'confirm'):
        indices = np.flatnonzero(cohort['phase'] == phase)
        data = {key: value[indices] for key, value in cohort.items()}
        directory = output/phase
        directory.mkdir()
        plain, plain_timing = attribute(target, data, 0., 'plain', 0, stop)
        perturb, exact, wt, skipped = build_perturbations(target, data, plain['observed_sensitivity'], stop)
        save_npz(directory/'perturbations.npz', **perturb, exact_delta=exact, wt_targets=wt, ids=data['ids'])
        write_json(directory/'skipped_blocks.json', skipped)
        methods = list(specs) if phase == 'tune' else list(dict.fromkeys(['plain', 'iid_0.1', winner]))
        reports = {}
        for name in methods:
            sigma, scheme = specs[name]
            repeats = (0,) if scheme == 'plain' else (0, 1)
            reports[name] = []
            first_maps = None
            for repeat in repeats:
                if stop(): raise TimeoutError('Refinement budget exhausted; completed variants preserved')
                values, timing = (plain, plain_timing) if scheme == 'plain' else attribute(target, data, sigma, scheme, repeat, stop)
                estimate = estimate_changes(values['gradient'], data, perturb)
                row = dict(phase=phase, method=name, repeat=repeat, sigma=sigma, scheme=scheme,
                    **timing, metrics=score_changes(exact, estimate, perturb, data))
                if scheme != 'plain':
                    row['prefix32_agreement'] = compare_maps(values['observed_prefix32'], values['observed_sensitivity'], data)
                if first_maps is not None:
                    row['independent_repeat_agreement'] = compare_maps(first_maps, values['observed_sensitivity'], data)
                else:
                    first_maps = values['observed_sensitivity']
                save_npz(directory/f'{name}_repeat{repeat}.npz', **values, mutation_estimate=estimate,
                    ids=data['ids'], targets=np.asarray(TARGETS))
                reports[name].append(row)
                write_json(directory/'progress.json', reports)
                event('refinement_variant_complete', phase=phase, method=name, repeat=repeat,
                    seconds=timing['seconds'], macro_native_pearson=row['metrics']['macro_native_pearson'])
        summaries[phase] = reports
        if phase == 'tune':
            ranking = choose_method(reports)
            winner = ranking[0]['method']
            write_json(output/'locked_selection.json', dict(winner=winner, ranking=ranking,
                confirmation_not_evaluated=True, tune_ids=list(map(str, data['ids']))))
            event('refinement_selection_locked', winner=winner)
    def enhancer_scores(name):
        return np.mean([[v['macro_native_pearson'] if v['macro_native_pearson'] is not None else np.nan
                         for v in row['metrics']['per_enhancer']]
                        for row in summaries['confirm'][name]], axis=0)
    confirmation = {baseline: bootstrap_confirmation(enhancer_scores(winner), enhancer_scores(baseline),
        data['labels'].sum(1)) for baseline in ('plain', 'iid_0.1')}
    write_json(output/'result.json', dict(**{**metadata, 'status': 'complete'}, winner=winner,
        confirmation=confirmation, phases=summaries, elapsed_seconds=time.monotonic()-began,
        files={str(p.relative_to(output)): digest(p) for p in output.rglob('*.npz')}))
    event('smoothgrad_refinement_complete', winner=winner, confirmation=confirmation,
          elapsed_seconds=time.monotonic()-began)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-preflight', action='store_true')
    args = parser.parse_args()
    operation = cpu_preflight if args.cpu_preflight else run
    operation(args.package.resolve(), args.checkpoint.resolve(), args.output.resolve())
