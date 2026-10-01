"""Small matched eight-context attribution-fidelity pilot, never training."""
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import time

import numpy as np
from scipy.stats import spearmanr
import torch

from classifier_motifs.attribution import one_hot, seed_for
from classifier_motifs.calibrated_attribution import CalibratedTargets, TARGETS, load_classifier
from classifier_motifs.calibrated_deeplift import (ProbabilityReadout, calibrate,
    paired_calibrated_deeplift)
from classifier_motifs.captum_deeplift import adapt_model, require_pinned_captum
from classifier_motifs.context_attribution import map_agreement
from classifier_motifs.smoothgrad import paired_gaussian_noise, smoothgrad
from .calibrated_context import make_references, save_npz
from .common import digest, event, write_json
from .smoothgrad_refinement import make_mutants


NAME = 'classifier_method_pilot_20260922_r1'
IG = 'results/classifier_calibrated_context_20260921/package/pilot_rtx2080/references100.npz'
CHECKPOINT = 'results/classifier_calibrated_context_20260921/inputs/best_model.pt'
SG = 'results/classifier_smoothgrad_20260922/local_run_r1'
NAMES = (*TARGETS[:8], 'expected_breadth')


def breadth(a):
    return np.concatenate((a, a[:, :8].sum(1, keepdims=True)), axis=1)


def prepare(project, root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Preparation is CPU only')
    if digest(project/IG) != '83d2401dc35dab6652a218e5c7024024598b76964a7f5147aa069817ab802d14':
        raise ValueError('Unexpected matched IG file')
    prior = project/'results/classifier_smoothgrad_20260922/package_local_r2'
    captum = project/'results/classifier_captum_highprecision_20260920/package'
    package = root/'package'
    package.mkdir(parents=True, exist_ok=False)
    for folder in ('code', 'src'):
        shutil.copytree(prior/folder, package/folder, ignore=shutil.ignore_patterns('__pycache__'))
    for folder in ('vendor', 'dependencies'):
        shutil.copytree(captum/folder, package/folder, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('classifier_motifs/calibrated_deeplift.py', 'classifier_motifs/captum_deeplift.py',
                 'classifier_motifs/dual_attribution.py', 'classifier_motifs/smoothgrad.py',
                 'classifier_modisco/method_pilot.py', 'classifier_modisco/smoothgrad_refinement.py'):
        shutil.copy2(project/'experiments'/name, package/'code'/name)
    (package/'tests').mkdir()
    for name in ('test_calibrated_deeplift.py', 'test_method_pilot.py',
                 'test_captum_deeplift.py', 'test_calibrated_attribution.py', 'test_smoothgrad.py'):
        shutil.copy2(project/'tests'/name, package/'tests'/name)
    sources = {'ig.npz': IG,
        'plain.npz': SG+'/sigma0_n1_repeat0.npz',
        'smoothgrad_iid.npz': SG+'/sigma0.1_n64_repeat0.npz',
        'smoothgrad_iid_repeat.npz': SG+'/sigma0.1_n64_repeat1.npz',
        'calibrators.json': 'results/classifier_calibrated_context_20260921/package/calibrators.json'}
    hashes = {CHECKPOINT: digest(project/CHECKPOINT)}
    old_report = json.loads((project/SG/'result.json').read_text())
    if old_report['status'] != 'complete' or old_report['checkpoint_sha256'] != hashes[CHECKPOINT]:
        raise ValueError('Completed same-model SmoothGrad pilot required')
    for destination, source in sources.items():
        if source.startswith(SG+'/') and digest(project/source) != old_report['files'][Path(source).name]:
            raise ValueError('Saved SmoothGrad output changed')
        shutil.copy2(project/source, package/destination)
        hashes[source] = digest(project/source)
    config = dict(checkpoint=CHECKPOINT, source_hashes=hashes,
        host=socket.gethostname(), targets=list(TARGETS[:8]), examples=16,
        references=100, seed=20260916, max_seconds=1680,
        scope='Authorized local CUDA pilot only, 30-minute hard cap; no CECAR changes.',
        hypotheses='Prespecified native SNV and signed 10bp-window perturbation fidelity; IG is not ground truth.',
        selection='Reuse original 16 balanced training enhancers, two per degree; exploratory, not held-out confirmation.',
        modisco='Native-only signed input exports and motif-scale proxies, not de novo discovery on 16 sequences.')
    write_json(package/'config.json', config)
    shutil.copy2(project/'scripts/run_method_pilot_local.sh', package/'run_local.sh')
    files = sorted(p for p in package.rglob('*') if p.is_file())
    (package/'MANIFEST.sha256').write_text(''.join(digest(p)+'  '+str(p.relative_to(package))+'\n' for p in files))
    event('method_pilot_prepared', root=str(root), manifest=digest(package/'MANIFEST.sha256'))


def load(package):
    with np.load(package/'ig.npz', allow_pickle=False) as z:
        data = dict(z)
    if (data['sequence'].shape != (16, 2048) or len(np.unique(data['ids'])) != 16
            or not (data['split'] == 'train').all()
            or not np.array_equal(np.bincount(data['labels'].sum(1).astype(int), minlength=9)[1:], [2]*8)
            or int(data['references']) != 100 or int(data['steps']) != 64
            or not np.array_equal(data['targets'][:8], TARGETS[:8])):
        raise ValueError('Wrong sixteen-example reference cohort')
    methods = {'ig': dict(hypothetical=data['hypothetical'][:, :8], actual=data['actual'][:, :8])}
    for name in ('plain', 'smoothgrad_iid', 'smoothgrad_iid_repeat'):
        with np.load(package/(name+'.npz'), allow_pickle=False) as z:
            np.testing.assert_array_equal(data['ids'], z['ids'])
            np.testing.assert_array_equal(data['targets'], z['targets'])
            methods[name] = dict(hypothetical=z['centered_sensitivity'][:, :8], actual=z['observed_sensitivity'][:, :8])
    return data, methods


def models(project, package, device):
    config = json.loads((package/'config.json').read_text())
    model = load_classifier(project/config['checkpoint'], device)
    calibration = json.loads((package/'calibrators.json').read_text())['enhancers_only']
    target = CalibratedTargets(model, calibration).to(device).eval()
    adapted, changes = adapt_model(model)
    adapted = adapted.double()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(adapted.state_dict()[key], value.to(adapted.state_dict()[key].dtype), atol=0, rtol=0)
    return target, adapted, changes


def check_real(target, adapted, x, baseline, stop=lambda: False):
    a, b = target.a.double(), target.b.double()
    errors = {}
    for name, points in [('wt', x), ('reference', baseline), ('path', .37*x+.63*baseline)]:
        if stop(): raise TimeoutError('Preflight stopped')
        points = points.detach().requires_grad_(True)
        original = target.endpoints(points)['calibrated_probabilities']
        modified = calibrate(ProbabilityReadout(adapted)(points.double()), a, b)
        torch.testing.assert_close(modified, original, atol=2e-5, rtol=1e-5, check_dtype=False)
        errors[name+'_forward'] = float((original-modified).abs().max())
        gradient_errors = []
        for t in range(8):
            g = torch.autograd.grad(original[:, t].sum(), points, retain_graph=True)[0]
            h = torch.autograd.grad(modified[:, t].sum(), points, retain_graph=True)[0]
            torch.testing.assert_close(g, h, atol=3e-5, rtol=2e-3)
            gradient_errors.append(float((g-h).abs().max()))
        errors[name+'_gradient'] = max(gradient_errors)
    got = paired_calibrated_deeplift(adapted, x.double(), baseline.double(), a, b, stop)
    singles = [paired_calibrated_deeplift(adapted, x[i:i+1].double(), baseline[i:i+1].double(), a, b, stop) for i in range(len(x))]
    for key in got:
        expected = torch.cat([r[key] for r in singles])
        torch.testing.assert_close(got[key], expected, atol=2e-5, rtol=2e-3)
    rc = paired_calibrated_deeplift(adapted, x.flip((1, 2)).double(), baseline.flip((1, 2)).double(), a, b, stop)
    torch.testing.assert_close(got['hypothetical'], rc['hypothetical'].flip((2, 3)), atol=2e-5, rtol=2e-3)
    repeat = paired_calibrated_deeplift(adapted, x.double(), baseline.double(), a, b, stop)
    torch.testing.assert_close(got['hypothetical'], repeat['hypothetical'], atol=0, rtol=0)
    if not (got['delta'].abs() <= .002+.05*got['difference'].abs()).all():
        raise ValueError('Calibrated DeepLIFT completeness failed')
    errors['maximum_pair_delta'] = float(got['delta'].abs().max())
    return dict(status='passed', errors=errors)


def preflight(project, root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU preflight must hide CUDA')
    torch.set_num_threads(2)
    package = root/'package'
    config = json.loads((package/'config.json').read_text())
    data, _ = load(package)
    refs, _ = make_references(data, np.arange(2), dict(config, references=1), None)
    target, adapted, changes = models(project, package, 'cpu')
    result = check_real(target, adapted, one_hot(data['sequence'][:2], 'cpu'), one_hot(refs[:, 0], 'cpu'))
    result.update(no_cuda=not torch.cuda.is_initialized(), changes=changes,
                  manifest_sha256=digest(package/'MANIFEST.sha256'))
    if (root/'cpu_preflight.json').exists(): raise ValueError('Preflight already exists')
    write_json(root/'cpu_preflight.json', result)
    event('real_model_cpu_preflight', **result)


def edits(data, methods):
    """Identical random mutations plus union of every method's signed top blocks."""
    sequences, metadata, selections, skipped = [], [], [], []
    for i, identifier in enumerate(data['ids']):
        codes = data['sequence'][i]
        lo, length = int(data['native_offset'][i]), int(data['native_length'][i])
        baseline, meta, omitted = make_mutants(codes, lo, length, str(identifier), methods['plain']['actual'][i])
        for seq, start, end, kind, region in zip(baseline, meta['start'], meta['end'], meta['kind'], meta['region']):
            if str(kind).startswith('block_top_'): continue
            sequences.append(seq)
            metadata.append(dict(example=i, start=int(start), end=int(end), kind=str(kind), region=str(region)))
        skipped.extend([dict(example=i, **v) for v in omitted])
        windows = {}
        for method, maps in methods.items():
            actual = breadth(maps['actual'][i:i+1])[0]
            for t, values in enumerate(actual):
                scores = np.convolve(values[lo:lo+length], np.ones(10), mode='valid')
                for sign, index in [('positive', int(np.argmax(scores))), ('negative', int(np.argmin(scores)))]:
                    if (sign == 'positive' and scores[index] <= 0) or (sign == 'negative' and scores[index] >= 0):
                        skipped.append(dict(example=i, method=method, target=t, sign=sign, reason='no_window_of_requested_sign'))
                        continue
                    windows.setdefault(lo+index, []).append(dict(example=i, method=method, target=t, sign=sign, score=float(scores[index])))
        for start, owners in windows.items():
            segment = codes[start:start+10]
            if len(np.unique(segment)) == 1:
                skipped.append(dict(example=i, start=start, reason='unshufflable_top_homopolymer'))
                continue
            rng = np.random.default_rng(seed_for('matched_top10', str(identifier), start, 20260922))
            for replicate in range(3):
                for attempt in range(100):
                    altered = rng.permutation(segment)
                    if not np.array_equal(altered, segment): break
                else: raise ValueError('No nonidentity top-window shuffle')
                seq = codes.copy(); seq[start:start+10] = altered
                row = len(sequences)
                sequences.append(seq)
                metadata.append(dict(example=i, start=start, end=start+10, kind='top_union', region='native'))
                selections.extend([dict(owner, mutation=row) for owner in owners])
    arrays = {key: np.asarray([r[key] for r in metadata]) for key in metadata[0]}
    return dict(sequence=np.stack(sequences), **arrays), selections, skipped


def mutation_estimate(hyp, data, mutations):
    estimated = []
    for i, changed in zip(mutations['example'], mutations['sequence']):
        wt = data['sequence'][i]
        positions = np.flatnonzero(changed != wt)
        g = hyp[i]
        estimated.append((g[:, changed[positions], positions]-g[:, wt[positions], positions]).sum(1))
    return np.asarray(estimated)


def scalar_metrics(exact, predicted):
    varying = len(exact) > 1 and np.std(exact) > 1e-10 and np.std(predicted) > 1e-10
    nonzero = np.abs(exact) > 1e-7
    return dict(n=len(exact), pearson=float(np.corrcoef(exact, predicted)[0, 1]) if varying else None,
        spearman=float(spearmanr(exact, predicted).statistic) if varying else None,
        mae=float(np.abs(exact-predicted).mean()) if len(exact) else None,
        sign_agreement=float((np.sign(exact[nonzero]) == np.sign(predicted[nonzero])).mean()) if nonzero.any() else None)


def fidelity(exact, predicted, mutations, data):
    a, b = breadth(exact), breadth(predicted)
    rows = []
    for kind, region in itertools.product(('snv', 'block_random', 'top_union'), ('native', 'flank')):
        mask = (mutations['kind'] == kind) & (mutations['region'] == region)
        if not mask.any(): continue
        for t, name in enumerate(NAMES):
            rows.append(dict(kind=kind, region=region, target=name, **scalar_metrics(a[mask, t], b[mask, t])))
    primary = []
    for i, identifier in enumerate(data['ids']):
        mask = (mutations['kind'] == 'snv') & (mutations['region'] == 'native') & (mutations['example'] == i)
        metrics = [scalar_metrics(a[mask, t], b[mask, t]) for t in range(8)]
        valid = [m['pearson'] if m['pearson'] is not None else 0. for t, m in enumerate(metrics) if np.std(a[mask, t]) > 1e-10]
        primary.append(dict(id=str(identifier), degree=int(data['labels'][i].sum()),
                            score=float(np.mean(valid)) if valid else None, contexts=metrics))
    valid = [r['score'] for r in primary if r['score'] is not None]
    return dict(macro_native_pearson=float(np.mean(valid)) if valid else None, per_enhancer=primary, groups=rows)


def top_effects(exact, mutations, selections):
    effects = breadth(exact)
    groups = {}
    for item in selections:
        key = (item['method'], item['example'], item['target'], item['sign'])
        groups.setdefault(key, []).append(item['mutation'])
    rows = []
    for (method, i, t, sign), indices in groups.items():
        random = ((mutations['example'] == i) & (mutations['kind'] == 'block_random') & (mutations['region'] == 'native'))
        factor = -1 if sign == 'positive' else 1
        top = float(factor*effects[indices, t].mean())
        null = float(factor*effects[random, t].mean()) if random.any() else None
        rows.append(dict(method=method, example=int(i), target=NAMES[t], sign=sign, n=len(indices),
            directional_effect=top, random_effect=null, excess_over_random=top-null if null is not None else None))
    return rows


def compare(first, second, data):
    rows = []
    for i, (lo, size) in enumerate(zip(data['native_offset'], data['native_length'])):
        for region, sl in [('native', slice(int(lo), int(lo+size))), ('full', slice(None))]:
            a, b = breadth(first['actual'])[i, :, sl], breadth(second['actual'])[i, :, sl]
            ah, bh = breadth(first['hypothetical'])[i, :, :, sl], breadth(second['hypothetical'])[i, :, :, sl]
            raw = map_agreement(ah.reshape(9, -1), bh.reshape(9, -1))
            centered = map_agreement((ah-ah.mean(1, keepdims=True)).reshape(9, -1), (bh-bh.mean(1, keepdims=True)).reshape(9, -1))
            for t, values in enumerate(map_agreement(a, b)):
                rows.append(dict(example=i, target=NAMES[t], region=region, **values,
                    hypothetical_cosine=raw[t]['cosine'], centered_hypothetical_cosine=centered[t]['cosine']))
    return rows


def native_export(data, maps):
    maximum = int(data['native_length'].max())
    sequences = np.zeros((len(data['ids']), maximum, 4), np.float32)
    hyp = np.zeros((len(data['ids']), 8, maximum, 4), np.float32)
    valid = np.zeros((len(data['ids']), maximum), bool)
    for i, (lo, size) in enumerate(zip(data['native_offset'], data['native_length'])):
        lo, size = int(lo), int(size)
        sequences[i, :size] = np.eye(4, dtype=np.float32)[data['sequence'][i, lo:lo+size]]
        hyp[i, :, :size] = maps['hypothetical'][i, :, :, lo:lo+size].transpose(0, 2, 1)
        valid[i, :size] = True
    return dict(one_hot=sequences, hypothetical=hyp, contribution=hyp*sequences[:, None],
                breadth_hypothetical=hyp.sum(1), breadth_contribution=hyp.sum(1)*sequences,
                valid=valid, lengths=data['native_length'], ids=data['ids'], targets=np.asarray(TARGETS[:8]))


def bootstrap_difference(first, second, degrees):
    diff = np.asarray(first, float)-np.asarray(second, float)
    valid = np.isfinite(diff)
    diff, degrees = diff[valid], np.asarray(degrees)[valid]
    if len(diff) < 2: return dict(mean=None, ci95=None, n=len(diff))
    rng = np.random.default_rng(20260922)
    groups = [np.flatnonzero(degrees == k) for k in np.unique(degrees)]
    draws = np.concatenate([rng.choice(g, (2000, len(g)), replace=True) for g in groups], axis=1)
    return dict(mean=float(diff.mean()), ci95=np.quantile(diff[draws].mean(1), [.025, .975]).tolist(), n=len(diff),
                units='Enhancers stratified by degree; exploratory, no multiplicity adjustment')


def run(project, root):
    package = root/'package'
    config = json.loads((package/'config.json').read_text())
    preflight_report = json.loads((root/'cpu_preflight.json').read_text())
    if (preflight_report['status'] != 'passed' or not preflight_report['no_cuda']
            or preflight_report['manifest_sha256'] != digest(package/'MANIFEST.sha256')
            or socket.gethostname() != config['host'] or os.environ.get('SLURM_JOB_ID')
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Requires matching CPU checks and the approved local CUDA GPU')
    for source, expected in config['source_hashes'].items():
        if digest(project/source) != expected: raise ValueError('Source changed: '+source)
    output = root/'run'
    output.mkdir(exist_ok=False)
    began, stopped = time.monotonic(), [False]
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def stop(): return stopped[0] or time.monotonic()-began >= config['max_seconds']
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = dict(status='running', pid=os.getpid(), gpu=torch.cuda.get_device_name(), targets=list(TARGETS[:8]),
        manifest_sha256=digest(package/'MANIFEST.sha256'), no_training=True,
        reused_ig=True, saved_ig_timing_not_hardware_matched=True, references=100)
    write_json(output/'runtime.json', report)
    event('matched_method_pilot_started', **report)
    data, methods = load(package)
    references, hashes = make_references(data, np.arange(16), config, None)
    np.testing.assert_array_equal(hashes, data['reference_hashes'])
    save_npz(output/'references.npz', codes=references, hashes=hashes, ids=data['ids'])
    target, adapted, report['adapter_changes'] = models(project, package, 'cuda')
    x = one_hot(data['sequence'], 'cuda')
    a, b = target.a.double(), target.b.double()
    report['gpu_preflight'] = check_real(target, adapted, x[:2], one_hot(references[:2, 0], 'cuda'), stop)
    event('matched_gpu_preflight_passed', **report['gpu_preflight'])
    sums = np.zeros((16, 8, 4, 2048), np.float64)
    squares, first_half = np.zeros_like(sums), None
    deltas = np.zeros((16, 100, 8), np.float64)
    differences = np.zeros_like(deltas)
    endpoint_errors, reference_outputs = [], np.zeros_like(deltas)
    torch.cuda.synchronize(); dl_start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for r in range(100):
        for first in range(0, 16, 8):
            if stop(): raise TimeoutError('Bounded pilot stopped; completed checkpoints retained')
            sl = slice(first, first+8)
            baseline = one_hot(references[sl, r], 'cuda')
            values = paired_calibrated_deeplift(adapted, x[sl].double(), baseline.double(), a, b, stop)
            with torch.no_grad():
                original_p = target.endpoints(baseline)['calibrated_probabilities']
                torch.testing.assert_close(values['reference_probabilities'], original_p, atol=2e-5, rtol=1e-5, check_dtype=False)
                original_wt = target.endpoints(x[sl])['calibrated_probabilities']
                torch.testing.assert_close(values['probabilities'], original_wt, atol=2e-5, rtol=1e-5, check_dtype=False)
            hyp = values['hypothetical'].cpu().numpy()
            sums[sl] += hyp; squares[sl] += hyp*hyp
            deltas[sl, r] = values['delta'].cpu().numpy()
            differences[sl, r] = values['difference'].cpu().numpy()
            reference_outputs[sl, r] = values['reference_probabilities'].cpu().numpy()
            endpoint_errors.append(float((values['reference_probabilities']-original_p).abs().max()))
            if not np.all(np.abs(deltas[sl, r]) <= .002+.05*np.abs(differences[sl, r])):
                raise ValueError('DeepLIFT per-reference completeness failure; do not adopt')
        if r == 49: first_half = sums/50
        if (r+1) % 10 == 0:
            save_npz(output/'deeplift_progress.npz', count=np.asarray(r+1), hypothetical=(sums/(r+1)).astype(np.float32),
                deltas=deltas[:, :r+1], differences=differences[:, :r+1])
            event('matched_deeplift_progress', references=r+1, total=100, elapsed_seconds=time.monotonic()-began)
    torch.cuda.synchronize()
    report['deeplift_seconds_including_checks_and_io'] = time.monotonic()-dl_start
    report['deeplift_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    hyp = sums/100
    actual = (hyp*x.cpu().numpy()[:, None]).sum(2)
    methods['deeplift'] = dict(hypothetical=hyp.astype(np.float32), actual=actual.astype(np.float32))
    save_npz(output/'deeplift.npz', **methods['deeplift'], hypothetical_first50=first_half.astype(np.float32),
        hypothetical_reference_se=np.sqrt(np.maximum(squares-sums*sums/100, 0)/(99*100)).astype(np.float32),
        deltas=deltas, differences=differences, reference_probabilities=reference_outputs, ids=data['ids'])
    report['deeplift_maximum_pair_delta'] = float(np.abs(deltas).max())
    report['deeplift_maximum_endpoint_error'] = max(endpoint_errors)
    def projected(h): return dict(hypothetical=h, actual=(h*x.cpu().numpy()[:, None]).sum(2))
    report['deeplift_first50_vs_last50'] = compare(projected(first_half), projected((sums-50*first_half)/50), data)
    report['deeplift_first50_vs100'] = compare(projected(first_half), methods['deeplift'], data)
    # Current paired-noise winner, fixed before this pilot; no tuning here.
    for repeat in (0, 1):
        if stop(): raise TimeoutError('Stopped before SmoothGrad')
        torch.cuda.synchronize(); sg_start = time.monotonic()
        noise = paired_gaussian_noise(data['ids'], 64, 2048, repeat=repeat)
        values = smoothgrad(target, x, torch.as_tensor(data['labels'], device='cuda'), noise, .1, (32, 64), 16, stop, paired=True, target_count=8)
        torch.cuda.synchronize()
        name = 'smoothgrad_paired' if repeat == 0 else 'smoothgrad_paired_repeat'
        methods[name] = dict(hypothetical=values[64]['centered_sensitivity'][:, :8], actual=values[64]['observed_sensitivity'][:, :8])
        save_npz(output/(name+'.npz'), **methods[name], ids=data['ids'])
        report[name+'_seconds'] = time.monotonic()-sg_start
    report['smoothgrad_repeat_agreement'] = compare(methods['smoothgrad_paired'], methods['smoothgrad_paired_repeat'], data)
    del methods['smoothgrad_paired_repeat'], methods['smoothgrad_iid_repeat']
    mutations, selections, skipped = edits(data, methods)
    write_json(output/'selected_windows.json', dict(selections=selections, skipped=skipped))
    event('matched_perturbations_started', mutants=len(mutations['example']))
    with torch.no_grad():
        wt = target.endpoints(x)['calibrated_probabilities'].cpu().numpy()
        np.testing.assert_allclose(wt, data['calibrated_probabilities'], atol=3e-5, rtol=2e-3)
        exact = []
        for first in range(0, len(mutations['example']), 16):
            if stop(): raise TimeoutError('Stopped during perturbation inference')
            sl = slice(first, first+16)
            prediction = target.endpoints(one_hot(mutations['sequence'][sl], 'cuda'))['calibrated_probabilities'].cpu().numpy()
            exact.append(prediction-wt[mutations['example'][sl]])
    exact = np.concatenate(exact)
    if not np.isfinite(exact).all(): raise ValueError('Nonfinite mutation predictions')
    estimates = {name: mutation_estimate(maps['hypothetical'], data, mutations) for name, maps in methods.items()}
    save_npz(output/'mutations.npz', **mutations, exact=exact, **estimates, ids=data['ids'])
    report['fidelity'] = {name: fidelity(exact, estimates[name], mutations, data) for name in methods}
    report['top_window_effects'] = top_effects(exact, mutations, selections)
    report['agreement_vs_ig'] = {name: compare(maps, methods['ig'], data) for name, maps in methods.items() if name != 'ig'}
    report['primary_differences'] = {}
    for left, right in itertools.combinations(methods, 2):
        scores = [[r['score'] if r['score'] is not None else np.nan for r in report['fidelity'][name]['per_enhancer']] for name in (left, right)]
        report['primary_differences'][left+'_minus_'+right] = bootstrap_difference(*scores, data['labels'].sum(1))
    for name, maps in methods.items():
        exported = native_export(data, maps)
        save_npz(output/(name+'_native_modisco_inputs.npz'), **exported)
    report.update(status='complete', elapsed_seconds=time.monotonic()-began,
        modisco_discovery_run=False, scope='16 training enhancers: exploratory method fidelity, not biological validation.',
        files={p.name: digest(p) for p in output.glob('*.npz')})
    write_json(output/'result.json', report)
    event('matched_method_pilot_complete', seconds=report['elapsed_seconds'],
          primary={name: row['macro_native_pearson'] for name, row in report['fidelity'].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'cpu', 'run'))
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = args.root.resolve() if args.root else project/'results'/NAME
    require_pinned_captum()
    {'prepare': prepare, 'cpu': preflight, 'run': run}[args.stage](project, root)


if __name__ == '__main__':
    main()
