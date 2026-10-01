"""Matched local GPU timing only; no attribution-quality or motif rerun."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

import numpy as np
import torch

from classifier_motifs.attribution import one_hot
from classifier_motifs.calibrated_attribution import integrate, TARGETS
from classifier_motifs.calibrated_deeplift import paired_calibrated_deeplift
from .calibrated_context import save_npz
from .common import digest, event, write_json
from .method_pilot import models


NAME = 'classifier_attribution_timing_20260922_r1'
PARENT = 'results/classifier_method_pilot_20260922_r1'
PARENT_SHA = '81a45f05096e886efcc1dc9e5640d451aa13a632a859c5fd56329f6e28d75712'
METHODS = ('ig64_batch64', 'deeplift', 'ig64_batch128')


def order(repeat):
    return METHODS[repeat % 3:]+METHODS[:repeat % 3]


def timing_summary(rows):
    result = {}
    for method in METHODS:
        values = [r for r in rows if r['method'] == method]
        if len(values) != 3 or {r['repeat'] for r in values} != {0, 1, 2}:
            raise ValueError('Three distinct timed repeats per method required')
        seconds = [r['seconds'] for r in values]
        if not all(np.isfinite(s) and s > 0 for s in seconds):
            raise ValueError('Invalid timing')
        result[method] = dict(median_seconds=float(np.median(seconds)),
            minimum_seconds=min(seconds), maximum_seconds=max(seconds),
            median_seconds_per_enhancer_reference=float(np.median(seconds)/32),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in values))
    best = min((m for m in METHODS if m.startswith('ig')), key=lambda m: result[m]['median_seconds'])
    return dict(methods=result, fastest_tested_ig=best,
        deeplift_seconds_divided_by_ig=result['deeplift']['median_seconds']/result[best]['median_seconds'],
        interpretation='Ratio >1 means DeepLIFT is slower on this GPU with these validated precisions. Not a universal algorithm ranking.')


def prepare(project, root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only preparation')
    prior = project/PARENT
    if digest(prior/'package/MANIFEST.sha256') != PARENT_SHA:
        raise ValueError('Unexpected validated parent')
    subprocess.run(['sha256sum', '--quiet', '-c', 'MANIFEST.sha256'], cwd=prior/'package', check=True)
    report = json.loads((prior/'run/result.json').read_text())
    if report['status'] != 'complete' or digest(prior/'run/references.npz') != report['files']['references.npz']:
        raise ValueError('Completed parent and verified references required')
    package = root/'package'
    package.mkdir(parents=True, exist_ok=False)
    for folder in ('code', 'src'):
        shutil.copytree(prior/'package'/folder, package/folder, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('classifier_motifs/calibrated_attribution.py', 'classifier_modisco/attribution_timing.py'):
        shutil.copy2(project/'experiments'/name, package/'code'/name)
    (package/'tests').mkdir()
    for name in ('test_calibrated_attribution.py', 'test_attribution_timing.py'):
        shutil.copy2(project/'tests'/name, package/'tests'/name)
    shutil.copy2(prior/'package/calibrators.json', package/'calibrators.json')
    shutil.copy2(project/'scripts/run_attribution_timing_local.sh', package/'run_local.sh')
    with np.load(prior/'package/ig.npz', allow_pickle=False) as z:
        chosen = [int(np.flatnonzero(z['labels'].sum(1) == degree)[0]) for degree in range(1, 9)]
        data = {key: z[key][chosen] for key in ('ids', 'labels', 'sequence')}
        reference_hashes = z['reference_hashes'][chosen][:, [0, 17, 49, 99]]
    with np.load(prior/'run/references.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'][chosen], data['ids'])
        np.testing.assert_array_equal(z['hashes'][chosen][:, [0, 17, 49, 99]], reference_hashes)
        references = z['codes'][chosen][:, [0, 17, 49, 99]]
    save_npz(package/'examples.npz', **data, references=references, reference_hashes=reference_hashes,
             original_indices=np.asarray(chosen), reference_indices=np.asarray([0, 17, 49, 99]))
    old = json.loads((prior/'package/config.json').read_text())
    config = dict(checkpoint=old['checkpoint'], checkpoint_sha256=old['source_hashes'][old['checkpoint']],
        parent=PARENT, parent_manifest_sha256=PARENT_SHA, host=socket.gethostname(),
        examples=8, references=4, pair_batch=8, repeats=3, targets=list(TARGETS[:8]),
        methods=list(METHODS), max_seconds=540, ig_steps=64,
        precisions=dict(ig='float32', deeplift='float64, preserving explicit internal FP32 casts'),
        scope='One local CUDA timing benchmark, ten-minute hard cap, no cluster or production changes.',
        timing='CUDA-synchronized wall time for attribution calls only, including each method internal validation. Excludes warmup, setup, input conversion, external verification and disk IO.')
    write_json(package/'config.json', config)
    files = sorted(p for p in package.rglob('*') if p.is_file())
    (package/'MANIFEST.sha256').write_text(''.join(digest(p)+'  '+str(p.relative_to(package))+'\n' for p in files))
    event('timing_prepared', manifest=digest(package/'MANIFEST.sha256'), root=str(root))


def cpu_check(project, root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Hide CUDA for CPU checks')
    torch.set_num_threads(2)
    package = root/'package'
    with np.load(package/'examples.npz', allow_pickle=False) as z:
        x = one_hot(z['sequence'][:1], 'cpu')
        baseline = one_hot(z['references'][:1, 0], 'cpu')
        labels = torch.as_tensor(z['labels'][:1])
    target, _, _ = models(project, package, 'cpu')
    nine = integrate(target, x, baseline, labels, steps=8, internal_batch=8)
    eight = integrate(target, x, baseline, labels, steps=8, internal_batch=8, target_count=8)
    errors = {}
    for key in nine:
        torch.testing.assert_close(eight[key], nine[key][:, :8], atol=3e-5, rtol=2e-3)
        errors[key] = float((eight[key]-nine[key][:, :8]).abs().max())
    report = dict(status='passed', no_cuda=not torch.cuda.is_initialized(),
        manifest_sha256=digest(package/'MANIFEST.sha256'), real_model_eight_vs_nine=errors,
        note='Eight quadrature points only for this CPU target-selection equivalence check; timed GPU runs use64.')
    if (root/'cpu_preflight.json').exists(): raise ValueError('CPU report already exists')
    write_json(root/'cpu_preflight.json', report)
    event('timing_cpu_passed', **report)


def run(project, root):
    package = root/'package'
    config = json.loads((package/'config.json').read_text())
    gate = json.loads((root/'cpu_preflight.json').read_text())
    if (gate['status'] != 'passed' or not gate['no_cuda']
            or gate['manifest_sha256'] != digest(package/'MANIFEST.sha256')
            or socket.gethostname() != config['host'] or os.environ.get('SLURM_JOB_ID')
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Approved local GPU and matching CPU gate required')
    if digest(project/config['checkpoint']) != config['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    output = root/'run'; output.mkdir(exist_ok=False)
    began, stopped = time.monotonic(), [None]
    def record_signal(signum, _frame):
        stopped[0] = signal.Signals(signum).name
        event('matched_timing_stop_signal', signal=stopped[0], elapsed_seconds=time.monotonic()-began)
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, record_signal)
    def stop():
        elapsed = time.monotonic()-began
        reason = stopped[0] or ('time_budget' if elapsed >= config['max_seconds'] else None)
        if reason:
            write_json(output/'stopped.json', dict(status='stopped', reason=reason,
                elapsed_seconds=elapsed, pid=os.getpid()))
        return reason is not None
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with np.load(package/'examples.npz', allow_pickle=False) as z: data = dict(z)
    if data['references'].shape != (8, 4, 2048): raise ValueError('Wrong paired input shape')
    target, adapted, _ = models(project, package, 'cuda')
    indices = np.repeat(np.arange(8), 4)
    x = one_hot(data['sequence'][indices], 'cuda')
    baseline = one_hot(data['references'].reshape(32, 2048), 'cuda')
    labels = torch.as_tensor(data['labels'][indices], device='cuda')
    double_x, double_baseline = x.double(), baseline.double()
    a, b = target.a.double(), target.b.double()
    with torch.no_grad():
        expected = target.endpoints(x)['calibrated_probabilities']-target.endpoints(baseline)['calibrated_probabilities']
    def calculate(method, sl):
        if method == 'deeplift':
            return paired_calibrated_deeplift(adapted, double_x[sl], double_baseline[sl], a, b, stop)
        return integrate(target, x[sl], baseline[sl], labels[sl], steps=64,
            internal_batch=int(method.removeprefix('ig64_batch')), should_stop=stop, target_count=8)
    report = dict(status='running', pid=os.getpid(), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda, no_training=True,
        config=config, manifest_sha256=digest(package/'MANIFEST.sha256'), rows=[])
    write_json(output/'runtime.json', report)
    event('matched_timing_started', gpu=report['gpu'], pid=report['pid'], examples=8, references=4, repeats=3)
    reference_maps = {}
    for method in METHODS:
        if stop(): raise TimeoutError('Stopped before warmup')
        torch.cuda.synchronize()
        warmup_started = time.perf_counter()
        value = calculate(method, slice(0, 8))
        torch.cuda.synchronize()
        event('matched_timing_warmup_method_complete', method=method,
              seconds=time.perf_counter()-warmup_started)
        if not (value['delta'].abs() <= .002+.05*expected[:8].abs()).all():
            raise ValueError('Warmup conservation failed: '+method)
        del value
    event('matched_timing_warmup_complete')
    for repeat in range(3):
        for method in order(repeat):
            if stop(): raise TimeoutError('Bounded timing benchmark stopped')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            outputs = []
            start = time.perf_counter()
            for first in range(0, 32, 8): outputs.append(calculate(method, slice(first, first+8)))
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            peak = torch.cuda.max_memory_allocated()
            # Verification and transfers occur AFTER the timed block.
            hyp = torch.cat([v['hypothetical'] for v in outputs])
            actual = torch.cat([v['actual'] for v in outputs])
            difference = torch.cat([v['difference'] if method == 'deeplift' else v['target_difference'] for v in outputs])
            torch.testing.assert_close(difference, expected, atol=2e-5, rtol=1e-5, check_dtype=False)
            delta = actual.sum(2)-expected
            passed = delta.abs() <= .002+.05*expected.abs()
            if not passed.all(): raise ValueError('Timed method conservation failed: '+method)
            cpu_hyp = hyp.cpu().numpy()
            if method not in reference_maps:
                reference_maps[method] = cpu_hyp
            else:
                np.testing.assert_allclose(cpu_hyp, reference_maps[method], atol=3e-5, rtol=2e-3)
            if method == 'ig64_batch128':
                np.testing.assert_allclose(cpu_hyp, reference_maps['ig64_batch64'], atol=3e-5, rtol=2e-3)
            row = dict(method=method, repeat=repeat, seconds=seconds,
                peak_allocated_bytes=peak, failed_conservation_checks=int((~passed).sum()),
                maximum_absolute_delta=float(delta.abs().max()))
            report['rows'].append(row)
            write_json(output/'progress.json', report)
            event('matched_timing_repeat_complete', **row)
            del outputs, hyp, actual, difference, delta, passed
    report.update(status='complete', summary=timing_summary(report['rows']), elapsed_seconds=time.monotonic()-began)
    save_npz(output/'maps_for_verification.npz', **reference_maps, ids=data['ids'], pair_example=indices,
             reference_indices=data['reference_indices'], targets=np.asarray(TARGETS[:8]))
    report['maps_sha256'] = digest(output/'maps_for_verification.npz')
    write_json(output/'result.json', report)
    event('matched_timing_complete', summary=report['summary'], elapsed_seconds=report['elapsed_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'cpu', 'run'))
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = args.root.resolve() if args.root else project/'results'/NAME
    {'prepare': prepare, 'cpu': cpu_check, 'run': run}[args.stage](project, root)
