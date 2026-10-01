"""Bounded local eight-context SmoothGrad cohort, separate from production IG."""
import argparse
import csv
import gzip
import hashlib
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

from classifier_motifs.attribution import active_weights, one_hot
from classifier_motifs.calibrated_attribution import CalibratedTargets, TARGETS, load_classifier
from classifier_motifs.smoothgrad import paired_gaussian_noise, smoothgrad
from .calibrated_context import save_npz
from .common import digest, event, write_json
from .original_intervals import join_intervals
from .smoothgrad_benchmark import require_benchmark_gpu


NAME = 'classifier_smoothgrad_context1000_20260923'
CHECKPOINT = 'results/classifier_calibrated_context_20260921/inputs/best_model.pt'
SEED = 20260923


def select_examples(data, per_degree=125):
    """Training-only, equal exact-degree counts, no overlapping input windows/RC duplicates."""
    n = len(data['ids'])
    if (per_degree < 1 or len(np.unique(data['ids'])) != n
            or data['sequence'].shape != (n, 2048)
            or not np.isin(data['sequence'], range(4)).all()
            or data['labels'].shape != (n, 8)
            or not np.isin(data['labels'], [0, 1]).all()
            or not data['labels'].any(1).all()):
        raise ValueError('Malformed enhancer-only training population')
    rng = np.random.default_rng(SEED)
    degree = data['labels'].sum(1)
    used, windows, groups = set(), [], {}
    for k in range(8, 0, -1):
        chosen = []
        for i in rng.permutation(np.flatnonzero(degree == k)):
            chrom, summit = str(data['chrom'][i]), int(data['summit'][i])
            if any(c == chrom and abs(s-summit) < 2048 for c, s in windows):
                continue
            sequence = data['sequence'][i].astype(np.uint8)
            key = min(hashlib.sha256(sequence.tobytes()).digest(),
                      hashlib.sha256((3-sequence[::-1]).tobytes()).digest())
            if key in used:
                continue
            chosen.append(int(i)); used.add(key); windows.append((chrom, summit))
            if len(chosen) == per_degree:
                break
        if len(chosen) != per_degree:
            raise ValueError(f'Insufficient non-overlapping enhancers for degree {k}')
        groups[k] = chosen
    # Every eight rows span all degrees, keeping partial outputs balanced too.
    return np.asarray([groups[k][j] for j in range(per_degree) for k in range(1, 9)])


def prepare(project, root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Preparation must hide CUDA')
    parent = project/'results/classifier_method_pilot_20260922_r1/package'
    subprocess.run(['sha256sum', '--quiet', '-c', 'MANIFEST.sha256'], cwd=parent, check=True)
    training = project/'results/classifier_background_training_20260916/data/train.npz'
    original = project/'results/classifier_modisco_original_20260917/package'
    catalog = original/'original_catalog.tsv.gz'
    if digest(catalog) != json.loads((original/'config.json').read_text())['catalog_sha256']:
        raise ValueError('Native catalog checksum changed')
    current = project/'results/classifier_calibrated_full8_20260923/package/calibrators.json'
    if digest(current) != digest(parent/'calibrators.json'):
        raise ValueError('Calibration differs from the ongoing eight-context IG run')
    with np.load(training, allow_pickle=False) as saved:
        data = dict(saved)
    selected = select_examples(data)
    with gzip.open(catalog, 'rt') as handle:
        intervals = join_intervals(data, csv.DictReader(handle, delimiter='\t'))
    examples = {key: value[selected] for key, value in data.items()}
    examples.update({f'native_{key}': intervals[key][selected]
                     for key in ('start', 'end', 'offset', 'length')})
    examples.update(source_train_rows=selected, split=np.asarray(['train']*len(selected)))
    package = root/'package'
    package.mkdir(parents=True, exist_ok=False)
    for folder in ('code', 'src'):
        shutil.copytree(parent/folder, package/folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(__file__), package/'code/classifier_modisco/smoothgrad_cohort.py')
    shutil.copy2(parent/'calibrators.json', package/'calibrators.json')
    replay = project/'results/classifier_smoothgrad_refinement_20260922/package_r1/endpoint_replay.npz'
    shutil.copy2(replay, package/'endpoint_replay.npz')
    (package/'tests').mkdir()
    for name in ('test_calibrated_attribution.py', 'test_smoothgrad.py',
                 'test_smoothgrad_refinement.py', 'test_smoothgrad_cohort.py'):
        shutil.copy2(project/'tests'/name, package/'tests'/name)
    shutil.copy2(project/'scripts/run_smoothgrad_context1000_local.sh', package/'run_local.sh')
    save_npz(package/'examples.npz', **examples)
    write_json(package/'provenance.json', dict(local_hostname=socket.gethostname().split('.')[0],
        parent_manifest_sha256=digest(parent/'MANIFEST.sha256'), checkpoint=CHECKPOINT,
        inputs={str(p.relative_to(project)): digest(p) for p in (training, catalog, replay, project/CHECKPOINT)},
        selection_seed=SEED, examples=1000, degree_counts=[125]*8,
        selection='Training only; exact degrees1-8; unique DNA/RC; non-overlapping2048bp inputs; round-robin degree order.',
        noise_seed=20260922, noise_samples=64, independent_pairs=32, sigma=.1, noise_repeat=0,
        targets=list(TARGETS[:8]), chunk_size=16, internal_batch=16,
        max_seconds=1680, hard_limit_seconds=1800,
        interpretation='Signed local sensitivities, not reference-based IG contributions. Raw-gradient SE only; no independence assumed across contexts.',
        scope='Authorized local CUDA1000-enhancer calculation; no training, remote mutations, automatic retry or motif discovery.'))
    files = sorted(p for p in package.rglob('*') if p.is_file())
    (package/'MANIFEST.sha256').write_text(''.join(
        digest(p)+'  '+str(p.relative_to(package))+'\n' for p in files))
    event('smoothgrad_cohort_prepared', root=str(root), examples=len(selected),
          manifest_sha256=digest(package/'MANIFEST.sha256'))


def check_maps(values, sequence):
    n, length = sequence.shape
    for key in ('gradient', 'gradient_noise_se', 'centered_sensitivity'):
        if values[key].shape != (n, 8, 4, length) or not np.isfinite(values[key]).all():
            raise ValueError('Malformed/nonfinite '+key)
    if (values['gradient_noise_se'] < 0).any():
        raise ValueError('Negative noise standard error')
    centered = values['gradient']-values['gradient'].mean(2, keepdims=True)
    np.testing.assert_allclose(values['centered_sensitivity'], centered, atol=3e-6, rtol=2e-3)
    observed = np.take_along_axis(values['centered_sensitivity'],
        sequence[:, None, None, :].astype(int), axis=2)[:, :, 0]
    np.testing.assert_allclose(values['observed_sensitivity'], observed, atol=3e-6, rtol=2e-3)


def device_check(target, package, device):
    """Real-model endpoint replay, batching, and signed breadth identity."""
    with np.load(package/'endpoint_replay.npz', allow_pickle=False) as saved:
        replay = dict(saved)
    predictions = []
    with torch.no_grad():
        for first in range(0, len(replay['sequence']), 4):
            predictions.append(target.endpoints(one_hot(replay['sequence'][first:first+4], device))[
                'probabilities'].cpu().numpy())
    error = float(np.abs(np.concatenate(predictions)-replay['probabilities']).max())
    if error > .005:
        raise ValueError('Checkpoint endpoint replay failed')
    with np.load(package/'examples.npz', allow_pickle=False) as saved:
        sequence, labels, ids = saved['sequence'][:2], saved['labels'][:2], saved['ids'][:2]
    x, labels = one_hot(sequence, device), torch.as_tensor(labels, device=device)
    noise = paired_gaussian_noise(ids, 4, 2048)
    small = smoothgrad(target, x, labels, noise, .1, (4,), 2, paired=True, target_count=8)[4]
    large = smoothgrad(target, x, labels, noise, .1, (4,), 4, paired=True, target_count=8)[4]
    for key in small:
        np.testing.assert_allclose(small[key], large[key], atol=3e-5, rtol=2e-3)
    direct = []
    for epsilon in noise:
        points = (x+.1*torch.as_tensor(epsilon, device=device)).requires_grad_(True)
        score = target(points, active_weights(labels))[:, :8].sum()
        direct.append(torch.autograd.grad(score, points)[0].cpu().numpy())
    summed = small['gradient'].sum(1)
    np.testing.assert_allclose(summed, np.mean(direct, 0), atol=3e-5, rtol=2e-3)
    check_maps(small, sequence)
    return dict(endpoint_replay_max_error=error,
        batch_gradient_max_error=float(np.abs(small['gradient']-large['gradient']).max()),
        breadth_sum_max_error=float(np.abs(summed-np.mean(direct, 0)).max()))


def execute(project, root, cpu=False):
    package = root/'package'
    subprocess.run(['sha256sum', '--quiet', '-c', 'MANIFEST.sha256'], cwd=package, check=True)
    meta = json.loads((package/'provenance.json').read_text())
    signature = digest(package/'MANIFEST.sha256')
    if cpu:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
            raise RuntimeError('CPU checks must hide CUDA')
        if (root/'cpu_preflight.json').exists():
            raise ValueError('Refusing to overwrite CPU checks')
    else:
        require_benchmark_gpu(True, meta)
        preflight = json.loads((root/'cpu_preflight.json').read_text())
        if preflight['status'] != 'passed' or preflight['manifest_sha256'] != signature:
            raise ValueError('Passing checks for this frozen package required')
    began, stopped = time.monotonic(), [False]
    output, files, completed = root/'run', {}, 0
    if not cpu:
        output.mkdir(exist_ok=False)
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def stop():
        return stopped[0] or time.monotonic()-began >= meta['max_seconds'] or (root/'STOP').exists()
    torch.set_num_threads(2 if cpu else 4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = 'cpu' if cpu else 'cuda'
    target = CalibratedTargets(load_classifier(project/meta['checkpoint'], device),
        json.loads((package/'calibrators.json').read_text())['enhancers_only']).to(device).eval()
    checks = device_check(target, package, 'cpu' if cpu else 'cuda')
    receipt = dict(status='passed', manifest_sha256=signature, checks=checks,
        checkpoint_sha256=digest(project/meta['checkpoint']), no_cuda=not torch.cuda.is_initialized())
    if cpu:
        write_json(root/'cpu_preflight.json', receipt)
        event('smoothgrad_cohort_cpu_passed', **receipt)
        return
    write_json(output/'gpu_preflight.json', receipt)
    runtime = dict(status='running', pid=os.getpid(), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, numpy=np.__version__, no_training=True,
        manifest_sha256=signature, targets=meta['targets'], max_seconds=meta['max_seconds'])
    write_json(output/'runtime.json', runtime)
    event('smoothgrad_cohort_started', **runtime)
    with np.load(package/'examples.npz', allow_pickle=False) as saved:
        data = dict(saved)
    if (len(data['ids']) != 1000 or not (data['split'] == 'train').all()
            or not np.array_equal(np.bincount(data['labels'].sum(1).astype(int), minlength=9)[1:], [125]*8)):
        raise ValueError('Expected1000 balanced training enhancers')
    try:
        for first in range(0, len(data['ids']), 16):
            if stop():
                raise TimeoutError('SmoothGrad cohort reached its time/stop boundary')
            sl = slice(first, first+16)
            batch = {key: value[sl] for key, value in data.items()}
            x = one_hot(batch['sequence'], 'cuda')
            labels = torch.as_tensor(batch['labels'], device='cuda')
            noise = paired_gaussian_noise(batch['ids'], 64, 2048)
            torch.cuda.synchronize()
            chunk_began = time.monotonic()
            values = smoothgrad(target, x, labels, noise, .1, (32, 64), 16,
                                stop, paired=True, target_count=8)
            check_maps(values[64], batch['sequence'])
            with torch.no_grad():
                endpoints = {key: value.cpu().numpy() for key, value in target.endpoints(x).items()}
            path = output/f'chunk_{first:06d}.npz'
            save_npz(path, **batch, **values[64], **endpoints,
                observed_prefix32=values[32]['observed_sensitivity'],
                targets=np.asarray(TARGETS[:8]), signature=np.asarray(signature))
            files[path.name] = digest(path)
            completed += len(batch['ids'])
            progress = dict(status='running', completed=completed, total=1000,
                elapsed_seconds=time.monotonic()-began, chunk_seconds=time.monotonic()-chunk_began,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), files=files)
            write_json(output/'progress.json', progress)
            event('smoothgrad_cohort_progress', **{k:v for k,v in progress.items() if k != 'files'})
    except TimeoutError as exc:
        write_json(output/'stopped.json', dict(status='stopped', reason=str(exc), completed=completed,
            elapsed_seconds=time.monotonic()-began, files=files, manifest_sha256=signature))
        event('smoothgrad_cohort_stopped', completed=completed, reason=str(exc))
        return
    write_json(output/'complete.json', dict(status='complete', completed=completed,
        elapsed_seconds=time.monotonic()-began, files=files, manifest_sha256=signature))
    event('smoothgrad_cohort_complete', completed=completed, elapsed_seconds=time.monotonic()-began)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'cpu', 'run'))
    parser.add_argument('--project', type=Path, default=Path('.'))
    args = parser.parse_args()
    project = args.project.resolve()
    root = project/'results'/NAME
    if args.stage == 'prepare':
        prepare(project, root)
    else:
        execute(project, root, cpu=args.stage == 'cpu')
