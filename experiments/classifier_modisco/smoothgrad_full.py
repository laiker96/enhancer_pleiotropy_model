"""Full-cohort eight-context SmoothGrad and three independent native motif fits.

GPU attribution is checkpointed in disjoint Slurm shards. CPU-only discovery
includes all data splits, without filtering on model correctness or activity.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time

import numpy as np

from .common import digest, event, require_allocation, write_json
from .motif_recovery import save
from .original_intervals import load_intervals

NAME = 'classifier_smoothgrad_full_20260923'
GROUPS = (('degree_1', 1, 1), ('degree_2_5', 2, 5), ('degree_6_8', 6, 8))
META = ('ids', 'sequence', 'labels', 'split', 'chrom', 'summit')


def group_indices(labels):
    if labels.ndim != 2 or labels.shape[1] != 8 or not np.isin(labels, [0, 1]).all() or not labels.any(1).all():
        raise ValueError('Expected eight binary labels and positive observed breadth')
    degree = labels.sum(1)
    return [np.flatnonzero((degree >= lo) & (degree <= hi)) for _, lo, hi in GROUPS]


def batches(n, shard, shards=4, batch=16):
    if n < 1 or not 0 <= shard < shards or batch < 1:
        raise ValueError('Invalid shard dimensions')
    return [np.arange(first, min(first+batch, n)) for i, first in enumerate(range(0, n, batch))
            if i % shards == shard]


def inputs(project, root):
    config = json.loads((root/'config.json').read_text())
    for name, expected in config['source_hashes'].items():
        if digest(project/name) != expected:
            raise ValueError('Changed source: '+name)
    with np.load(project/config['cohort'], allow_pickle=False) as stored:
        data = {k:stored[k] for k in META}
    n = len(data['ids'])
    if (n != 40338 or len(np.unique(data['ids'])) != n or data['sequence'].shape != (n, 2048)
            or not np.isin(data['sequence'], range(4)).all()
            or not np.isin(data['split'], ['train', 'validation', 'test']).all()
            or [len(i) for i in group_indices(data['labels'])] != [23100, 15524, 1714]):
        raise ValueError('Invalid full enhancer cohort')
    intervals = load_intervals(project/config['original'], data)
    data.update({f'native_{k}': intervals[k] for k in ('start', 'end', 'offset', 'length')})
    if (not np.array_equal(data['native_end']-data['native_start'], data['native_length'])
            or (data['native_length'] < 30).any() or (data['native_offset'] < 0).any()
            or (data['native_offset']+data['native_length'] > 2048).any()):
        raise ValueError('Invalid native coordinates')
    return config, data


def validate_chunk(values, indices, data, signature):
    if str(values['signature']) != signature:
        raise ValueError('Wrong SmoothGrad package signature')
    np.testing.assert_array_equal(values['indices'], indices)
    np.testing.assert_array_equal(values['targets'],
        ['calibrated_probability_'+c for c in ('ab', 'e13', 'e5', 'ead', 'hid', 'lb', 'o', 'wid')])
    for name in data:
        np.testing.assert_array_equal(values[name], data[name][indices])
    for name in ('gradient', 'gradient_noise_se', 'centered_sensitivity'):
        if values[name].shape != (len(indices), 8, 4, 2048) or not np.isfinite(values[name]).all():
            raise ValueError('Malformed maps: '+name)
    for name in ('observed_sensitivity', 'observed_prefix32'):
        if values[name].shape != (len(indices), 8, 2048) or not np.isfinite(values[name]).all():
            raise ValueError('Malformed maps: '+name)
    if (values['gradient_noise_se'] < 0).any():
        raise ValueError('Negative noise standard error')
    centered = values['gradient']-values['gradient'].mean(2, keepdims=True)
    np.testing.assert_allclose(centered, values['centered_sensitivity'], atol=3e-6, rtol=2e-3)
    observed = np.take_along_axis(centered, values['sequence'][:, None, None, :].astype(int), axis=2)[:, :, 0]
    np.testing.assert_allclose(observed, values['observed_sensitivity'], atol=3e-6, rtol=2e-3)
    for name in ('probabilities', 'calibrated_probabilities'):
        p = values[name]
        if p.shape != (len(indices), 8) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Invalid endpoint probabilities')


def execution_contract(config, local_cuda=False):
    """Local batch64 is opt-in; keep the original remote contract unchanged."""
    expected = (1, 64) if local_cuda else (4, 16)
    if (config['samples'] != 64 or config['sigma'] != .1 or config['seed'] != 20260922
            or config['repeat'] != 0 or config['batch'] != 16
            or (config['shards'], config['internal_batch']) != expected
            or bool(config.get('local_hostname')) != local_cuda):
        raise ValueError('Changed validated SmoothGrad execution contract')


def run_gpu(project, root, shard, local_cuda=False):
    import torch
    from classifier_motifs.attribution import one_hot
    from classifier_motifs.calibrated_attribution import CalibratedTargets, TARGETS, load_classifier
    from classifier_motifs.smoothgrad import paired_gaussian_noise, smoothgrad
    from .smoothgrad_cohort import device_check

    config = json.loads((root/'config.json').read_text())
    execution_contract(config, local_cuda)
    host = socket.gethostname().split('.')[0]
    if local_cuda:
        from .smoothgrad_benchmark import require_benchmark_gpu
        require_benchmark_gpu(True, config)
        if 'GTX 1060' not in torch.cuda.get_device_name():
            raise RuntimeError('The local run requires the benchmarked GTX1060')
        benchmark = json.loads((project/config['batch_benchmark']).read_text())
        if (benchmark['status'] != 'complete' or benchmark['best']['batch'] != 64
                or not benchmark['best']['eligible'] or benchmark['best']['repetitions'] != 3):
            raise ValueError('Passing batch64 benchmark required')
    elif (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST')
            or host not in ('a100', 'xg07', 'xg08', 'xg09')
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('One compatible Slurm GPU required; never a login node')
    config, data = inputs(project, root)
    if config['targets'] != list(TARGETS[:8]):
        raise ValueError('Changed eight calibrated output contract')
    selected = batches(len(data['ids']), shard, config['shards'], config['batch'])
    signature = digest(root/'MANIFEST.sha256')
    directory = root/'chunks'; directory.mkdir(exist_ok=True)
    # Exclusive shard lock prevents accidental simultaneous writes. On forced
    # termination a stale lock is intentionally left for manual inspection.
    lock = root/f'lock_{shard}'
    job = str(os.getpid()) if local_cuda else os.environ['SLURM_JOB_ID']
    with lock.open('x') as handle:
        handle.write(job+'\n')
    began, stopping = time.monotonic(), [False]
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    def stop():
        return stopping[0] or (root/'STOP').exists() or time.monotonic()-began >= config['max_seconds']
    files, completed = {}, 0
    identity = f'{shard}_{job}'
    try:
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        target = CalibratedTargets(load_classifier(project/config['checkpoint'], 'cuda'),
            json.loads((root/'calibrators.json').read_text())['enhancers_only']).to('cuda').eval()
        runtime = dict(job=job, pid=os.getpid(), local_cuda=local_cuda, shard=shard, node=host,
            gpu=torch.cuda.get_device_name(), torch=torch.__version__, numpy=np.__version__,
            no_training=True, lr=None, signature=signature)
        write_json(root/f'runtime_{identity}.json', runtime); event('smoothgrad_full_started', **runtime)
        checked = device_check(target, root, 'cuda')
        write_json(root/f'gate_{identity}.json', dict(status='passed', **checked, signature=signature))
        event('smoothgrad_full_gate_passed', **checked)
        for indices in selected:
            if stop(): raise TimeoutError('Stopped before next checkpoint chunk')
            path = directory/f'chunk_{indices[0]:06d}.npz'
            chunk_began = time.monotonic()
            if path.exists():
                with np.load(path, allow_pickle=False) as stored: values = dict(stored)
                validate_chunk(values, indices, data, signature)
            else:
                batch = {k:v[indices] for k,v in data.items()}
                x = one_hot(batch['sequence'], 'cuda')
                labels = torch.as_tensor(batch['labels'], device='cuda')
                noise = paired_gaussian_noise(batch['ids'], 64, 2048, config['seed'], config['repeat'])
                maps = smoothgrad(target, x, labels, noise, .1, (32, 64), config['internal_batch'],
                                  stop, paired=True, target_count=8)
                with torch.no_grad():
                    endpoints = {k:v.cpu().numpy() for k,v in target.endpoints(x).items()}
                values = dict(**batch, **maps[64], **endpoints, indices=indices,
                    observed_prefix32=maps[32]['observed_sensitivity'],
                    signature=np.asarray(signature), targets=np.asarray(TARGETS[:8]))
                validate_chunk(values, indices, data, signature)
                save(path, **values)
            files[str(path.relative_to(root))] = digest(path)
            completed += len(indices)
            progress = dict(shard=shard, completed=completed, total=sum(map(len, selected)),
                elapsed_seconds=time.monotonic()-began, chunk_seconds=time.monotonic()-chunk_began,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), signature=signature)
            write_json(root/f'progress_{shard}.json', dict(progress, files=files))
            event('smoothgrad_full_progress', **progress)
        write_json(root/f'shard_{shard}.json', dict(status='complete', **progress, files=files))
        event('smoothgrad_full_shard_complete', **progress)
    except TimeoutError as exc:
        write_json(root/f'stopped_{identity}.json', dict(status='stopped', reason=str(exc),
            completed=completed, files=files, signature=signature))
        raise  # nonzero: dependent motif analysis must not start on partial data
    finally:
        lock.unlink()


def prepare_motifs(project, root):
    require_allocation('cpu')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Hide CUDA for CPU motif preparation')
    from .smoothgrad_motifs import native_batch
    config, data = inputs(project, root)
    signature = digest(root/'MANIFEST.sha256')
    hashes = {}
    for shard in range(config['shards']):
        done = json.loads((root/f'shard_{shard}.json').read_text())
        expected = sum(map(len, batches(len(data['ids']), shard, config['shards'], config.get('batch', 16))))
        if (done['status'] != 'complete' or done['signature'] != signature or done['shard'] != shard
                or done['completed'] != expected or hashes.keys() & done['files'].keys()):
            raise ValueError('Incomplete or conflicting attribution shard')
        hashes.update(done['files'])
    width = int(data['native_length'].max())
    groups = group_indices(data['labels'])
    arrays = [dict(sequence=np.zeros((len(g), width, 4), np.float32),
                   summed_sensitivity=np.zeros((len(g), width, 4), np.float32),
                   observed_context=np.zeros((len(g), 8, width), np.float32)) for g in groups]
    owner = np.empty(len(data['ids']), int); row = np.empty_like(owner)
    for i, g in enumerate(groups): owner[g] = i; row[g] = np.arange(len(g))
    covered = np.zeros(len(data['ids']), bool)
    for name, expected in sorted(hashes.items()):
        path = root/name
        if digest(path) != expected: raise ValueError('Changed attribution chunk: '+name)
        with np.load(path, allow_pickle=False) as stored: values = dict(stored)
        indices = values['indices']
        if (indices.ndim != 1 or not len(indices) or (indices < 0).any()
                or (indices >= len(covered)).any() or len(np.unique(indices)) != len(indices)
                or covered[indices].any()): raise ValueError('Duplicate/invalid chunk indices')
        validate_chunk(values, indices, data, signature)
        cropped = native_batch(values, width)
        for i in range(3):
            mask = owner[indices] == i
            for key in cropped: arrays[i][key][row[indices[mask]]] = cropped[key][mask]
        covered[indices] = True
    if not covered.all(): raise ValueError('Missing enhancer maps')
    parent = root/'motifs'; parent.mkdir(exist_ok=False)
    for (name, lo, hi), indices, values in zip(GROUPS, groups, arrays):
        folder = parent/name; folder.mkdir()
        save(folder/'inputs.npz', **values, ids=data['ids'][indices], labels=data['labels'][indices],
            lengths=data['native_length'][indices], split=data['split'][indices],
            chrom=data['chrom'][indices], summit=data['summit'][indices],
            source_indices=indices, native_start=data['native_start'][indices], native_end=data['native_end'][indices])
        group = dict(config['motifs'], examples=len(indices), group_name=name,
            degree_counts=np.bincount(data['labels'][indices].sum(1).astype(int), minlength=9)[1:].tolist(),
            source_manifest_sha256=signature, source_files=hashes,
            split_counts={s:int((data['split'][indices] == s).sum()) for s in ('train', 'validation', 'test')},
            description=f'{len(indices):,} enhancers, observed degree of pleiotropy {lo}–{hi}; all splits. '
                'Independent TF-MoDISco fit on the sum of all eight calibrated-probability sensitivity maps. '
                'Native enhancer intervals only; no extra clustering.',
            population_note='Full-dataset descriptive discovery, not independent held-out validation. '
                'Unequal group sizes and the 20,000 seqlets/sign cap affect discovery support; '
                'support is not scan prevalence or an unbiased between-group enrichment test.')
        write_json(folder/'config.json', group)
        (folder/'MANIFEST.sha256').write_text(''.join(digest(folder/p)+'  '+p+'\n' for p in ('config.json', 'inputs.npz')))
        event('smoothgrad_full_motif_group_prepared', group=name, examples=len(indices), splits=group['split_counts'])
    write_json(parent/'prepared.json', dict(status='complete', examples=int(covered.sum()),
        groups={name:len(g) for (name, _, _), g in zip(GROUPS, groups)}, signature=signature))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('gpu', 'prepare_motifs', 'motifs'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--local-cuda', action='store_true')
    args = parser.parse_args()
    if args.local_cuda and args.stage != 'gpu': raise ValueError('Local flag only applies to GPU attribution')
    if args.stage == 'gpu': run_gpu(args.project, args.root, args.task, local_cuda=args.local_cuda)
    elif args.stage == 'prepare_motifs': prepare_motifs(args.project, args.root)
    else:
        import subprocess
        from .smoothgrad_motifs import run
        if not 0 <= args.task < 3: raise ValueError('Invalid group')
        folder = args.root/'motifs'/GROUPS[args.task][0]
        subprocess.run(['sha256sum', '--quiet', '-c', 'MANIFEST.sha256'], cwd=folder, check=True)
        run(args.project, folder)
