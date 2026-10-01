"""Eight-output IG64/ref100 continuation on all compatible CECAR GPUs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import time

import numpy as np
import torch

from classifier_motifs.attribution import one_hot
from classifier_motifs.calibrated_attribution import CalibratedTargets, TARGETS, integrate, load_classifier
from . import calibrated_context as old
from .common import digest, event, write_json
from .original_intervals import load_intervals

NAME = 'classifier_calibrated_full8_20260923'
PARENT = 'classifier_calibrated_context_20260921_rtx2080'
PARENT_SHA = '7d9ad051c88b4c30b0d844241e5fdb970b5d02ac0bf6bd06e98757030980a25d'
OUTPUT_AXES = dict(hypothetical=1, actual=1, hypothetical_reference_se=1,
    actual_reference_se=1, actual_reference_halves=2, delta=2, target_difference=2,
    reference_quality_pass=2, quality_pass=1)
STATE_AXES = dict(sum_hyp=1, sum_sq_hyp=1, sum_half_actual=2, delta=2, target_difference=2)


def select_eight(arrays, signature, state=False):
    result = {key: value.copy() for key, value in arrays.items()}
    for key, axis in (STATE_AXES if state else OUTPUT_AXES).items():
        if arrays[key].shape[axis] != 9: raise ValueError('Expected historical nine-target arrays')
        result[key] = np.take(arrays[key], np.arange(8), axis=axis)
    result['signature'] = np.asarray(signature)
    if not state: result['targets'] = np.asarray(TARGETS[:8])
    return result


def import_chunk(parent, destination, indices, config, signature, data, intervals):
    source = parent/'chunks'/destination.name
    if destination.exists() or destination.with_suffix('.state.npz').exists(): return None
    historical = dict(config, targets=list(TARGETS))
    if source.exists():
        with np.load(source, allow_pickle=False) as z: original = dict(z)
        old.validate_output(original, indices, PARENT_SHA, historical)
        for key in ('ids', 'sequence', 'labels', 'split', 'chrom', 'summit'):
            np.testing.assert_array_equal(original[key], data[key][indices])
        for key in ('start', 'end', 'offset', 'length'):
            np.testing.assert_array_equal(original['native_'+key], intervals[key][indices])
        result = select_eight(original, signature)
        old.validate_output(result, indices, signature, config)
        old.save_npz(destination, **result)
        return dict(kind='completed', source=str(source), sha256=digest(source), elements=len(indices))
    source = source.with_suffix('.state.npz')
    if source.exists():
        with np.load(source, allow_pickle=False) as z: original = dict(z)
        template = old.new_state(len(indices), 2048, 100, PARENT_SHA, indices)
        old.validate_state(original, template, PARENT_SHA, indices, 100)
        result = select_eight(original, signature, state=True)
        template = old.new_state(len(indices), 2048, 100, signature, indices, 8)
        old.validate_state(result, template, signature, indices, 100)
        old.save_npz(destination.with_suffix('.state.npz'), **result)
        return dict(kind='partial', source=str(source), sha256=digest(source), references=int(result['count']))
    return None


def hardware():
    host = socket.gethostname().split('.')[0]
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST')
            or host not in ('a100', 'xg07', 'xg08', 'xg09')
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Requires one compatible Slurm GPU; never login, local or xg05')
    name = torch.cuda.get_device_name()
    if host == 'a100' and 'A100' in name: return 'A100'
    if host.startswith('xg') and 'RTX 2080' in name: return 'rtx2080'
    raise ValueError('Unexpected GPU: '+name)


def gate(target, data, root, parent, config, family, stop):
    # The old RTX pilot already checked full100-reference completeness and IG64
    # versus128. Check exact same model/readouts and new batching on every job.
    reports = old.resolve_pilots(parent, PARENT_SHA, json.loads((parent/'config.json').read_text()))
    with np.load(root/'endpoint_replay.npz', allow_pickle=False) as z: replay = dict(z)
    with torch.no_grad():
        got = target.endpoints(one_hot(replay['sequence'][:16], 'cuda'))['probabilities'].cpu().numpy()
    endpoint_error = float(np.abs(got-replay['probabilities'][:16]).max())
    if endpoint_error > .005: raise ValueError('Checkpoint endpoint replay failed')
    parent_config = json.loads((parent/'config.json').read_text())
    pilot_root = parent.parent/parent_config['pilot_parent']
    with np.load(pilot_root/'pilot_rtx2080/grid_check.npz', allow_pickle=False) as z:
        indices, saved_actual64 = z['indices'], z['actual64'][:, :8]
    references = np.stack([old.shuffled((data['sequence'][i], str(data['ids'][i]), 0, config['seed']))[0]
                           for i in indices])
    x, b = one_hot(data['sequence'][indices], 'cuda'), one_hot(references, 'cuda')
    labels = torch.as_tensor(data['labels'][indices], device='cuda')
    anchor = {k: v.cpu() for k, v in integrate(target, x, b, labels, 64, 32, 1, stop).items()}
    np.testing.assert_allclose(anchor['actual'][:, :8].numpy(), saved_actual64, atol=3e-5, rtol=2e-3)
    candidates = [reports['rtx2080']['settings']]
    if family == 'A100':
        candidates += [dict(pair_batch=8, internal_batch=512, target_batch=1),
                       dict(pair_batch=16, internal_batch=1024, target_batch=1)]
    timings = []
    for settings in candidates:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            torch.cuda.synchronize(); began = time.monotonic()
            batches = [integrate(target, x[i:i+settings['pair_batch']], b[i:i+settings['pair_batch']],
                labels[i:i+settings['pair_batch']], 64, settings['internal_batch'], settings['target_batch'],
                stop, target_count=8) for i in range(0, len(x), settings['pair_batch'])]
            torch.cuda.synchronize(); elapsed = time.monotonic()-began
            for key in anchor:
                torch.testing.assert_close(torch.cat([r[key] for r in batches]).cpu(), anchor[key][:, :8],
                                           atol=3e-5, rtol=2e-3)
            peak = torch.cuda.max_memory_reserved()
            if peak <= .85*torch.cuda.get_device_properties(0).total_memory:
                timings.append(dict(settings=settings, seconds=elapsed, peak_reserved_bytes=peak))
            del batches
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
    if not timings: raise ValueError('No equivalent batch with memory headroom')
    best = min(timings, key=lambda r:r['seconds'])['settings']
    return dict(status='passed', settings=best, endpoint_error=endpoint_error, candidates=timings,
                prior_100_reference_pilot_family='rtx2080', cross_hardware_saved_ig64='passed',
                targets=list(TARGETS[:8]))


def run(project, root, shard):
    family = hardware()
    config = json.loads((root/'config.json').read_text())
    if (config['targets'] != list(TARGETS[:8]) or config['steps'] != 64 or config['references'] != 100
            or config['refinement_steps'] != [] or config['seed'] != 20260916
            or config['shards'] != 80 or not 0 <= shard < 80):
        raise ValueError('Changed IG64/ref100/eight-output contract')
    parent = root.parent/PARENT
    if digest(parent/'MANIFEST.sha256') != PARENT_SHA: raise ValueError('Historical package changed')
    original = json.loads((parent/'config.json').read_text())
    for key in ('cohort','original','checkpoint','references','steps','seed','source_hashes',
                'input_bp','enhancer_batch','reference_block','calibration_population'):
        if config[key] != original[key]: raise ValueError('Incompatible continuation: '+key)
    if digest(root/'calibrators.json') != digest(parent/'calibrators.json'):
        raise ValueError('Calibration changed')
    for file, expected in config['source_hashes'].items():
        if digest(project/file) != expected: raise ValueError('Input changed: '+file)
    signature = digest(root/'MANIFEST.sha256')
    with np.load(project/config['cohort'], allow_pickle=False) as z: data = dict(z)
    if (len(data['ids']) != 40338 or len(np.unique(data['ids'])) != 40338
            or data['sequence'].shape != (40338,2048) or not np.isin(data['sequence'],range(4)).all()
            or data['labels'].shape != (40338,8) or not np.isin(data['labels'],[0,1]).all()
            or not data['labels'].any(1).all()): raise ValueError('Invalid full cohort')
    intervals = load_intervals(project/config['original'], data)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    target = CalibratedTargets(load_classifier(project/config['checkpoint'], 'cuda'),
        json.loads((root/'calibrators.json').read_text())['enhancers_only']).to('cuda').eval()
    began = time.monotonic(); stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGUSR1): signal.signal(sig, lambda *_:stopped.__setitem__(0, True))
    def stop(): return stopped[0] or (root/'STOP').exists() or time.monotonic()-began >= 85800
    identity = f'{shard}_{os.environ["SLURM_JOB_ID"]}'
    runtime = dict(shard=shard, job=os.environ['SLURM_JOB_ID'], node=socket.gethostname(),
        family=family, gpu=torch.cuda.get_device_name(), targets=list(TARGETS[:8]), signature=signature,
        supported_cuda_arches=torch.cuda.get_arch_list(), no_training=True)
    write_json(root/f'runtime_{identity}.json', runtime); event('full8_started', **runtime)
    try:
        checked = gate(target, data, root, parent, config, family, stop)
        write_json(root/f'gate_{identity}.json', checked); event('full8_gate_passed', **checked)
        directory = root/'chunks'; directory.mkdir(exist_ok=True)
        processed = 0; failures = np.zeros(8, int); hashes = {}; imports = {}
        with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context('spawn')) as pool:
            for batch, begin in enumerate(range(0, len(data['ids']), config['enhancer_batch'])):
                if batch % 80 != shard: continue
                if stop(): raise TimeoutError('Stopped at chunk boundary')
                indices = np.arange(begin, min(begin+config['enhancer_batch'], len(data['ids'])))
                path = directory/f'chunk_{begin:06d}.npz'
                imported = import_chunk(parent, path, indices, config, signature, data, intervals)
                if imported:
                    imports[path.name] = imported
                    write_json(root/f'imports_{shard}.json', imports)
                result = old.score_chunk(target, data, intervals, indices, config, checked['settings'],
                                         path, signature, pool, stop)
                processed += len(indices); failures += (~result['quality_pass']).sum(0)
                hashes[path.name] = digest(path)
                progress = dict(**runtime, elements=processed, quality_failures=failures.tolist(),
                                last_chunk=path.name, elapsed_seconds=time.monotonic()-began)
                write_json(root/f'progress_{shard}.json', progress); event('full8_progress', **progress)
                if processed >= 256 and (1-failures/processed < .95).any():
                    raise ValueError('Completeness below95%; retain outputs, do not silently filter')
        write_json(root/f'shard_{shard}.json', dict(status='complete', **runtime, elements=processed,
                                                  quality_failures=failures.tolist(), chunks=hashes))
    except Exception as error:
        write_json(root/f'stopped_{identity}.json', dict(**runtime,
            status='stopped' if isinstance(error, TimeoutError) else 'failed',
            reason=type(error).__name__+': '+str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args(); run(args.project.resolve(), args.root.resolve(), args.shard)
