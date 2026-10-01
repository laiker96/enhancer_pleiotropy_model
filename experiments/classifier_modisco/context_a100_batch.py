"""One bounded A100 pair-batching benchmark; never launches production work."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import time
import warnings

import numpy as np
import torch

from classifier_motifs.attribution import dinucleotide_shuffle, load_model, one_hot, seed_for
from classifier_motifs.context_attribution import context_integrated_gradients, region_masks
from .common import digest, event, write_json
from .context_cuda_check import compare, measure
from .context_pair_batch import refine_pair_batch
from .context_pilot import refine_reference


def require_a100():
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST')
            or socket.gethostname().split('.')[0] != 'a100'):
        raise RuntimeError('Requires a CECAR A100 Slurm compute allocation, never login')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('Expected exactly one allocated A100 CUDA device')


def chunks(order, size):
    if size < 1:
        raise ValueError('Positive pair batch required')
    return [list(map(int, order[start:start+size])) for start in range(0, len(order), size)]


def run(project, root):
    require_a100()
    began = time.monotonic()
    config = json.loads((root/'config.json').read_text())
    if config['max_seconds'] != 720 or config['repeats'] != 3 or config['references'] != [100, 101]:
        raise ValueError('Unexpected bounded A100 benchmark contract')
    if (root/'result.json').exists():
        raise ValueError('Benchmark already started; no automatic repeat')
    for relative, expected_hash in config['source_hashes'].items():
        if digest(project/relative) != expected_hash:
            raise ValueError('Frozen source changed: '+relative)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    stopped = [False]
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def stop():
        return stopped[0] or (root/'STOP').exists() or time.monotonic()-began >= config['max_seconds']
    spec = importlib.util.spec_from_file_location('classifier_motifs._a100_original', root/'reference_context_attribution.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    model = load_model(project/config['checkpoint'], 'cuda')
    with np.load(root/'examples.npz', allow_pickle=False) as saved:
        data = dict(saved)
    cases, inputs, references, masks = [], [], [], []
    for j, index in enumerate(data['indices']):
        for reference in config['references']:
            name, sequence = str(data['ids'][j]), data['sequence'][j]
            seed = seed_for(20260919, name, 'context_pilot', 0, reference)
            inputs.append(sequence)
            references.append(dinucleotide_shuffle(sequence, seed))
            masks.append(region_masks(int(data['offset'][j]), int(data['length'][j])))
            cases.append(dict(index=int(index), id=name, degree=int(data['labels'][j].sum()),
                              reference=reference, seed=seed))
    x, baseline = one_hot(np.stack(inputs), 'cuda'), one_hot(np.stack(references), 'cuda')
    result = dict(status='in_progress', job_id=os.environ['SLURM_JOB_ID'],
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        manifest_sha256=digest(root/'MANIFEST.sha256'), steps=64, repeats=3, contexts=8,
        input_bp=2048, checkpoint=config['checkpoint'], cases=cases, variants=[])
    def save(status=None):
        if status:
            result['status'] = status
        result['elapsed_seconds'] = time.monotonic()-began
        write_json(root/'result.json', result)
    save()
    event('a100_batch_started', gpu=result['gpu'], pairs=len(cases), max_seconds=config['max_seconds'])
    try:
        expected = []
        for i in range(len(cases)):
            if stop():
                raise TimeoutError('Ground-truth budget exhausted')
            value = original.context_integrated_gradients(model, x[i:i+1], baseline[i:i+1], 64, 4)
            expected.append({k:v.cpu() for k,v in value.items()})
            del value
        def batch_expected(indices):
            return {k:torch.cat([expected[i][k] for i in indices]) for k in expected[0]}
        for variant in config['variants']:
            if stop():
                raise TimeoutError('Variant budget exhausted')
            row = dict(variant, status='in_progress', trials=[], repeat_seconds_per_pair=[])
            result['variants'].append(row)
            save()
            event('a100_variant_started', variant=row['name'], pair_batch=row['pair_batch'], settings=row['settings'])
            warning_state = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
            torch._C._debug_only_display_vmap_fallback_warnings(True)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    warm = list(range(min(row['pair_batch'], len(cases))))
                    value = context_integrated_gradients(model, x[warm], baseline[warm], steps=64, **row['settings'])
                    compare(value, batch_expected(warm))
                    del value
                    for repeat in range(config['repeats']):
                        elapsed = 0.
                        for indices in chunks(np.roll(np.arange(len(cases)), repeat), row['pair_batch']):
                            if stop():
                                raise TimeoutError('Timed comparison budget exhausted')
                            # Pair assembly is outside the synchronized kernel timer.
                            bx, br = x[indices], baseline[indices]
                            value, timing = measure(context_integrated_gradients, model, bx, br, row['settings'])
                            differences, passed = compare(value, batch_expected(indices))
                            del value, bx, br
                            elapsed += timing['seconds']
                            row['trials'].append(dict(cases=indices, repeat=repeat,
                                max_absolute_differences=differences, completeness_pass=passed, **timing))
                        row['repeat_seconds_per_pair'].append(elapsed/len(cases))
                        save()
                        event('a100_repeat_finished', variant=row['name'], repeat=repeat,
                              seconds_per_pair=elapsed/len(cases))
                    row['fallback_warnings'] = sorted({str(w.message) for w in caught if 'batching rule' in str(w.message)})
                row['status'] = 'passed'
                row['median_seconds_per_pair'] = statistics.median(row['repeat_seconds_per_pair'])
                row['peak_allocated_bytes'] = max(t['peak_allocated_bytes'] for t in row['trials'])
                row['peak_reserved_bytes'] = max(t['peak_reserved_bytes'] for t in row['trials'])
            except torch.cuda.OutOfMemoryError:
                row['status'] = 'out_of_memory'
            except AssertionError as error:
                row['status'] = 'equivalence_failed'
                row['error'] = str(error)[:1500]
            finally:
                torch._C._debug_only_display_vmap_fallback_warnings(warning_state)
            if 'value' in locals():
                del value
            if 'bx' in locals():
                del bx, br
            gc.collect()
            torch.cuda.empty_cache()
            save()
            event('a100_variant_finished', variant=row['name'], status=row['status'],
                  seconds_per_pair=row.get('median_seconds_per_pair'))
        passed = [r for r in result['variants'] if r['status'] == 'passed']
        if not passed or result['variants'][0]['status'] != 'passed':
            save('failed'); return
        best = min(passed, key=lambda r:r['median_seconds_per_pair'])
        result['fastest_passing_variant'] = best['name']
        result['speedup_vs_same_gpu_single_pair'] = result['variants'][0]['median_seconds_per_pair']/best['median_seconds_per_pair']
        save()
        # Independent adaptive decisions on four pairs (two enhancers, two refs).
        indices = [0, 1, len(cases)-2, len(cases)-1]
        cfg = dict(integration_steps=[64, 128, 256, 512], absolute_tolerance=.01,
                   relative_tolerance=.01, internal_batch=max(4, best['settings']['internal_batch']),
                   context_batch=best['settings']['context_batch'])
        anchors = [True, False, True, False]
        adaptive_expected = []
        for i, anchor in zip(indices, anchors):
            if stop():
                raise TimeoutError('Adaptive check budget exhausted')
            adaptive_expected.append(refine_reference(model, x[i:i+1], baseline[i:i+1],
                masks[i], dict(cfg, internal_batch=32, context_batch=1), anchor))
        actual = refine_pair_batch(model, x[indices], baseline[indices], np.asarray(masks)[indices],
                                   cfg, anchors, stop)
        checks = []
        for i, a, b in zip(indices, actual, adaptive_expected):
            differences, flags = compare(a[0], {k:v.cpu() for k,v in b[0].items()})
            if a[1] != b[1] or [(v['from_steps'], v['to_steps']) for v in a[3]] != [(v['from_steps'], v['to_steps']) for v in b[3]]:
                raise AssertionError('Per-pair adaptive integration decisions changed')
            checks.append(dict(case=i, steps=a[1], max_absolute_differences=differences, completeness_pass=flags))
        result['adaptive_check'] = dict(status='passed', cases=checks)
        result['interpretation'] = 'Same-A100 FP32 kernel benchmark; excludes shuffling, pair assembly, serialization and full reference averaging. No production launch.'
        save('complete')
        event('a100_batch_complete', fastest=best['name'], speedup=result['speedup_vs_same_gpu_single_pair'])
    except TimeoutError as error:
        save('budget_stopped')
        event('a100_batch_budget_stopped', reason=str(error))
    except Exception as error:
        result['error'] = type(error).__name__+': '+str(error)[:1500]
        save('failed')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.project.resolve(), args.root.resolve())


if __name__ == '__main__':
    main()
