"""One bounded CUDA timing/equivalence check, never training or production attribution."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import time
import warnings

import numpy as np
import torch

from classifier_motifs.attribution import dinucleotide_shuffle, load_model, one_hot, seed_for
from classifier_motifs.context_attribution import context_integrated_gradients
from .common import digest, event, require_allocation, write_json


def compare(actual, expected):
    differences = {}
    for key in expected:
        value = actual[key].detach().cpu()
        torch.testing.assert_close(value, expected[key], atol=2e-5, rtol=1e-4)
        differences[key] = float((value-expected[key]).abs().max())
    tolerance = .01+.01*expected['target_difference'].abs()
    passed = actual['delta'].detach().cpu().abs() <= tolerance
    torch.testing.assert_close(passed, expected['delta'].abs() <= tolerance)
    return differences, passed.tolist()


def measure(function, model, x, baseline, settings):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    result = function(model, x, baseline, steps=64, **settings)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-began
    return result, dict(seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())


def run(project, root):
    require_allocation('gpu')
    began = time.monotonic()
    config = json.loads((root/'config.json').read_text())
    if config['max_seconds'] not in (600, 720) or config['repeats'] != 3 or config['references'] != [100, 101]:
        raise ValueError('Unexpected bounded benchmark contract')
    if (root/'result.json').exists():
        raise ValueError('Benchmark already started/completed; no automatic repeat')
    for relative, expected in config['source_hashes'].items():
        if digest(project/relative) != expected:
            raise ValueError('Frozen source changed: '+relative)
    original_path = root/'reference_context_attribution.py'
    spec = importlib.util.spec_from_file_location('classifier_motifs._cuda_original', original_path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA GPU is unavailable')
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
    model = load_model(project/config['checkpoint'], 'cuda')
    with np.load(root/'examples.npz', allow_pickle=False) as saved:
        data = dict(saved)
    cases = []
    for index, name, sequence in zip(data['indices'], data['ids'], data['sequence']):
        for reference in config['references']:
            seed = seed_for(20260919, str(name), 'context_pilot', 0, reference)
            baseline = dinucleotide_shuffle(sequence, seed)
            cases.append(dict(index=int(index), id=str(name), reference=reference, seed=seed,
                x=one_hot(sequence[None], 'cuda'), baseline=one_hot(baseline[None], 'cuda')))
    result = dict(status='in_progress', job_id=os.environ['SLURM_JOB_ID'],
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        manifest_sha256=digest(root/'MANIFEST.sha256'), steps=64, repeats=3,
        contexts=8, input_bp=2048, checkpoint=config['checkpoint'],
        cases=[{k:v for k,v in row.items() if k not in ('x', 'baseline')} for row in cases], variants=[])
    def save(status=None):
        if status:
            result['status'] = status
        result['elapsed_seconds'] = time.monotonic()-began
        write_json(root/'result.json', result)
    save()
    event('cuda_check_started', gpu=result['gpu'], cases=len(cases), variants=len(config['variants']), max_seconds=config['max_seconds'])
    # Ground truth always uses the old function on THIS GPU, not CPU or another device.
    expected = []
    for case in cases:
        if stop():
            save('budget_stopped'); return
        value = original.context_integrated_gradients(model, case['x'], case['baseline'], 64, 4)
        expected.append({k:v.cpu() for k,v in value.items()})
        del value
    torch.cuda.synchronize()
    for variant in config['variants']:
        if stop():
            save('budget_stopped'); return
        settings = variant['settings']
        function = original.context_integrated_gradients if variant['name'] == 'frozen_pilot' else context_integrated_gradients
        row = dict(name=variant['name'], settings=settings, status='in_progress', trials=[])
        result['variants'].append(row)
        event('cuda_variant_started', variant=row['name'], settings=settings)
        warning_state = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
        torch._C._debug_only_display_vmap_fallback_warnings(True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                value = function(model, cases[0]['x'], cases[0]['baseline'], steps=64, **settings)
                compare(value, expected[0])
                del value
                torch.cuda.synchronize()
                for repeat in range(3):
                    # Deterministic rotation avoids always measuring one enhancer first.
                    for j in np.roll(np.arange(len(cases)), repeat):
                        if stop():
                            row['status'] = 'budget_stopped'; save('budget_stopped'); return
                        case = cases[int(j)]
                        value, timing = measure(function, model, case['x'], case['baseline'], settings)
                        differences, passed = compare(value, expected[int(j)])
                        del value
                        row['trials'].append(dict(case=int(j), repeat=repeat,
                            max_absolute_differences=differences, completeness_pass=passed, **timing))
                    save()
                row['fallback_warnings'] = sorted({str(w.message) for w in caught if 'batching rule' in str(w.message)})
            row['status'] = 'passed'
            row['median_seconds_per_reference'] = statistics.median(t['seconds'] for t in row['trials'])
            row['peak_allocated_bytes'] = max(t['peak_allocated_bytes'] for t in row['trials'])
            row['peak_reserved_bytes'] = max(t['peak_reserved_bytes'] for t in row['trials'])
        except torch.cuda.OutOfMemoryError:
            row['status'] = 'out_of_memory'
        except AssertionError as error:
            row['status'] = 'equivalence_failed'
            row['error'] = str(error)[:1500]
        finally:
            torch._C._debug_only_display_vmap_fallback_warnings(warning_state)
        # Drop exception frames/temporary graphs before attempting a smaller/different variant.
        if 'value' in locals():
            del value
        gc.collect()
        torch.cuda.empty_cache()
        save()
        event('cuda_variant_finished', variant=row['name'], status=row['status'],
              median_seconds=row.get('median_seconds_per_reference'), peak_bytes=row.get('peak_allocated_bytes'))
    passed = [row for row in result['variants'] if row['status'] == 'passed']
    if not passed or result['variants'][0]['status'] != 'passed':
        save('failed'); return
    best = min(passed, key=lambda row: row['median_seconds_per_reference'])
    result['fastest_passing_variant'] = best['name']
    result['measured_speedup_vs_original'] = result['variants'][0]['median_seconds_per_reference']/best['median_seconds_per_reference']
    result['interpretation'] = 'Small repeated same-GPU kernel check; excludes shuffle/I/O and full reference averaging. No production launch or automatic resubmission.'
    save('complete')
    event('cuda_check_complete', fastest=best['name'], speedup=result['measured_speedup_vs_original'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.project.resolve(), args.root.resolve())


if __name__ == '__main__':
    main()
