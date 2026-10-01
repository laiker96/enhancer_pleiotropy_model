"""Bounded comparison of EG sampling with the same 50-reference IG target."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from classifier_motifs.attribution import dinucleotide_shuffle, ensemble, load_model, one_hot, seed_for
from classifier_motifs.context_attribution import CONTEXTS, REGIONS, context_integrated_gradients, map_agreement, region_masks
from classifier_motifs.expected_gradients import context_sampled_gradients, quadrature_plan, sampling_plan, stack_plans
from .common import digest, event, write_json
from .context_a100_batch import require_a100
from .context_pilot import save_npz


def compare_maps(candidate, reference, masks, labels):
    """Signed actual/hypothetical agreement; 10-bp windows never cross gaps."""
    rows = []
    for i in range(len(labels)):
        weights = labels[i]/labels[i].sum()
        a = candidate['actual'][i]
        b = reference['actual'][i]
        ah = candidate['hypothetical'][i]
        bh = reference['hypothetical'][i]
        a = np.concatenate([a, (a*weights[:, None]).sum(0, keepdims=True)])
        b = np.concatenate([b, (b*weights[:, None]).sum(0, keepdims=True)])
        ah = np.concatenate([ah, (ah*weights[:, None, None]).sum(0, keepdims=True)])
        bh = np.concatenate([bh, (bh*weights[:, None, None]).sum(0, keepdims=True)])
        for region, mask in [('full', np.ones(a.shape[-1], bool)), *zip(REGIONS, masks[i])]:
            if not mask.any():
                continue
            values = map_agreement(a[:, mask], b[:, mask])
            hypothetical = map_agreement(ah[:, :, mask].reshape(9, -1), bh[:, :, mask].reshape(9, -1))
            valid_windows = np.convolve(mask.astype(int), np.ones(10, int), mode='valid') == 10
            window_values = None
            if valid_windows.any():
                wa = np.stack([np.convolve(v, np.ones(10), mode='valid')[valid_windows] for v in a])
                wb = np.stack([np.convolve(v, np.ones(10), mode='valid')[valid_windows] for v in b])
                window_values = map_agreement(wa, wb)
            for k, context in enumerate([*CONTEXTS, 'active_mean']):
                rms = float(np.sqrt(np.mean(b[k, mask].astype(float)**2)))
                rows.append(dict(example=i, degree=int(labels[i].sum()), context=context, region=region,
                    **values[k], hypothetical_cosine=hypothetical[k]['cosine'],
                    normalized_rmse=float(np.sqrt(np.mean((a[k, mask]-b[k, mask]).astype(float)**2))/rms) if rms else None,
                    window10_cosine=window_values[k]['cosine'] if window_values else None,
                    window10_top10pct_jaccard=window_values[k]['top10pct_jaccard'] if window_values else None))
    return rows


def summarize(rows):
    summary = {}
    fields = ('cosine', 'hypothetical_cosine', 'weighted_sign_agreement', 'top10pct_jaccard',
              'normalized_rmse', 'window10_cosine', 'window10_top10pct_jaccard')
    for region in ['full', *REGIONS]:
        summary[region] = {}
        for target in ('separate_contexts', 'active_mean'):
            selected = [r for r in rows if r['region'] == region and
                        (r['context'] == 'active_mean') == (target == 'active_mean')]
            summary[region][target] = {}
            for field in fields:
                values = [r[field] for r in selected if r[field] is not None]
                summary[region][target][field] = dict(n=len(values),
                    median=float(np.median(values)) if values else None,
                    p10=float(np.quantile(values, .1)) if values else None,
                    p90=float(np.quantile(values, .9)) if values else None)
    return summary


def completeness(actual, difference):
    delta = actual.sum(-1)-difference
    passed = np.abs(delta) <= .01+.01*np.abs(difference)
    return dict(delta=delta.tolist(), passed=passed.tolist(), failed=int((~passed).sum()),
                maximum_absolute_delta=float(np.abs(delta).max()))


def run(project, root):
    require_a100()
    began = time.monotonic()
    config = json.loads((root/'config.json').read_text())
    if (config['max_seconds'] != 720 or config['references'] != 50
            or config['ig_steps'] != [8, 32, 64] or config['samples'] != [256, 512]
            or config['repeats'] != 3 or config['schemes'] != ['iid', 'stratified']):
        raise ValueError('Unexpected bounded EG benchmark contract')
    if (root/'result.json').exists():
        raise ValueError('Already started: no automatic repeat')
    for relative, expected in config['source_hashes'].items():
        if digest(project/relative) != expected:
            raise ValueError('Frozen input changed: '+relative)
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
    with np.load(root/'examples.npz', allow_pickle=False) as f:
        data = dict(f)
    model = load_model(project/config['checkpoint'], 'cuda')
    result = dict(status='in_progress', job_id=os.environ['SLURM_JOB_ID'],
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda,
        manifest_sha256=digest(root/'MANIFEST.sha256'), references=50, examples=len(data['ids']),
        ids=data['ids'].tolist(), indices=data['indices'].tolist(), degrees=data['labels'].sum(1).tolist(), variants=[])
    def save(status=None):
        if status:
            result['status'] = status
        result['elapsed_seconds'] = time.monotonic()-began
        write_json(root/'result.json', result)
    save()
    event('expected_gradients_started', examples=len(data['ids']), references=50, max_seconds=config['max_seconds'])
    try:
        prepared = time.perf_counter()
        codes = np.stack([[dinucleotide_shuffle(sequence, seed_for(config['reference_seed'], str(name), r))
                           for r in range(50)] for name, sequence in zip(data['ids'], data['sequence'])])
        x = one_hot(data['sequence'], 'cuda')
        references = one_hot(codes.reshape(-1, codes.shape[-1]), 'cuda').reshape(len(x), 50, 4, x.shape[-1])
        masks = np.stack([region_masks(int(o), int(n)) for o,n in zip(data['offset'], data['length'])])
        with torch.no_grad():
            logits = ensemble(model, x)[0].cpu().numpy()
            baseline_logits = torch.cat([ensemble(model, v)[0] for v in references.flatten(0, 1).split(64)]).reshape(len(x), 50, 8).cpu().numpy()
        difference = logits-baseline_logits.mean(1)
        result['preparation_seconds'] = time.perf_counter()-prepared
        # Verify new weighted-projection integration against the untouched IG
        # implementation before using it as this benchmark's numerical reference.
        calibration = context_sampled_gradients(model, x[:1], references[:1, :2],
            stack_plans([quadrature_plan(2, 64)]), config['internal_batch'], stop)
        original = [context_integrated_gradients(model, x[:1], references[:1, r], 64, 64, context_batch=1) for r in range(2)]
        maximum = {}
        for key in calibration:
            expected = torch.stack([v[key] for v in original]).mean(0)
            torch.testing.assert_close(calibration[key], expected, atol=2e-5, rtol=1e-4)
            maximum[key] = float((calibration[key]-expected).abs().max())
        result['old_kernel_equivalence'] = dict(status='passed', maximum_differences=maximum)
        del calibration, original
        specs = [dict(name=f'ig{steps}', scheme='quadrature', steps=steps) for steps in (64, 32, 8)]
        specs += [dict(name=f'eg_{scheme}_{samples}_r{repeat}', scheme=scheme, samples=samples, repeat=repeat)
                  for scheme in config['schemes'] for samples in config['samples'] for repeat in range(3)]
        reference_maps, first_seed_maps = None, {}
        for spec in specs:
            if stop():
                raise TimeoutError('Variant budget exhausted')
            plans = [quadrature_plan(50, spec['steps']) if spec['scheme'] == 'quadrature'
                     else sampling_plan(50, spec['samples'], seed_for(config['sampling_seed'], str(name),
                          'expected_gradients', spec['scheme'], spec['samples'], spec['repeat']), spec['scheme'])
                     for name in data['ids']]
            plan = stack_plans(plans)
            row = dict(spec, status='in_progress', points_per_enhancer=plan['alpha'].shape[1])
            result['variants'].append(row)
            save()
            event('expected_gradients_variant_started', variant=spec['name'], points=row['points_per_enhancer'])
            # A full-shape short warm-up uses the same input/reference pool.
            width = min(plan['alpha'].shape[1], config['internal_batch']//len(x))
            warm = {k:v[:, :width].copy() for k,v in plan.items()}
            warm['weights'] /= warm['weights'].sum(1, keepdims=True)
            context_sampled_gradients(model, x, references, warm, config['internal_batch'], stop)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            value = context_sampled_gradients(model, x, references, plan, config['internal_batch'], stop)
            torch.cuda.synchronize()
            row['seconds'] = time.perf_counter()-started
            row['seconds_per_enhancer'] = row['seconds']/len(x)
            row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            row['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            copied = time.perf_counter()
            maps = {k:v.cpu().numpy() for k,v in value.items()}
            del value
            row['device_to_host_seconds'] = time.perf_counter()-copied
            row['completeness_uniform_50'] = completeness(maps['actual'], difference)
            sampled_reference = np.sum(baseline_logits[np.arange(len(x))[:, None], plan['reference']]*plan['weights'][:, :, None], axis=1)
            row['completeness_sampled_reference_mix'] = completeness(maps['actual'], logits-sampled_reference)
            row['distinct_references_per_enhancer'] = [len(np.unique(p['reference'])) for p in plans]
            if reference_maps is None:
                reference_maps = maps
            metrics = compare_maps(maps, reference_maps, masks, data['labels'])
            row['agreement_vs_ig64'] = summarize(metrics)
            io_started = time.perf_counter()
            folder = root/spec['name']; folder.mkdir()
            save_npz(folder/'maps.npz', **maps, **{f'plan_{k}':v for k,v in plan.items()})
            write_json(folder/'agreement_vs_ig64.json', metrics)
            row['output_io_seconds'] = time.perf_counter()-io_started
            row['maps_sha256'] = digest(folder/'maps.npz')
            if spec['scheme'] != 'quadrature':
                key = (spec['scheme'], spec['samples'])
                if key in first_seed_maps:
                    agreement = compare_maps(maps, first_seed_maps[key], masks, data['labels'])
                    row['agreement_vs_repeat0'] = summarize(agreement)
                    write_json(folder/'agreement_vs_repeat0.json', agreement)
                else:
                    first_seed_maps[key] = maps
            row['status'] = 'complete'
            row['speedup_vs_ig64'] = result['variants'][0]['seconds']/row['seconds']
            save()
            event('expected_gradients_variant_complete', variant=spec['name'], seconds=row['seconds'],
                  completeness_failures=row['completeness_uniform_50']['failed'],
                  native_context_cosine=row['agreement_vs_ig64']['native']['separate_contexts']['cosine']['median'])
        result['interpretation'] = 'Finite fixed-50-reference numerical pilot; inspect accuracy and sampling stability, not speed alone. No production adoption. Window10 scores are motif-scale proxies, not TF matches or MoDISco discoveries.'
        save('complete')
        event('expected_gradients_complete', variants=len(result['variants']))
    except TimeoutError as error:
        save('budget_stopped')
        event('expected_gradients_budget_stopped', reason=str(error))
    except Exception as error:
        result['error'] = type(error).__name__+': '+str(error)[:1500]
        save('failed')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    run(args.project.resolve(), args.root.resolve())


if __name__ == '__main__':
    main()
