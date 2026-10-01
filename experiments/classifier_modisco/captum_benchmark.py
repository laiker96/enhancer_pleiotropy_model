"""One bounded, fixed-reference Captum-versus-IG comparison; no adoption."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import warnings

import numpy as np
import torch

from classifier_motifs.attribution import active_weights, dinucleotide_shuffle, load_model, one_hot, seed_for
from classifier_motifs.captum_deeplift import CAPTUM_COMMIT, DEEPLIFT_EPS, adapt_model, paired_deeplift, require_pinned_captum
from classifier_motifs.context_attribution import REGIONS, map_agreement, region_masks
from classifier_motifs.dual_attribution import TARGETS, dual_integrated_gradients, targets
from .common import digest, event, write_json
from .context_a100_batch import require_a100
from .context_pilot import save_npz


def completeness(actual, difference):
    delta = np.asarray(actual).sum(-1)-np.asarray(difference)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite completeness residual')
    passed = np.abs(delta) <= .02+.05*np.abs(difference)
    return dict(delta=delta.tolist(), passed=passed.tolist(), failed=int((~passed).sum()),
                maximum_absolute_delta=float(np.abs(delta).max()))


def compare_maps(candidate, reference, data):
    rows = []
    for i, index in enumerate(data['indices']):
        a, b = candidate['actual'][i], reference['actual'][i]
        ah, bh = candidate['hypothetical'][i], reference['hypothetical'][i]
        masks = region_masks(int(data['offset'][i]), int(data['length'][i]))
        for region, mask in [('full', np.ones(a.shape[-1], bool)), *zip(REGIONS, masks)]:
            if not mask.any():
                continue
            metrics = map_agreement(a[:, mask], b[:, mask])
            hyp = map_agreement(ah[:, :, mask].reshape(2, -1), bh[:, :, mask].reshape(2, -1))
            valid = np.convolve(mask.astype(int), np.ones(10, int), mode='valid') == 10
            window = None
            if valid.any():
                wa = np.stack([np.convolve(v, np.ones(10), mode='valid')[valid] for v in a])
                wb = np.stack([np.convolve(v, np.ones(10), mode='valid')[valid] for v in b])
                window = map_agreement(wa, wb)
            for k, target in enumerate(TARGETS):
                rms = float(np.sqrt(np.mean(b[k, mask].astype(float)**2)))
                rows.append(dict(index=int(index), degree=int(data['labels'][i].sum()), target=target,
                    region=region, **metrics[k], hypothetical_cosine=hyp[k]['cosine'],
                    normalized_rmse=float(np.sqrt(np.mean((a[k, mask]-b[k, mask]).astype(float)**2))/rms) if rms else None,
                    window10_cosine=window[k]['cosine'] if window else None,
                    window10_top10pct_jaccard=window[k]['top10pct_jaccard'] if window else None))
    return rows


def summarize(rows):
    fields = ('cosine', 'hypothetical_cosine', 'weighted_sign_agreement', 'top10pct_jaccard',
              'normalized_rmse', 'window10_cosine', 'window10_top10pct_jaccard')
    result = {}
    for region in ['full', *REGIONS]:
        result[region] = {}
        for target in TARGETS:
            selected = [r for r in rows if r['target'] == target and r['region'] == region]
            result[region][target] = {}
            for field in fields:
                values = [r[field] for r in selected if r[field] is not None]
                result[region][target][field] = dict(n=len(values),
                    median=float(np.median(values)) if values else None,
                    minimum=float(np.min(values)) if values else None,
                    maximum=float(np.max(values)) if values else None)
    return result


def check_pair_batching(model, x, baseline, labels):
    """Real-device pre-timing gate; preserve the original CPU test tolerances."""
    got = paired_deeplift(model, x, baseline, labels)
    rows = [paired_deeplift(model, x[i:i+1], baseline[i:i+1], labels[i:i+1])
            for i in range(min(4,len(x)))]
    maximum = {}
    for key in got:
        expected = torch.cat([r[key] for r in rows])
        actual = got[key][:len(rows)]
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-3)
        maximum[key] = float((actual-expected).abs().max())
    reverse = paired_deeplift(model, x.flip((1,2)), baseline.flip((1,2)), labels)
    torch.testing.assert_close(got['hypothetical'], reverse['hypothetical'].flip((2,3)), atol=1e-6, rtol=1e-5)
    repeat = paired_deeplift(model, x, baseline, labels)
    torch.testing.assert_close(got['hypothetical'], repeat['hypothetical'], atol=0, rtol=0)
    return dict(status='passed', batched_examples=len(x), individual_examples=len(rows),
                maximum_differences=maximum, deeplift_eps=DEEPLIFT_EPS)


def run(project, root):
    require_a100()
    require_pinned_captum()
    began = time.monotonic()
    config = json.loads((root/'config.json').read_text())
    if (config['max_seconds'], config['references'], config['reference_seed'],
            config['ig_steps'], config['internal_batch']) != (720, 50, 20260916, 32, 512):
        raise ValueError('Unexpected authorized benchmark contract')
    if config.get('deeplift_eps') != DEEPLIFT_EPS:
        raise ValueError('Record the tested DeepLIFT numerical safeguard explicitly')
    precision = config.get('deeplift_precision', 'float32')
    if precision not in ('float32', 'float64'):
        raise ValueError('Expected an explicit FP32 or higher-precision DeepLIFT candidate')
    dl_dtype = getattr(torch, precision)
    if (root/'result.json').exists():
        raise ValueError('Already started; no automatic repeat')
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
        return stopped[0] or (root/'STOP').exists() or time.monotonic()-began >= 720
    with np.load(root/'examples.npz', allow_pickle=False) as f:
        data = dict(f)
    if (data['sequence'].shape != (16, 2048) or len(np.unique(data['indices'])) != 16
            or not np.array_equal(data['labels'].sum(1), np.tile(np.arange(1, 9), 2))):
        raise ValueError('Expected the same sixteen balanced training examples')
    result = dict(status='in_progress', job_id=os.environ['SLURM_JOB_ID'],
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda,
        captum_commit=CAPTUM_COMMIT, deeplift_eps=DEEPLIFT_EPS, manifest_sha256=digest(root/'MANIFEST.sha256'),
        deeplift_precision=precision, ig_precision='float32',
        precision_note='Higher precision retains original explicit FP32 internal casts; not full FP64.',
        targets=list(TARGETS), references=50, indices=data['indices'].tolist(),
        ids=data['ids'].tolist(), degrees=data['labels'].sum(1).tolist(), variants=[])
    def save(status=None):
        if status:
            result['status'] = status
        result['elapsed_seconds'] = time.monotonic()-began
        write_json(root/'result.json', result)
    save()
    event('captum_benchmark_started', examples=16, references=50, max_seconds=720)
    try:
        model = load_model(project/config['checkpoint'], 'cuda')
        adapted, result['changes'] = adapt_model(model)
        adapted = adapted.to(dtype=dl_dtype)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value.to(adapted.state_dict()[name].dtype),
                                       adapted.state_dict()[name], atol=0, rtol=0)
        codes = np.stack([[dinucleotide_shuffle(seq, seed_for(20260916, str(name), r))
                          for r in range(50)] for seq, name in zip(data['sequence'], data['ids'])])
        save_npz(root/'reference_codes.npz', codes=codes, indices=data['indices'])
        result['reference_codes_sha256'] = digest(root/'reference_codes.npz')
        x = one_hot(data['sequence'], 'cuda')
        refs = one_hot(codes.reshape(-1, 2048), 'cuda').reshape(16, 50, 4, 2048)
        dl_x, dl_refs = x.to(dl_dtype), refs.to(dl_dtype)
        labels = torch.as_tensor(data['labels'], device=x.device)
        weights = active_weights(labels)
        result['forward_checks'] = []
        # Check every reference and WT; ordinary input gradients on WT,
        # reference0 and one interior path point. No attribution graph involved.
        with torch.no_grad():
            wt = targets(model, x, weights)
            dl_wt = targets(adapted, dl_x, weights)
            torch.testing.assert_close(dl_wt, wt, atol=2e-5, rtol=1e-5, check_dtype=False)
            base_outputs, dl_base_outputs, errors = [], [], []
            for r in range(50):
                if stop():
                    raise TimeoutError('Forward-check budget exhausted')
                original = targets(model, refs[:, r], weights)
                changed = targets(adapted, dl_refs[:, r], weights)
                torch.testing.assert_close(changed, original, atol=2e-5, rtol=1e-5, check_dtype=False)
                errors.append(float((changed-original).abs().max()))
                base_outputs.append(original)
                dl_base_outputs.append(changed)
            difference = (wt[:, None]-torch.stack(base_outputs, 1)).cpu().numpy()
            dl_difference = (dl_wt[:, None]-torch.stack(dl_base_outputs, 1)).cpu().numpy()
            result['reference_forward_max_difference'] = max(errors)
        for name, points in [('wt', x), ('reference0', refs[:, 0]), ('path0.37', .37*x+.63*refs[:, 0])]:
            points = points.detach().requires_grad_(True)
            original, changed = targets(model, points, weights), targets(adapted, points.to(dl_dtype), weights)
            torch.testing.assert_close(changed, original, atol=2e-5, rtol=1e-5, check_dtype=False)
            row = dict(input=name, maximum_target_difference=float((changed-original).abs().max()))
            for k, target in enumerate(TARGETS):
                a = torch.autograd.grad(original[:, k].sum(), points, retain_graph=True)[0]
                b = torch.autograd.grad(changed[:, k].sum(), points, retain_graph=True)[0]
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-3)
                row[target+'_max_gradient_difference'] = float((a-b).abs().max())
            result['forward_checks'].append(row)
        del original, changed, points, a, b, base_outputs, dl_base_outputs
        event('captum_cuda_forward_equivalence_passed'); save()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Setting forward, backward hooks.*')
            result['cuda_pair_batching'] = check_pair_batching(adapted,dl_x,dl_refs[:,0],labels)
        event('captum_cuda_pair_batching_passed', **result['cuda_pair_batching']); save()
        maps = {}
        for method in ('deeplift', 'ig32'):
            if stop():
                raise TimeoutError('Attribution budget exhausted')
            def attribute(r):
                if method == 'deeplift':
                    return paired_deeplift(adapted, dl_x, dl_refs[:, r], labels)
                return dual_integrated_gradients(model, x, refs[:, r], labels, 32, 512, stop)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Setting forward, backward hooks.*')
                # Full pair batch warmup, excluded from method timing.
                warm = attribute(0); del warm
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                sums = None
                method_dtype = dl_dtype if method == 'deeplift' else x.dtype
                method_difference = dl_difference if method == 'deeplift' else difference
                pair_sums = torch.empty(16, 50, 2, device=x.device, dtype=method_dtype)
                row = dict(method=method, precision=str(method_dtype), status='running', references_completed=0)
                result['variants'].append(row); save()
                kernel_seconds = 0.
                for r in range(50):
                    if stop():
                        raise TimeoutError('Stopped at reference boundary')
                    torch.cuda.synchronize(); started = time.perf_counter()
                    value = attribute(r)
                    if sums is None:
                        sums = {k:torch.zeros_like(value[k]) for k in ('hypothetical', 'actual')}
                    for k in sums:
                        sums[k] += value[k]
                    pair_sums[:, r] = value['actual'].sum(-1)
                    del value
                    torch.cuda.synchronize(); kernel_seconds += time.perf_counter()-started
                    if (r+1) % 10 == 0:
                        row.update(references_completed=r+1, seconds_so_far=kernel_seconds)
                        save(); event('captum_benchmark_progress', method=method, references=r+1, seconds=kernel_seconds)
                row.update(seconds=kernel_seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
                transferred = time.perf_counter()
                maps[method] = {k:(v/50).cpu().numpy() for k, v in sums.items()}
                actual_sums = pair_sums.cpu().numpy()
                row['device_to_host_seconds'] = time.perf_counter()-transferred
                row['per_pair_completeness'] = completeness(actual_sums[..., None], method_difference)
                row['mean_50_completeness'] = completeness(maps[method]['actual'], method_difference.mean(1))
                row['per_pair_completeness_vs_original_fp32'] = completeness(actual_sums[..., None], difference)
                io_started = time.perf_counter()
                folder = root/method; folder.mkdir()
                save_npz(folder/'maps.npz', **maps[method], reference_target_difference=method_difference,
                         original_fp32_target_difference=difference,
                         per_reference_actual_sum=actual_sums, indices=data['indices'])
                row.update(output_io_seconds=time.perf_counter()-io_started,
                           maps_sha256=digest(folder/'maps.npz'), status='complete')
                save(); event('captum_benchmark_variant_complete', method=method, seconds=kernel_seconds,
                              failed_pairs=row['per_pair_completeness']['failed'])
                del sums, pair_sums
        rows = compare_maps(maps['deeplift'], maps['ig32'], data)
        write_json(root/'agreement_vs_ig32.json', rows)
        result['agreement_vs_ig32'] = summarize(rows)
        result['speedup_vs_ig32'] = result['variants'][1]['seconds']/result['variants'][0]['seconds']
        result['interpretation'] = ('Method comparison only: IG32 is not biological ground truth. '
            'Completeness is necessary, not proof of equal maps. Window10 metrics are motif-scale '
            'proxies, not discovered motifs or perturbation validation. No production adoption.')
        save('complete'); event('captum_benchmark_complete', speedup=result['speedup_vs_ig32'])
    except TimeoutError as error:
        result['reason'] = str(error); save('budget_stopped')
        event('captum_benchmark_budget_stopped', reason=str(error))
    except Exception as error:
        result['error'] = type(error).__name__+': '+str(error)[:1500]
        save('failed'); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.project.resolve(), args.root.resolve())


if __name__ == '__main__':
    main()
