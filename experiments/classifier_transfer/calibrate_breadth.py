"""CPU-only, pair-blocked calibration of frozen validation probabilities."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata

CONTEXTS = ('ab', 'e13', 'e5', 'ead', 'hid', 'lb', 'o', 'wid')
MODEL = 'background__cnn_finetune_20260914'
METHODS = ('raw', 'temperature', 'sigmoid')
POPULATIONS = ('enhancers_only', 'enhancers_plus_background_1to1')
EPSILON = 1e-7


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate(p, y=None):
    p = np.asarray(p, float)
    if p.ndim != 2 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Finite probability matrix required')
    if y is not None:
        y = np.asarray(y)
        if y.shape != p.shape or not np.isin(y, (0, 1)).all():
            raise ValueError('Aligned binary labels required')
        if np.any(y.sum(0) == 0) or np.any(y.sum(0) == len(y)):
            raise ValueError('Both classes required for every calibration fit')
    return p


def fit(p, y, method):
    p = validate(p, y)
    if method not in METHODS:
        raise ValueError('Unknown method')
    x = logit(np.clip(p, EPSILON, 1-EPSILON))
    a, b, optimization = [], [], []
    for j in range(p.shape[1]):
        if method == 'raw':
            a.append(1.); b.append(0.); optimization.append(dict(status='identity')); continue
        def objective(parameters):
            z = parameters[0]*x[:, j] + (parameters[1] if method == 'sigmoid' else 0.)
            error = expit(z) - y[:, j]
            gradient = [float(np.mean(error*x[:, j]))]
            if method == 'sigmoid': gradient.append(float(error.mean()))
            return float(np.mean(np.logaddexp(0, z) - y[:, j]*z)), np.asarray(gradient)
        initial = [1., 0.] if method == 'sigmoid' else [1.]
        bounds = [(1e-6, None)] + ([(None, None)] if method == 'sigmoid' else [])
        result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=1000, ftol=1e-12, gtol=1e-8))
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError('Calibration optimizer failed: '+str(result.message))
        a.append(float(result.x[0])); b.append(float(result.x[1]) if method == 'sigmoid' else 0.)
        optimization.append(dict(status='converged', nll=float(result.fun), iterations=int(result.nit),
                                 slope_at_lower_bound=bool(result.x[0] <= 1.00001e-6)))
    return dict(method=method, a=a, b=b, probability_clip=EPSILON,
                formula='sigmoid(a * logit(clip(mean_RC_probability, eps, 1-eps)) + b)',
                application_order='Calibrate AFTER mean forward/RC probabilities, not mean logits.',
                optimization=optimization)


def apply(p, calibration):
    p = validate(p)
    if len(calibration['a']) != p.shape[1]: raise ValueError('Wrong context count')
    if calibration['method'] == 'raw': return p.copy()
    x = logit(np.clip(p, calibration['probability_clip'], 1-calibration['probability_clip']))
    return expit(x*np.asarray(calibration['a']) + np.asarray(calibration['b']))


def overlaps(chrom, start, end, held_chrom, held_start, held_end):
    """Half-open overlap with the union of held-out DNA windows."""
    found = np.zeros(len(start), bool)
    for c in np.unique(held_chrom):
        source = np.flatnonzero(held_chrom == c)
        intervals = sorted(zip(held_start[source], held_end[source]))
        merged = []
        for lo, hi in intervals:
            if not merged or lo > merged[-1][1]: merged.append([lo, hi])
            else: merged[-1][1] = max(hi, merged[-1][1])
        lower, upper = np.asarray(merged).T
        query = np.flatnonzero(chrom == c)
        index = np.searchsorted(lower, end[query], side='left')-1
        good = index >= 0
        found[query[good]] = upper[index[good]] > start[query[good]]
    return found


def make_folds(chrom, summit, bg_chrom, bg_start, bg_end):
    if len(np.unique(chrom)) != 1:
        raise ValueError('This frozen validation contract expects one chromosome')
    n = len(summit); fold = np.empty(n, int)
    for number, indices in enumerate(np.array_split(np.argsort(summit, kind='stable'), 5)):
        fold[indices] = number
    records, masks = [], []
    for number in range(5):
        held = fold == number
        hc = np.r_[chrom[held], bg_chrom[held]]
        hs = np.r_[summit[held]-1024, bg_start[held]]
        he = np.r_[summit[held]+1024, bg_end[held]]
        conflict = overlaps(chrom, summit-1024, summit+1024, hc, hs, he)
        conflict |= overlaps(bg_chrom, bg_start, bg_end, hc, hs, he)
        train = ~held & ~conflict
        if not train.any(): raise ValueError('Overlap purge left no calibration pairs')
        records.append(dict(fold=number, heldout_pairs=int(held.sum()),
                            training_pairs=int(train.sum()), purged_pairs=int((~held & conflict).sum()),
                            minimum_heldout_enhancer_summit=int(summit[held].min()),
                            maximum_heldout_enhancer_summit=int(summit[held].max())))
        masks.append(train)
    return fold, masks, records


def average_precision(y, p):
    if not y.any(): return None
    order = np.argsort(-p, kind='stable'); yy, pp = y[order], p[order]
    last = np.r_[pp[1:] != pp[:-1], True]
    tp = yy.cumsum()[last]; rank = np.arange(1, len(y)+1)[last]
    return float(np.sum(np.diff(np.r_[0., tp/y.sum()]) * tp/rank))


def correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.ptp(x) > 0 and np.ptp(y) > 0 else None


def evaluate(y, p):
    p = validate(p); clipped = np.clip(p, 1e-15, 1-1e-15)
    nll = -(y*np.log(clipped)+(1-y)*np.log1p(-clipped))
    context_metrics = {}
    for j, context in enumerate(CONTEXTS):
        bin_index = np.minimum((p[:, j]*10).astype(int), 9)
        bins, ece = [], 0.
        for k in range(10):
            selected = bin_index == k; n = int(selected.sum())
            pred = float(p[selected, j].mean()) if n else None
            truth = float(y[selected, j].mean()) if n else None
            if n: ece += n/len(y)*abs(pred-truth)
            bins.append(dict(bin=k, n=n, predicted=pred, observed=truth))
        context_metrics[context] = dict(prevalence=float(y[:, j].mean()), predicted_mean=float(p[:, j].mean()),
            brier=float(np.square(p[:, j]-y[:, j]).mean()), log_loss=float(nll[:, j].mean()),
            ece10=float(ece), average_precision=average_precision(y[:, j], p[:, j]), reliability=bins)
    truth, prediction = y.sum(1), p.sum(1)
    mse = float(np.square(prediction-truth).mean())
    return dict(n=len(y), brier=float(np.square(p-y).mean()), log_loss=float(nll.mean()),
        macro_ece10=float(np.mean([m['ece10'] for m in context_metrics.values()])),
        macro_ap=float(np.mean([m['average_precision'] for m in context_metrics.values()])) if y.any() else None,
        contexts=context_metrics,
        breadth=dict(observed_mean=float(truth.mean()), predicted_mean=float(prediction.mean()),
            bias=float((prediction-truth).mean()), mae=float(np.abs(prediction-truth).mean()), rmse=mse**.5,
            r2=1-mse/float(np.var(truth)) if np.var(truth)>0 else None,
            pearson=correlation(truth, prediction), spearman=correlation(rankdata(truth), rankdata(prediction))),
        observed_breadth=[dict(degree=int(k), n=int((truth==k).sum()),
            predicted_mean=float(prediction[truth==k].mean())) for k in np.unique(truth)])


def run(bundle_root, data_root, inventory, output):
    if output.exists(): raise FileExistsError('Preserve prior calibration')
    provenance = json.loads((bundle_root/'provenance.json').read_text())
    for name, expected in provenance['outputs'].items():
        if sha(bundle_root/name) != expected: raise ValueError('Changed inference cache: '+name)
    records = json.loads(inventory.read_text())['records']
    ranked = sorted((r for r in records if r['model_type']=='classifier'
        and (r['selected'].get('background_metrics') or {}).get('enhancers_plus_background')),
        key=lambda r:-r['selected']['background_metrics']['enhancers_plus_background']['macro_average_precision'])
    winner = ranked[0]
    if winner['model_id'] != MODEL: raise ValueError('Selected model changed; inspect before fitting')
    lock = json.loads((bundle_root/'selection_lock.json').read_text())['2048']['20260914']
    if (lock['checkpoint_sha256'] != winner['checkpoint_sha256']
            or lock['best_epoch'] != winner['selected']['epoch']):
        raise ValueError('Wrong classifier checkpoint')
    source_files = [bundle_root/name for name in ('provenance.json','selection_lock.json','predictions.npz')]
    source_files += [inventory, data_root/'validation.npz', data_root/'background/validation.npz']
    inputs = {str(path): sha(path) for path in source_files}
    for name in ('validation.npz','background/validation.npz'):
        key = 'experiments/classifier_background_20260916/data/'+name
        if inputs[str(data_root/name)] != provenance['source_sha256'][key]:
            raise ValueError('Changed validation metadata')
    # Explicitly request validation keys only. Test entries are never loaded for calibration.
    with np.load(bundle_root/'predictions.npz', allow_pickle=False) as bundle:
        ids, y = bundle['validation__ids'], bundle['validation__labels']
        p = bundle['validation__2048__20260914__enhancer'].astype(float)
        bg = bundle['validation__2048__20260914__background'].astype(float)
        bg_ids = bundle['validation__background_ids']
        paired_index = bundle['validation__background_enhancer_index']
    with np.load(data_root/'validation.npz', allow_pickle=False) as meta:
        np.testing.assert_array_equal(ids, meta['ids']); np.testing.assert_array_equal(y, meta['labels'])
        chrom, summit = meta['chrom'], meta['summit']
    with np.load(data_root/'background/validation.npz', allow_pickle=False) as meta:
        np.testing.assert_array_equal(bg_ids, meta['ids'])
        np.testing.assert_array_equal(paired_index, meta['enhancer_index'])
        np.testing.assert_array_equal(ids[paired_index], meta['enhancer_ids'])
        if meta['labels'].any(): raise ValueError('Expected catalogue-negative background labels')
        order = np.argsort(paired_index)
        bg_chrom, bg_start, bg_end = (meta[k][order] for k in ('chrom','start','end'))
    n = len(ids)
    if (n != 4062 or len(set(ids)) != n or len(set(bg_ids)) != n
            or not np.array_equal(np.sort(paired_index), np.arange(n))):
        raise ValueError('Invalid validation cohort/pairing')
    bg, bg_ids = bg[order], bg_ids[order]
    validate(p, y); validate(bg)
    probabilities, labels = np.vstack((p, bg)), np.vstack((y, np.zeros_like(y)))
    baseline = evaluate(y, p)
    np.testing.assert_allclose(baseline['macro_ap'], lock['best_macro_ap'], atol=1e-12)
    np.testing.assert_allclose(evaluate(labels, probabilities)['macro_ap'],
        winner['selected']['background_metrics']['enhancers_plus_background']['macro_average_precision'], atol=1e-12)
    fold, train_masks, fold_records = make_folds(chrom, summit, bg_chrom, bg_start, bg_end)
    output.mkdir(parents=True)
    all_results, final_calibrators = {}, {}
    saved = dict(ids=np.r_[ids, bg_ids], labels=labels, raw_probabilities=probabilities,
                 fold=np.r_[fold, fold], population=np.asarray(['enhancer']*n+['background']*n))
    for population in POPULATIONS:
        all_results[population] = {}
        for method in METHODS:
            oof = np.full_like(probabilities, np.nan); fitted = []
            for k, mask in enumerate(train_masks):
                train = np.r_[mask, mask if population.endswith('1to1') else np.zeros(n, bool)]
                held = np.r_[fold==k, fold==k]
                parameters = fit(probabilities[train], labels[train], method)
                oof[held] = apply(probabilities[held], parameters)
                fitted.append(dict(fold=k, calibration_examples=int(train.sum()), parameters=parameters))
            if not np.isfinite(oof).all(): raise ValueError('Missing OOF predictions')
            saved[population+'__'+method] = oof
            all_results[population][method] = dict(
                enhancers_only=evaluate(y, oof[:n]), background_only=evaluate(np.zeros_like(y), oof[n:]),
                enhancers_plus_background_1to1=evaluate(labels, oof), folds=fitted)
            print(json.dumps(dict(event='calibration_scored',population=population,method=method,
                log_loss=all_results[population][method][population]['log_loss'])),flush=True)
        # Method choice is validation-development, not an independent test result.
        chosen = min(METHODS, key=lambda m:all_results[population][m][population]['log_loss'])
        eligible = np.r_[np.ones(n,bool), np.full(n,population.endswith('1to1'))]
        parameters = fit(probabilities[eligible], labels[eligible], chosen)
        final_calibrators[population] = dict(parameters, checkpoint_sha256=winner['checkpoint_sha256'],
            model_id=MODEL, contexts=CONTEXTS, fitting_population=population,
            selected_by='minimum aggregate-context OOF log loss within fitting population',
            scalar='sum of eight calibrated probabilities; no observed-label mask',
            training_examples=int(eligible.sum()), enhancer_fraction=1. if population==POPULATIONS[0] else .5)
    report = dict(status='complete', model_id=MODEL, checkpoint_sha256=winner['checkpoint_sha256'],
        checkpoint_path=winner['checkpoint_path'], epoch=lock['best_epoch'], contexts=CONTEXTS,
        methods=METHODS, populations=POPULATIONS, folds=fold_records, results=all_results,
        selected={p:c['method'] for p,c in final_calibrators.items()},
        fold_rule='Five contiguous enhancer-coordinate folds; keep matched backgrounds with enhancers; purge any training pair with either 2048bp input overlapping any held-out input.',
        test_predictions_used=False, inference_run=False, attributions_recomputed=False,
        caveats=['Validation previously used for checkpoint/model selection; cross-fitting is development evidence, not a pristine final holdout.',
                 'Genome background is sampled and assigned zero labels, not proven inactive.',
                 '1:1 enhancer/background calibration is not a genome-wide activity probability.',
                 'Sigmoid/temperature fit on logit of RC-mean probability, not RC-mean logit.',
                 'OOF fold-specific calibration can alter pooled rankings; a fixed positive-slope calibrator preserves per-context ranks except clipping/ties.',
                 'No per-context F1 thresholds or family-aware attribution targets are fitted.',
                 'Different model and nonlinear target require new IG; legacy motif maps cannot be converted.'])
    write_json(output/'metrics.json', report)
    write_json(output/'calibrators.json', final_calibrators)
    np.savez_compressed(output/'validation_oof.npz', **saved)
    lines = ['# Frozen-classifier probability calibration', '',
        'Model: '+MODEL+', epoch '+str(lock['best_epoch'])+'. Local CPU only.', '',
        'Five paired, spatially blocked folds with overlapping training DNA windows purged.',
        'All numbers below are out-of-fold validation-development estimates, not test results.', '',
        '| Calibration population | Method | Evaluation population | Log loss | Brier | ECE10 | Breadth MAE | Breadth bias |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for population in POPULATIONS:
        for method in METHODS:
            for view in POPULATIONS:
                r = all_results[population][method][view]
                lines.append(f"| {population} | {method} | {view} | {r['log_loss']:.5f} | {r['brier']:.5f} | {r['macro_ece10']:.5f} | {r['breadth']['mae']:.4f} | {r['breadth']['bias']:+.4f} |")
    lines += ['', 'Selected methods: '+json.dumps(report['selected'])+'.', '',
              '## Limitations', '', *['- '+s for s in report['caveats']], '',
              'Per-context reliability bins, observed-breadth summaries and fold coefficients: `metrics.json`.',
              'Final coefficients fitted to all validation examples of each population: `calibrators.json`.',
              'Final-fit coefficients are for a subsequent frozen pilot, not their own evaluation.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    for path, expected in inputs.items():
        if sha(Path(path)) != expected: raise ValueError('Input changed during fitting')
    write_json(output/'complete.json', dict(status='complete', inputs=inputs,
        code_sha256=sha(Path(__file__)), numpy=np.__version__,
        files={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(json.dumps(dict(event='calibration_complete', selected=report['selected'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle-root','data-root','inventory','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    run(a.bundle_root, a.data_root, a.inventory, a.output)
