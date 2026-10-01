"""Validation-only, spatially cross-fitted thresholds for binary context breadth."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .calibrate_breadth import CONTEXTS, apply, correlation, make_folds, sha, validate, write_json


def choose_threshold(probability, labels):
    """Exact maximum F1 over distinct scores; ties prefer the highest threshold.

    Positive means probability >= threshold. Identical probabilities are never
    split. Both label classes are required. This optimizes context F1, not a
    breadth correlation or agreement with each enhancer's observed degree.
    """
    p, y = np.asarray(probability, float), np.asarray(labels)
    if (p.ndim != 1 or y.shape != p.shape or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1)) or not np.isin(y, (0,1)).all()
            or not 0 < y.sum() < len(y)):
        raise ValueError('Finite aligned scores and both binary classes required')
    order = np.argsort(-p, kind='stable')
    pp, yy = p[order], y[order]
    ends = np.flatnonzero(np.r_[pp[1:] != pp[:-1], True])
    tp = yy.cumsum()[ends].astype(np.int64)
    predicted = ends+1
    f1 = 2*tp/(predicted+y.sum())
    best = int(np.argmax(f1))
    return dict(threshold=float(pp[ends[best]]), fitting_f1=float(f1[best]),
        fitting_n=len(y), fitting_positives=int(y.sum()), fitting_predicted_positives=int(predicted[best]))


def fit_fold(raw, labels, training, held, calibration):
    if np.any(training & held) or not training.any() or not held.any():
        raise ValueError('Nonempty disjoint threshold-fitting and evaluation sets required')
    q_fit, q_held = apply(raw[training],calibration), apply(raw[held],calibration)
    fitted = [choose_threshold(q_fit[:,j],labels[training,j]) for j in range(8)]
    thresholds = np.asarray([r['threshold'] for r in fitted])
    return q_held >= thresholds, thresholds, fitted


def breadth_metrics(labels, score, discrete):
    truth, prediction = labels.sum(1).astype(float), np.asarray(score,float)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError('Invalid predicted breadth')
    error = prediction-truth
    out = dict(n=len(truth),observed_mean=float(truth.mean()),predicted_mean=float(prediction.mean()),
        bias=float(error.mean()),mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),
        r2=float(1-np.mean(error**2)/np.var(truth)),pearson=correlation(truth,prediction),
        spearman=correlation(rankdata(truth),rankdata(prediction)))
    if discrete:
        if not np.isin(prediction,np.arange(9)).all(): raise ValueError('Counts must be0..8')
        out.update(exact_breadth_accuracy=float(np.mean(truth==prediction)),
            within_one_accuracy=float(np.mean(np.abs(error)<=1)),
            zero_context_predictions=int(np.sum(prediction==0)))
    groups=[]
    for degree in range(1,9):
        values=prediction[truth==degree]
        if not len(values): raise ValueError('Missing experimental degree')
        row=dict(degree=degree,n=len(values),mean=float(values.mean()),median=float(np.median(values)),
                 q1=float(np.quantile(values,.25)),q3=float(np.quantile(values,.75)))
        if discrete: row['counts_0_to_8']=np.bincount(values.astype(int),minlength=9).tolist()
        groups.append(row)
    out['groups']=groups
    return out


def binary_metrics(labels, predicted):
    if labels.shape!=predicted.shape or not np.isin(predicted,(0,1)).all():
        raise ValueError('Eight binary predictions required')
    y,p=labels.astype(bool),predicted.astype(bool)
    rows={}
    for j,c in enumerate(CONTEXTS):
        tp,fp,fn,tn=(int(v.sum()) for v in (p[:,j]&y[:,j],p[:,j]&~y[:,j],~p[:,j]&y[:,j],~p[:,j]&~y[:,j]))
        rows[c]=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,
            recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),
            specificity=tn/(tn+fp) if tn+fp else None)
    tp=sum(v['tp'] for v in rows.values()); fp=sum(v['fp'] for v in rows.values()); fn=sum(v['fn'] for v in rows.values())
    return dict(contexts=rows,macro_f1=float(np.mean([v['f1'] for v in rows.values()])),
        micro_f1=2*tp/(2*tp+fp+fn),label_accuracy=float(np.mean(p==y)),
        exact_label_set_accuracy=float(np.mean(np.all(p==y,axis=1))),
        breadth=breadth_metrics(labels,p.sum(1),True))


def brute_threshold(p,y):
    """Independent exhaustive verifier, not the sorting/cumulative fitting code."""
    candidates=np.unique(p)[::-1]
    best=None
    for first in range(0,len(candidates),128):
        thresholds=candidates[first:first+128]
        calls=p[None,:]>=thresholds[:,None]
        score=2*(calls*y[None,:]).sum(1)/(calls.sum(1)+y.sum())
        j=int(np.argmax(score))
        if best is None or score[j]>best['f1']:
            best=dict(threshold=float(thresholds[j]),f1=float(score[j]))
    return best


def run(source,data_root,output):
    if output.exists(): raise FileExistsError('Preserve existing threshold analysis')
    completion=json.loads((source/'complete.json').read_text())
    if completion['status']!='complete' or json.loads((source/'verification.json').read_text())['status']!='passed':
        raise ValueError('Verified calibration required')
    for name,expected in completion['files'].items():
        if sha(source/name)!=expected: raise ValueError('Changed calibration output: '+name)
    for name in ('validation.npz','background/validation.npz'):
        suffix='classifier_background_training_20260916/data/'+name
        expected=[v for k,v in completion['inputs'].items() if k.endswith(suffix)]
        if len(expected)!=1 or sha(data_root/name)!=expected[0]:
            raise ValueError('Changed calibration metadata: '+name)
    metrics=json.loads((source/'metrics.json').read_text())
    final=json.loads((source/'calibrators.json').read_text())['enhancers_only']
    if metrics['selected']['enhancers_only']!='sigmoid' or tuple(final['contexts'])!=CONTEXTS:
        raise ValueError('Unexpected frozen calibration')
    with np.load(source/'validation_oof.npz',allow_pickle=False) as f:
        keep=f['population']=='enhancer'
        ids,labels,raw,fold,calibrated=(f[k][keep] for k in
            ('ids','labels','raw_probabilities','fold','enhancers_only__sigmoid'))
    validate(raw,labels); validate(calibrated)
    if labels.shape!=(4062,8) or len(np.unique(ids))!=4062 or not labels.any(1).all():
        raise ValueError('Wrong enhancer cohort')
    with np.load(data_root/'validation.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(ids,f['ids']); np.testing.assert_array_equal(labels,f['labels'])
        chrom,summit=f['chrom'],f['summit']
    with np.load(data_root/'background/validation.npz',allow_pickle=False) as f:
        index=f['enhancer_index']; np.testing.assert_array_equal(np.sort(index),np.arange(len(ids)))
        np.testing.assert_array_equal(ids[index],f['enhancer_ids'])
        order=np.argsort(index); bc,bs,be=(f[k][order] for k in ('chrom','start','end'))
    reconstructed,masks,details=make_folds(chrom,summit,bc,bs,be)
    np.testing.assert_array_equal(reconstructed,fold)
    if details!=metrics['folds']: raise ValueError('Changed calibration folds')
    fits=metrics['results']['enhancers_only']['sigmoid']['folds']
    thresholds=np.empty((5,8)); binary=np.zeros_like(labels,bool); seen=np.zeros(len(ids),int)
    fold_records=[]; threshold_checks=0
    for k,training in enumerate(masks):
        held=fold==k; calibration=fits[k]['parameters']
        if fits[k]['fold']!=k or fits[k]['calibration_examples']!=int(training.sum()):
            raise ValueError('Wrong fold calibrator')
        np.testing.assert_array_equal(apply(raw[held],calibration),calibrated[held])
        binary[held],thresholds[k],rows=fit_fold(raw,labels,training,held,calibration)
        # Independently verify all candidate thresholds and both-window purging.
        q=apply(raw[training],calibration)
        for j,row in enumerate(rows):
            expected=brute_threshold(q[:,j],labels[training,j])
            np.testing.assert_allclose([row['threshold'],row['fitting_f1']],
                                       [expected['threshold'],expected['f1']],atol=1e-14,rtol=0)
            threshold_checks+=1
        starts=np.r_[summit[training]-1024,bs[training]]; ends=np.r_[summit[training]+1024,be[training]]
        cs=np.r_[chrom[training],bc[training]]
        hs=np.r_[summit[held]-1024,bs[held]]; he=np.r_[summit[held]+1024,be[held]]
        hc=np.r_[chrom[held],bc[held]]
        for first in range(0,len(starts),128):
            conflict=(starts[first:first+128,None]<he)&(ends[first:first+128,None]>hs)&(cs[first:first+128,None]==hc)
            if conflict.any(): raise ValueError('Fitting DNA overlaps held-out DNA')
        seen[held]+=1
        fold_records.append(dict(fold=k,fitting_n=int(training.sum()),heldout_n=int(held.sum()),
            thresholds=thresholds[k].tolist(),fitting_metrics=rows))
    np.testing.assert_array_equal(seen,np.ones(len(ids)))
    np.testing.assert_array_equal(binary,calibrated>=thresholds[fold])
    # Freeze deployment thresholds on ALL validation; do not evaluate them on
    # those same rows and present the result as held-out performance.
    final_q=apply(raw,final)
    final_rows=[choose_threshold(final_q[:,j],labels[:,j]) for j in range(8)]
    for j,row in enumerate(final_rows):
        expected=brute_threshold(final_q[:,j],labels[:,j])
        np.testing.assert_allclose(row['threshold'],expected['threshold'],atol=0,rtol=0)
        threshold_checks+=1
    baseline=calibrated>=.5
    soft=breadth_metrics(labels,calibrated.sum(1),False)
    for key in ('pearson','spearman','mae','rmse','r2'):
        np.testing.assert_allclose(soft[key],metrics['results']['enhancers_only']['sigmoid']['enhancers_only']['breadth'][key],atol=1e-12)
    report=dict(status='complete',model_id=metrics['model_id'],checkpoint_sha256=metrics['checkpoint_sha256'],
        contexts=list(CONTEXTS),population='known enhancers only',split='validation',
        threshold_criterion='Maximum F1 separately in each context; highest threshold breaks ties.',
        rule='active if calibrated_probability >= context_threshold; predicted_breadth = sum(active calls). Zero calls allowed.',
        evaluation='Five outer spatial folds; calibrator and threshold fitted without the held-out fold and with original DNA-overlap purge.',
        folds=fold_records,results=dict(continuous_calibrated_sum=soft,
            calibrated_threshold_05=binary_metrics(labels,baseline),calibrated_max_f1=binary_metrics(labels,binary)),
        caveats=['Validation previously selected model and calibration method; these remain development results.',
                 'Maximum context F1 is not an optimization of breadth correlation, breadth MAE or genomic-background specificity.',
                 'Probability calibration and decision-threshold tuning are distinct; this does not retrain the model.',
                 'No minimum-one-context correction, test fitting, new inference or attribution changes.'],
        method_reference='https://scikit-learn.org/stable/modules/classification_threshold.html')
    output.mkdir(parents=True)
    write_json(output/'metrics.json',report)
    write_json(output/'thresholds.json',dict(contexts=list(CONTEXTS),thresholds=[r['threshold'] for r in final_rows],
        fitting_scope='All4062validation enhancers; deploy with exact frozen enhancer-only calibrator, not an OOF performance estimate.',
        criterion=report['threshold_criterion'],comparison='>=',checkpoint_sha256=metrics['checkpoint_sha256'],
        calibrators_sha256=sha(source/'calibrators.json'),calibrator_population='enhancers_only'))
    np.savez_compressed(output/'validation_binary_oof.npz',ids=ids,labels=labels,fold=fold,
        calibrated_probabilities=calibrated,binary_predictions=binary,binary_threshold_05=baseline,
        predicted_breadth=binary.sum(1).astype(np.uint8),experimental_breadth=labels.sum(1).astype(np.uint8),
        fold_thresholds=thresholds,fold_training_masks=np.stack(masks))
    write_json(output/'verification.json',dict(status='passed',n=len(ids),thresholds_exhaustively_checked=threshold_checks,
        heldout_prediction_coverage='Every enhancer exactly once; all8calls replayed.',
        calibration_replay='Five saved held-out calibrators reproduced exactly.',
        overlap_check='Independent brute-force: no fitting enhancer/background window overlaps held-out windows.',
        continuous_metric_parity='All five breadth metrics match previous report.',no_test_data=True,no_cuda=True))
    paths=[source/n for n in ('complete.json','verification.json','metrics.json','calibrators.json','validation_oof.npz')]
    paths += [data_root/'validation.npz',data_root/'background/validation.npz']
    write_json(output/'complete.json',dict(status='complete',inputs={str(p):sha(p) for p in paths},
        code_sha256=sha(Path(__file__)),files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(event='binary_breadth_complete',output=str(output),
        macro_f1=report['results']['calibrated_max_f1']['macro_f1'],
        breadth=report['results']['calibrated_max_f1']['breadth']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('results/classifier_calibration_20260921'))
    p.add_argument('--data-root',type=Path,default=Path('results/classifier_background_training_20260916/data'))
    p.add_argument('--output',type=Path,default=Path('results/classifier_binary_breadth_20260921'))
    a=p.parse_args(); run(a.source,a.data_root,a.output)
