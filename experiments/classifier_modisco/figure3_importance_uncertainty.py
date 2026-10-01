"""Pointwise enhancer-bootstrap uncertainty for the fixed FlyFactorSurvey GAF mean."""
import numpy as np

from .common import digest

REPLICATES=10000
SEED=20260930
GAF_ID='FBgn0013263'


def bootstrap_mean(values, *, seed, replicates=REPLICATES):
    """Resample complete enhancer scores, not correlated bases or motif instances."""
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or not np.isfinite(values).all():
        raise ValueError('Finite one-dimensional enhancer scores required')
    if replicates<2:raise ValueError('At least two bootstrap replicates required')
    if len(values)<2:return None
    rng=np.random.default_rng(seed);means=np.empty(replicates)
    for start in range(0,replicates,128):
        n=min(128,replicates-start)
        samples=rng.integers(0,len(values),size=(n,len(values)))
        means[start:start+n]=values[samples].mean(1)
    return np.quantile(means,[.025,.975],method='linear').tolist()


def gaf_intervals(summary,scores_path):
    if summary['profiles']!=656:raise ValueError('Expected FlyFactorSurvey catalogue')
    row=next(r for r in summary['exact'] if r['id']==GAF_ID)
    with np.load(scores_path,allow_pickle=False) as z:
        if len(np.unique(z['ids']))!=len(z['ids']):raise ValueError('Duplicate enhancer IDs')
        ix=np.flatnonzero(z['motif_ids']==GAF_ID)
        if len(ix)!=1:raise ValueError('Expected one GAF profile')
        scores=z['scores'][ix[0]];covered=z['covered_bases'][ix[0]]
        degree=z['breadth'];eligible=z['eligible']
        np.testing.assert_array_equal(np.isfinite(scores),covered>0)
        if np.isfinite(scores[~eligible]).any():raise ValueError('QC-excluded scores present')
        groups=[]
        for k in range(1,9):
            mask=eligible&(degree==k);values=scores[mask];values=values[np.isfinite(values)]
            if len(values)!=row['motif_containing_enhancers'][k-1] or int(mask.sum())!=row['group_enhancers'][k-1]:
                raise ValueError('Carrier or cohort count mismatch')
            mean=float(values.mean(dtype=np.float64)) if len(values) else None
            if mean!=row['means'][k-1]:raise ValueError('Saved mean mismatch')
            interval=bootstrap_mean(values,seed=np.random.SeedSequence([SEED,k]))
            groups.append(dict(degree=k,n=len(values),mean=mean,
                               lower=None if interval is None else interval[0],
                               upper=None if interval is None else interval[1]))
    return dict(motif_id=GAF_ID,confidence=.95,replicates=REPLICATES,seed=SEED,groups=groups,
        method='Pointwise percentile bootstrap, independently within each exact degree',
        resampling_unit='One signed mean importance score per motif-containing enhancer',
        estimator='Active-context mean IG, union of hit bases per enhancer, equal-weight carrier mean',
        missing_policy='Noncarriers excluded, never zero-filled; n<2 has no interval',
        uncertainty_scope='Enhancer-resampling uncertainty conditional on fixed model, PWM, scans and maps; '
                          'treats enhancers as independent. Not between-model, attribution-reference or '
                          'biological-replicate uncertainty; not a simultaneous confidence band.',
        scores_sha256=digest(scores_path))
