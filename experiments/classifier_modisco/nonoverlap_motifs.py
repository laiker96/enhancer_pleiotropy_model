"""Non-overlapping 1 / 2-5 / 6-8 motif analysis from existing IG64 x100 maps."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .common import digest, event, require_allocation, write_json
from .dual_motif_pipeline import TARGETS, group_indices, report, verify_sources
from .jaspar_importance import union_importance
from .original_intervals import discover_original

GROUPS = (('exact_1', 1, 1), ('between_2_5', 2, 5), ('ge_6', 6, 8))
LABELS = ('1', '2-5', '6-8')


def partition(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 8 or not np.isin(labels, [0, 1]).all():
        raise ValueError('Eight binary activity labels required')
    breadth = labels.sum(1)
    masks = [(breadth >= lo) & (breadth <= hi) for _, lo, hi in GROUPS]
    if not np.all(np.asarray(masks).sum(0) == 1):
        raise ValueError('Groups must partition every enhancer exactly once')
    return masks


def verify_receipt(directory, name='complete.json'):
    receipt = json.loads((directory/name).read_text())
    if receipt['status'] != 'complete':
        raise ValueError('Incomplete input: '+str(directory))
    for filename, expected in receipt['files'].items():
        if digest(directory/filename) != expected:
            raise ValueError('Changed input: '+str(directory/filename))
    return receipt


def setup(project, root, config):
    verify_sources(project, config)
    parent = project/config['parent']
    audit = json.loads((parent/'audit_complete.json').read_text())
    gate = json.loads((parent/'synthetic_passed.json').read_text())
    if (audit['status'] != 'complete' or gate['status'] != 'passed'
            or gate['config_sha256'] != digest(parent/'config.json')):
        raise ValueError('Parent tensor audit or native-boundary gate failed')
    for filename in ('metadata.npz', 'native_actual.npy', 'native_hypothetical.npy'):
        if digest(parent/filename) != audit['files'][filename]:
            raise ValueError('Changed assembled attribution')
        (root/filename).symlink_to(parent/filename)
    with np.load(root/'metadata.npz', allow_pickle=False) as saved:
        metadata = dict(saved)
    if len(np.unique(metadata['ids'])) != len(metadata['ids']):
        raise ValueError('Duplicate enhancer IDs')
    masks = partition(metadata['labels'])
    counts = []
    for target, target_name in enumerate(TARGETS):
        for number, (name, lo, hi) in enumerate(GROUPS):
            indices, eligible, excluded = group_indices(metadata, target, lo, hi, config['discovery_parameters']['seed'])
            counts.append(dict(target=target_name, group=name, eligible_train=eligible,
                passing_train=len(indices), excluded_train=excluded,
                all=int(masks[number].sum()), validation=int((masks[number] & (metadata['split']=='validation')).sum()),
                test=int((masks[number] & (metadata['split']=='test')).sum())))
        directory = root/target_name/'groups'; directory.mkdir(parents=True)
        for name, lo, hi in (GROUPS[0], GROUPS[2]):
            source = parent/target_name/'groups'/name
            verify_receipt(source)
            selection = json.loads((source/'selection.json').read_text())
            with np.load(source/'examples.npz', allow_pickle=False) as saved:
                got = saved['indices']
                expected, _, _ = group_indices(metadata, target, lo, hi, selection['seed'])
                np.testing.assert_array_equal(got, expected)
                np.testing.assert_array_equal(saved['ids'], metadata['ids'][expected])
            if selection['attribution_sha256'] != digest(parent/'audit_complete.json'):
                raise ValueError('Reused fit has different attribution inputs')
            (directory/name).symlink_to(source, target_is_directory=True)
    write_json(root/'setup_complete.json', dict(status='complete', counts=counts,
        parent_audit_sha256=digest(parent/'audit_complete.json'), parent_synthetic_gate_sha256=digest(parent/'synthetic_passed.json'),
        config_sha256=digest(root/'config.json'),
        reuse='Exact1 and ge6 discovery copied by read-only reference, with original seed/order; only between2and5 is new.',
        files={name:digest(root/name) for name in ('metadata.npz','native_actual.npy','native_hypothetical.npy')}))
    event('nonoverlap_setup_complete', counts=counts)


def discover(project, root, config, target):
    ready = json.loads((root/'setup_complete.json').read_text())
    if ready['config_sha256'] != digest(root/'config.json') or ready['status'] != 'complete':
        raise ValueError('Missing non-overlap preparation gate')
    for filename in ('metadata.npz', 'native_hypothetical.npy'):
        if digest(root/filename) != ready['files'][filename]:
            raise ValueError('Changed assembled input')
    with np.load(root/'metadata.npz', allow_pickle=False) as f:
        metadata = dict(f)
    parameters = dict(config['discovery_parameters'], seed=config['discovery_parameters']['seed']+1)
    name, lo, hi = GROUPS[1]
    indices, eligible, excluded = group_indices(metadata, target, lo, hi, parameters['seed'])
    directory = root/TARGETS[target]/'groups'/name; directory.mkdir()
    lengths = metadata['length'][indices]; width = int(lengths.max())
    source = np.load(root/'native_hypothetical.npy', mmap_mode='r', allow_pickle=False)
    hyp = np.asarray(source[indices, target, :width]).copy()
    sequence = np.eye(5, 4, dtype=np.float32)[metadata['sequence'][indices, :width]]
    np.savez_compressed(directory/'examples.npz', indices=indices, ids=metadata['ids'][indices], lengths=lengths)
    np.savez_compressed(directory/'discovery_inputs.npz', sequence=sequence, hypothetical=hyp, lengths=lengths)
    selection = dict(group=name, target=TARGETS[target], minimum_breadth=lo, maximum_breadth=hi,
        elements=len(indices), eligible_before_quality=eligible, quality_excluded=excluded,
        train_only=True, enhancer_downsampling=False, interval='original_catalog_bounds',
        parameters=parameters, seed=parameters['seed'], attribution_sha256=ready['parent_audit_sha256'])
    write_json(directory/'selection.json', selection); event('nonoverlap_discovery_start', **selection)
    result = discover_original(sequence, hyp, lengths, parameters, directory/'motifs.h5',
                               parameters['max_seqlets_per_metacluster'])
    write_json(directory/'complete.json', dict(status='complete', target=TARGETS[target], group=name,
        audit=result, reclustered=False, files={p.name:digest(p) for p in directory.iterdir() if p.is_file()}))
    event('nonoverlap_discovery_complete', target=TARGETS[target],
        positive=result['positive']['patterns'], negative=result['negative']['patterns'])


def wilson(hits, n):
    if not n:
        return None
    z = 1.959963984540054; fraction = hits/n
    center = (fraction+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,float(center-radius)), min(1.,float(center+radius))]


def summarize_metric(best, scores, metadata, target, low, high, split, quantifiable, width):
    breadth = metadata['labels'].sum(1)
    mask = (metadata['split']==split) & (breadth>=low) & (breadth<=high)
    hits = mask & (best <= 1e-4)
    quality = metadata['quality_pass'][:, target]
    values = scores[mask & quality & np.isfinite(scores)]
    # A noncarrier is missing, not an attribution score of zero.
    if np.any(np.isfinite(scores) & (~quality | (best > 1e-4))):
        raise ValueError('Attribution score without a passing scan carrier')
    n = int(mask.sum()); count = int(hits.sum())
    return dict(n=n, sequence_carriers=count, fraction=count/n if n and quantifiable else None,
        prevalence_wilson95=wilson(count,n) if quantifiable else None,
        length_eligible=int((mask & (metadata['length']>=width)).sum()),
        attribution_carriers=len(values), quality_excluded=int((mask & ~quality).sum()),
        mean_importance=float(values.mean()) if len(values) else None,
        sd_importance=float(values.std(ddof=1)) if len(values)>1 else None)


def compare_groups(project, root, config, target):
    ready = json.loads((root/'setup_complete.json').read_text())
    for name in ('metadata.npz','native_actual.npy'):
        if digest(root/name) != ready['files'][name]:
            raise ValueError('Changed attribution input')
    with np.load(root/'metadata.npz',allow_pickle=False) as f:
        metadata = dict(f)
    partition(metadata['labels'])
    actual = np.load(root/'native_actual.npy',mmap_mode='r',allow_pickle=False)
    result = root/TARGETS[target]
    verify_receipt(result, 'report_complete.json')
    audit = json.loads((result/'report_audit.json').read_text())
    metrics, all_scores, all_best, motif_ids = [], [], [], []
    for group in audit['groups']:
        rows = group['rows']
        if not rows:
            continue
        scan = result/'scans'/group['name']
        with np.load(scan/'minimum_p.npz',allow_pickle=False) as f:
            best = f['minimum_p']; query_ids = f['query_ids'].tolist()
        expected = [r['id'].replace('/','__') for r in rows]
        if query_ids != expected or best.shape != (len(rows),len(metadata['ids'])):
            raise ValueError('Motif/sequence order differs from scan inputs')
        lookup = {key:i for i,key in enumerate(expected)}; sites = {}
        with gzip.open(scan/'hits.tsv.gz','rt') as stream:
            for hit in csv.DictReader((s for s in stream if not s.startswith('#')),delimiter='\t'):
                ident = hit['motif_id']; index = int(hit['sequence_name'][1:])
                if ident not in lookup or not 0<=index<len(metadata['ids']) or float(hit['p-value'])>1e-4:
                    raise ValueError('Unexpected scan hit')
                start,end=int(hit['start'])-1,int(hit['stop'])
                if not 0<=start<end<=metadata['length'][index]:
                    raise ValueError('Scan hit outside native enhancer')
                if metadata['quality_pass'][index,target]:
                    sites.setdefault((lookup[ident],index),[]).append((start,end))
        scores = np.full(best.shape,np.nan,dtype=np.float32)
        for (j,index),positions in sites.items():
            scores[j,index],_ = union_importance(actual[index,target,:metadata['length'][index]], positions)
        for j,row in enumerate(rows):
            expected_carriers = (best[j]<=1e-4) & metadata['quality_pass'][:,target]
            np.testing.assert_array_equal(np.isfinite(scores[j]),expected_carriers)
            comparisons,exact = [], []
            for split in ('train','validation','test'):
                for name,low,high in GROUPS:
                    comparisons.append(dict(group=name,split=split,**summarize_metric(best[j],scores[j],metadata,
                        target,low,high,split,row['frequency_quantifiable'],len(row['trimmed_pwm']))))
                for degree in range(1,9):
                    exact.append(dict(degree=degree,split=split,**summarize_metric(best[j],scores[j],metadata,
                        target,degree,degree,split,row['frequency_quantifiable'],len(row['trimmed_pwm']))))
            metrics.append(dict(id=row['id'],origin=group['name'],sign=row['sign'],rank=row['rank'],
                discovery_support=row['attribution_support'],frequency_quantifiable=row['frequency_quantifiable'],
                groups=comparisons,exact_degrees=exact))
            motif_ids.append(row['id'])
        all_scores.append(scores);all_best.append(best)
    shape=(0,len(metadata['ids']))
    np.savez_compressed(result/'cross_group_scores.npz', motif_ids=np.array(motif_ids,dtype=str),ids=metadata['ids'],
        mean_importance=np.concatenate(all_scores) if all_scores else np.empty(shape),
        minimum_p=np.concatenate(all_best) if all_best else np.empty(shape),
        split=metadata['split'],breadth=metadata['labels'].sum(1))
    summary=dict(status='complete',target=TARGETS[target],groups=list(GROUPS),references=100,steps=64,
        rows=metrics,site_p=1e-4,interval='native enhancer',
        units='logit/base' if target==0 else 'probability-sum/base',
        importance='Mean signed attribution over union of hit bases, equally averaged over quality-passing carriers; noncarriers missing.',
        occurrence='Each enhancer counted once per PWM; all enhancer denominators retained, independent of attribution quality.',
        caveats=['Independent origin-labelled PWMs are not unique TF families and are not merged.',
            'Discovery support is separate from sequence occurrence; a scan hit is not predictive occupancy.',
            'Wilson intervals are descriptive binomial intervals, not genomic-block-aware significance tests.',
            'Unequal group sizes, enhancer lengths, GC and tissue composition can affect comparisons.',
            'Validation/test are reported separately and were previously examined; not untouched confirmation.',
            'Positive/negative labels describe source clusters, not biological activation/repression.'],
        source_report_sha256=digest(result/'report_audit.json'),parent_audit_sha256=ready['parent_audit_sha256'])
    write_json(result/'cross_group_summary.json',summary)
    write_json(result/'comparison_complete.json',dict(status='complete',files={name:digest(result/name)
        for name in ('cross_group_summary.json','cross_group_scores.npz')}))
    event('nonoverlap_comparison_complete',target=TARGETS[target],motifs=len(metrics))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['setup','discover','report','figure'])
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--task',type=int,default=0);args=parser.parse_args()
    require_allocation('cpu')
    config=json.loads((args.root/'config.json').read_text())
    if args.stage=='setup':setup(args.project,args.root,config)
    else:
        if args.task not in (0,1):raise ValueError('Only two attribution targets')
        if args.stage=='discover':discover(args.project,args.root,config,args.task)
        elif args.stage=='report':
            report(args.project,args.root,config,args.task,groups=GROUPS)
            compare_groups(args.project,args.root,config,args.task)
        else:
            from .nonoverlap_motif_figure import render
            render(args.project,args.root,config,args.task)


if __name__=='__main__':main()
