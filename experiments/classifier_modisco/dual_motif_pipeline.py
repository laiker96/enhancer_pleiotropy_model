"""CPU-only native enhancer discovery from completed IG64/100-reference maps.

New outputs only. Frozen attribution, native boundary adapter and original
TF-MoDISco clustering are reused; no post-discovery merging or GPU inference.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import partial
import gzip
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .common import digest, event, require_allocation, write_json
from .original_intervals import GROUPS, discover_original, load_intervals

TARGETS = ('mean_active_logit', 'soft_breadth')
CORE_RULE = ('First-to-last qualifying 5-position window with >=4 columns >0.5 bits; '
             'trim terminal columns <=0.5 bits; preserve internal columns. '
             'Require width 5-30bp, mean >=0.5 bits/base, total >=5 bits and concordant sign.')


def core_rule(flank_threshold=.5):
    if not np.isfinite(flank_threshold) or not 0 < flank_threshold <= .5:
        raise ValueError('Flank threshold must be in (0, 0.5] bits')
    if flank_threshold == .5:
        return CORE_RULE
    return (f'First-to-last qualifying 5-position window with >=4 columns >0.5 bits; '
            f'trim terminal columns <={flank_threshold:g} bits and extend into immediately '
            f'adjacent columns >{flank_threshold:g} bits, stopping at the first weaker column; '
            'preserve internal columns. Require width 5-30bp, mean >=0.5 bits/base, '
            'total >=5 bits and concordant sign.')


def informative_core(pwm, *, flank_threshold=.5, window_size=5, min_informative=4):
    core_rule(flank_threshold)
    if (not isinstance(window_size, int) or isinstance(window_size, bool)
            or not isinstance(min_informative, int) or isinstance(min_informative, bool)
            or not 1 <= min_informative <= window_size):
        raise ValueError('Require integer 1 <= min_informative <= window_size')
    pwm = np.asarray(pwm, dtype=float)
    if (pwm.ndim != 2 or pwm.shape[1] != 4 or not len(pwm)
            or not np.isfinite(pwm).all() or (pwm < 0).any()
            or not np.allclose(pwm.sum(1), 1, atol=1e-6)):
        raise ValueError('Invalid PWM')
    pwm = pwm / pwm.sum(1, keepdims=True)
    info = 2 + (pwm*np.log2(np.maximum(pwm, 1e-300))).sum(1)
    starts = [i for i in range(len(pwm)-window_size+1)
              if (info[i:i+window_size] > .5).sum() >= min_informative]
    if not starts:
        reason = ('no_four_of_five_columns_above_0.5_bits' if (window_size,min_informative)==(5,4)
                  else f'no_{min_informative}_of_{window_size}_columns_above_0.5_bits')
        return dict(passed=False, reason=reason)
    start, end = starts[0], starts[-1]+window_size
    while start < end and info[start] <= flank_threshold:
        start += 1
    while end > start and info[end-1] <= flank_threshold:
        end -= 1
    if flank_threshold < .5:
        while start > 0 and info[start-1] > flank_threshold:
            start -= 1
        while end < len(info) and info[end] > flank_threshold:
            end += 1
    core = info[start:end]
    reason = ('core_shorter_than_5_bp' if len(core) < 5 else
              'core_longer_than_30_bp' if len(core) > 30 else
              'mean_information_below_0.5_bits' if core.mean() < .5 else
              'total_information_below_5_bits' if core.sum() < 5 else 'passed')
    return dict(start=int(start), end=int(end), width=int(end-start),
                mean_bits=float(core.mean()), total_bits=float(core.sum()),
                passed=reason == 'passed', reason=reason)


def validate_chunk(values, data, indices, signature):
    n, width = len(indices), data['sequence'].shape[1]
    if (str(values['signature']) != signature or int(values['references']) != 100
            or list(values['targets']) != list(TARGETS)):
        raise ValueError('Wrong attribution protocol')
    for name, expected in dict(indices=indices, ids=data['ids'][indices],
            labels=data['labels'][indices], split=data['split'][indices]).items():
        np.testing.assert_array_equal(values[name], expected)
    shapes = dict(actual=(n, 2, width), hypothetical=(n, 2, 4, width),
        delta=(n, 100, 2), target_difference=(n, 100, 2), steps=(n, 100, 2),
        quality_pass=(n, 2), logits=(n, 8), probabilities=(n, 8))
    for key, shape in shapes.items():
        if values[key].shape != shape or not np.isfinite(values[key]).all():
            raise ValueError('Malformed/nonfinite '+key)
    if not (values['steps'] == 64).all():
        raise ValueError('Unexpected integration resolution')
    observed = np.take_along_axis(values['hypothetical'],
        data['sequence'][indices, None, None, :], axis=2)[:, :, 0]
    np.testing.assert_allclose(observed, values['actual'], atol=1e-7, rtol=1e-5)
    quality = (np.abs(values['delta']) <= .02+.05*np.abs(values['target_difference'])).all(1)
    np.testing.assert_array_equal(values['quality_pass'], quality)
    if not ((values['probabilities'] >= 0) & (values['probabilities'] <= 1)).all():
        raise ValueError('Invalid probabilities')
    return quality


def group_indices(metadata, target_index, low, high, seed):
    breadth = metadata['labels'].sum(1)
    eligible = ((metadata['split'] == 'train') & (breadth >= low) & (breadth <= high))
    passing = eligible & metadata['quality_pass'][:, target_index]
    indices = np.random.default_rng(seed).permutation(np.flatnonzero(passing))
    return indices, int(eligible.sum()), int((eligible & ~passing).sum())


def verify_sources(project, config):
    for name, expected in config['source_hashes'].items():
        if digest(project/name) != expected:
            raise ValueError('Pinned source changed: '+name)


def audit(project, root, config):
    verify_sources(project, config)
    if (root/'audit_complete.json').exists():
        raise FileExistsError('Audit already complete')
    source, original = project/config['attribution'], project/config['original']
    source_config = json.loads((source/'config.json').read_text())
    if (source_config['steps'] != 64 or source_config['references'] != 100
            or source_config['refinement_steps'] != [] or source_config['targets'] != list(TARGETS)):
        raise ValueError('Unexpected source configuration')
    signature = digest(source/'MANIFEST.sha256')
    with np.load(project/config['cohort'], allow_pickle=False) as saved:
        data = dict(saved)
    n = len(data['ids'])
    if (n != 40338 or len(np.unique(data['ids'])) != n
            or data['sequence'].shape != (n, 2048) or not np.isin(data['sequence'], range(4)).all()
            or data['labels'].shape != (n, 8) or not np.isin(data['labels'], [0, 1]).all()
            or not data['labels'].any(1).all()):
        raise ValueError('Invalid source cohort')
    intervals = load_intervals(original, data)
    width = int(intervals['length'].max())
    hyp_path, actual_path = root/'native_hypothetical.partial.npy', root/'native_actual.partial.npy'
    hyp = np.lib.format.open_memmap(hyp_path, mode='w+', dtype='float32', shape=(n, 2, width, 4))
    actual = np.lib.format.open_memmap(actual_path, mode='w+', dtype='float32', shape=(n, 2, width))
    hyp[:] = 0; actual[:] = 0
    native_codes = np.full((n, width), 4, np.uint8)
    quality = np.zeros((n, 2), bool); seen = np.zeros(n, bool)
    chunks, receipts, flagged = {}, {}, []
    for shard in range(40):
        receipt_path = source/f'attribution_shard_{shard}.json'
        receipt = json.loads(receipt_path.read_text())
        if (receipt['status'] != 'complete' or receipt['signature'] != signature
                or receipt['shard'] != shard):
            raise ValueError('Invalid shard receipt')
        receipts[receipt_path.name] = digest(receipt_path)
        shard_n, shard_failed = 0, np.zeros(2, int)
        for name, expected in receipt['chunks'].items():
            if Path(name).name != name or name in chunks:
                raise ValueError('Unsafe or duplicate chunk')
            path = source/'chunks'/name
            if digest(path) != expected:
                raise ValueError('Chunk checksum mismatch: '+name)
            begin = int(name.removeprefix('chunk_').removesuffix('.npz'))
            indices = np.arange(begin, min(begin+16, n))
            if begin % 16 or begin//16 % 40 != shard or seen[indices].any():
                raise ValueError('Chunk/shard coverage disagreement')
            with np.load(path, allow_pickle=False) as saved:
                values = dict(saved)
            quality[indices] = validate_chunk(values, data, indices, signature)
            for row, index in enumerate(indices):
                offset, length = int(intervals['offset'][index]), int(intervals['length'][index])
                hyp[index, :, :length] = values['hypothetical'][row, :, :, offset:offset+length].transpose(0, 2, 1)
                actual[index, :, :length] = values['actual'][row, :, offset:offset+length]
                native_codes[index, :length] = data['sequence'][index, offset:offset+length]
                for target in np.flatnonzero(~quality[index]):
                    tol = .02+.05*np.abs(values['target_difference'][row, :, target])
                    failed_refs = np.flatnonzero(np.abs(values['delta'][row, :, target]) > tol)
                    flagged.append(dict(index=int(index), id=str(data['ids'][index]), split=str(data['split'][index]),
                        target=TARGETS[target], references=failed_refs.tolist(),
                        residuals=values['delta'][row, failed_refs, target].tolist(),
                        tolerances=tol[failed_refs].tolist()))
            seen[indices] = True; chunks[name] = expected
            shard_n += len(indices); shard_failed += (~quality[indices]).sum(0)
        if shard_n != receipt['elements'] or shard_failed.tolist() != receipt['quality_failures']:
            raise ValueError('Receipt counts disagree with tensors')
        event('dual_motif_audit_shard', shard=shard, verified=int(seen.sum()))
    if not seen.all() or set(chunks) != {f'chunk_{i:06d}.npz' for i in range(0, n, 16)}:
        raise ValueError('Incomplete or extra attribution coverage')
    hyp.flush(); actual.flush(); del hyp, actual
    hyp_path.replace(root/'native_hypothetical.npy'); actual_path.replace(root/'native_actual.npy')
    np.savez_compressed(root/'metadata.npz', ids=data['ids'], labels=data['labels'], split=data['split'],
        chrom=data['chrom'], sequence=native_codes, quality_pass=quality,
        **{key:intervals[key] for key in ('start', 'end', 'offset', 'length')})
    # Audit the revised rule against all prior raw patterns, without TF-specific rescue.
    from .simple_report import filter_native
    filter_rows = []
    for name, _, _ in GROUPS:
        directory = original/'groups'/name
        previous = json.loads((directory/'motifs.audit.json').read_text())
        counts = {s:previous[s]['patterns'] for s in ('positive', 'negative')}
        strict, strict_failed = filter_native(directory/'motifs.h5', name, counts, threshold=.5)
        revised, revised_failed = filter_native(directory/'motifs.h5', name, counts, core_filter=informative_core)
        strict_ids, revised_ids = {r['id'] for r in strict}, {r['id'] for r in revised}
        filter_rows.append(dict(group=name, strict_retained=len(strict), revised_retained=len(revised),
            added=sorted(revised_ids-strict_ids), removed=sorted(strict_ids-revised_ids),
            strict_exclusions=strict_failed, revised_exclusions=revised_failed,
            revised_cores=[{k:r[k] for k in ('id', 'sign', 'quality', 'supporting_discovery_enhancers')} for r in revised]))
    write_json(root/'filter_audit_previous50.json', dict(rule=CORE_RULE, groups=filter_rows,
        note='Exploratory sensitivity motivated by Grh exclusions; applied to every TF/sign/group before new discovery.'))
    output_names = ['metadata.npz', 'native_actual.npy', 'native_hypothetical.npy', 'filter_audit_previous50.json']
    write_json(root/'audit_complete.json', dict(status='complete', elements=n, signature=signature,
        chunks=chunks, shard_receipts=receipts, targets=list(TARGETS), references=100, steps=64,
        quality_failures=(~quality).sum(0).tolist(), flagged=flagged,
        exclusions='Target-specific quality failures excluded from discovery/importance, retained in backup and tensors',
        files={name:digest(root/name) for name in output_names}, source_hashes=config['source_hashes'],
        job_id=os.environ['SLURM_JOB_ID']))
    event('dual_motif_audit_complete', elements=n, quality_failures=(~quality).sum(0).tolist())


def smoke(root, config):
    rng = np.random.default_rng(471)
    lengths = rng.integers(100, 161, size=512)
    codes = rng.integers(0, 4, (512, 160))
    hyp = rng.normal(0, .01, (512, 160, 4)).astype(np.float32)
    motif = np.array([0, 1, 2, 0, 3, 1, 1, 2, 3, 0, 2, 2])
    for i, length in enumerate(lengths):
        start = int(rng.integers(25, length-40))
        codes[i, start:start+len(motif)] = motif
        hyp[i, np.arange(start, start+len(motif)), motif] += 1 if i < 256 else -1
        hyp[i, length:] = 1e6
    directory = root/'synthetic_smoke'; directory.mkdir(exist_ok=False)
    result = discover_original(np.eye(4, dtype=np.float32)[codes], hyp, lengths,
        config['discovery_parameters'], directory/'motifs.h5', 1000)
    if min(result[s]['patterns'] for s in ('positive', 'negative')) < 1:
        raise ValueError('Boundary-aware synthetic discovery did not recover both signs')
    write_json(root/'synthetic_passed.json', dict(status='passed', audit=result,
        config_sha256=digest(root/'config.json'), files={'synthetic_smoke/motifs.h5':digest(directory/'motifs.h5')}))


def discover(root, config, task):
    target, group_number = divmod(task, len(GROUPS))
    name, low, high = GROUPS[group_number]
    complete = json.loads((root/'audit_complete.json').read_text())
    ready = json.loads((root/'synthetic_passed.json').read_text())
    if ready['status'] != 'passed' or ready['config_sha256'] != digest(root/'config.json'):
        raise ValueError('Synthetic gate failed')
    for filename in ('metadata.npz', 'native_hypothetical.npy'):
        if digest(root/filename) != complete['files'][filename]:
            raise ValueError('Assembled input changed')
    with np.load(root/'metadata.npz', allow_pickle=False) as saved:
        metadata = dict(saved)
    parameters = dict(config['discovery_parameters'], seed=config['discovery_parameters']['seed']+group_number)
    indices, eligible, excluded = group_indices(metadata, target, low, high, parameters['seed'])
    directory = root/TARGETS[target]/'groups'/name; directory.mkdir(parents=True, exist_ok=False)
    lengths = metadata['length'][indices]; width = int(lengths.max())
    source = np.load(root/'native_hypothetical.npy', mmap_mode='r', allow_pickle=False)
    hyp = np.asarray(source[indices, target, :width]).copy()
    sequences = np.eye(5, 4, dtype=np.float32)[metadata['sequence'][indices, :width]]
    np.savez_compressed(directory/'examples.npz', indices=indices, ids=metadata['ids'][indices], lengths=lengths)
    np.savez_compressed(directory/'discovery_inputs.npz', sequence=sequences, hypothetical=hyp, lengths=lengths)
    selection = dict(group=name, target=TARGETS[target], elements=len(indices), eligible_before_quality=eligible,
        quality_excluded=excluded, train_only=True, enhancer_downsampling=False, interval='original_catalog_bounds',
        parameters=parameters, seed=parameters['seed'], attribution_sha256=digest(root/'audit_complete.json'))
    write_json(directory/'selection.json', selection); event('dual_discovery_start', **selection)
    result = discover_original(sequences, hyp, lengths, parameters,
        directory/'motifs.h5', parameters['max_seqlets_per_metacluster'])
    write_json(directory/'complete.json', dict(status='complete', target=TARGETS[target], group=name,
        audit=result, reclustered=False, job_id=os.environ['SLURM_JOB_ID'],
        files={p.name:digest(p) for p in directory.iterdir() if p.is_file()}))
    event('dual_discovery_complete', target=TARGETS[target], group=name,
          positive=result['positive']['patterns'], negative=result['negative']['patterns'])


def report(project, root, config, target, *, groups=GROUPS):
    from .simple_report import filter_native, scan, combgap
    from .tomtom_atlas import run as tomtom
    from .report_breadth import COLORS, GLYPHS
    verify_sources(project, config)
    result_root = root/TARGETS[target]
    previous = project/config['sequence_report']
    prepared = json.loads((previous/'motif_audit.json').read_text())
    with np.load(previous/'cohort_metadata.npz', allow_pickle=False) as saved:
        metadata = dict(saved)
    with np.load(root/'metadata.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(metadata['ids'], saved['ids'])
        np.testing.assert_array_equal(metadata['length'], saved['length'])
    for name, checksum in prepared['prepared_files'].items():
        if digest(previous/name) != checksum:
            raise ValueError('Sequence-scan input changed')
    for name in ('bin', 'references', 'fonts'):
        (result_root/name).symlink_to(project/config['assets']/name, target_is_directory=True)
    if subprocess.check_output([str(result_root/'bin/fimo'),'--version'],text=True).strip() != '5.5.9':
        raise ValueError('Expected FIMO 5.5.9')
    definitions = tuple(groups)
    flank_threshold = config.get('flank_threshold', .5)
    rule = core_rule(flank_threshold)
    groups, exclusions = [], []
    for name, low, high in definitions:
        directory = result_root/'groups'/name
        receipt = json.loads((directory/'complete.json').read_text())
        for filename, checksum in receipt['files'].items():
            if digest(directory/filename) != checksum:
                raise ValueError('Completed group changed')
        selection = json.loads((directory/'selection.json').read_text())
        rows, excluded = filter_native(directory/'motifs.h5', name,
            {s:receipt['audit'][s]['patterns'] for s in ('positive', 'negative')},
            core_filter=partial(informative_core, flank_threshold=flank_threshold))
        exclusions.extend(excluded)
        groups.append(dict(name=name, minimum_breadth=low, maximum_breadth=high,
            n=selection['eligible_before_quality'], discovery_elements=selection['elements'],
            quality_excluded=selection['quality_excluded'], rows=rows, rank_by='native_support'))
    audit_data = dict(groups=groups, exclusions=exclusions, colors=COLORS, glyphs=GLYPHS,
        background=prepared['background'], top_n=5, target=TARGETS[target], references=100, steps=64,
        rules=dict(trim=rule, reclustering=False, rank='Distinct original-cluster enhancer support, within group/sign',
            frequency='Sequence-scan prevalence separate from support', site_p=1e-4),
        report_scope=dict(rank_by='native_support', native_discovery='New native-enhancer fits; fixed IG64, 100 shuffled references'))
    if flank_threshold != .5:
        audit_data['flank_note'] = f'Flank cutoff >{flank_threshold:g} bits; strong core >0.5 bits'
        audit_data['report_scope']['native_discovery'] = 'Reused exact native-enhancer fits; only postprocessing changed'
    (result_root/'scans').mkdir()
    write_json(result_root/'cores.json', audit_data)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda g:scan(result_root, previous, g, metadata,
            np.array(prepared['background']['probabilities'])), groups))
    write_json(result_root/'frequency_audit.json', audit_data)
    annotation = result_root/'annotation'; annotation.mkdir()
    (annotation/'references').symlink_to(result_root/'references', target_is_directory=True)
    tomtom(annotation, result_root/'frequency_audit.json', result_root/'bin/tomtom', query_rule=rule)
    combgap(result_root, audit_data)
    write_json(result_root/'report_audit.json', audit_data)
    with (result_root/'motif_metrics.tsv').open('x') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['motif','sign','rank','support_hits','discovery_n','support_fraction','split','exact_degree','scan_hits','scan_n','scan_fraction'])
        for group in groups:
            if not group['rows']:
                continue
            with np.load(result_root/'scans'/group['name']/'minimum_p.npz', allow_pickle=False) as saved:
                best = saved['minimum_p']
            for j, row in enumerate(group['rows']):
                support = row['attribution_support']
                for split in ('train','validation','test'):
                    for degree in range(1,9):
                        mask = (metadata['split']==split) & (metadata['breadth']==degree)
                        hits, n = int((best[j,mask] <= 1e-4).sum()), int(mask.sum())
                        writer.writerow([row['id'],row['sign'],row['rank'],support['hits'],support['n'],support['fraction'],split,degree,
                            hits,n,hits/n if n and row['frequency_quantifiable'] else ''])
    files = ['cores.json','frequency_audit.json','report_audit.json','motif_metrics.tsv','combgap.json','combgap_matches.tsv']
    files += [str(p.relative_to(result_root)) for p in annotation.rglob('*') if p.is_file()]
    files += [str(p.relative_to(result_root)) for p in (result_root/'scans').rglob('*') if p.is_file()]
    write_json(result_root/'report_complete.json', dict(status='complete',target=TARGETS[target],reclustered=False,
        files={name:digest(result_root/name) for name in files}, figure_status='pending local render and visual QA'))
    event('dual_report_complete',target=TARGETS[target],retained=sum(len(g['rows']) for g in groups))


def importance(project, root, config, task):
    """Rescore checksum-pinned sequence hits: no changed PWM or site threshold."""
    from .tomtom_atlas import DATABASES, read_meme
    from .jaspar_importance import union_importance, summarize, EXACT_GROUPS
    verify_sources(project, config)
    target, database_index = divmod(task, 2)
    db = ('jaspar','flyfactorsurvey')[database_index]
    cached = project/config['cached_importance'][db]
    previous = json.loads((cached/'summary.json').read_text())
    if previous['site_p'] != 1e-4:
        raise ValueError('Unexpected cached sequence-scan threshold')
    profiles = list(read_meme(project/config['assets']/'references'/DATABASES[db]['file']).values())
    if [r['id'] for r in profiles] != [r['id'] for r in previous['trajectories']]:
        raise ValueError('Cached reference profile order changed')
    audit_complete = json.loads((root/'audit_complete.json').read_text())
    for filename in ('metadata.npz','native_actual.npy'):
        if digest(root/filename) != audit_complete['files'][filename]:
            raise ValueError('Assembled attribution changed')
    with np.load(root/'metadata.npz',allow_pickle=False) as f:metadata=dict(f)
    metadata['breadth'] = metadata['labels'].sum(1)
    actual = np.load(root/'native_actual.npy',mmap_mode='r',allow_pickle=False)
    output = root/TARGETS[target]/'importance'/db;output.mkdir(parents=True,exist_ok=False)
    lookup={p['id']:i for i,p in enumerate(profiles)}
    scores=np.full((len(profiles),len(metadata['ids'])),np.nan,np.float32)
    covered=np.zeros(scores.shape,np.uint16)
    hit_counts={p['id']:0 for p in profiles}
    for number in range(8):
        sites={}
        path=cached/f'hits_{number}.tsv.gz'
        with gzip.open(path,'rt') as stream:
            rows=csv.DictReader((line for line in stream if not line.startswith('#')),delimiter='\t')
            for row in rows:
                ident=row['motif_id'];index=int(row['sequence_name'][1:])
                if (ident not in lookup or metadata['split'][index]!='train'
                        or float(row['p-value'])>1e-4):
                    raise ValueError('Unexpected cached FIMO hit')
                start,end=int(row['start'])-1,int(row['stop'])
                if not 0<=start<end<=metadata['length'][index]:raise ValueError('Site outside enhancer')
                hit_counts[ident]+=1
                if metadata['quality_pass'][index,target]:
                    sites.setdefault((ident,index),[]).append((start,end))
        for (ident,index),positions in sites.items():
            j=lookup[ident]
            if covered[j,index]:raise ValueError('Reference profile repeated across scan shards')
            scores[j,index],covered[j,index]=union_importance(actual[index,target,:metadata['length'][index]],positions)
        event('dual_importance_cached_shard',target=TARGETS[target],database=db,shard=number)
    if hit_counts != previous['hit_counts']:
        raise ValueError('Cached hit counts do not reproduce previous scan')
    trajectories=summarize(scores,metadata,profiles,groups=EXACT_GROUPS)
    summary=dict(status='complete',target=TARGETS[target],profiles=len(profiles),groups=list(EXACT_GROUPS),
        group_labels=[str(i) for i in range(1,9)],grouping='exact',database_key=db,split='train',
        references=100,steps=64,site_p=1e-4,interval='original enhancer',trajectories=trajectories,
        metric='Mean signed target contribution per matched base, equally averaged over passing motif carriers',
        quality_policy='Target-specific failed enhancers excluded, noncarriers missing rather than zero',
        raw_attribution_units='logit' if target==0 else 'sum of probabilities',
        hit_counts=hit_counts,cached_scan_directory=config['cached_importance'][db],
        attribution_audit_sha256=digest(root/'audit_complete.json'),source_hashes=config['source_hashes'])
    write_json(output/'summary.json',summary)
    np.savez_compressed(output/'importance.npz',motif_ids=np.array(list(lookup)),scores=scores,
                        covered_bases=covered,ids=metadata['ids'],split=metadata['split'],breadth=metadata['breadth'])
    write_json(output/'complete.json',dict(status='complete',files={name:digest(output/name)
        for name in ('summary.json','importance.npz')}))
    event('dual_importance_complete',target=TARGETS[target],database=db)


def compare(project,root,config):
    """Paired per-enhancer actual-map similarity; not independent reference halves."""
    verify_sources(project,config)
    with np.load(root/'metadata.npz',allow_pickle=False) as f:metadata=dict(f)
    actual=np.load(root/'native_actual.npy',mmap_mode='r',allow_pickle=False)
    original=project/config['original']
    receipt=json.loads((original/'attribution_complete.json').read_text())
    if receipt['references']!=50:raise ValueError('Expected previous 50-reference maps')
    n=len(metadata['ids']);cosine=np.full(n,np.nan);mae=np.full(n,np.nan);seen=np.zeros(n,bool)
    for number,(name,checksum) in enumerate(receipt['chunks'].items(),1):
        path=original/'chunks'/name
        if digest(path)!=checksum:raise ValueError('Old chunk changed')
        with np.load(path,allow_pickle=False) as f:
            indices=f['indices'];old=f['actual'];quality=f['quality_pass']
        if seen[indices].any():raise ValueError('Repeated previous enhancer')
        for row,index in enumerate(indices):
            offset,length=int(metadata['offset'][index]),int(metadata['length'][index])
            if quality[row] and metadata['quality_pass'][index,0]:
                a=np.asarray(old[row,offset:offset+length],float);b=np.asarray(actual[index,0,:length],float)
                norm=np.linalg.norm(a)*np.linalg.norm(b)
                if norm>0:cosine[index]=a.dot(b)/norm
                mae[index]=np.abs(a-b).mean()
        seen[indices]=True
        if number%200==0:event('dual_previous50_comparison',chunks=number)
    if not seen.all():raise ValueError('Incomplete paired comparison')
    rows=[]
    for split in ['train','validation','test','all']:
        for degree in ['all',*range(1,9)]:
            mask=np.isfinite(cosine)
            if split!='all':mask&=metadata['split']==split
            if degree!='all':mask&=metadata['labels'].sum(1)==degree
            values=cosine[mask]
            rows.append(dict(split=split,degree=degree,n=len(values),cosine_quantiles=np.quantile(values,[0,.01,.05,.5,.95,1]).tolist(),
                mean_absolute_difference=float(mae[mask].mean()),below_point9=int((values<.9).sum())))
    np.savez_compressed(root/'comparison50.npz',ids=metadata['ids'],cosine=cosine,mean_absolute_difference=mae)
    write_json(root/'comparison50.json',dict(status='complete',region='native enhancer',target='mean_active_logit',rows=rows,
        quantile_order=[0,.01,.05,.5,.95,1],caveat='Same model, nested references: old adaptive IG32/64/128 x50 versus fixed IG64 x100; not independent reproducibility or an isolated reference-count effect',
        old_attribution_sha256=digest(original/'attribution_complete.json'),new_audit_sha256=digest(root/'audit_complete.json')))
    event('dual_previous50_comparison_complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('audit','smoke','discover','report','importance','compare'))
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--task',type=int,default=0)
    args = parser.parse_args(); require_allocation('cpu')
    config = json.loads((args.root/'config.json').read_text())
    if args.stage == 'audit': audit(args.project,args.root,config)
    elif args.stage == 'smoke': smoke(args.root,config)
    elif args.stage == 'compare': compare(args.project,args.root,config)
    elif args.stage == 'importance':
        if not 0<=args.task<4:raise ValueError('Importance task must be 0..3')
        importance(args.project,args.root,config,args.task)
    elif args.stage == 'discover':
        if not 0 <= args.task < 16: raise ValueError('Discovery task must be 0..15')
        discover(args.root,config,args.task)
    else:
        if args.task not in (0,1): raise ValueError('Report task must be 0 or 1')
        report(args.project,args.root,config,args.task)


if __name__ == '__main__':
    main()
