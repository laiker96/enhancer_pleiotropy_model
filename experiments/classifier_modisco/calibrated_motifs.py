"""Native-enhancer TF-MoDISco from linear combinations of frozen eight-output IG.

CPU only. Preserve raw positive/negative patterns, no filtering or extra merging.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .common import digest, event, require_allocation, write_json
from .family_breadth import CONTEXTS, FAMILIES, GROUPS, grouping
from .original_intervals import discover_original, load_intervals

NAME = 'classifier_calibrated_motifs_20260927'
SOURCE = 'experiments/classifier_calibrated_full8_20260923'
SOURCE_SHA = '763e0ed8ba1adcbae5c5745c8402417b9471c9be5d6ba2b47a83a3ed65785ee8'
DEGREES = (('degree_1', 1, 1), ('degree_2_5', 2, 5), ('degree_6_8', 6, 8))


def family_weights(scheme):
    result = {}
    for name, contexts in FAMILIES[scheme].items():
        row = np.zeros(8)
        row[[CONTEXTS.index(c) for c in contexts]] = 1/len(contexts)
        result[name] = row
    return result


def targets():
    out = {'observed_active_sum': dict(weights=[1.]*8, observed_mask=True),
           'all_context_sum': dict(weights=[1.]*8, observed_mask=False)}
    for scheme in FAMILIES:
        means = family_weights(scheme)
        out['balanced_'+scheme] = dict(weights=sum(means.values()).tolist(), observed_mask=False)
    for name in ('embryo', 'CNS', 'imaginal_discs', 'ovary', 'adult_brain', 'larval_brain'):
        scheme = 'separate_brains' if name in ('adult_brain', 'larval_brain') else 'four_families'
        means = family_weights(scheme)
        rest = np.mean([w for k, w in means.items() if k != name], axis=0)
        out[name+'_vs_other_families'] = dict(weights=(means[name]-rest).tolist(), observed_mask=False)
    for name, positive, negative in [('adult_vs_larval_brain', 'ab', 'lb'), ('e13_vs_e5', 'e13', 'e5')]:
        row = np.zeros(8); row[CONTEXTS.index(positive)] = 1; row[CONTEXTS.index(negative)] = -1
        out[name] = dict(weights=row.tolist(), observed_mask=False)
    return out


def tasks():
    result = []
    for target in ('observed_active_sum', 'all_context_sum'):
        for name, low, high in DEGREES:
            result.append(dict(target=target, group=name, scheme='raw_degree', low=low, high=high))
    for scheme in FAMILIES:
        for group in GROUPS:
            result.append(dict(target='balanced_'+scheme, group=group, scheme=scheme))
    for target in targets():
        if '_vs_' in target:
            result.append(dict(target=target, group='all_enhancers', scheme='all'))
    return [dict(task=i, **r) for i, r in enumerate(result)]


def coefficients(labels, target, *, spec=None):
    grouping(labels)  # Validate the exact eight-context order/binary-label contract.
    spec = targets()[target] if spec is None else spec
    if spec.get('masked_mean') and (not spec['observed_mask'] or spec.get('observed_mean')
            or spec.get('active_member_scheme') or spec.get('observed_family_scheme')):
        raise ValueError('Masked subset mean requires only the individual observed-label mask')
    if spec.get('mean_active_families') and not spec.get('active_member_scheme'):
        raise ValueError('Family averaging requires active-member family means')
    if spec.get('observed_mean') and (not spec['observed_mask'] or
            spec.get('active_member_scheme') or spec.get('observed_family_scheme')):
        raise ValueError('Observed-context mean requires only the individual observed-label mask')
    weights = np.broadcast_to(spec['weights'], labels.shape).copy()
    if spec.get('active_member_scheme'):
        if spec['observed_mask'] or spec.get('observed_family_scheme'):
            raise ValueError('Active-member normalization cannot be combined with another mask')
        weights = np.zeros(labels.shape, dtype=float)
        for members in FAMILIES[spec['active_member_scheme']].values():
            columns = [CONTEXTS.index(c) for c in members]
            active = labels[:, columns].astype(float)
            count = active.sum(1, keepdims=True)
            weights[:, columns] = np.divide(active, count, out=np.zeros_like(active), where=count > 0)
        if spec.get('mean_active_families'):
            weights /= grouping(labels, scheme=spec['active_member_scheme'])['breadth'][:, None]
        return weights
    if spec.get('observed_family_scheme'):
        scheme = spec['observed_family_scheme']
        active = grouping(labels, scheme=scheme)['active']
        if spec['observed_mask']: raise ValueError('Cannot combine context and family masks')
        for i, members in enumerate(FAMILIES[scheme].values()):
            weights[:, [CONTEXTS.index(c) for c in members]] *= active[:, i, None]
    weights = weights*labels if spec['observed_mask'] else weights
    if spec.get('observed_mean'):
        weights /= labels.sum(1, keepdims=True)
    if spec.get('masked_mean'):
        count = weights.sum(1, keepdims=True)
        if (weights < 0).any() or (count <= 0).any():
            raise ValueError('Every selected enhancer needs an active context in the selected subset')
        weights /= count
    return weights


def selection(metadata, task, seed):
    labels = metadata['labels']; degree = labels.sum(1)
    if task['scheme'] == 'raw_degree':
        eligible = (degree >= task['low']) & (degree <= task['high'])
    elif task['scheme'] == 'observed_contexts':
        contexts = task['contexts']
        if not contexts or len(contexts) != len(set(contexts)):
            raise ValueError('Expected a nonempty unique context subset')
        eligible = labels[:, [CONTEXTS.index(c) for c in contexts]].any(1)
    elif task['scheme'] == 'all':
        eligible = np.ones(len(labels), bool)
    elif task.get('family_degree'):
        breadth = grouping(labels, scheme=task['scheme'])['breadth']
        eligible = (breadth >= task['low']) & (breadth <= task['high'])
    else:
        eligible = grouping(labels, scheme=task['scheme'])['group'] == task['group']
    # Identical eight-map QC policy and ordering for all matched comparisons.
    passing = metadata['quality_pass'].all(1)
    order = np.random.default_rng(seed).permutation(len(labels))
    return order[(eligible & passing)[order]], eligible


def combine_native(hypothetical, actual, sequence, offsets, lengths, weights, width):
    """Fixed per-enhancer coefficients commute with IG/reference averaging."""
    n, channels, _, full_width = hypothetical.shape
    if (channels != 8 or hypothetical.shape[2] != 4 or actual.shape != (n, 8, full_width)
            or sequence.shape != (n, full_width) or weights.shape != (n, 8)
            or not np.isfinite(hypothetical).all() or not np.isfinite(actual).all()
            or not np.isin(sequence, range(4)).all() or not np.isfinite(weights).all()):
        raise ValueError('Invalid aligned maps/sequence/coefficients')
    projected = np.take_along_axis(hypothetical, sequence[:, None, None, :].astype(int), axis=2)[:, :, 0]
    np.testing.assert_allclose(projected, actual, atol=1e-7, rtol=1e-5)
    codes = np.full((n, width), 4, np.uint8)
    hyp = np.zeros((n, 8, width, 4), np.float32)
    observed = np.zeros((n, 8, width), np.float32)
    for i, (offset, length) in enumerate(zip(offsets, lengths)):
        offset, length = int(offset), int(length)
        if not 30 <= length <= width or not 0 <= offset <= full_width-length:
            raise ValueError('Invalid native interval')
        codes[i, :length] = sequence[i, offset:offset+length]
        hyp[i, :, :length] = hypothetical[i, :, :, offset:offset+length].transpose(0, 2, 1)
        observed[i, :, :length] = actual[i, :, offset:offset+length]
    combined = np.einsum('nc,nclb->nlb', weights, hyp, dtype=np.float64).astype(np.float32)
    expected = np.einsum('nc,ncl->nl', weights, observed)
    np.testing.assert_allclose((np.eye(5, 4)[codes]*combined).sum(2), expected, atol=3e-7, rtol=1e-5)
    return codes, hyp, observed


def validate_chunk(z, data, intervals, indices, signature):
    n = len(indices)
    expected_targets = ['calibrated_probability_'+c for c in CONTEXTS]
    if (str(z['signature']) != signature or list(z['targets']) != expected_targets
            or int(z['references']) != 100 or int(z['steps']) != 64):
        raise ValueError('Unexpected attribution protocol')
    np.testing.assert_array_equal(z['indices'], indices)
    for key in ('ids', 'labels', 'split', 'chrom', 'summit', 'sequence'):
        np.testing.assert_array_equal(z[key], data[key][indices])
    for key in ('start', 'end', 'offset', 'length'):
        np.testing.assert_array_equal(z['native_'+key], intervals[key][indices])
    for key, shape in dict(hypothetical=(n,8,4,2048), actual=(n,8,2048),
            actual_reference_halves=(n,2,8,2048), delta=(n,100,8), target_difference=(n,100,8),
            quality_pass=(n,8), calibrated_probabilities=(n,8), reference_calibrated_probabilities=(n,100,8)).items():
        if z[key].shape != shape or not np.isfinite(z[key]).all(): raise ValueError('Invalid '+key)
    probability, reference = z['calibrated_probabilities'], z['reference_calibrated_probabilities']
    if any(((a < 0) | (a > 1)).any() for a in (probability, reference)):
        raise ValueError('Invalid calibrated probabilities')
    np.testing.assert_allclose(z['target_difference'], probability[:, None]-reference, atol=2e-6, rtol=1e-5)
    quality = (np.abs(z['delta']) <= .002+.05*np.abs(z['target_difference'])).all(1)
    np.testing.assert_array_equal(z['quality_pass'], quality)
    np.testing.assert_allclose(z['actual_reference_halves'].mean(1), z['actual'], atol=2e-7, rtol=1e-4)
    np.testing.assert_allclose(z['actual'].sum(2, dtype=np.float64),
        (z['target_difference']+z['delta']).mean(1, dtype=np.float64), atol=3e-6, rtol=1e-4)
    return quality


def verify_ready(root):
    ready = json.loads((root/'prepared.json').read_text())
    if ready['status'] != 'complete' or ready['manifest_sha256'] != digest(root/'MANIFEST.sha256'):
        raise ValueError('Missing/changed preparation contract')
    return ready


def prepare(project, root, config):
    source = project/SOURCE
    if digest(source/'MANIFEST.sha256') != SOURCE_SHA: raise ValueError('Wrong IG package')
    subprocess.run(['sha256sum','--quiet','-c','MANIFEST.sha256'],cwd=source,check=True)
    original = json.loads((source/'config.json').read_text())
    if (original['references'] != 100 or original['steps'] != 64 or original['refinement_steps']
            or original['targets'] != ['calibrated_probability_'+c for c in CONTEXTS]):
        raise ValueError('Wrong IG configuration')
    for name, expected in original['source_hashes'].items():
        if digest(project/name) != expected: raise ValueError('Changed source '+name)
    with np.load(project/original['cohort'], allow_pickle=False) as f: data = dict(f)
    n = len(data['ids'])
    if n != 40338 or len(np.unique(data['ids'])) != n: raise ValueError('Incomplete/duplicate cohort')
    grouping(data['labels'])
    intervals = load_intervals(project/original['original'], data)
    width = int(intervals['length'].max())
    destination = root/'native'; destination.mkdir(exist_ok=False)
    maps = np.lib.format.open_memmap(destination/'hypothetical.npy', mode='w+', dtype='float32', shape=(n,8,width,4))
    actual = np.lib.format.open_memmap(destination/'actual.npy', mode='w+', dtype='float32', shape=(n,8,width))
    codes = np.full((n, width), 4, np.uint8); quality = np.zeros((n,8), bool)
    probabilities = np.zeros((n,8), np.float32); seen = np.zeros(n, bool)
    specs = list(targets()); readouts = np.zeros((n,len(specs)), np.float32)
    residual = np.zeros((n,len(specs)), np.float32)
    hashes, receipts, flags = {}, {}, []
    for shard in range(80):
        path = source/f'shard_{shard}.json'; done = json.loads(path.read_text())
        if done['status'] != 'complete' or done['signature'] != SOURCE_SHA or done['shard'] != shard:
            raise ValueError('Incomplete/mismatched shard')
        receipts[path.name] = digest(path); count = 0; failed = np.zeros(8, int)
        for name, expected in sorted(done['chunks'].items()):
            if Path(name).name != name or name in hashes: raise ValueError('Unsafe/duplicate chunk')
            path = source/'chunks'/name
            if digest(path) != expected: raise ValueError('Chunk checksum failed: '+name)
            begin = int(name.removeprefix('chunk_').removesuffix('.npz'))
            indices = np.arange(begin, min(begin+16, n))
            if begin % 16 or begin//16 % 80 != shard or seen[indices].any(): raise ValueError('Invalid chunk ownership')
            keys = ('signature','indices','targets','references','steps','ids','labels','split','chrom','summit',
                    'sequence','native_start','native_end','native_offset','native_length','hypothetical','actual',
                    'actual_reference_halves','delta','target_difference','quality_pass','calibrated_probabilities',
                    'reference_calibrated_probabilities')
            with np.load(path, allow_pickle=False) as f: z = {k:f[k] for k in keys}
            q = validate_chunk(z, data, intervals, indices, SOURCE_SHA)
            c, h, a = combine_native(z['hypothetical'], z['actual'], z['sequence'],
                z['native_offset'], z['native_length'], np.ones((len(indices),8)), width)
            maps[indices] = h; actual[indices] = a; codes[indices] = c; quality[indices] = q
            probabilities[indices] = z['calibrated_probabilities']
            for t, target in enumerate(specs):
                w = coefficients(z['labels'], target)
                readouts[indices,t] = (w*z['calibrated_probabilities']).sum(1)
                residual[indices,t] = np.abs(np.einsum('nc,nrc->nr', w, z['delta'])).max(1)
            for i in np.flatnonzero(~q.all(1)):
                flags.append(dict(index=int(indices[i]), id=str(z['ids'][i]),
                    contexts=[CONTEXTS[t] for t in np.flatnonzero(~q[i])]))
            seen[indices] = True; hashes[name] = expected; count += len(indices); failed += (~q).sum(0)
        if count != done['elements'] or failed.tolist() != done['quality_failures']:
            raise ValueError('Receipt/tensor counts disagree')
        event('calibrated_motifs_audit', shard=shard, verified=int(seen.sum()))
    if not seen.all() or set(hashes) != {f'chunk_{i:06d}.npz' for i in range(0,n,16)}:
        raise ValueError('Missing enhancer attribution coverage')
    maps.flush(); actual.flush(); del maps, actual
    np.savez_compressed(destination/'metadata.npz', **{k:data[k] for k in ('ids','labels','split','chrom','summit')},
        **{k:v for k,v in intervals.items() if k!='ids'}, sequence=codes, quality_pass=quality, calibrated_probabilities=probabilities,
        target_names=np.asarray(specs), target_scores=readouts, target_max_reference_residual=residual)
    metadata = dict(labels=data['labels'], quality_pass=quality)
    counts = []
    for task in tasks():
        selected, eligible = selection(metadata, task, config['discovery_parameters']['seed'])
        counts.append(dict(**task, eligible=int(eligible.sum()), selected=len(selected),
            quality_excluded=int(eligible.sum()-len(selected)),
            splits={s:int((data['split'][selected] == s).sum()) for s in ('train','validation','test')}))
    write_json(root/'prepared.json', dict(status='complete', manifest_sha256=digest(root/'MANIFEST.sha256'),
        source_manifest_sha256=SOURCE_SHA, elements=n, width=width, source_chunks=hashes, source_receipts=receipts,
        files={str(p.relative_to(root)):digest(p) for p in destination.iterdir()}, counts=counts,
        quality_failures=(~quality).sum(0).tolist(), excluded_enhancers=flags,
        exclusion='Common cohort passing all eight original per-reference completeness checks; nothing deleted.',
        references=100, steps=64, calibrated_probability_population='enhancers_only',
        job=os.environ['SLURM_JOB_ID']))
    event('calibrated_motifs_prepared', elements=n, excluded=len(flags), fits=len(counts))


def synthetic(root, config):
    rng = np.random.default_rng(471); lengths = rng.integers(100,161,512)
    codes = rng.integers(0,4,(512,160)); hyp = rng.normal(0,.01,(512,160,4)).astype(np.float32)
    word = np.asarray([0,1,2,0,3,1,1,2,3,0,2,2])
    for i, length in enumerate(lengths):
        start = int(rng.integers(25,length-40)); codes[i,start:start+len(word)] = word
        hyp[i,np.arange(start,start+len(word)),word] += 1 if i<256 else -1
        hyp[i,length:] = 1e6  # Adversarial padding must not become a seqlet or affect thresholds.
    directory = root/'synthetic'; directory.mkdir(exist_ok=False)
    result = discover_original(np.eye(4,dtype=np.float32)[codes], hyp, lengths,
        config['discovery_parameters'], directory/'motifs.h5', 1000)
    if min(result[s]['patterns'] for s in ('positive','negative')) < 1:
        raise ValueError('Synthetic discovery must recover both signs')
    write_json(root/'synthetic_passed.json', dict(status='passed', result=result,
        manifest_sha256=digest(root/'MANIFEST.sha256'), h5_sha256=digest(directory/'motifs.h5')))


def catalogue(path, indices, metadata, actual):
    """Raw-pattern support/context signatures, not sequence-scan occurrence."""
    import h5py
    rows = []
    with h5py.File(path,'r') as h5:
        for prefix in ('pos_patterns','neg_patterns'):
            for key, pattern in h5.get(prefix,{}).items():
                seqlets = pattern['seqlets']; members = {}
                for i,start,end in zip(seqlets['example_idx'][:],seqlets['start'][:],seqlets['end'][:]):
                    i,start,end = int(i),int(start),int(end)
                    if not 0<=i<len(indices) or not 0<=start<end<=metadata['length'][indices[i]]:
                        raise ValueError('Raw seqlet outside native enhancer')
                    members.setdefault(i, np.zeros(int(metadata['length'][indices[i]]),bool))[start:end] = True
                scores = np.asarray([actual[indices[i],:,:len(mask)][:,mask].mean(1)
                                     for i,mask in members.items()])
                ppm = pattern['sequence'][:]
                rows.append(dict(pattern=prefix+'/'+key, sign='positive' if prefix=='pos_patterns' else 'negative',
                    width=len(ppm), consensus=''.join('ACGT'[i] for i in ppm.argmax(1)),
                    seqlets=len(seqlets['example_idx']), supporting_enhancers=len(members),
                    discovery_enhancers=len(indices), support_fraction=len(members)/len(indices),
                    mean_context_contribution_per_bp=scores.mean(0).tolist() if len(scores) else None,
                    support_interpretation='Fraction contributing at least one discovery seqlet, not a sequence-scan frequency.'))
    return rows


def discover(root, config, task_number, *, task=None, spec=None):
    ready = verify_ready(root)
    gate = json.loads((root/'synthetic_passed.json').read_text())
    if gate['status'] != 'passed' or gate['manifest_sha256'] != digest(root/'MANIFEST.sha256'):
        raise ValueError('Missing synthetic boundary gate')
    for name, expected in ready['files'].items():
        if digest(root/name) != expected: raise ValueError('Changed native input: '+name)
    with np.load(root/'native/metadata.npz', allow_pickle=False) as f: metadata = dict(f)
    task = tasks()[task_number] if task is None else task
    spec = targets()[task['target']] if spec is None else spec
    indices, eligible = selection(metadata, task, config['discovery_parameters']['seed'])
    if not len(indices): raise ValueError('Empty discovery group')
    directory = root/'fits'/f'{task_number:02d}_{task["target"]}__{task["group"]}'
    directory.mkdir(parents=True, exist_ok=False)
    lengths = metadata['length'][indices]; width = int(lengths.max())
    hyp = np.zeros((len(indices),width,4),np.float32)
    source = np.load(root/'native/hypothetical.npy',mmap_mode='r',allow_pickle=False)
    actual = np.load(root/'native/actual.npy',mmap_mode='r',allow_pickle=False)
    weights = coefficients(metadata['labels'][indices],task['target'],spec=spec)
    for begin in range(0,len(indices),64):
        end=min(begin+64,len(indices)); ix=indices[begin:end]
        hyp[begin:end] = np.einsum('nc,nclb->nlb',weights[begin:end],source[ix,:,:width],dtype=np.float64)
    sequence = np.eye(5,4,dtype=np.float32)[metadata['sequence'][indices,:width]]
    np.savez_compressed(directory/'examples.npz', indices=indices, ids=metadata['ids'][indices],
        lengths=lengths, split=metadata['split'][indices], labels=metadata['labels'][indices], weights=weights,
        chrom=metadata['chrom'][indices], native_start=metadata['start'][indices],
        native_end=metadata['end'][indices], native_offset=metadata['offset'][indices])
    np.savez_compressed(directory/'discovery_inputs.npz', sequence=sequence, hypothetical=hyp, lengths=lengths)
    write_json(directory/'selection.json', dict(**task, spec=spec, elements=len(indices),
        eligible=int(eligible.sum()), quality_excluded=int(eligible.sum()-len(indices)),
        all_splits=True, enhancer_downsampling=False, seed=config['discovery_parameters']['seed'],
        interval='original_catalog_bounds', parameters=config['discovery_parameters'],
        attribution_sha256=digest(root/'prepared.json'),
        interpretation='Descriptive full-cohort discovery; not held-out validation. Labels are fixed on the IG path.'))
    event('calibrated_motifs_discovery_start', **task, elements=len(indices))
    result = discover_original(sequence,hyp,lengths,config['discovery_parameters'],directory/'motifs.h5',
        config['discovery_parameters']['max_seqlets_per_metacluster'])
    rows = catalogue(directory/'motifs.h5',indices,metadata,actual)
    write_json(directory/'raw_catalogue.json',dict(patterns=rows,contexts=list(CONTEXTS),
        filtered=False,reclustered=False,task=task))
    write_json(directory/'complete.json',dict(status='complete',task=task,audit=result,
        files={p.name:digest(p) for p in directory.iterdir() if p.is_file()},
        modisco_version=importlib.metadata.version('modisco'),prepared_sha256=digest(root/'prepared.json')))
    event('calibrated_motifs_discovery_complete',**task,patterns=len(rows))


def finalize(root, *, task_list=None):
    verify_ready(root); fits=[]
    for task in tasks() if task_list is None else task_list:
        directory=root/'fits'/f'{task["task"]:02d}_{task["target"]}__{task["group"]}'
        done=json.loads((directory/'complete.json').read_text())
        if done['status']!='complete' or done['task']!=task: raise ValueError('Incomplete discovery')
        for name,expected in done['files'].items():
            if digest(directory/name)!=expected: raise ValueError('Changed fit file')
        fits.append(dict(**task, directory=str(directory.relative_to(root)),
            positive=done['audit']['positive']['patterns'],negative=done['audit']['negative']['patterns'],
            complete_sha256=digest(directory/'complete.json')))
    write_json(root/'complete.json',dict(status='complete',fits=fits,prepared_sha256=digest(root/'prepared.json'),
        filtered=False,reclustered=False,tomtom_run=False))
    event('calibrated_motifs_all_complete',fits=len(fits))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','discover','finalize'))
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--task',type=int,default=0);args=parser.parse_args()
    require_allocation('cpu')
    config=json.loads((args.root/'config.json').read_text())
    if config['targets']!=targets() or config['tasks']!=tasks(): raise ValueError('Changed analysis design')
    if importlib.metadata.version('modisco')!='2.5.2': raise ValueError('Use existing pinned TF-MoDISco')
    if args.stage=='prepare':
        synthetic(args.root,config);prepare(args.project,args.root,config)
    elif args.stage=='discover':
        if not 0<=args.task<len(tasks()):raise ValueError('Invalid task')
        discover(args.root,config,args.task)
    else:finalize(args.root)
