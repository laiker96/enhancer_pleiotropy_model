"""Bounded, matched DeepLIFT/IG motif-recovery pilot on CECAR compute nodes."""
import argparse
from functools import partial
import html
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from .common import digest, event, require_allocation, write_json

NAME = 'classifier_deeplift_motif_recovery_20260922'
METHODS = ('ig', 'deeplift')
META = ('ids', 'labels', 'split', 'chrom', 'summit', 'sequence', 'native_start',
        'native_end', 'native_offset', 'native_length', 'reference_hashes',
        'calibrated_probabilities', 'reference_calibrated_probabilities')


def save(path, **arrays):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def receipt(root, names, **metadata):
    return dict(status='complete', **metadata, files={n: digest(root/n) for n in names})


def verify(root, name):
    value = json.loads((root/name).read_text())
    if value['status'] != 'complete':
        raise ValueError('Incomplete upstream stage: '+str(root/name))
    for filename, expected in value['files'].items():
        if digest(root/filename) != expected:
            raise ValueError('Changed upstream file: '+filename)
    return value


def select(data, quality, per_degree=18, seed=20260922):
    labels = data['labels']
    if (labels.shape != (len(data['ids']), 8) or not np.isin(labels, [0, 1]).all()
            or not labels.any(1).all() or len(np.unique(data['ids'])) != len(labels)):
        raise ValueError('Invalid or duplicate cohort')
    rng = np.random.default_rng(seed)
    chosen, counts = [], []
    for degree in range(1, 9):
        candidates = np.flatnonzero((data['split'] == 'train') & quality &
                                    (labels.sum(1) == degree))
        counts.append(len(candidates))
        if len(candidates) < per_degree:
            raise ValueError(f'Insufficient degree {degree}: {len(candidates)} < {per_degree}')
        chosen.extend(rng.choice(candidates, per_degree, replace=False).tolist())
    return rng.permutation(chosen), counts


def native_inputs(data, hypothetical):
    n, width = len(data['ids']), int(data['native_length'].max())
    if hypothetical.shape != (n, 8, 4, 2048) or not np.isfinite(hypothetical).all():
        raise ValueError('Expected eight finite full-input maps')
    sequence = np.zeros((n, width, 4), np.float32)
    hyp = np.zeros_like(sequence)
    for i, (offset, length) in enumerate(zip(data['native_offset'], data['native_length'])):
        offset, length = int(offset), int(length)
        if not 30 <= length <= width or not 0 <= offset <= 2048-length:
            raise ValueError('Invalid native enhancer interval')
        sequence[i, :length] = np.eye(4, dtype=np.float32)[data['sequence'][i, offset:offset+length]]
        hyp[i, :length] = hypothetical[i, :, :, offset:offset+length].sum(0).T
    return sequence, hyp


def prepare(project, root, config):
    require_allocation('cpu')
    from .calibrated_context import validate_output
    parent = project/config['ig_parent']
    if digest(parent/'MANIFEST.sha256') != config['ig_manifest_sha256']:
        raise ValueError('Wrong IG parent package')
    old = json.loads((parent/'config.json').read_text())
    if digest(root/'calibrators.json') != digest(parent/'calibrators.json'):
        raise ValueError('Calibration changed')
    if digest(project/config['checkpoint']) != config['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    sources, metadata, qualities, origins = {}, [], [], []
    paths = sorted(p for p in (parent/'chunks').glob('chunk_*.npz') if '.state.' not in p.name)
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            if str(z['signature']) != config['ig_manifest_sha256']:
                raise ValueError('Wrong completed IG signature')
            metadata.append({key: z[key] for key in META})
            qualities.append(z['quality_pass'][:, :8].all(1) & z['breadth_quality_pass'])
            origins.extend((path, j) for j in range(len(z['ids'])))
    data = {key: np.concatenate([v[key] for v in metadata]) for key in META}
    quality = np.concatenate(qualities)
    indices, counts = select(data, quality, config['per_degree'], config['selection_seed'])
    selected = {key: value[indices] for key, value in data.items()}
    hyp = np.zeros((len(indices), 8, 4, 2048), np.float32)
    selected_by_file = {}
    for destination, index in enumerate(indices):
        path, row = origins[index]
        selected_by_file.setdefault(path, []).append((destination, row))
    for path, rows in selected_by_file.items():
        with np.load(path, allow_pickle=False) as z:
            chunk = dict(z)
        validate_output(chunk, chunk['indices'], config['ig_manifest_sha256'], old)
        sources[str(path.relative_to(project))] = digest(path)
        for destination, row in rows:
            hyp[destination] = chunk['hypothetical'][row, :8]
    if not np.isfinite(hyp).all() or not (selected['split'] == 'train').all():
        raise ValueError('Bad selected maps or split')
    # Inputs and maps are paired by the identical permuted selection for both methods.
    save(root/'examples.npz', **selected)
    save(root/'ig.npz', hypothetical=hyp, ids=selected['ids'])
    sequence, native = native_inputs(selected, hyp)
    save(root/'ig_native.npz', sequence=sequence, hypothetical=native,
         lengths=selected['native_length'], ids=selected['ids'])
    write_json(root/'selection.json', dict(elements=len(indices), per_degree=config['per_degree'],
        eligible_by_degree=counts, available_completed_elements=len(data['ids']),
        excluded_quality=int((~quality).sum()), sources=sources,
        ids=selected['ids'].tolist(), seed=config['selection_seed'], train_only=True,
        pooled_discovery=True, note='Matched exploratory sample of completed IG, not population prevalence or held-out validation.'))
    write_json(root/'prepared.json', receipt(root, ['examples.npz', 'ig.npz', 'ig_native.npz', 'selection.json'],
        config_sha256=digest(root/'config.json'), elements=len(indices)))
    event('motif_recovery_prepared', elements=len(indices), eligible_by_degree=counts)


def load_data(root):
    verify(root, 'prepared.json')
    with np.load(root/'examples.npz', allow_pickle=False) as z:
        return dict(z)


def gpu_models(project, root, config):
    import torch
    from .method_pilot import models
    from .calibrated_context import require_gpu
    family = require_gpu()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if digest(project/config['checkpoint']) != config['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    return *models(project, root, 'cuda')[:2], family


def attribute(project, root, config, shard):
    import torch
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from classifier_motifs.attribution import one_hot
    from classifier_motifs.calibrated_deeplift import paired_calibrated_deeplift
    from .calibrated_context import make_references
    from .method_pilot import check_real
    if shard not in (0, 1):
        raise ValueError('Exactly two GPU shards')
    data = load_data(root)
    target, adapted, family = gpu_models(project, root, config)
    directory = root/'deeplift_chunks'; directory.mkdir(exist_ok=True)
    stopped, began = [False], time.monotonic()
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def stop(): return stopped[0] or time.monotonic()-began > 21000
    write_json(root/f'runtime_{shard}.json', dict(job=os.environ['SLURM_JOB_ID'], pid=os.getpid(),
        gpu=torch.cuda.get_device_name(), family=family, no_training=True, shard=shard))
    a, b = target.a.double(), target.b.double()
    chunks = {}
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context('spawn')) as pool:
        for batch, first in enumerate(range(0, len(data['ids']), 8)):
            if batch % 2 != shard: continue
            path = directory/f'chunk_{first:03d}.npz'
            index = np.arange(first, min(first+8, len(data['ids'])))
            references, hashes = make_references(data, index, config, pool)
            np.testing.assert_array_equal(hashes, data['reference_hashes'][index])
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], data['ids'][index])
                    if str(z['signature']) != digest(root/'MANIFEST.sha256') or int(z['count']) != 100:
                        raise ValueError('Incompatible completed chunk')
                chunks[path.name] = digest(path); continue
            x = one_hot(data['sequence'][index], 'cuda')
            if not chunks:
                gate = check_real(target, adapted, x[:2], one_hot(references[:2, 0], 'cuda'), stop)
                write_json(root/f'gpu_preflight_{shard}.json', gate)
                event('motif_recovery_gpu_preflight', shard=shard, **gate)
            state_path = path.with_suffix('.state.npz')
            sums = np.zeros((len(index), 8, 4, 2048), np.float64)
            deltas, reference_p = np.zeros((len(index), 100, 8)), np.zeros((len(index), 100, 8))
            count = 0
            if state_path.exists():
                with np.load(state_path, allow_pickle=False) as z:
                    if str(z['signature']) != digest(root/'MANIFEST.sha256'):
                        raise ValueError('Changed restart configuration')
                    np.testing.assert_array_equal(z['ids'], data['ids'][index])
                    if (z['sums'].shape != sums.shape or z['deltas'].shape != deltas.shape
                            or z['reference_p'].shape != reference_p.shape or not 0 <= int(z['count']) <= 100
                            or any(not np.isfinite(z[k]).all() for k in ('sums', 'deltas', 'reference_p'))):
                        raise ValueError('Malformed reference checkpoint')
                    sums, deltas, reference_p, count = z['sums'], z['deltas'], z['reference_p'], int(z['count'])
            def checkpoint():
                save(state_path, sums=sums, deltas=deltas, reference_p=reference_p,
                     count=np.asarray(count), ids=data['ids'][index], signature=np.asarray(digest(root/'MANIFEST.sha256')))
            try:
                for r in range(count, 100):
                    values = paired_calibrated_deeplift(adapted, x.double(),
                        one_hot(references[:, r], 'cuda').double(), a, b, stop)
                    torch.testing.assert_close(values['probabilities'],
                        torch.as_tensor(data['calibrated_probabilities'][index], device='cuda'),
                        atol=2e-5, rtol=1e-5, check_dtype=False)
                    torch.testing.assert_close(values['reference_probabilities'],
                        torch.as_tensor(data['reference_calibrated_probabilities'][index, r], device='cuda'),
                        atol=2e-5, rtol=1e-5, check_dtype=False)
                    if not (values['delta'].abs() <= .002+.05*values['difference'].abs()).all():
                        raise ValueError('Per-reference conservation failure; no silent filtering')
                    sums += values['hypothetical'].cpu().numpy()
                    deltas[:, r] = values['delta'].cpu().numpy()
                    reference_p[:, r] = values['reference_probabilities'].cpu().numpy()
                    count = r+1
                    if count % 10 == 0:
                        checkpoint()
                        event('motif_recovery_references', shard=shard, first=first, references=count)
            except Exception:
                checkpoint()
                raise
            save(path, hypothetical=(sums/100).astype(np.float32), ids=data['ids'][index],
                indices=index, deltas=deltas, reference_probabilities=reference_p,
                count=np.asarray(count), signature=np.asarray(digest(root/'MANIFEST.sha256')))
            chunks[path.name] = digest(path)
            write_json(root/f'progress_{shard}.json', dict(elements=8*len(chunks), chunks=chunks,
                shard=shard, elapsed_seconds=time.monotonic()-began))
            event('motif_recovery_chunk_complete', shard=shard, first=first, elements=len(index))
    write_json(root/f'attribution_{shard}.json', receipt(directory, list(chunks), shard=shard,
        elapsed_seconds=time.monotonic()-began))


def collect_deeplift(root, data):
    hyp = np.zeros((len(data['ids']), 8, 4, 2048), np.float32)
    seen = np.zeros(len(hyp), bool)
    directory = root/'deeplift_chunks'
    for shard in (0, 1):
        r = json.loads((root/f'attribution_{shard}.json').read_text())
        if r['status'] != 'complete': raise ValueError('Incomplete attribution shard')
        for name, checksum in r['files'].items():
            if digest(directory/name) != checksum: raise ValueError('Changed DeepLIFT chunk')
            with np.load(directory/name, allow_pickle=False) as z:
                indices = z['indices']
                if seen[indices].any(): raise ValueError('Duplicate attributed elements')
                np.testing.assert_array_equal(z['ids'], data['ids'][indices])
                hyp[indices] = z['hypothetical']; seen[indices] = True
    if not seen.all(): raise ValueError('Missing DeepLIFT elements')
    return hyp


def occurrence(start, end, reverse, trim_start, trim_end, offset, length):
    left, right = ((end-trim_end, end-trim_start) if reverse else
                   (start+trim_start, start+trim_end))
    if not 0 <= left < right <= length:
        raise ValueError('Motif core crossed native enhancer boundary')
    return int(offset+left), int(offset+right)


def discover(project, root, config, task):
    require_allocation('cpu')
    import h5py
    from .original_intervals import discover_original
    from .dual_motif_pipeline import informative_core, core_rule
    from .simple_report import filter_native, rank_rows
    method = METHODS[task]
    data = load_data(root)
    if method == 'ig':
        with np.load(root/'ig.npz', allow_pickle=False) as z: hyp = z['hypothetical']
    else:
        hyp = collect_deeplift(root, data)
        save(root/'deeplift.npz', hypothetical=hyp, ids=data['ids'])
    sequence, native = native_inputs(data, hyp)
    directory = root/method; directory.mkdir(exist_ok=False)
    save(directory/'inputs.npz', sequence=sequence, hypothetical=native,
         lengths=data['native_length'], ids=data['ids'])
    audit = discover_original(sequence, native, data['native_length'],
        config['discovery_parameters'], directory/'motifs.h5', 20000)
    counts = {s: audit[s]['patterns'] for s in ('positive', 'negative')}
    rows, exclusions = filter_native(directory/'motifs.h5', method, counts,
        core_filter=partial(informative_core, flank_threshold=.2))
    group = dict(name=method, rows=rows, rank_by='native_support', discovery_elements=len(data['ids']))
    rank_rows(group)
    degrees = data['labels'].sum(1)
    with h5py.File(directory/'motifs.h5', 'r') as handle:
        for row in rows:
            seqlets = handle[row['pattern']]['seqlets']
            members = np.unique(seqlets['example_idx'][:])
            row['member_ids'] = data['ids'][members].tolist()
            row['support_by_degree'] = [dict(degree=k, hits=int((degrees[members] == k).sum()),
                n=int((degrees == k).sum())) for k in range(1, 9)]
            row['support_by_group'] = [dict(group=name,
                hits=int(((degrees[members] >= lo) & (degrees[members] <= hi)).sum()),
                n=int(((degrees >= lo) & (degrees <= hi)).sum()))
                for name, lo, hi in (('1', 1, 1), ('2-5', 2, 5), ('6-8', 6, 8))]
            row['occurrences'] = []
            for i, start, end, reverse in zip(seqlets['example_idx'][:], seqlets['start'][:],
                                             seqlets['end'][:], seqlets['is_revcomp'][:]):
                left, right = occurrence(int(start), int(end), bool(reverse), row['quality']['start'],
                    row['quality']['end'], int(data['native_offset'][i]), int(data['native_length'][i]))
                row['occurrences'].append(dict(example=int(i), start=left, end=right, rc=bool(reverse)))
    write_json(directory/'cores.json', dict(groups=[group], exclusions=exclusions,
        rule=core_rule(.2), reclustered=False, audit=audit))
    write_json(directory/'complete.json', receipt(directory,
        ['inputs.npz', 'motifs.h5', 'motifs.audit.json', 'cores.json'], method=method))
    event('motif_recovery_discovery_complete', method=method, raw=counts, retained=len(rows))


def jaccard(first, second):
    a, b = set(first), set(second)
    return len(a & b)/len(a | b) if a | b else None


def compare(project, root, config):
    require_allocation('cpu')
    import subprocess
    from .tomtom_atlas import meme_queries, parse_matches, query_id, read_meme, DATABASES
    from .report_breadth import sequence_logo
    audits = {}
    for method in METHODS:
        verify(root/method, 'complete.json')
        audits[method] = json.loads((root/method/'cores.json').read_text())
    output = root/'comparison'; output.mkdir(exist_ok=False)
    binary = project/config['tomtom']
    if digest(binary) != config['tomtom_sha256']:
        raise ValueError('Changed Tomtom executable')
    common = [str(binary), '-text', '-dist', 'pearson', '-min-overlap', '5',
              '-motif-pseudo', '.1', '-thresh', '1', '-verbosity', '2']
    def match(query, target, name):
        queries, targets = read_meme(query), read_meme(target)
        completed = subprocess.run(common+[str(query), str(target)], capture_output=True, text=True, check=True)
        (output/(name+'.tsv')).write_text(completed.stdout)
        (output/(name+'.stderr.log')).write_text(completed.stderr)
        best, counts = parse_matches(completed.stdout, queries, targets)
        return {key: dict(value, reference_name=targets[value['target_id']]['name']) for key, value in best.items()}
    # Match methods separately by sign; no post-hoc merging or clustering.
    matches, annotations = {}, {}
    for method in METHODS:
        annotations[method] = {}
        rows = audits[method]['groups'][0]['rows']
        if not rows: continue
        query = output/(method+'.meme'); query.write_text(meme_queries(audits[method]))
        for db, meta in DATABASES.items():
            target = project/config['database_root']/meta['file']
            if digest(target) != config['database_sha256'][db]:
                raise ValueError('Changed reference database: '+db)
            annotations[method][db] = match(query, target, method+'_'+db)
        for sign in ('positive', 'negative'):
            group = dict(rows=[r for r in rows if r['sign'] == sign])
            (output/(method+'_'+sign+'.meme')).write_text(meme_queries(dict(groups=[group])))
    rows_by_id = {query_id(r): r for a in audits.values() for r in a['groups'][0]['rows']}
    for method, other in (('ig', 'deeplift'), ('deeplift', 'ig')):
        for sign in ('positive', 'negative'):
            key = method+'_to_'+other+'_'+sign
            selected = [r for r in audits[method]['groups'][0]['rows'] if r['sign'] == sign]
            target_rows = [r for r in audits[other]['groups'][0]['rows'] if r['sign'] == sign]
            if not selected or not target_rows:
                matches[key] = {}; continue
            hits = match(output/(method+'_'+sign+'.meme'), output/(other+'_'+sign+'.meme'), key)
            for q, value in hits.items():
                value['support_jaccard'] = jaccard(rows_by_id[q]['member_ids'], rows_by_id[value['target_id']]['member_ids'])
            matches[key] = hits
    result = dict(method_matches=matches, database_matches=annotations,
        rule='Separate positive/negative Tomtom, both directions. q<=0.05 is descriptive; databases/search sizes differ.',
        support='Distinct native discovery-seqlet member enhancers, not sequence-scan prevalence.',
        limitations='Small training-only balanced pilot; absence is not evidence of biological absence; no automatic adoption.')
    write_json(output/'matches.json', result)
    cards = ['<!doctype html><meta charset="utf-8"><title>DeepLIFT / IG motif recovery</title>',
        '<style>body{font:15px sans-serif;max-width:1100px;margin:30px auto}article{border-top:1px solid #ccc;padding:15px}svg{max-width:700px}table{border-collapse:collapse}td,th{padding:5px 12px;text-align:left}</style>',
        '<h1>Matched DeepLIFT / IG motif recovery</h1><p>144 training enhancers; 100 shared references; summed eight calibrated probabilities. Native enhancer intervals. Positive and negative motifs are ranked separately by distinct seqlet-member enhancers. No extra clustering.</p>']
    for method in METHODS:
        cards.append('<h2>'+method+'</h2>')
        rows = sorted(audits[method]['groups'][0]['rows'], key=lambda r: (r['sign'], r['rank']))
        if not rows: cards.append('<p>No motifs passed the reporting filter; see raw discovery audit.</p>')
        for row in rows:
            key = query_id(row)
            cards.append('<article><h3>'+html.escape(f"{row['sign']} #{row['rank']} — {row['id']}")+'</h3>')
            cards.append(sequence_logo(np.asarray(row['trimmed_pwm']), row['id']))
            cards.append(f"<p>Support: {row['supporting_discovery_enhancers']}/144 enhancers</p><ul>")
            for db, hits in annotations[method].items():
                m = hits[key]
                cards.append('<li>'+html.escape(f"{db}: {m['reference_name']} ({m['target_id']}), q={m['q']:.3g}")+'</li>')
            cards.append('</ul><p>Exact-degree support: '+', '.join(f"{v['degree']}: {v['hits']}/{v['n']}" for v in row['support_by_degree'])+'</p>')
            cards.append('<p>Non-overlapping groups: '+', '.join(f"{v['group']}: {v['hits']}/{v['n']}" for v in row['support_by_group'])+'</p></article>')
    cards.append('<p>Motif matches do not identify bound TFs or establish cooperativity. Perturbation results are saved separately in perturbations.json.</p>')
    (root/'report.html').write_text('\n'.join(cards))
    write_json(root/'comparison_complete.json', receipt(root, ['comparison/matches.json', 'report.html']))
    event('motif_recovery_comparison_complete')


def shuffled_core(codes, start, end, rng):
    source = codes[start:end]
    if len(np.unique(source)) < 2: return None
    for _ in range(100):
        value = rng.permutation(source)
        if not np.array_equal(value, source):
            result = codes.copy(); result[start:end] = value
            return result
    raise ValueError('Failed to draw nonidentity shuffle')


def perturbation_summary(records, effects):
    summaries = []
    for method in METHODS:
        for motif in sorted({r['motif'] for r in records if r['method'] == method}):
            rows = [r for r in records if r['method'] == method and r['motif'] == motif]
            values = {}
            for kind in ('motif', 'control'):
                # Average shuffles within enhancer, then enhancers; not independent replicates.
                per_enhancer = [np.mean([effects[r['mutant']] for r in rows if r['kind'] == kind and r['example'] == i], axis=0)
                    for i in sorted({r['example'] for r in rows if r['kind'] == kind})]
                values[kind] = dict(n=len(per_enhancer), mean_mutant_minus_wt=np.mean(per_enhancer, axis=0).tolist() if per_enhancer else None,
                    mean_breadth_change=float(np.mean(np.sum(per_enhancer, axis=1))) if per_enhancer else None)
            summaries.append(dict(method=method, motif=motif, sign=rows[0]['sign'], effects=values))
    return summaries


def perturb(project, root, config):
    import torch
    from classifier_motifs.attribution import one_hot, seed_for
    data = load_data(root)
    target, _, family = gpu_models(project, root, config)
    mutations, lookup, records, skipped = [], {}, [], []
    for method in METHODS:
        verify(root/method, 'complete.json')
        audit = json.loads((root/method/'cores.json').read_text())
        for row in audit['groups'][0]['rows']:
            if row['rank'] > 5: continue
            seen = set()
            for site in sorted(row['occurrences'], key=lambda s: (s['example'], s['start'])):
                i, start, end = site['example'], site['start'], site['end']
                if i in seen: continue
                seen.add(i)
                rng = np.random.default_rng(seed_for(config['selection_seed'], str(data['ids'][i]), start))
                lo, length = int(data['native_offset'][i]), int(data['native_length'][i])
                # Same-width, non-overlapping within-enhancer random controls.
                candidates = [j for j in range(lo, lo+length-(end-start)+1)
                              if j+end-start <= start or j >= end]
                control = int(rng.choice(candidates)) if candidates else None
                for kind, left in [('motif', start), ('control', control)]:
                    if left is None: skipped.append(dict(motif=row['id'], example=i, reason='no_nonoverlapping_control')); continue
                    for repeat in range(3):
                        key = (i, left, end-start, repeat)
                        if key not in lookup:
                            local_rng = np.random.default_rng(seed_for(config['selection_seed'], str(key), 0))
                            value = shuffled_core(data['sequence'][i], left, left+end-start, local_rng)
                            if value is None:
                                skipped.append(dict(motif=row['id'], example=i, kind=kind, reason='homopolymer')); continue
                            lookup[key] = len(mutations); mutations.append((i, value))
                        records.append(dict(method=method, motif=row['id'], sign=row['sign'], example=i,
                            kind=kind, repeat=repeat, mutant=lookup[key]))
                if len(seen) >= 12: break
    with torch.no_grad():
        wt = np.concatenate([target.endpoints(one_hot(data['sequence'][i:i+16], 'cuda'))['calibrated_probabilities'].cpu().numpy()
                             for i in range(0, len(data['ids']), 16)])
        effects = []
        for i in range(0, len(mutations), 16):
            batch = mutations[i:i+16]
            scores = target.endpoints(one_hot(np.stack([v for _, v in batch]), 'cuda'))['calibrated_probabilities'].cpu().numpy()
            effects.extend(scores-wt[[j for j, _ in batch]])
    summaries = perturbation_summary(records, effects)
    save(root/'perturbations.npz', effects=np.asarray(effects).reshape(-1, 8), wt=wt, ids=data['ids'])
    write_json(root/'perturbations.json', dict(status='complete', mutants=len(mutations), records=records,
        skipped=skipped, summary=summaries, gpu=family,
        caveat='Composition-preserving core shuffles may retain motif features; descriptive pilot, no causal or exact-mutation-attribution claim.'))
    verify(root, 'comparison_complete.json')
    write_json(root/'complete.json', receipt(root, ['prepared.json', 'ig/complete.json', 'deeplift/complete.json',
        'comparison/matches.json', 'report.html', 'perturbations.npz', 'perturbations.json'], no_training=True))
    event('motif_recovery_complete', mutants=len(mutations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'attribute', 'discover', 'compare', 'perturb'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    config = json.loads((args.root/'config.json').read_text())
    if args.stage in ('attribute', 'discover'):
        globals()[args.stage](args.project, args.root, config, args.task)
    else:
        globals()[args.stage](args.project, args.root, config)


if __name__ == '__main__': main()
