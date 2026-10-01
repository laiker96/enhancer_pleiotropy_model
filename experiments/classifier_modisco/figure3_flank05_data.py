"""CPU-only 0.5-bit motif reannotation and all-JASPAR current-map importance."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import os
from pathlib import Path

import numpy as np

from .common import digest, event, require_allocation, write_json
from .cumulative_motifs import directory
from .dual_motif_pipeline import informative_core
from .family_motifs import core_profiles
from .jaspar_importance import scan_shard
from .simple_report import filter_native
from .tomtom_atlas import read_meme, run as tomtom_run

NAME = 'figure3_context_flank05_20260929'
SOURCE = 'experiments/classifier_cumulative_mean_motifs_20260929'
ASSETS = 'experiments/classifier_modisco_support50ref_20260918'


def active_mean(actual, labels):
    labels = np.asarray(labels)
    if not np.isin(labels, (0, 1)).all() or (labels.sum(-1) == 0).any():
        raise ValueError('Nonempty observed binary labels required')
    return np.einsum('nc,ncl->nl', labels / labels.sum(1, keepdims=True), actual)


def summarize(scores, breadth, eligible, profiles, cumulative=False):
    rows = []
    for j, motif in enumerate(profiles):
        means, counts, totals, sd = [], [], [], []
        for k in range(1, 9):
            mask = eligible & ((breadth >= k) if cumulative and k > 1 else (breadth == k))
            values = scores[j, mask]
            values = values[np.isfinite(values)]
            means.append(float(values.mean(dtype=np.float64)) if len(values) else None)
            counts.append(len(values)); totals.append(int(mask.sum()))
            sd.append(float(values.std(ddof=1, dtype=np.float64)) if len(values) > 1 else None)
        rows.append(dict(id=motif['id'], name=motif['name'], means=means,
                         motif_containing_enhancers=counts, group_enhancers=totals,
                         standard_deviation=sd))
    return rows


def update_examples(source, groups):
    result = []
    for original in source['examples']:
        if original['scheme'] != 'context':
            continue
        e = copy.deepcopy(original)
        row = next(r for r in groups[e['group_index']]['rows'] if r['id'] == e['motif_id'])
        if row['rank'] > 3 or row['sign'] != 'positive':
            raise ValueError('Example is no longer a top-three positive motif')
        a, b = row['quality']['start'], row['quality']['end']
        left, right = ((e['untrimmed_end']-b, e['untrimmed_end']-a) if e['reverse'] else
                       (e['untrimmed_start']+a, e['untrimmed_start']+b))
        values = active_mean(np.asarray(e['actual'])[None], np.asarray(e['labels'])[None])[0]
        outside = np.ones(len(values), bool); outside[max(0, left-2):right+2] = False
        strength = float(values[left:right].mean())
        contrast = strength / max(float(np.abs(values[outside]).mean()), 1e-6)
        if strength <= 0 or contrast < 2:
            raise ValueError('Example fails revised core contrast gate')
        match = row['match']; name = match['reference']['name']
        e.update(start=left, end=right, rank=row['rank'], motif=name, match_q=match['q'],
                 core_strength=strength, contrast=contrast,
                 sites=[dict(start=left, end=right, reverse=e['reverse'], name=name)])
        np.testing.assert_allclose(values, e['mean_active_context_ig'], atol=1e-14, rtol=0)
        result.append(e)
    if len(result) != 8:
        raise ValueError('Expected eight native examples')
    return dict(examples=result, policy='Same frozen examples, updated trim bounds and JASPAR matches; '
                'top-three rank and >=2x positive-core contrast rechecked. No mutation selection.')


def run(project, root):
    require_allocation('cpu')
    config = json.loads((root/'config.json').read_text())
    for name, expected in config['source_hashes'].items():
        if digest(project/name) != expected:
            raise ValueError('Changed source '+name)
    source = project/SOURCE
    prepared = json.loads((project/'experiments/classifier_calibrated_motifs_20260927/prepared.json').read_text())
    for name in ('actual.npy', 'metadata.npz'):
        if digest(source/'native'/name) != prepared['files']['native/'+name]:
            raise ValueError('Changed native maps '+name)
    if prepared['references'] != 100 or prepared['steps'] != 64:
        raise ValueError('Expected IG64/100-reference calibrated probabilities')
    with np.load(source/'native/metadata.npz', allow_pickle=False) as z:
        meta = dict(z)
    actual = np.load(source/'native/actual.npy', mmap_mode='r', allow_pickle=False)
    eligible = meta['quality_pass'].all(1)
    if len(meta['ids']) != 40338 or eligible.sum() != 40309 or len(np.unique(meta['ids'])) != 40338:
        raise ValueError('Unexpected cohort')
    out = root/'output'; out.mkdir(exist_ok=False)
    audit = json.loads((source/'output/audit.json').read_text())
    groups, changes, arrays, fit_hashes = [], [], {}, {}
    for old in audit['groups']:
        fit = directory(source, old['task'])
        done = json.loads((fit/'complete.json').read_text())
        for file in ('motifs.h5', 'examples.npz'):
            if digest(fit/file) != done['files'][file]:
                raise ValueError('Changed discovery '+file)
        fit_hashes[str(fit.relative_to(project))] = digest(fit/'complete.json')
        counts = {sign:done['audit'][sign]['patterns'] for sign in ('positive', 'negative')}
        rows, exclusions = filter_native(fit/'motifs.h5', fit.name, counts,
                                         core_filter=partial(informative_core, flank_threshold=.5))
        # Filter all raw patterns first: do not restrict to previously retained PWMs.
        rows = sorted((r for r in rows if r['sign'] == 'positive'),
                      key=lambda r:(-r['supporting_discovery_enhancers'], r['id']))
        for rank, row in enumerate(rows, 1):
            row.update(rank=rank, group_n=old['elements'],
                       support_fraction=row['supporting_discovery_enhancers']/old['elements'],
                       consensus=''.join('ACGT'[i] for i in np.asarray(row['trimmed_pwm']).argmax(1)))
        with np.load(fit/'examples.npz', allow_pickle=False) as z:
            indices = z['indices']; np.testing.assert_array_equal(z['ids'], meta['ids'][indices])
        profiles = core_profiles(fit/'motifs.h5', rows, indices, meta, actual)
        k = old['task']['task']
        for j, (row, profile) in enumerate(zip(rows, profiles)):
            labels = meta['labels'][profile['indices']]
            partition = profile['values'] * labels / labels.sum(1, keepdims=True)
            row['profile'] = partition.mean(0).tolist()
            row['mean_ig'] = float(partition.sum(1).mean())
            np.testing.assert_allclose(sum(row['profile']), row['mean_ig'], atol=1e-14)
            prefix = f'g{k}_p{j}'; row['array_prefix'] = prefix
            for key in ('indices', 'values', 'covered_bp'):
                arrays[prefix+'_'+key] = profile[key]
            arrays[prefix+'_labels'] = labels
            arrays[prefix+'_context_partition'] = partition
        old_ids = {r['id']:r for r in old['rows'] if r['sign'] == 'positive'}
        new_ids = {r['id']:r for r in rows}
        changes.append(dict(task=k, retained=len(rows), lost=sorted(old_ids.keys()-new_ids.keys()),
                            gained=sorted(new_ids.keys()-old_ids.keys()), exclusions=exclusions,
                            changed_bounds=[i for i in old_ids.keys() & new_ids.keys()
                                if old_ids[i]['quality'] != new_ids[i]['quality']]))
        groups.append(dict(task=old['task'], elements=old['elements'], rows=rows))
        event('flank05_profiles', task=k, positive_motifs=len(rows))
    data = dict(groups=groups, changes=changes, fit_hashes=fit_hashes,
                rule='4/5 positions >0.5 bits; terminal cutoff >0.5 bits; width 5-30; '
                     'mean >=0.5 and total >=5 bits; concordant CWM sign; no reclustering.',
                source_hashes=config['source_hashes'], native_hashes={
                    name:prepared['files']['native/'+name] for name in ('actual.npy','metadata.npz')})
    write_json(out/'audit.json', data)
    np.savez_compressed(out/'motif_profiles.npz', **arrays)
    annotation = out/'annotation'; annotation.mkdir()
    tomtom_run(annotation, out/'audit.json', project/ASSETS/'bin/tomtom', data['rule'],
               references=project/ASSETS/'references', database_keys=['jaspar'])
    matches = json.loads((annotation/'jaspar/matches.json').read_text())['best']
    for group in groups:
        for row in group['rows']:
            row['match'] = matches[row['id'].replace('/', '__')]
    write_json(out/'analysis.json', dict(context=groups, **{k:v for k,v in data.items() if k != 'groups'}))
    examples = update_examples(json.loads((root/'native_examples.json').read_text()), groups)
    write_json(out/'native_examples.json', examples)
    event('flank05_annotation_complete', examples=len(examples['examples']))

    # Reuse the reference scan definition, but scan all QC enhancers, not just train.
    importance = out/'importance'; importance.mkdir()
    fasta = importance/'enhancers.fa'
    with fasta.open('x') as handle:
        for i in np.flatnonzero(eligible):
            seq = meta['sequence'][i, :int(meta['length'][i])]
            if not np.isin(seq, (0, 1, 2, 3)).all():
                raise ValueError('Ambiguous base inside native enhancer')
            handle.write(f'>e{i}\n'+''.join(np.array(list('ACGT'))[seq])+'\n')
    mean_maps = np.empty((len(meta['ids']), actual.shape[-1]), dtype=np.float32)
    for start in range(0, len(meta['ids']), 512):
        mean_maps[start:start+512] = active_mean(actual[start:start+512], meta['labels'][start:start+512])
    database = project/ASSETS/'references/jaspar2026_insects.meme'
    profiles = list(read_meme(database).values()); ids = [p['id'] for p in profiles]
    if len(profiles) != 296:
        raise ValueError('Expected 296 JASPAR insect profiles')
    # scan_shard checks for "train": use a scan eligibility field, preserve true splits in outputs.
    scan_meta = dict(meta, split=np.where(eligible, 'train', 'excluded'))
    workers = min(8, int(os.environ['SLURM_CPUS_PER_TASK']))
    shards = [ids[j::workers] for j in range(workers)]
    def scan(j):
        return scan_shard(j, shards[j], project/ASSETS/'bin/fimo', database,
                         root/'training_background.txt', fasta, scan_meta, mean_maps, importance)
    scores = np.full((296, len(meta['ids'])), np.nan, dtype=np.float32)
    covered = np.zeros(scores.shape, dtype=np.uint16)
    commands = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for names, values, bases, counts, command in pool.map(scan, range(workers)):
            for j, name in enumerate(names):
                scores[ids.index(name)] = values[j]; covered[ids.index(name)] = bases[j]
            commands.append(command)
    np.testing.assert_array_equal(np.isfinite(scores), covered > 0)
    if np.isfinite(scores[:, ~eligible]).any():
        raise ValueError('QC-excluded enhancer entered importance')
    breadth = meta['labels'].sum(1)
    summary = dict(profiles=296, target='Mean observed-active calibrated probabilities',
        references=100, steps=64, population='All 40309 common-QC enhancers, all splits exploratory',
        definition='Signed IG per union of FIMO hit bases, equally averaged across motif-containing '
                   'enhancers. Noncarriers missing, not zero. Untrimmed reference PWMs; FIMO p<=1e-4.',
        exact=summarize(scores, breadth, eligible, profiles),
        cumulative=summarize(scores, breadth, eligible, profiles, cumulative=True), commands=commands,
        database_sha256=digest(database), background_sha256=digest(root/'training_background.txt'))
    write_json(importance/'summary.json', summary)
    np.savez_compressed(importance/'scores.npz', scores=scores, covered_bases=covered,
                        ids=meta['ids'], motif_ids=np.asarray(ids), breadth=breadth,
                        split=meta['split'], eligible=eligible)
    write_json(out/'complete.json', dict(status='complete',
        summary=dict(groups=8, positive_motifs=sum(len(g['rows']) for g in groups), profiles=296,
                     new_attributions=0, enhancers=int(eligible.sum())),
        manifest_sha256=digest(root/'MANIFEST.sha256'),
        files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}))
    event('flank05_data_complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); run(args.project, args.root)
