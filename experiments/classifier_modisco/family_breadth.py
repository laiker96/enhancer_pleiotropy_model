"""Regroup frozen motif scans; no discovery, inference, or target redefinition."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import digest, event, write_json

CONTEXTS = ('ab', 'e13', 'e5', 'ead', 'hid', 'lb', 'o', 'wid')
TARGETS = ('mean_active_logit', 'soft_breadth')
FAMILIES = {
    'four_families': {'embryo': ('e5', 'e13'), 'CNS': ('ab', 'lb'),
                      'imaginal_discs': ('ead', 'hid', 'wid'), 'ovary': ('o',)},
    'separate_brains': {'embryo': ('e5', 'e13'), 'adult_brain': ('ab',),
                        'larval_brain': ('lb',),
                        'imaginal_discs': ('ead', 'hid', 'wid'), 'ovary': ('o',)},
}
GROUPS = ('context_specific', 'family_restricted_multicontext',
          'two_families', 'three_or_more_families')
SPLITS = ('train', 'validation', 'test')
THRESHOLDS = (.7, .8, .9)


def grouping(labels, contexts=CONTEXTS, scheme='four_families'):
    labels = np.asarray(labels)
    if (tuple(contexts) != CONTEXTS or labels.ndim != 2 or labels.shape[1] != 8
            or not np.isin(labels, (0, 1)).all() or not labels.any(1).all()):
        raise ValueError('Expected ordered, nonempty eight-context binary labels')
    families = FAMILIES[scheme]
    members = [c for values in families.values() for c in values]
    if sorted(members) != sorted(contexts):
        raise ValueError('Families must partition contexts')
    active = np.column_stack([labels[:, [contexts.index(c) for c in values]].any(1)
                              for values in families.values()])
    raw, breadth = labels.sum(1), active.sum(1)
    masks = np.column_stack((raw == 1, (raw > 1) & (breadth == 1),
                             breadth == 2, breadth >= 3))
    if not np.all(masks.sum(1) == 1) or np.any(breadth > raw):
        raise ValueError('Non-partitioning family groups')
    return dict(raw=raw, breadth=breadth, active=active,
                group=np.asarray(GROUPS)[masks.argmax(1)],
                pattern=np.asarray(['+'.join(name for name, yes in zip(families, row) if yes)
                                    for row in active]))


def load_metadata(path):
    with np.load(path, allow_pickle=False) as saved:
        data = dict(saved)
    n = len(data['ids'])
    if (len(set(data['ids'])) != n or data['labels'].shape != (n, 8)
            or not np.isin(data['split'], SPLITS).all()
            or data['quality_pass'].shape != (n, 2)
            or not np.isin(data['quality_pass'], (False, True)).all()):
        raise ValueError('Invalid IDs, labels, split, or attribution QC')
    codes, lengths = data['sequence'], data['length']
    valid = np.arange(codes.shape[1])[None, :] < lengths[:, None]
    if (np.any(lengths <= 0) or np.any(lengths > codes.shape[1])
            or not np.isin(codes[valid], range(4)).all() or np.any(codes[~valid] != 4)
            or not np.array_equal(data['end'] - data['start'], lengths)):
        raise ValueError('Invalid native enhancer sequence/bounds')
    data['gc_fraction'] = (((codes == 1) | (codes == 2)) & valid).sum(1) / lengths
    return data


def read_scan(path, metadata):
    with np.load(path, allow_pickle=False) as saved:
        scan = dict(saved)
    np.testing.assert_array_equal(scan['ids'], metadata['ids'])
    np.testing.assert_allclose(scan['thresholds'], THRESHOLDS, rtol=0, atol=1e-12)
    n = len(metadata['ids'])
    best, null, importance = (scan[k] for k in
                              ('best_fraction', 'null_best_fraction', 'mean_importance'))
    if best.shape != (n,) or null.shape != (10, n) or importance.shape != (n,):
        raise ValueError('Unaligned scan or wrong number of null replicates')
    if (np.isnan(best).any() or np.isnan(null).any() or np.isposinf(best).any()
            or np.isposinf(null).any() or np.isinf(importance).any()
            or np.any(best > 1.00001) or np.any(null > 1.00001)):
        raise ValueError('Invalid motif scores')
    return scan


def summarize(scan, selected, quality, threshold):
    """Noncarriers/QC failures do not become zero-importance observations."""
    n = int(selected.sum())
    carriers = selected & (scan['best_fraction'] >= threshold)
    null_counts = (scan['null_best_fraction'][:, selected] >= threshold).sum(1)
    means = scan['mean_importance']
    eligible = selected & quality & np.isfinite(means)
    hits = int(carriers.sum())
    expected = float(null_counts.mean())
    return dict(n=n, sequence_carriers=hits, fraction=hits/n if n else None,
                null_mean_carriers=expected,
                null_fraction=expected/n if n else None,
                excess_fraction=(hits-expected)/n if n else None,
                observed_over_null=hits/expected if expected else None,
                attribution_carriers_at_80=int(eligible.sum()),
                mean_signed_IG_per_bp_at_80=float(means[eligible].mean()) if eligible.any() else None,
                median_signed_IG_per_bp_at_80=float(np.median(means[eligible])) if eligible.any() else None,
                quality_excluded=int((selected & ~quality).sum()),
                finemo_fraction=None, finemo_status='not_in_this_sequence_cache')


def write_table(path, rows):
    if not rows:
        raise ValueError('Cannot infer columns for an empty table')
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def selections(data, groups):
    for scheme, group in groups.items():
        for split in SPLITS:
            member = data['split'] == split
            for name in GROUPS:
                yield (scheme, split, 'group', name, member & (group['group'] == name))
            for degree in range(1, len(FAMILIES[scheme])+1):
                yield (scheme, split, 'exact_family_breadth', str(degree),
                       member & (group['breadth'] == degree))
            # Same number of positive context labels, different family diversity.
            for degree in (1, 2, 3):
                yield (scheme, split, 'raw_context_breadth_3', str(degree),
                       member & (group['raw'] == 3) & (group['breadth'] == degree))
        # Family identities remain available, rather than being pooled irreversibly.
        for pattern in sorted(set(group['pattern'])):
            yield (scheme, 'validation', 'family_identity', pattern,
                   (data['split'] == 'validation') & (group['pattern'] == pattern))


def run(parent, scans, context_provenance, output):
    if output.exists():
        raise FileExistsError('Preserve previous output: '+str(output))
    contexts = tuple(json.loads(context_provenance.read_text())['contexts'])
    data = load_metadata(parent/'metadata.npz')
    groups = {scheme: grouping(data['labels'], contexts, scheme) for scheme in FAMILIES}
    inputs = {str(path): digest(path) for path in
              (parent/'metadata.npz', context_provenance, scans/'config.json')}
    config = json.loads((scans/'config.json').read_text())
    # The cached scans must refer to precisely the frozen PWMs/metadata in use.
    for name, expected in config['source_hashes'].items():
        path = parent/name
        if digest(path) != expected:
            raise ValueError('Changed motif source: '+name)
        inputs[str(path)] = expected
    setup = json.loads((parent/'setup_complete.json').read_text())
    if (setup['status'] != 'complete'
            or setup['files']['metadata.npz'] != inputs[str(parent/'metadata.npz')]):
        raise ValueError('Metadata differs from the frozen source used for scans')
    members, counts, transitions, correlations = [], [], [], []
    for i, identifier in enumerate(data['ids']):
        row = dict(id=str(identifier), split=str(data['split'][i]),
                   chrom=str(data['chrom'][i]), start=int(data['start'][i]), end=int(data['end'][i]),
                   native_length_bp=int(data['length'][i]), gc_fraction=float(data['gc_fraction'][i]),
                   context_breadth=int(groups['four_families']['raw'][i]),
                   active_contexts='+'.join(c for c, yes in zip(contexts, data['labels'][i]) if yes),
                   mean_logit_quality_pass=bool(data['quality_pass'][i, 0]),
                   soft_breadth_quality_pass=bool(data['quality_pass'][i, 1]))
        for scheme, group in groups.items():
            row.update({scheme+'_breadth': int(group['breadth'][i]),
                        scheme+'_group': str(group['group'][i]),
                        scheme+'_active': str(group['pattern'][i])})
        members.append(row)
    for scheme, group in groups.items():
        for split in ('all', *SPLITS):
            mask = np.ones(len(members), bool) if split == 'all' else data['split'] == split
            for name in GROUPS:
                select = mask & (group['group'] == name)
                counts.append(dict(scheme=scheme, split=split, group=name, n=int(select.sum()),
                    mean_native_length_bp=float(data['length'][select].mean()) if select.any() else None,
                    mean_gc_fraction=float(data['gc_fraction'][select].mean()) if select.any() else None,
                    mean_logit_qc_pass=int((select & data['quality_pass'][:, 0]).sum()),
                    soft_breadth_qc_pass=int((select & data['quality_pass'][:, 1]).sum())))
            for raw in range(1, 9):
                for family in range(1, len(FAMILIES[scheme])+1):
                    transitions.append(dict(scheme=scheme, split=split, context_breadth=raw,
                        family_breadth=family, n=int((mask & (group['raw'] == raw)
                                                    & (group['breadth'] == family)).sum())))
    training = data['labels'][data['split'] == 'train'].astype(bool)
    phi = np.corrcoef(training.T)
    for i, a in enumerate(contexts):
        for j in range(i+1, 8):
            inter = int((training[:, i] & training[:, j]).sum())
            union = int((training[:, i] | training[:, j]).sum())
            correlations.append(dict(context_a=a, context_b=contexts[j], split='train',
                                     phi=float(phi[i, j]), jaccard=inter/union,
                                     intersection=inter, union=union))
    output.mkdir(parents=True)
    for name, rows in [('enhancer_groups', members), ('group_counts', counts),
                       ('context_to_family_breadth', transitions), ('context_correlations', correlations)]:
        write_table(output/(name+'.tsv'), rows)
    summaries, catalogue = [], []
    selected_groups = list(selections(data, groups))
    for target_index, target in enumerate(TARGETS):
        audit_path = parent/target/'report_audit.json'
        audit = json.loads(audit_path.read_text())
        receipt_path = scans/target/'sequence/complete.json'
        receipt = json.loads(receipt_path.read_text())
        if receipt['status'] != 'complete':
            raise ValueError('Sequence scan incomplete')
        inputs[str(receipt_path)] = digest(receipt_path)
        matches = {}
        for database in ('jaspar', 'flyfactorsurvey', 'flyreg'):
            path = parent/target/'annotation'/database/'matches.json'
            matches[database] = json.loads(path.read_text())['best']
            inputs[str(path)] = digest(path)
        expected_files = {r['id'].replace('/', '__')+'.npz'
                          for g in audit['groups'] for r in g['rows']}
        if set(receipt['files']) != expected_files:
            raise ValueError('Scan catalogue differs from retained motif catalogue')
        for origin in audit['groups']:
            for row in origin['rows']:
                identifier = row['id']; key = identifier.replace('/', '__')
                path = scans/target/'sequence'/(key+'.npz')
                if digest(path) != receipt['files'][path.name]:
                    raise ValueError('Changed scan: '+str(path))
                inputs[str(path)] = receipt['files'][path.name]
                scan = read_scan(path, data)
                expected_importance = (scan['best_fraction'] >= .8) & data['quality_pass'][:, target_index]
                np.testing.assert_array_equal(np.isfinite(scan['mean_importance']), expected_importance)
                info = dict(target=target, motif_id=identifier, discovery_group=origin['name'],
                            contribution_sign=row['sign'], original_discovery_rank=row['rank'],
                            core_width_bp=row['quality']['width'])
                for database, best in matches.items():
                    hit = best[key]
                    info.update({database+'_name': hit['reference']['name'],
                                 database+'_id': hit['target_id'], database+'_q': hit['q']})
                catalogue.append(info)
                for scheme, split, comparison, group_name, selected in selected_groups:
                    for threshold in THRESHOLDS:
                        summaries.append(dict(info, scheme=scheme, split=split, comparison=comparison,
                            group=group_name, threshold=threshold,
                            **summarize(scan, selected, data['quality_pass'][:, target_index], threshold)))
                event('motif_regrouped', target=target, motif=identifier)
    write_table(output/'motif_catalogue.tsv', catalogue)
    write_table(output/'motif_metrics.tsv', summaries)
    # Compact typed input for the human-readable workbook, not a second analysis.
    primary = [r for r in summaries if r['split'] == 'validation' and r['threshold'] == .8
               and r['comparison'] in ('group', 'raw_context_breadth_3')]
    write_json(output/'workbook_data.json', dict(counts=counts, correlations=correlations,
        motifs=primary, catalogue=catalogue, input_metadata=str(parent/'metadata.npz')))
    for path, expected in inputs.items():
        if digest(Path(path)) != expected:
            raise ValueError('Input changed during analysis: '+path)
    code = Path(__file__)
    write_json(output/'complete.json', dict(status='complete', n=len(members), motifs=len(catalogue),
        metric_rows=len(summaries), contexts=contexts, families=FAMILIES, groups=GROUPS,
        inputs=inputs, code_sha256=digest(code),
        files={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        no_inference=True, no_new_attribution=True, no_discovery=True,
        attribution_targets_unchanged=True,
        limitations=['Family OR is a biological grouping, not statistical independence.',
            'Motifs were discovered in the original raw-breadth groups; ranks are not re-estimated.',
            'Sequence occurrence is not Fi-NeMo support, TF occupancy, or causality.',
            'Importance remains signed saved IG/bp over union of >=80%-score hit bases, carrier-only.',
            'Comparisons are descriptive, not adjusted for length, GC, family identity or genomic dependence.',
            'Same-raw-breadth comparisons control label count only, not tissue composition.',
            'Test chromosome was examined earlier; not a pristine holdout.']))
    event('family_breadth_complete', enhancers=len(members), motifs=len(catalogue), rows=len(summaries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'scans', 'context-provenance', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.parent, args.scans, args.context_provenance, args.output)
