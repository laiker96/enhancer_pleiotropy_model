"""Fixed Trl PWM prevalence and eight-context IG at motif sites; CPU only."""
import argparse
import html
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.stats import beta, binomtest

from .common import digest, event, require_allocation, write_json
from .family_breadth import CONTEXTS, FAMILIES, GROUPS, grouping
from .motif_occurrence import pssm

NAME = 'classifier_trl_family_20260928'
PARENT = 'experiments/classifier_calibrated_motifs_20260927'
PREPARED_SHA = '281dc4e4bbc4900b270a5c3dae22419f632bd570847f832bc83ee5de7f692814'
THRESHOLDS = np.array([.7, .8, .9])


def select_denovo(audit, matches):
    """Outcome-independent choice, fixed before scanning degrees 2/3."""
    candidates = []
    for group in audit['groups']:
        if (group['task']['target'], group['task']['group']) != ('all_context_sum', 'degree_1'):
            continue
        for row in group['rows']:
            match = matches['best'][row['id'].replace('/', '__')]
            if (row['sign'] == 'positive' and row['quality']['passed']
                    and match['target_id'] == 'MA0205.3' and match['q'] <= .05):
                candidates.append((row, match))
    if not candidates:
        raise ValueError('No eligible fixed Trl-like discovery PWM')
    return min(candidates, key=lambda item: (-item[0]['supporting_discovery_enhancers'], item[0]['id']))


@njit(parallel=True)
def scan_sites(codes, lengths, weights, maximum, actual, thresholds):
    """One hit per start if either strand passes; overlapping bases counted once."""
    n, width = len(codes), len(weights)
    covered = np.zeros((n, len(thresholds), codes.shape[1]), np.bool_)
    starts = np.zeros((n, len(thresholds)), np.int32)
    best = np.full(n, -np.inf)
    sums = np.zeros((n, len(thresholds), 8), np.float64)
    for i in prange(n):
        for start in range(lengths[i]-width+1):
            forward, reverse = 0., 0.
            for j in range(width):
                base = codes[i, start+j]
                forward += weights[j, base]
                reverse += weights[width-1-j, 3-base]
            score = max(forward, reverse)/maximum
            best[i] = max(best[i], score)
            for t in range(len(thresholds)):
                if score >= thresholds[t]:
                    starts[i, t] += 1
                    covered[i, t, start:start+width] = True
        for t in range(len(thresholds)):
            for j in range(lengths[i]):
                if covered[i, t, j]:
                    for context in range(8):
                        sums[i, t, context] += actual[i, context, j]
    return best, starts, covered, sums


def load_metadata(path, actual):
    with np.load(path, allow_pickle=False) as saved:
        data = dict(saved)
    codes, lengths = data['sequence'], data['length']
    n = len(codes)
    grouping(data['labels'])
    if (len(set(data['ids'])) != n or actual.shape != (n, 8, codes.shape[1])
            or data['quality_pass'].shape != (n, 8)
            or not np.isin(data['quality_pass'], [False, True]).all()
            or not np.isin(data['split'], ['train', 'validation', 'test']).all()
            or not np.array_equal(data['end']-data['start'], lengths)
            or np.any(lengths <= 0) or np.any(lengths > codes.shape[1])
            or data['calibrated_probabilities'].shape != (n, 8)
            or not np.isfinite(data['calibrated_probabilities']).all()
            or np.any((data['calibrated_probabilities'] < 0) | (data['calibrated_probabilities'] > 1))):
        raise ValueError('Invalid native metadata/maps')
    for start in range(0, n, 1024):
        sl = slice(start, start+1024)
        valid = np.arange(codes.shape[1])[None, :] < lengths[sl, None]
        if (not np.isin(codes[sl][valid], range(4)).all() or np.any(codes[sl][~valid] != 4)
                or not np.isfinite(actual[sl]).all()
                or np.any(actual[sl][np.broadcast_to(~valid[:, None, :], actual[sl].shape)] != 0)):
            raise ValueError('Invalid native sequence or map padding')
    data['gc'] = ((codes == 1) | (codes == 2)).sum(1)/lengths
    data['qc'] = data['quality_pass'].all(1)
    return data


def match_pairs(data, groups, degree, length_ratio=1.1, gc_caliper=.025):
    """Deterministic greedy 1:1 matching, without seeing PWM scores or IG."""
    eligible = groups['raw'] == degree
    broad = np.flatnonzero(eligible & (groups['breadth'] >= 2))
    restricted = np.flatnonzero(eligible & (groups['breadth'] == 1))
    candidates = []
    for control in restricted:
        distance_length = np.abs(np.log(data['length'][broad]/data['length'][control]))/np.log(length_ratio)
        distance_gc = np.abs(data['gc'][broad]-data['gc'][control])/gc_caliper
        keep = ((data['chrom'][broad] == data['chrom'][control])
                & (data['split'][broad] == data['split'][control])
                & (distance_length <= 1+1e-12) & (distance_gc <= 1+1e-12)
                & ((data['end'][broad] <= data['start'][control])
                   | (data['start'][broad] >= data['end'][control])))
        choices = broad[keep]
        distance = distance_length[keep]**2 + distance_gc[keep]**2
        choices = choices[np.lexsort((data['ids'][choices], distance))]
        candidates.append((control, choices))
    # Scarce controls first; ties stable by enhancer ID. Not optimal full matching.
    candidates.sort(key=lambda item: (len(item[1]), str(data['ids'][item[0]])))
    used, pairs = set(), []
    for control, choices in candidates:
        case = next((int(i) for i in choices if int(i) not in used), None)
        if case is not None:
            pairs.append((case, int(control))); used.add(case)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def matched_prevalence(carriers, pairs):
    a, b = pairs.T
    only_broad = int((carriers[a] & ~carriers[b]).sum())
    only_restricted = int((~carriers[a] & carriers[b]).sum())
    discordant = only_broad+only_restricted
    low = float(beta.ppf(.025, only_broad, only_restricted+1)) if only_broad else 0.
    high = float(beta.ppf(.975, only_broad+1, only_restricted)) if only_restricted else 1.
    # Infinity and unidentified estimates are explicit strings, not silent NaN/zeros.
    odds = only_broad/only_restricted if only_restricted else ('infinity' if only_broad else None)
    return dict(pairs=len(pairs), broad_carriers=int(carriers[a].sum()),
        restricted_carriers=int(carriers[b].sum()), broad_only=only_broad,
        restricted_only=only_restricted, both=int((carriers[a] & carriers[b]).sum()),
        neither=int((~carriers[a] & ~carriers[b]).sum()), matched_odds_ratio=odds,
        nominal_exact_95ci=[low/(1-low), high/(1-high) if high < 1 else 'infinity'] if discordant else None,
        nominal_mcnemar_p=float(binomtest(only_broad, discordant, .5).pvalue) if discordant else None)


def readouts(sums, labels, scheme):
    """Family means use every member context; observed-active sum is separate."""
    values = [sums, sums.sum(1, keepdims=True), (sums*labels).sum(1, keepdims=True)]
    names = [*CONTEXTS, 'all_context_sum', 'observed_active_sum']
    for name, members in FAMILIES[scheme].items():
        values.append(sums[:, [CONTEXTS.index(c) for c in members]].mean(1, keepdims=True))
        names.append(name+'_mean')
    return names, np.concatenate(values, axis=1)


def mean_or_none(values):
    return values.mean(0).tolist() if len(values) else None


def paired_contribution(values, covered, quality, pairs):
    a, b = pairs.T
    keep = quality[a] & quality[b] & (covered[a] > 0) & (covered[b] > 0)
    a, b = a[keep], b[keep]
    difference = values[a]-values[b]
    per_bp = values[a]/covered[a, None]-values[b]/covered[b, None]
    return dict(both_carrier_qc_pairs=len(a), broad_mean=mean_or_none(values[a]),
        restricted_mean=mean_or_none(values[b]), paired_mean_difference=mean_or_none(difference),
        paired_per_covered_base_difference=mean_or_none(per_bp))


def balance(data, pairs):
    a, b = pairs.T
    out = dict(pairs=len(pairs))
    for name, values in [('length_bp', data['length']), ('gc', data['gc'])]:
        pooled = np.sqrt((np.var(values[a])+np.var(values[b]))/2) if len(a) else 0
        out[name] = dict(broad_mean=float(values[a].mean()) if len(a) else None,
            restricted_mean=float(values[b].mean()) if len(a) else None,
            standardized_mean_difference=float((values[a].mean()-values[b].mean())/pooled) if pooled else None,
            max_absolute_pair_difference=float(np.max(np.abs(values[a]-values[b]))) if len(a) else None)
    return out


def summarize(data, groups, pairs, covered, sums):
    rows, comparisons = [], []
    for scheme, group in groups.items():
        names, values = readouts(sums, data['labels'], scheme)
        selections = [('group', name, group['group'] == name) for name in GROUPS]
        selections += [('exact_degree', str(d), group['raw'] == d) for d in range(1, 9)]
        selections += [('cumulative_degree', '>='+str(d), group['raw'] >= d) for d in range(2, 9)]
        selections += [('family_breadth', str(d), group['breadth'] == d) for d in range(1, len(FAMILIES[scheme])+1)]
        for kind, name, selected in selections:
            carrier = selected & (covered > 0)
            valid = carrier & data['qc']
            rows.append(dict(scheme=scheme, kind=kind, group=name, n=int(selected.sum()),
                sequence_carriers=int(carrier.sum()), qc_eligible=int((selected & data['qc']).sum()),
                qc_sequence_carriers=int(valid.sum()), carrier_readout_names=names,
                carrier_mean_signed_ig=mean_or_none(values[valid]),
                carrier_mean_signed_ig_per_covered_bp=mean_or_none(values[valid]/covered[valid, None]),
                carrier_mean_covered_bp=float(covered[valid].mean()) if valid.any() else None))
        for degree in (2, 3, '2+3'):
            p = pairs[scheme][str(degree)]
            common = p[data['qc'][p].all(1)]
            comparisons.append(dict(scheme=scheme, degree=degree, readout_names=names,
                sequence=matched_prevalence(covered > 0, p),
                common_qc=matched_prevalence(covered > 0, common),
                contribution=paired_contribution(values, covered, data['qc'], p)))
    return dict(groups=rows, comparisons=comparisons)


def table(headers, rows):
    def cell(value):
        if value is None: return '—'
        if isinstance(value, float): return f'{value:.4g}'
        return str(value)
    return '<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(cell(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def render(report):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Trl: cross-family breadth</title>',
        '<style>body{font:15px system-ui;max-width:1250px;margin:36px auto;padding:0 20px;color:#20242b}table{border-collapse:collapse;margin:20px 0}th,td{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}summary{cursor:pointer;margin:20px 0}p{max-width:1050px;line-height:1.5}</style>',
        '<h1>Trl-like sites and cross-family enhancer activity</h1>',
        '<p>Fixed JASPAR MA0205.3 is primary; one fixed degree-1 de novo PWM is a sensitivity analysis. '
        'Both strands, native enhancer intervals, union of overlapping hit bases. Primary scan cutoff: '
        '0.8 of maximum log-odds; 0.7 and 0.9 are sensitivities, not p-values.</p>',
        '<p>Cross-family means ≥2 families; restricted means multiple contexts in one family. '
        'Matching is 1:1 without replacement, exact raw degree, chromosome and dataset split; '
        'length ratio ≤1.10 and GC difference ≤0.025. Overlapping paired intervals are excluded. '
        'At degree 3 the restricted controls are necessarily the three imaginal discs. '
        'There is no same-degree restricted comparator at degree ≥4.</p>',
        '<p>Sequence prevalence uses all 40,338 enhancers; signed attribution uses the 40,309 '
        'passing all eight map QC checks. Carrier-only means do not assign zero to noncarriers. '
        'Paired intensity comparisons require both enhancers to carry the motif and pass QC. '
        'IG site contributions are baseline-relative model explanations, not mutation effects or evidence of TF binding.</p>',
        '<p>All splits are exploratory. Matched odds-ratio intervals and McNemar p-values are nominal '
        '(pairs treated as independent; no multiplicity correction or genomic-dependence adjustment). '
        'Family definitions do not establish statistical independence. Contribution differences are descriptive, without significance claims.</p>',
        '<p><a href="report.json">Full group summaries and methods</a> · <a href="enhancers.npz">Enhancer metadata</a> · '
        '<a href="pairs.npz">Matched pairs</a> · <a href="config.json">Frozen configuration</a></p>']
    for scheme, matches in report['matching'].items():
        parts.append('<h2>'+html.escape(scheme)+'</h2>')
        parts.append(table(['Degree','Eligible broad','Eligible restricted','Matched pairs','Length SMD','GC SMD'],[
            [d, m['broad'],m['restricted'],m['balance']['pairs'],m['balance']['length_bp']['standardized_mean_difference'],
             m['balance']['gc']['standardized_mean_difference']] for d,m in matches.items()]))
    for result in report['results']:
        primary = result['threshold'] == .8
        if not primary: parts.append('<details><summary>'+html.escape(result['motif'])+f' — cutoff {result["threshold"]}</summary>')
        else: parts.append('<h2>'+html.escape(result['motif'])+' — primary cutoff 0.8</h2>')
        parts.append(table(['Families','Degree','Pairs','Broad carriers','Restricted carriers','Matched OR','Nominal 95% CI','Nominal p'],[
            [c['scheme'],c['degree'],c['sequence']['pairs'],c['sequence']['broad_carriers'],c['sequence']['restricted_carriers'],
             c['sequence']['matched_odds_ratio'],c['sequence']['nominal_exact_95ci'],c['sequence']['nominal_mcnemar_p']]
            for c in result['comparisons']]))
        if primary:
            for c in result['comparisons']:
                if c['degree'] != '2+3': continue
                effect = c['contribution']
                parts.append('<h3>'+html.escape(c['scheme'])+f': signed IG among {effect["both_carrier_qc_pairs"]} paired motif carriers</h3>')
                parts.append(table(['Readout','Cross-family mean','Restricted mean','Paired difference','Difference per covered bp'],[
                    [name,*[effect[key][i] if effect[key] is not None else None for key in
                     ('broad_mean','restricted_mean','paired_mean_difference','paired_per_covered_base_difference')]]
                    for i,name in enumerate(c['readout_names'])]))
        else: parts.append('</details>')
    parts.append('</html>')
    return '\n'.join(parts)


def run(project, root):
    require_allocation('cpu')
    config = json.loads((root/'config.json').read_text())
    parent = project/PARENT
    if digest(parent/'prepared.json') != PREPARED_SHA:
        raise ValueError('Changed parent preparation')
    ready = json.loads((parent/'prepared.json').read_text())
    if ready['status'] != 'complete' or ready['elements'] != 40338:
        raise ValueError('Incomplete parent cohort')
    inputs = {'prepared.json': PREPARED_SHA}
    for name in ('native/metadata.npz', 'native/actual.npy'):
        inputs[name] = digest(parent/name)
        if inputs[name] != ready['files'][name]: raise ValueError('Changed native input: '+name)
    actual = np.load(parent/'native/actual.npy', mmap_mode='r', allow_pickle=False)
    data = load_metadata(parent/'native/metadata.npz', actual)
    if int(data['qc'].sum()) != 40309: raise ValueError('Unexpected common QC cohort')
    groups = {scheme: grouping(data['labels'], scheme=scheme) for scheme in FAMILIES}
    output = root/'output'; output.mkdir(exist_ok=False)
    write_json(output/'config.json', config)
    np.savez_compressed(output/'enhancers.npz', **{k:data[k] for k in
        ('ids','labels','split','chrom','start','end','length','gc','qc','quality_pass','calibrated_probabilities')},
        contexts=np.asarray(CONTEXTS))
    pairs, matching = {}, {}
    for scheme, group in groups.items():
        pairs[scheme] = {str(d): match_pairs(data, group, d, **config['matching']) for d in (2,3)}
        pairs[scheme]['2+3'] = np.concatenate(list(pairs[scheme].values()))
        matching[scheme] = {str(d): dict(broad=int(((group['raw']==d)&(group['breadth']>=2)).sum()),
            restricted=int(((group['raw']==d)&(group['breadth']==1)).sum()),
            balance=balance(data,pairs[scheme][str(d)])) for d in (2,3)}
    np.savez_compressed(output/'pairs.npz', **{scheme+'__'+degree:p for scheme,rows in pairs.items() for degree,p in rows.items()})
    event('trl_matching_complete', matching=matching)
    report = dict(config=config, inputs=inputs, elements=len(data['ids']), qc_elements=int(data['qc'].sum()),
        duplicate_native_intervals=len(data['ids'])-len(set(zip(data['chrom'], data['start'], data['end']))),
        matching=matching, results=[])
    for motif in config['motifs']:
        weights, maximum = pssm(motif['pwm'], [.25]*4)
        best, starts, mask, sums = scan_sites(data['sequence'], data['length'], weights, maximum, actual, THRESHOLDS)
        covered = mask.sum(2)
        if not np.array_equal(covered > 0, best[:, None] >= THRESHOLDS):
            raise ValueError('Scan mask/score mismatch')
        sums[~data['qc']] = np.nan
        np.savez_compressed(output/(motif['key']+'.npz'), ids=data['ids'], thresholds=THRESHOLDS,
            best_fraction=best, passing_starts=starts, covered_bp=covered,
            union_mask_packed=np.packbits(mask, axis=2), native_padded_width=mask.shape[2],
            context_signed_ig=sums, weights=weights, maximum_logodds=maximum)
        for t, threshold in enumerate(THRESHOLDS):
            report['results'].append(dict(motif=motif['key'], threshold=float(threshold),
                **summarize(data, groups, pairs, covered[:,t], sums[:,t])))
        event('trl_scan_complete', motif=motif['key'], carriers=(covered > 0).sum(0).tolist())
        del mask
    write_json(output/'report.json', report)
    (output/'index.html').write_text(render(report))
    files = {p.name:digest(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'complete.json', dict(status='complete', manifest_sha256=digest(root/'MANIFEST.sha256'),
        files=files, versions={name:importlib.metadata.version(name) for name in ('numpy','numba','scipy')},
        job=os.environ['SLURM_JOB_ID']))
    event('trl_family_complete', report=str(output/'index.html'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.root)
