"""One unmasked family-balanced catalogue; sharing at individual native seqlet cores.

Reuse saved IG64/100 calibrated-output maps. CPU only; no new model evaluations.
"""
import argparse
import html
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np

from . import calibrated_motifs as native
from . import cumulative_motifs as cumulative
from .calibrated_tomtom import RULE, alignment_logos, page_start, render
from .common import digest, event, require_allocation, write_json
from .family_breadth import CONTEXTS, FAMILIES, grouping
from .motif_sharing import block_interval, enhancer_means
from .tomtom_atlas import DATABASES, meme_queries, query_id
from .trl_family import balance, match_pairs, table

NAME = 'classifier_hierarchical_motifs_20260929'
TARGET = 'unmasked_four_family_mean'
SCHEME = 'four_families'
SEED = 20260929
NUMERIC_FLOOR = 1e-6  # IG units per core base; not biological significance.
NULL_QUANTILES = (.90, .95, .99)
MIN_NULL_WINDOWS = 20
BOOTSTRAPS = 2000


def spec():
    return dict(weights=(sum(native.family_weights(SCHEME).values())/4).tolist(), observed_mask=False)


def tasks():
    return [dict(task=0, target=TARGET, group='all_enhancers', scheme='all')]


def family_scores(contexts):
    x = np.asarray(contexts, dtype=float)
    if x.ndim != 2 or x.shape[1] != 8 or not np.isfinite(x).all():
        raise ValueError('Expected finite occurrence-by-eight signed context scores')
    return np.column_stack([x[:, [CONTEXTS.index(c) for c in members]].mean(1)
                            for members in FAMILIES[SCHEME].values()])


def sharing(contexts, lengths, sign):
    """Net signed family mean FIRST; split signs second; exp(Shannon) per site."""
    if sign not in ('positive', 'negative'): raise ValueError('Invalid sign')
    lengths = np.asarray(lengths)
    family = family_scores(contexts)
    if lengths.shape != (len(family),) or (lengths <= 0).any(): raise ValueError('Invalid core widths')
    magnitudes = np.maximum(family * (1 if sign == 'positive' else -1), 0)
    strength = magnitudes.sum(1)
    fractions = np.divide(magnitudes, strength[:, None], out=np.zeros_like(magnitudes), where=strength[:, None] > 0)
    logs = np.zeros_like(fractions); np.log(fractions, out=logs, where=fractions > 0)
    neff = np.exp(-(fractions*logs).sum(1))
    neff[strength/lengths < NUMERIC_FLOOR] = np.nan
    return dict(family_signed=family, strength=strength, strength_per_bp=strength/lengths,
                fractions=fractions, effective_families=neff)


def core_occurrences(path, rows, indices, metadata, actual):
    """Retain every raw seqlet, plus a deterministic nonoverlap counting subset.

    Coordinates are native, forward-strand half-open intervals. RC changes core
    coordinates, never the sign or order of the context outputs. Never unite sites.
    """
    import h5py
    profiles = []
    with h5py.File(path, 'r') as h5:
        for row in rows:
            node = h5[row['pattern']+'/seqlets']; a, b = row['quality']['start'], row['quality']['end']
            records = []
            for j, (local, start, end, rc) in enumerate(zip(node['example_idx'][:], node['start'][:],
                                                          node['end'][:], node['is_revcomp'][:])):
                local, start, end, rc = int(local), int(start), int(end), bool(rc)
                if not 0 <= local < len(indices): raise ValueError('Invalid example index')
                i = int(indices[local]); length = int(metadata['length'][i])
                if not 0 <= start < end <= length or end-start != len(row['full_pwm']):
                    raise ValueError('Invalid native seqlet bounds or PWM alignment')
                if not 0 <= a < b <= end-start: raise ValueError('Invalid core slice')
                sequence = np.eye(4)[metadata['sequence'][i, start:end]]
                if rc: sequence = sequence[::-1, ::-1]
                np.testing.assert_array_equal(sequence[a:b], node['sequence'][j, a:b])
                left, right = (end-b, end-a) if rc else (start+a, start+b)
                records.append((i, left, right, int(rc), j))
            records = np.asarray(sorted(records), dtype=np.int64).reshape(-1, 5)
            keep = np.zeros(len(records), bool); last_end = {}
            for j, (i, left, right, rc, source_row) in enumerate(records):
                if left >= last_end.get(i, -1): keep[j] = True; last_end[i] = right
            values = np.asarray([actual[i, :, left:right].sum(1, dtype=np.float64)
                                 for i, left, right, _, _ in records]).reshape(-1, 8)
            if len(np.unique(records[:, 0])) != row['supporting_discovery_enhancers']:
                raise ValueError('Seqlet carrier count changed')
            if not np.isfinite(values).all(): raise ValueError('Nonfinite motif contribution')
            profiles.append(dict(indices=records[:, 0], start=records[:, 1], end=records[:, 2],
                is_revcomp=records[:, 3].astype(bool), source_seqlet_row=records[:, 4],
                selected=keep, context_signed_sum=values))
    return profiles


def background_thresholds(actual, width, occupied):
    """All same-width windows outside retained motif cores in the SAME enhancer.

    Overlapping background windows are allowed: this is an empirical specificity
    screen, not independent null replicates, a p-value, or calibrated motif FDR.
    """
    length = len(occupied)
    if actual.shape != (8, length) or not 0 < width <= length: raise ValueError('Invalid background window')
    mask_sum = np.r_[0, np.cumsum(occupied)]
    starts = np.flatnonzero(mask_sum[width:] == mask_sum[:-width])
    thresholds = np.full((2, len(NULL_QUANTILES), 4), np.nan)
    if len(starts) < MIN_NULL_WINDOWS: return thresholds, len(starts)
    cumulative_sum = np.pad(np.cumsum(actual, axis=1, dtype=float), ((0, 0), (1, 0)))
    values = (cumulative_sum[:, starts+width]-cumulative_sum[:, starts]).T/width
    family = family_scores(values)
    for s, factor in enumerate((1, -1)):
        thresholds[s] = np.maximum(np.quantile(factor*family, NULL_QUANTILES, axis=0), NUMERIC_FLOOR)
    return thresholds, len(starts)


def attach_support(profiles, metadata, actual):
    occupied = {}
    for profile in profiles:
        for i, a, b in zip(profile['indices'], profile['start'], profile['end']):
            occupied.setdefault(int(i), np.zeros(int(metadata['length'][i]), bool))[a:b] = True
    cache = {}
    for profile in profiles:
        thresholds, counts = [], []
        for i, a, b in zip(profile['indices'], profile['start'], profile['end']):
            key = (int(i), int(b-a))
            if key not in cache:
                cache[key] = background_thresholds(actual[i, :, :metadata['length'][i]], int(b-a), occupied[int(i)])
            threshold, count = cache[key]; thresholds.append(threshold); counts.append(count)
        profile['background_thresholds_per_bp'] = np.asarray(thresholds)
        profile['background_windows'] = np.asarray(counts)
        widths = profile['end']-profile['start']
        for s, (sign, factor) in enumerate((('positive', 1), ('negative', -1))):
            values = sharing(profile['context_signed_sum'], widths, sign)
            for key, value in values.items(): profile[sign+'_'+key] = value
            valid = profile['background_windows'] >= MIN_NULL_WINDOWS
            supported = factor*values['family_signed'][:, None, :]/widths[:, None, None] > profile['background_thresholds_per_bp'][:, s]
            counts = supported.sum(2).astype(np.int16); counts[~valid] = -1
            profile[sign+'_supported_families'] = counts
        event('hierarchical_site_support', sites=len(widths), with_background=int(valid.sum()))


def group_masks(labels):
    g = grouping(labels)
    out = {'all': np.ones(len(labels), bool)}
    for k in range(1, 5):
        out['family_1' if k == 1 else f'family_ge_{k}'] = g['breadth'] == 1 if k == 1 else g['breadth'] >= k
        out[f'exact_family_{k}'] = g['breadth'] == k
    for k in range(1, 9):
        out['context_1' if k == 1 else f'context_ge_{k}'] = g['raw'] == 1 if k == 1 else g['raw'] >= k
    for k in (2, 3):
        out[f'context_{k}_one_family'] = (g['raw'] == k) & (g['breadth'] == 1)
        out[f'context_{k}_cross_family'] = (g['raw'] == k) & (g['breadth'] >= 2)
    return out


def enhancer_metrics(profile, n, sign):
    """Equal occurrence weight WITHIN enhancer; equal enhancer weight downstream."""
    selected = profile['selected']; indices = profile['indices'][selected]
    counts = np.bincount(indices, minlength=n)
    support = profile[sign+'_supported_families'][selected]
    strength = profile[sign+'_strength'][selected]
    neff = profile[sign+'_effective_families'][selected].copy()
    # Shannon is retained for all numerically eligible sites in NPZ, but primary
    # sharing summaries require at least one family above the 95th-percentile screen.
    neff[support[:, 1] < 1] = np.nan
    columns = [strength, profile[sign+'_strength_per_bp'][selected], neff,
               (support[:, 1] >= 0).astype(float)]
    names = ['strength', 'strength_per_bp', 'effective_families_supported', 'background_eligible_fraction']
    for q, quantile in enumerate(NULL_QUANTILES):
        for k in (2, 3, 4):
            value = (support[:, q] >= k).astype(float); value[support[:, q] < 0] = np.nan
            columns.append(value); names.append(f'fraction_sites_ge_{k}_families_q{quantile:g}')
    # Profiles are summarized per site before taking enhancer-level means.
    for c, context in enumerate(CONTEXTS):
        columns.append(profile['context_signed_sum'][selected, c]); names.append('signed_context_'+context)
    for f, family in enumerate(FAMILIES[SCHEME]):
        columns.append(profile[sign+'_family_signed'][selected, f]); names.append('signed_family_'+family)
    means, _ = enhancer_means(indices, np.column_stack(columns), n)
    return counts, names, means


def summarize(profiles, metadata):
    qc = metadata['quality_pass'].all(1); n = len(qc)
    masks = {key: value & qc for key, value in group_masks(metadata['labels']).items()}
    blocks = np.asarray([str(c)+':'+str(int(s)//1000000) for c, s in zip(metadata['chrom'], metadata['summit'])])
    # Match using labels/length/GC only, before inspecting any motif outcomes.
    ix = np.flatnonzero(qc)
    data = {key: metadata[key][ix] for key in ('ids', 'labels', 'length', 'chrom', 'split', 'start', 'end')}
    data['gc'] = ((metadata['sequence'][ix] == 1) | (metadata['sequence'][ix] == 2)).sum(1)/data['length']
    pairs = {str(k): match_pairs(data, grouping(data['labels']), k) for k in (2, 3)}
    matching = {key: dict(**balance(data, p), pair_indices=p.tolist()) for key, p in pairs.items()}
    result = []
    for pattern, profile in enumerate(profiles):
        signs = {}; counts = np.bincount(profile['indices'][profile['selected']], minlength=n)
        for sign in ('positive', 'negative'):
            _, names, values = enhancer_metrics(profile, n, sign)
            measures = np.column_stack([(counts > 0).astype(float), counts,
                                        1000*counts/metadata['length'], values])
            names = ['selected_seqlet_carrier_fraction', 'nonoverlap_core_count_per_enhancer',
                     'nonoverlap_core_count_per_kb']+names
            summaries = {}
            for group, mask in masks.items():
                estimate = block_interval(measures[mask], blocks[mask], repeats=BOOTSTRAPS, seed=SEED)
                summaries[group] = dict(enhancers=int(mask.sum()), carriers=int((counts[mask] > 0).sum()),
                    raw_seqlets=int(np.isin(profile['indices'], np.flatnonzero(mask)).sum()),
                    counted_cores=int(counts[mask].sum()), **estimate)
            signs[sign] = dict(metrics=names, groups=summaries)
        matched = {}
        for degree, p in pairs.items():
            a, b = ix[p[:, 0]], ix[p[:, 1]]
            matched[degree] = dict(pairs=len(p), broad_carriers=int((counts[a] > 0).sum()),
                restricted_carriers=int((counts[b] > 0).sum()),
                broad_carrier_fraction=float((counts[a] > 0).mean()) if len(p) else None,
                restricted_carrier_fraction=float((counts[b] > 0).mean()) if len(p) else None,
                broad_count_per_kb=float((1000*counts[a]/metadata['length'][a]).mean()) if len(p) else None,
                restricted_count_per_kb=float((1000*counts[b]/metadata['length'][b]).mean()) if len(p) else None)
        result.append(dict(pattern_index=pattern, signs=signs, matched_seqlet_support=matched))
        event('hierarchical_pattern_summary', pattern=pattern, carriers=int((counts > 0).sum()))
    return dict(patterns=result, matching=matching, matched_metadata_indices=ix.tolist(),
                group_counts={k:int(v.sum()) for k,v in masks.items()})


def display(value, percent=False):
    if value is None: return 'NA'
    return f'{100*value:.1f}%' if percent else f'{value:.3g}'


def report(audit, matches, summary, output):
    page = page_start('Family-balanced motif discovery and same-site sharing', ('jaspar',))
    page.append('<p>One unmasked catalogue; discovery = mean of four family means. Experimental labels only '
        'stratify results. Original enhancer boundaries; no new attribution. All splits are exploratory.</p>'
        '<p>Frequency below means <b>selected discovery-seqlet support</b>, not exhaustive PWM occurrence. '
        'One global fit, capped at 20,000 seqlets per sign. No extra motif clustering. Overlapping cores of '
        'the same motif are counted once (leftmost-first); distinct nonoverlapping occurrences stay separate.</p>'
        '<p>Sharing is exp(Shannon entropy) of positive or negative net family contributions, calculated per '
        'occurrence before enhancer averaging. Family means precede sign separation. Primary sharing requires '
        'at least one family above its within-enhancer background 95th percentile. Background uses all same-width '
        'windows excluding retained motif cores, minimum 20 windows; 90/99% sensitivities are saved. This is a '
        'descriptive support screen, not a p-value/FDR. Missing background is NA, not absence.</p>'
        '<p>Numeric JSON includes context/family profiles, cumulative and exact groups, 1-Mb block-bootstrap '
        '95% intervals (minimum 10 blocks), and 90/95/99% support sensitivity. Noncarriers count as zero for '
        'frequency/counts, but not for per-instance strength/sharing. Strength is not proof of causality.</p>'
        '<p><a href="jaspar.html">All aligned JASPAR matches</a> · <a href="summary.json">All statistics</a> · '
        '<a href="occurrences.npz">Individual site maps and background thresholds</a> · '
        '<a href="metadata.npz">Enhancer IDs, coordinates, labels and QC</a></p>')
    rows = audit['groups'][0]['rows']
    for row, stats in zip(rows, summary['patterns']):
        match = matches['best'][query_id(row)]
        page.append('<h2>'+html.escape(row['sign']+' · '+row['consensus']+' · '+match['reference']['name'])+
                    ' (Tomtom q='+display(match['q'])+')</h2>')
        page.append(alignment_logos(row, match))
        for sign in ('positive', 'negative'):
            profile = stats['signs'][sign]; names = profile['metrics']; records = []
            for group in ('family_1', 'family_ge_2', 'family_ge_3', 'family_ge_4'):
                s = profile['groups'][group]
                def mean(name): return s['mean'][names.index(name)]
                records.append([group.replace('family_ge_', '≥').replace('family_', ''), s['enhancers'],
                    f'{s["carriers"]} ({display(mean("selected_seqlet_carrier_fraction"), True)})',
                    s['raw_seqlets'], s['counted_cores'], display(mean('strength_per_bp')),
                    display(mean('effective_families_supported')),
                    display(mean('fraction_sites_ge_2_families_q0.95'), True),
                    display(mean('fraction_sites_ge_3_families_q0.95'), True),
                    display(mean('fraction_sites_ge_4_families_q0.95'), True)])
            page.append('<h3>'+sign.capitalize()+' family contributions at these sites</h3>')
            page.append(table(['Active families', 'Enhancers', 'Seqlet carriers', 'Raw seqlets', 'Counted cores',
                               'Strength/bp', 'Effective families', '≥2 supported', '≥3 supported', '4 supported'], records))
        records = []
        for degree, m in stats['matched_seqlet_support'].items():
            records.append([degree, m['pairs'], display(m['broad_carrier_fraction'], True),
                display(m['restricted_carrier_fraction'], True), display(m['broad_count_per_kb']),
                display(m['restricted_count_per_kb'])])
        page.append('<h3>Matched cross-family versus within-family comparison</h3><p>Same context count, '
            'chromosome and split; length ratio ≤1.1, GC difference ≤0.025; nonoverlapping pair members. '
            'Descriptive estimates, no significance claim. At degree 3, within-family controls are three discs.</p>')
        page.append(table(['Context count', 'Pairs', 'Cross-family carriers', 'Within-family carriers',
                           'Cross-family cores/kb', 'Within-family cores/kb'], records))
    (output/'index.html').write_text(''.join(page)+'</body></html>')


def finalize(project, root, config):
    task = tasks()[0]; native.finalize(root, task_list=tasks())
    folder = cumulative.directory(root, task)
    cumulative.annotate(project, root, task, spec=spec(), composites=False)
    audit = json.loads((folder/'report/audit.json').read_text())
    with np.load(folder/'examples.npz', allow_pickle=False) as z: examples = dict(z)
    indices = examples['indices']
    with np.load(root/'native/metadata.npz', allow_pickle=False) as z: metadata = dict(z)
    expected, _ = native.selection(metadata, task, config['discovery_parameters']['seed'])
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(examples['ids'], metadata['ids'][indices])
    np.testing.assert_allclose(examples['weights'], native.coefficients(metadata['labels'][indices], TARGET, spec=spec()), rtol=0, atol=0)
    actual = np.load(root/'native/actual.npy', mmap_mode='r', allow_pickle=False)
    rows = audit['groups'][0]['rows']
    profiles = core_occurrences(folder/'motifs.h5', rows, indices, metadata, actual)
    attach_support(profiles, metadata, actual)
    output = root/'output'; output.mkdir(exist_ok=False)
    arrays = {f'pattern_{j}_{k}':v for j,p in enumerate(profiles) for k,v in p.items()}
    np.savez_compressed(output/'occurrences.npz', **arrays)
    np.savez_compressed(output/'metadata.npz', **{k:metadata[k] for k in
        ('ids','labels','split','chrom','summit','start','end','length','quality_pass','calibrated_probabilities')})
    summary = summarize(profiles, metadata)
    summary.update(contexts=list(CONTEXTS), families=list(FAMILIES[SCHEME]), config=config,
        patterns_index=[dict(index=j,id=r['id'],pattern=r['pattern'],sign=r['sign']) for j,r in enumerate(rows)])
    matches = json.loads((folder/'report/jaspar/matches.json').read_text())
    audit.update(rule=RULE, discovery_note='One new pooled unmasked mean-of-family-means fit; weights total one. '
        'Labels only stratify summaries; selected seqlet support is not exhaustive occurrence prevalence.')
    write_json(output/'audit.json', audit); write_json(output/'summary.json', summary)
    write_json(output/'config.json', config); write_json(output/'jaspar_matches.json', matches)
    (output/'queries_4of5.meme').write_text(meme_queries(audit))
    shutil.copytree(folder/'report/jaspar', output/'jaspar')
    (output/'jaspar.html').write_text(render(audit, dict(jaspar=matches), 'jaspar', database_keys=('jaspar',)))
    report(audit, matches, summary, output)
    counts = dict(fits=1, retained=len(rows), positive=sum(r['sign']=='positive' for r in rows),
                  negative=sum(r['sign']=='negative' for r in rows), selected_enhancers=len(indices),
                  raw_seqlets=sum(len(p['indices']) for p in profiles),
                  counted_cores=sum(int(p['selected'].sum()) for p in profiles))
    write_json(output/'complete.json', dict(status='complete', summary=counts,
        manifest_sha256=digest(root/'MANIFEST.sha256'),
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}))
    event('hierarchical_complete', summary=counts)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare','discover','finalize'))
    p.add_argument('--project', type=Path, required=True); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); require_allocation('cpu')
    if importlib.metadata.version('modisco') != '2.5.2': raise ValueError('Pinned TF-MoDISco required')
    config = json.loads((args.root/'config.json').read_text())
    if config['tasks'] != tasks() or config['spec'] != spec(): raise ValueError('Changed hierarchy contract')
    if args.stage == 'prepare': cumulative.prepare(args.project, args.root, config, task_list=tasks())
    elif args.stage == 'discover': native.discover(args.root, config, 0, task=tasks()[0], spec=spec())
    else: finalize(args.project, args.root, config)
