"""Within-locus, within/between-family sharing of saved calibrated-probability IG."""
import argparse
import html
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
from numba import njit, prange

from .common import digest, event, require_allocation, write_json
from .family_breadth import CONTEXTS, FAMILIES, grouping
from .motif_occurrence import pssm
from .trl_family import PARENT, PREPARED_SHA, THRESHOLDS, load_metadata, scan_sites, table

NAME = 'classifier_motif_sharing_20260928'
TRL = 'MA0205.3'
FLOORS = (1e-6, .01)


def select_profiles(audit, matches, references):
    """Fixed reference profiles; inclusion never uses the sharing outcome."""
    support = {}
    for group in audit['groups']:
        for row in group['rows']:
            match = matches['best'][row['id'].replace('/', '__')]
            if row['quality']['passed'] and match['q'] <= .05 and row['supporting_discovery_enhancers'] >= 50:
                support.setdefault(match['target_id'], []).append(dict(pattern=row['id'],
                    q=match['q'], support=row['supporting_discovery_enhancers'], sign=row['sign']))
    result = [dict(**references[key], discovery_support=support[key]) for key in sorted(support)]
    if not {TRL, 'MA2107.1', 'MA1457.2', 'MA0460.1'}.issubset(support):
        raise ValueError('Expected Trl/cg/grh/ttk profiles absent from frozen panel')
    return result


@njit(parallel=True)
def locus_sums(actual, rows, starts, ends):
    out = np.zeros((len(rows), 8), np.float64)
    for i in prange(len(rows)):
        for pos in range(starts[i], ends[i]):
            for context in range(8):
                out[i, context] += actual[rows[i], context, pos]
    return out


def extract_loci(mask, actual, quality):
    """Contiguous union components merge overlapping OR touching PWM hits."""
    usable = mask & quality[:, None]
    previous = np.zeros_like(usable); previous[:, 1:] = usable[:, :-1]
    following = np.zeros_like(usable); following[:, :-1] = usable[:, 1:]
    rows, starts = np.nonzero(usable & ~previous)
    end_rows, last = np.nonzero(usable & ~following)
    np.testing.assert_array_equal(rows, end_rows)
    ends = last+1
    values = locus_sums(actual, rows, starts, ends)
    return rows, starts, ends, values


def effective_count(values, floor):
    total = values.sum(1)
    result = np.full(len(values), np.nan)
    keep = total >= floor
    result[keep] = total[keep]**2 / (values[keep]**2).sum(1)
    return result


def locus_metrics(signed, lengths, scheme, sign, floor):
    """Positive/negative magnitudes are split per context BEFORE family averaging."""
    if signed.ndim != 2 or signed.shape[1] != 8 or not np.isfinite(signed).all():
        raise ValueError('Expected eight finite signed site contributions')
    x = np.maximum(signed*(1 if sign == 'positive' else -1), 0)
    families = FAMILIES[scheme]
    family = np.column_stack([x[:, [CONTEXTS.index(c) for c in members]].mean(1)
                              for members in families.values()])
    strength = family.sum(1)
    neff = effective_count(family, floor)
    names = ['strength', 'strength_per_bp', 'effective_families', 'between_sharing', 'eligible_locus_fraction']
    columns = [strength, strength/lengths, neff, (neff-1)/(len(families)-1), (strength >= floor).astype(float)]
    for name, members in families.items():
        if len(members) == 1: continue  # No within-family replication for ovary/individual brains.
        part = x[:, [CONTEXTS.index(c) for c in members]]
        within = (effective_count(part, floor*len(members))-1)/(len(members)-1)
        names.append('within_'+name); columns.append(within)
    return names, np.column_stack(columns)


def enhancer_means(rows, values, n):
    """Equal locus weights within an enhancer; equal enhancer weights downstream."""
    result = np.full((n, values.shape[1]), np.nan)
    counts = np.zeros(result.shape, np.int32)
    for j in range(values.shape[1]):
        valid = np.isfinite(values[:, j])
        counts[:, j] = np.bincount(rows[valid], minlength=n)
        sums = np.bincount(rows[valid], weights=values[valid, j], minlength=n)
        np.divide(sums, counts[:, j], out=result[:, j], where=counts[:, j] > 0)
    return result, counts


def describe(values):
    out = dict(n=[], mean=[], median=[])
    for col in values.T:
        col = col[np.isfinite(col)]
        out['n'].append(len(col))
        out['mean'].append(float(col.mean()) if len(col) else None)
        out['median'].append(float(np.median(col)) if len(col) else None)
    return out


def block_interval(values, blocks, repeats=2000, seed=20260928):
    """Paired enhancer differences, bootstrap complete 1-Mb genomic blocks."""
    answer = describe(values)
    intervals, numbers = [], []
    for col in values.T:
        keep = np.isfinite(col)
        keys, inverse = np.unique(blocks[keep], return_inverse=True)
        numbers.append(len(keys))
        if len(keys) < 10:
            intervals.append(None); continue
        count = np.bincount(inverse)
        total = np.bincount(inverse, weights=col[keep])
        weights = np.random.default_rng(seed).multinomial(len(keys), np.ones(len(keys))/len(keys), size=repeats)
        means = (weights@total)/(weights@count)
        intervals.append(np.quantile(means, [.025,.975]).tolist())
    answer.update(blocks=numbers, descriptive_block_95ci=intervals)
    return answer


def metric_key(t, scheme, sign, f):
    return f't{t}__{scheme}__{sign}__f{f}'


def selections(data, scheme, detailed):
    yield 'all', np.ones(len(data['ids']), bool)
    if not detailed: return
    group = grouping(data['labels'], scheme=scheme)
    for degree in range(1,9): yield 'degree_'+str(degree), group['raw'] == degree
    for degree in range(2,9): yield 'degree_ge_'+str(degree), group['raw'] >= degree
    for category in ('context_specific','family_restricted_multicontext','two_families','three_or_more_families'):
        yield category, group['group'] == category
    # Exact active-context patterns, not merely the number of active contexts.
    for pattern in np.unique(data['label_pattern']):
        yield 'activity_pattern_'+str(pattern), data['label_pattern'] == pattern


def analyze_motif(motif, data, actual, output):
    n = len(data['ids'])
    weights, maximum = pssm(motif['pwm'], [.25]*4)
    best, starts_count, mask, union_sum = scan_sites(data['sequence'], data['length'], weights, maximum, actual, THRESHOLDS)
    cache = dict(ids=data['ids'], thresholds=THRESHOLDS, best_fraction=best,
        passing_starts=starts_count, covered_bp=mask.sum(2), union_mask_packed=np.packbits(mask,axis=2),
        native_padded_width=mask.shape[2], context_union_ig=union_sum)
    cache['context_union_ig'][~data['qc']] = np.nan
    summary = []
    for t, threshold in enumerate(THRESHOLDS):
        rows, start, end, signed = extract_loci(mask[:,t], actual, data['qc'])
        rebuilt = np.column_stack([np.bincount(rows,weights=signed[:,c],minlength=n) for c in range(8)])
        np.testing.assert_allclose(rebuilt[data['qc']], union_sum[data['qc'],t],rtol=1e-9,atol=1e-10)
        cache.update({f't{t}_locus_enhancer':rows, f't{t}_locus_start':start, f't{t}_locus_end':end,
                      f't{t}_locus_context_ig':signed})
        # Raw signed profiles are mean site IG, not averages over different sets of contexts.
        profile, _ = enhancer_means(rows, signed, n)
        cache[f't{t}_context_mean_site_ig'] = profile
        for scheme in FAMILIES:
            for sign in ('positive','negative'):
                for f, floor in enumerate(FLOORS):
                    names, site = locus_metrics(signed, end-start, scheme, sign, floor)
                    values, counts = enhancer_means(rows, site, n)
                    key = metric_key(t,scheme,sign,f)
                    cache[key] = values; cache[key+'_counts'] = counts
                    detailed = t == 1 and f == 0 and scheme == 'four_families'
                    for label, selected in selections(data,scheme,detailed):
                        if label != 'all' and not np.any(np.isfinite(values[selected,0])): continue
                        summary.append(dict(motif=motif['id'], name=motif['name'], threshold=float(threshold),
                            scheme=scheme, sign=sign, floor=floor, group=label, metrics=names,
                            **describe(values[selected])))
    np.savez_compressed(output/(motif['id']+'.npz'), **cache)
    write_json(output/(motif['id']+'.json'), summary)
    event('motif_sharing_scan_complete', motif=motif['id'], motif_name=motif['name'],
          sequence_carriers=(cache['covered_bp']>0).sum(0).tolist())
    return summary


def paired_comparisons(motifs, data, output, config, *, groups=None):
    result = []
    blocks = np.array([str(c)+':'+str(int(s)//1_000_000) for c,s in zip(data['chrom'],data['start'])])
    with np.load(output/(TRL+'.npz'),allow_pickle=False) as saved: trl = dict(saved)
    for motif in motifs:
        if motif['id'] == TRL: continue
        with np.load(output/(motif['id']+'.npz'),allow_pickle=False) as saved: other = dict(saved)
        np.testing.assert_array_equal(trl['ids'],other['ids'])
        for t, threshold in enumerate(THRESHOLDS):
            carriers = data['qc'] & (trl['covered_bp'][:,t]>0) & (other['covered_bp'][:,t]>0)
            overlap = np.any(trl['union_mask_packed'][:,t] & other['union_mask_packed'][:,t],axis=1)
            for scheme in FAMILIES:
                for sign in ('positive','negative'):
                    for f, floor in enumerate(FLOORS):
                        key = metric_key(t,scheme,sign,f)
                        a, b = trl[key], other[key]
                        names, _ = locus_metrics(np.zeros((0,8)),np.zeros(0),scheme,sign,floor)
                        for policy in ('disjoint','all_co_carriers'):
                            subsets = groups if groups is not None else {'all':np.ones(len(carriers),bool)}
                            for group_name, selected in subsets.items():
                                selected_carriers = carriers & selected
                                valid = selected_carriers & (~overlap if policy=='disjoint' else True)
                                selected_a, selected_b = a[valid], b[valid]
                                paired = np.isfinite(selected_a) & np.isfinite(selected_b)
                                x, y = np.where(paired,selected_a,np.nan), np.where(paired,selected_b,np.nan)
                                delta = x-y
                                ci = t == 1 and scheme == 'four_families' and policy == 'disjoint'
                                stats = block_interval(delta,blocks[valid],config['bootstrap_repeats'],config['seed']) if ci else describe(delta)
                                row = dict(other=motif['id'],name=motif['name'],threshold=float(threshold),
                                    scheme=scheme,sign=sign,floor=floor,policy=policy,metrics=names,
                                    co_carriers=int(selected_carriers.sum()),overlap_excluded=int((selected_carriers&overlap).sum()) if policy=='disjoint' else 0,
                                    trl=describe(x),other_motif=describe(y),difference=stats)
                                if groups is not None: row['group'] = group_name
                                result.append(row)
        event('motif_sharing_pair_complete',other=motif['id'])
    return result


def render(report, output, *, group='all', subset=None, cache_directory=None, title=None, artifact_prefix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    primary = [r for r in report['summary'] if r['threshold']==.8 and r['scheme']=='four_families'
               and r['floor']==FLOORS[0] and r['group']==group]
    motifs = sorted(report['config']['motifs'],key=lambda m:(m['id']!=TRL,m['name'].lower()))
    sharing = ['within_embryo','within_CNS','within_imaginal_discs','between_sharing']
    labels = ['Within embryo','Within CNS','Within discs','Between families']
    fig,axes=plt.subplots(1,2,figsize=(12,9),layout='constrained',sharey=True)
    for ax,sign in zip(axes,('positive','negative')):
        matrix=[]
        for motif in motifs:
            row=next(r for r in primary if r['motif']==motif['id'] and r['sign']==sign)
            matrix.append([row['mean'][row['metrics'].index(m)] if row['mean'][row['metrics'].index(m)] is not None else np.nan for m in sharing])
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set_xticks(range(4),labels,rotation=35,ha='right');ax.set_title(sign.capitalize()+' contributions')
        for i,vals in enumerate(matrix):
            for j,value in enumerate(vals):
                if np.isfinite(value):ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=8,color='white' if value<.55 else 'black')
    axes[0].set_yticks(range(len(motifs)),[m['name'] for m in motifs])
    fig.colorbar(im,ax=axes,label='Sharing (0: concentrated; 1: equally distributed)',shrink=.65)
    fig.savefig(output/'sharing.svg');plt.close(fig)
    profiles=[]
    for motif in motifs:
        with np.load((cache_directory or output)/(motif['id']+'.npz'),allow_pickle=False) as saved:
            covered=saved['covered_bp'][:,1]
            signed=saved['context_union_ig'][:,1]
            valid=(covered>0)&np.isfinite(signed).all(1)
            if subset is not None: valid &= subset
            mean=(signed[valid]/covered[valid,None]).mean(0) if valid.any() else np.full(8,np.nan)
        profiles.append(dict(motif=motif['id'],name=motif['name'],carriers=int(valid.sum()),
            contexts=list(CONTEXTS),mean_signed_ig_per_covered_bp=[float(v) if np.isfinite(v) else None for v in mean]))
    write_json(output/'profiles.json',profiles)
    context_matrix=np.asarray([p['mean_signed_ig_per_covered_bp'] for p in profiles],dtype=float)
    family_matrix=np.column_stack([context_matrix[:,[CONTEXTS.index(c) for c in members]].mean(1)
                                   for members in FAMILIES['four_families'].values()])
    limit=float(np.nanmax(np.abs(context_matrix))) if np.isfinite(context_matrix).any() else 1.
    limit=max(limit,1e-12)
    fig,axes=plt.subplots(1,2,figsize=(13,9),layout='constrained',sharey=True,gridspec_kw={'width_ratios':[2,1]})
    for ax,matrix,labels in zip(axes,(context_matrix,family_matrix),(list(CONTEXTS),list(FAMILIES['four_families']))):
        im=ax.imshow(matrix,vmin=-limit,vmax=limit,cmap='RdBu_r',aspect='auto')
        ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right')
    axes[0].set_yticks(range(len(motifs)),[m['name'] for m in motifs])
    fig.colorbar(im,ax=axes,label='Mean signed IG per covered base',shrink=.65)
    fig.savefig(output/'contribution_profiles.svg');plt.close(fig)
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Motif contribution sharing</title>',
        '<style>body{font:15px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;color:#222}p{max-width:1100px;line-height:1.5}table{border-collapse:collapse;margin:20px 0;font-size:13px}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{width:100%;max-width:1200px}summary{cursor:pointer}</style>',
        '<h1>'+html.escape(title or 'Motif contributions within and between context families')+'</h1>',
        '<p>Each contiguous motif locus is measured across all eight saved calibrated-probability IG maps, without masking inactive contexts. '
        'Overlapping or touching hits form one locus. Calculate sharing at each locus first, then average loci within each enhancer and enhancers equally. '
        'Do not infer broad sharing from a population-average contribution profile.</p>',
        '<p>Positive and negative net context contributions are analyzed separately. Effective family count is squared total family support divided by the sum of squared supports. '
        'Sharing rescales it to 0–1; within-family sharing uses member contexts. Families receive means, not sums. Ovary has no within-family replication. '
        'Strength is shown separately; high sharing alone does not imply a strong effect. Primary floor 10⁻⁶ excludes numerical zeros; a 0.01-strength sensitivity is saved.</p>',
        '<p>Fixed JASPAR panel selected from the preceding 22 fits: at least one passing discovery with ≥50 contributing enhancers and best JASPAR match q≤0.05. '
        'One PWM per profile, no outcome-based selection. Scan cutoff0.8, sensitivities0.7/0.9. Matched score fractions do not imply equal false-positive rates across PWMs. '
        'No new inference or attribution; no motif reclustering.</p>',
        '<h2>Contribution magnitude by context and family</h2><img src="contribution_profiles.svg" alt="Signed contribution magnitude across eight contexts and four families">',
        '<p>Signed IG per covered base, averaged equally across carrier enhancers; family columns average member contexts. '
        'This is a magnitude summary, not the sharing statistic. Both heatmaps on this page use one common color scale. '
        '<a href="profiles.json">Exact profile values and carrier counts</a>.</p>',
        '<h2>Sharing measured at individual loci</h2>',
        '<img src="sharing.svg" alt="Positive and negative motif-sharing heatmaps">',
        '<p>Heatmap entries have different available-carrier counts. These pooled summaries alone do not adjust for enhancer composition. '
        'The comparisons below pair Trl and another motif within the SAME enhancer, excluding enhancers where their site unions overlap. '
        'Thus activity labels, sequence length and GC are identical within each comparison; occurrence numbers/widths may still differ.</p>']
    for sign in ('positive','negative'):
        parts.append('<h2>'+sign.capitalize()+' contribution strength and sharing</h2>')
        rows=[]
        for motif in motifs:
            r=next(r for r in primary if r['motif']==motif['id'] and r['sign']==sign)
            metrics=r['metrics'];b=metrics.index('between_sharing')
            rows.append([motif['name'],r['n'][0],r['mean'][0],r['mean'][1],r['n'][b],r['mean'][metrics.index('effective_families')],r['mean'][b]])
        parts.append(table(['Motif','Carriers','Mean strength/locus','Strength/bp','Sharing carriers','Effective families','Between sharing'],rows))
        rows=[]
        for r in report['paired']:
            if r.get('group','all') != group: continue
            if not (r['threshold']==.8 and r['scheme']=='four_families' and r['sign']==sign and r['floor']==FLOORS[0] and r['policy']=='disjoint'):continue
            j=r['metrics'].index('between_sharing');d=r['difference']
            rows.append([r['name'],d['n'][j],r['overlap_excluded'],r['trl']['mean'][j],r['other_motif']['mean'][j],d['mean'][j],d['descriptive_block_95ci'][j]])
        parts.append('<h3>Paired Trl − comparator, between-family sharing</h3>')
        parts.append(table(['Comparator','Paired enhancers','Overlap excluded','Trl sharing','Comparator sharing','Difference','Block-bootstrap95% interval'],rows))
    parts += ['<p>Intervals resample 1-Mb genomic blocks (2,000 replicates), with at least10 occupied blocks; descriptive, not multiplicity-adjusted. '
              'Different comparator motifs use different enhancer subsets. Similar PWMs may represent indistinguishable TF preferences. '
              'This is an exploratory model explanation, not TF binding, a knockout effect, or causal evidence. All splits were used.</p>',
              '<p><a href="report.json">All metrics and sensitivities for this report</a> · '
              f'<a href="{artifact_prefix}config.json">Frozen protocol and PWMs</a> · <a href="{artifact_prefix}enhancers.npz">Enhancer metadata</a></p>','</html>']
    (output/'index.html').write_text('\n'.join(parts))


def run(project,root):
    require_allocation('cpu')
    config=json.loads((root/'config.json').read_text());parent=project/PARENT
    if digest(parent/'prepared.json')!=PREPARED_SHA:raise ValueError('Changed source receipt')
    receipt=json.loads((parent/'prepared.json').read_text())
    inputs={name:digest(parent/name) for name in ('native/metadata.npz','native/actual.npy')}
    if any(sha!=receipt['files'][name] for name,sha in inputs.items()):raise ValueError('Changed source arrays')
    actual=np.load(parent/'native/actual.npy',mmap_mode='r',allow_pickle=False)
    data=load_metadata(parent/'native/metadata.npz',actual)
    if len(data['ids'])!=40338 or int(data['qc'].sum())!=40309:raise ValueError('Unexpected cohort')
    data['label_pattern']=(data['labels'].astype(np.int64)*(2**np.arange(8))).sum(1)
    output=root/'output';output.mkdir(exist_ok=False)
    write_json(output/'config.json',config)
    np.savez_compressed(output/'enhancers.npz',**{k:data[k] for k in
        ('ids','labels','label_pattern','split','chrom','start','end','length','gc','qc','calibrated_probabilities')},contexts=np.asarray(CONTEXTS))
    report=dict(config=config,inputs=inputs,summary=[],paired=[])
    for motif in config['motifs']:
        report['summary'].extend(analyze_motif(motif,data,actual,output))
    report['paired']=paired_comparisons(config['motifs'],data,output,config)
    write_json(output/'report.json',report);render(report,output)
    files={p.name:digest(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'complete.json',dict(status='complete',manifest_sha256=digest(root/'MANIFEST.sha256'),
        files=files,job=os.environ['SLURM_JOB_ID'],versions={name:importlib.metadata.version(name) for name in ('numpy','numba','scipy','matplotlib')}))
    event('motif_sharing_complete',motifs=len(config['motifs']),report=str(output/'index.html'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();run(args.project,args.root)
