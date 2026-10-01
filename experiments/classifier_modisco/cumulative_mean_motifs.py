"""Cumulative context-degree motifs from mean-active IG, with per-enhancer profiles."""
import argparse
import html
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np

from . import calibrated_motifs as native
from . import cumulative_motifs as cumulative
from .calibrated_tomtom import RULE, label, page_start, render
from .common import digest, event, require_allocation, write_json
from .family_breadth import CONTEXTS, FAMILIES
from .family_motifs import core_profiles
from .tomtom_atlas import DATABASES, meme_queries, query_id

NAME = 'classifier_cumulative_mean_motifs_20260929'
TARGET = 'observed_active_mean'
SHARING_FLOOR = 1e-6  # IG probability units per covered core base, not a scan cutoff.


def spec():
    return dict(weights=[1.]*8, observed_mask=True, observed_mean=True)


def tasks():
    return [dict(task=k-1, target=TARGET, group='degree_1' if k == 1 else f'degree_ge_{k}',
                 scheme='raw_degree', low=k, high=1 if k == 1 else 8) for k in range(1, 9)]


def sharing_summary(magnitudes):
    """Compute sharing within EACH enhancer before taking equal-enhancer averages."""
    if magnitudes.ndim != 2 or not np.isfinite(magnitudes).all() or (magnitudes < 0).any():
        raise ValueError('Finite nonnegative enhancer-by-context/family matrix required')
    total = magnitudes.sum(1)
    selected = total >= SHARING_FLOOR
    fractions = magnitudes[selected]/total[selected, None]
    effective = 1/(fractions**2).sum(1)
    n = magnitudes.shape[1]
    return dict(eligible_enhancers=int(selected.sum()), total_enhancers=len(total),
        mean_strength=float(total.mean()),
        mean_fraction=fractions.mean(0).tolist() if selected.any() else [None]*n,
        mean_effective_count=float(effective.mean()) if selected.any() else None,
        mean_sharing=float(((effective-1)/(n-1)).mean()) if selected.any() and n > 1 else None)


def phenotype(values):
    if values.ndim != 2 or values.shape[1] != 8 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty finite enhancer-by-eight-context profiles required')
    result = dict(contexts=list(CONTEXTS), schemes={}, sharing_floor=SHARING_FLOOR,
                  mean_signed_context_ig_per_bp=values.mean(0).tolist(), signs={})
    for scheme, families in FAMILIES.items():
        columns = [[CONTEXTS.index(c) for c in members] for members in families.values()]
        family = np.column_stack([values[:, ix].mean(1) for ix in columns])
        result['schemes'][scheme] = dict(families=list(families),
            mean_signed_family_ig_per_bp=family.mean(0).tolist(), signs={})
    for sign, factor in (('positive', 1), ('negative', -1)):
        # Split net core contributions by context BEFORE averaging family members.
        magnitudes = np.maximum(factor*values, 0)
        result['signs'][sign] = sharing_summary(magnitudes)
        for scheme, families in FAMILIES.items():
            columns = [[CONTEXTS.index(c) for c in members] for members in families.values()]
            family = np.column_stack([magnitudes[:, ix].mean(1) for ix in columns])
            result['schemes'][scheme]['signs'][sign] = dict(between=sharing_summary(family),
                within={name:sharing_summary(magnitudes[:, ix]) for (name, members), ix
                        in zip(families.items(), columns) if len(members) > 1})
    return result


def annotate(project, root, task):
    cumulative.annotate(project, root, task, spec=spec(), composites=False)
    folder = cumulative.directory(root, task)
    audit = json.loads((folder/'report/audit.json').read_text())
    with np.load(folder/'examples.npz', allow_pickle=False) as z: indices = z['indices']
    with np.load(root/'native/metadata.npz', allow_pickle=False) as z: metadata = dict(z)
    actual = np.load(root/'native/actual.npy', mmap_mode='r', allow_pickle=False)
    rows = audit['groups'][0]['rows']
    profiles = core_profiles(folder/'motifs.h5', rows, indices, metadata, actual)
    arrays = {}
    for j, (row, profile) in enumerate(zip(rows, profiles)):
        selected = profile['indices']; values = profile['values']
        weights = native.coefficients(metadata['labels'][selected], TARGET, spec=spec())
        readout = np.sum(values*weights, axis=1)
        row['contribution_profile'] = dict(**phenotype(values),
            array_prefix=f'pattern_{j}', mean_discovery_scalar_ig_per_bp=float(readout.mean()),
            averaging='Union trimmed cores per enhancer; mean per covered base, then equal enhancer weight. '
                      'All eight outputs unmasked in context/family diagnostics; experimental active mean only for discovery.')
        for key in ('indices', 'values', 'covered_bp'):
            arrays[f'pattern_{j}_{key}'] = profile[key]
        for key in ('ids', 'labels'):
            arrays[f'pattern_{j}_{key}'] = metadata[key][selected]
        arrays[f'pattern_{j}_discovery_scalar'] = readout
    np.savez_compressed(folder/'report/core_context_profiles.npz', **arrays)
    write_json(folder/'report/audit.json', audit)
    write_json(folder/'report_complete.json', dict(status='complete', task=task,
        files={str(p.relative_to(folder)):digest(p) for p in (folder/'report').rglob('*') if p.is_file()}))
    event('cumulative_mean_profiles_complete', task=task['task'], motifs=len(rows))


def profile_report(groups, matches, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none', 'font.size':9})
    page = page_start('Motif contributions across contexts and families', ('jaspar',))
    page.append('<p>Groups use experimental context degree, not family degree: 1, ≥2, …, ≥8. '
        'Discovery averages active-context maps. Diagnostic profiles retain all eight outputs at the '
        'same native trimmed seqlet cores; overlapping cores are counted once per enhancer. '
        'Heatmaps show signed IG per covered base, equally averaged over supporting enhancers. '
        'Families average all member contexts, without an activity mask. Color scales are symmetric '
        'within each fit; use numeric data for comparisons between fits.</p>'
        '<p>Sharing is calculated per enhancer before averaging, separately for positive and negative '
        'net context contributions. Positive/negative magnitudes are separated BEFORE family averaging. '
        'Effective count = squared total / sum of squared contributions; sharing rescales this from '
        'zero (one context/family) to one (equal contributions across all). Strength below 1e-6 '
        'IG units/base is excluded from sharing, not set to zero. This is not a probability or '
        'evidence that an individual binding event acts across tissues. Distinct occurrences in '
        'one enhancer are pooled. Weak Tomtom matches remain explicitly marked.</p>')
    for group in groups:
        task = group['task']; rows = group['rows']
        page.append('<h2>'+html.escape(label(task['group']))+'</h2>')
        if not rows: page.append('<p>No motifs pass the fixed information filter.</p>'); continue
        matrix = np.asarray([r['contribution_profile']['mean_signed_context_ig_per_bp'] for r in rows])
        matrices = [matrix]+[np.asarray([r['contribution_profile']['schemes'][s]['mean_signed_family_ig_per_bp']
                    for r in rows]) for s in FAMILIES]
        names = [list(CONTEXTS)]+[list(f) for f in FAMILIES.values()]
        limit = max(float(np.abs(matrix).max()), 1e-12)
        labels = []
        for row in rows:
            match = matches[query_id(row)]
            labels.append(('+' if row['sign']=='positive' else '−')+' #'+str(row['rank'])+' '+
                          match['reference']['name']+(' ?' if match['q'] > .05 else '')+' | '+row['consensus'])
        fig, axes = plt.subplots(1, 3, figsize=(17, max(3, .35*len(rows))), layout='constrained',
            sharey=True, gridspec_kw={'width_ratios':[8, 4, 5]})
        for ax, values, titles in zip(axes, matrices, names):
            im = ax.imshow(values, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
            ax.set_xticks(range(len(titles)), titles, rotation=45, ha='right')
        axes[0].set_yticks(range(len(rows)), labels)
        fig.colorbar(im, ax=axes, label='Mean signed IG per core base', shrink=.7)
        name=f'profiles_{task["task"]:02d}.svg'; fig.savefig(output/name); plt.close(fig)
        page.append(f'<img style="width:100%" src="{name}" alt="Context and family contribution profiles">')
        page.append('<table><tr><th>Motif</th><th>Sign</th><th>Support</th><th>Mean discovery contribution/bp</th>'
            '<th>Positive family sharing (n)</th><th>Negative family sharing (n)</th></tr>')
        for row in rows:
            p = row['contribution_profile']; cells = []
            for sign in ('positive', 'negative'):
                s = p['schemes']['four_families']['signs'][sign]['between']
                cells.append(('NA' if s['mean_sharing'] is None else f'{s["mean_sharing"]:.3f}')+
                             f' ({s["eligible_enhancers"]})')
            page.append('<tr><td><a href="jaspar.html#'+query_id(row)+'">'+html.escape(row['consensus'])+
                '</a></td><td>'+row['sign']+'</td><td>'+str(row['supporting_discovery_enhancers'])+
                '</td><td>'+f'{p["mean_discovery_scalar_ig_per_bp"]:.4g}'+'</td><td>'+cells[0]+'</td><td>'+cells[1]+'</td></tr>')
        page.append('</table><p><a href="annotation/fit_'+f'{task["task"]:02d}'+
                    '/core_context_profiles.npz">Per-enhancer profiles, IDs, labels and core lengths</a></p>')
    page.append('<p><a href="audit.json">All numeric profiles, strengths, within-family and between-family sharing</a></p>')
    (output/'context_profiles.html').write_text(''.join(page)+'</body></html>')


def finalize(root, config):
    native.finalize(root, task_list=tasks())
    output = root/'output'; output.mkdir(exist_ok=False)
    groups, best, provenance = [], {}, []
    for task in tasks():
        folder = cumulative.directory(root, task)
        receipt = json.loads((folder/'report_complete.json').read_text())
        if receipt['status'] != 'complete' or receipt['task'] != task: raise ValueError('Incomplete fit report')
        for name, sha in receipt['files'].items():
            if digest(folder/name) != sha: raise ValueError('Changed fit report')
        groups.extend(json.loads((folder/'report/audit.json').read_text())['groups'])
        matches = json.loads((folder/'report/jaspar/matches.json').read_text())
        if best.keys() & matches['best'].keys(): raise ValueError('Duplicate motif query')
        best.update(matches['best']); provenance.append(dict(task=task, **{k:v for k,v in matches.items() if k != 'best'}))
        shutil.copytree(folder/'report', output/'annotation'/f'fit_{task["task"]:02d}')
    audit = dict(groups=groups, rule=RULE, contexts=list(CONTEXTS), discovery_note=
        'Experimental context degree groups: exactly1, ≥2, …, ≥8; cumulative groups overlap. '
        'Discovery uses mean IG over experimentally active contexts, with coefficients totaling one '
        'per enhancer and fixed experimental labels along the IG path. NOT a sum, family mean or predicted breadth. '
        'Degree1 reuses an exactly identical fit after index/ID/weight/hash checks; seven new fits use saved maps. '
        'Context/family profiles retain all eight outputs. No new attribution, inference, scanning or reclustering. '
        'Independent motifs across groups are not automatically the same motif; Tomtom labels do not merge them. '
        'All splits are exploratory; support is discovery-seqlet support, not scan prevalence.')
    results = dict(jaspar=dict(database=DATABASES['jaspar'], best=best, fit_provenance=provenance))
    write_json(output/'audit.json', audit); write_json(output/'jaspar_matches.json', results['jaspar'])
    write_json(output/'config.json', config)
    (output/'queries_4of5.meme').write_text(meme_queries(audit))
    for database, name in ((None, 'index.html'), ('jaspar', 'jaspar.html')):
        content = render(audit, results, database, database_keys=('jaspar',))
        content = content.replace('<nav>', '<p><a href="context_profiles.html">Motif context/family profiles and sharing</a></p><nav>', 1)
        (output/name).write_text(content)
    profile_report(groups, best, output)
    summary = dict(fits=8, new_discoveries=7, reused=1, retained=len(best),
        positive=sum(r['sign']=='positive' for g in groups for r in g['rows']),
        negative=sum(r['sign']=='negative' for g in groups for r in g['rows']))
    write_json(output/'complete.json', dict(status='complete', summary=summary,
        manifest_sha256=digest(root/'MANIFEST.sha256'),
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}))
    event('cumulative_mean_complete', summary=summary)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare','discover','finalize'))
    p.add_argument('--project', type=Path, required=True); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', type=int, default=0); args=p.parse_args(); require_allocation('cpu')
    if importlib.metadata.version('modisco') != '2.5.2': raise ValueError('Pinned TF-MoDISco required')
    config=json.loads((args.root/'config.json').read_text())
    if config['tasks'] != tasks() or config['spec'] != spec(): raise ValueError('Changed mean-active design')
    if args.stage == 'prepare': cumulative.prepare(args.project, args.root, config, task_list=tasks())
    elif args.stage == 'discover':
        task=tasks()[args.task]
        if task['low'] == 1:
            cumulative.reuse_exact_one(args.project, args.root, config, task, spec=spec(), source_task=native.tasks()[0])
        else: native.discover(args.root, config, args.task, task=task, spec=spec())
        annotate(args.project, args.root, task)
    else: finalize(args.root, config)
