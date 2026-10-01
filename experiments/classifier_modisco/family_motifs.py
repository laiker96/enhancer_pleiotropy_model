"""Family-degree TF-MoDISco, using only saved calibrated-probability IG maps."""
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
from .family_breadth import CONTEXTS, FAMILIES, grouping
from .tomtom_atlas import DATABASES, meme_queries

NAME = 'classifier_family_active_members_retry1_20260929'
PREVIOUS = 'experiments/classifier_family_motifs_20260928'
PARENT = cumulative.PARENT


def specs():
    result = {}
    for scheme in FAMILIES:
        weights = sum(native.family_weights(scheme).values()).tolist()
        result['balanced_'+scheme] = dict(weights=weights, observed_mask=False)
        result['active_member_means_'+scheme] = dict(weights=[1.]*8, observed_mask=False,
                                                    active_member_scheme=scheme)
    return result


def tasks():
    result = []
    for scheme, families in FAMILIES.items():
        for target in ('balanced_'+scheme, 'active_member_means_'+scheme):
            for degree in range(1, len(families)+1):
                result.append(dict(task=len(result), target=target, scheme=scheme, family_degree=True,
                    group='family_degree_1' if degree == 1 else 'family_degree_ge_'+str(degree),
                    low=degree, high=1 if degree == 1 else len(families)))
    return result


def reuse_task(task):
    """Reuse only fits whose selected enhancers AND weights are verified identical."""
    if task['target'].startswith('balanced_'):
        return PREVIOUS, task
    return None


def unfinished_reuse_jobs(array_job, accounting):
    """Do not depend on completed tasks already purged from Slurm's controller."""
    states = {}
    for line in accounting.splitlines():
        job, state, exit_code = line.strip().split('|')[:3]
        if job in states: raise ValueError('Duplicate accounting job '+job)
        states[job] = state, exit_code
    pending = []
    for task in tasks():
        if not reuse_task(task): continue
        job = str(array_job)+'_'+str(task['task'])
        state, exit_code = states[job]  # Missing accounting must fail closed.
        if state == 'COMPLETED' and exit_code == '0:0': continue
        if state not in ('PENDING', 'RUNNING', 'COMPLETING'):
            raise ValueError('Unsuccessful reuse source '+job+': '+state+' '+exit_code)
        pending.append(job)
    return pending


def family_metadata(metadata):
    quality = metadata['quality_pass'].all(1)
    result = {}
    for scheme in FAMILIES:
        g = grouping(metadata['labels'], scheme=scheme)
        counts = []
        for degree in range(1, len(FAMILIES[scheme])+1):
            mask = g['breadth'] == 1 if degree == 1 else g['breadth'] >= degree
            counts.append(dict(group='family_degree_1' if degree == 1 else 'family_degree_ge_'+str(degree),
                all_enhancers=int(mask.sum()), qc_enhancers=int((mask & quality).sum())))
        # JSON uses lists, so metadata must compare identically after loading.
        result[scheme] = dict(families={name:list(members) for name,members in FAMILIES[scheme].items()}, counts=counts,
            raw_by_family_counts=[[int(((g['raw'] == r) & (g['breadth'] == f) & quality).sum())
                                  for f in range(1, len(FAMILIES[scheme])+1)] for r in range(1, 9)])
    return result


def reuse(project, root, config, task, source_spec, *, copy_report=False):
    source_root, source_task = source_spec
    source = cumulative.directory(project/source_root, source_task)
    done = json.loads((source/'complete.json').read_text())
    if done['status'] != 'complete' or done['task'] != source_task:
        raise ValueError('Incomplete reuse source')
    with np.load(root/'native/metadata.npz', allow_pickle=False) as z: metadata = dict(z)
    ix, _ = native.selection(metadata, task, config['discovery_parameters']['seed'])
    weights = native.coefficients(metadata['labels'][ix], task['target'], spec=specs()[task['target']])
    with np.load(source/'examples.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'], ix)
        np.testing.assert_array_equal(z['ids'], metadata['ids'][ix])
        np.testing.assert_array_equal(z['weights'], weights)
    dest = cumulative.directory(root, task); dest.mkdir(parents=True, exist_ok=False)
    for name, sha in done['files'].items():
        if digest(source/name) != sha: raise ValueError('Changed reuse file '+name)
        if name != 'selection.json': (dest/name).symlink_to(source/name)
    selection = json.loads((source/'selection.json').read_text())
    selection.update(task); selection.update(spec=specs()[task['target']], reused_from=str(source))
    write_json(dest/'selection.json', selection)
    write_json(dest/'complete.json', dict(status='complete', task=task, audit=done['audit'],
        files={p.name:digest(p) for p in dest.iterdir() if p.is_file()},
        modisco_version=done['modisco_version'], prepared_sha256=digest(root/'prepared.json'),
        reused_from=str(source), source_complete_sha256=digest(source/'complete.json')))
    if copy_report:
        receipt=json.loads((source/'report_complete.json').read_text())
        if receipt['status'] != 'complete' or receipt['task'] != source_task:
            raise ValueError('Incomplete reusable annotation')
        for name,sha in receipt['files'].items():
            if digest(source/name) != sha: raise ValueError('Changed reusable annotation')
        shutil.copytree(source/'report',dest/'report')
        for name,sha in receipt['files'].items():
            if digest(dest/name) != sha: raise ValueError('Copied annotation checksum mismatch')
        write_json(dest/'report_complete.json',dict(receipt,reused_from=str(source)))
    event('family_fit_reused', task=task['task'], source=str(source))


def core_profiles(path, rows, indices, metadata, actual):
    """Map trimmed pattern columns back to native coordinates; union per enhancer."""
    import h5py
    profiles = []
    with h5py.File(path, 'r') as h5:
        for row in rows:
            node = h5[row['pattern']+'/seqlets']; a, b = row['quality']['start'], row['quality']['end']
            masks = {}
            for j, (local, start, end, rc) in enumerate(zip(node['example_idx'][:], node['start'][:],
                                                          node['end'][:], node['is_revcomp'][:])):
                local, start, end = int(local), int(start), int(end)
                if not 0 <= local < len(indices): raise ValueError('Invalid seqlet example')
                i = int(indices[local]); length = int(metadata['length'][i])
                if not 0 <= start < end <= length or end-start != len(row['full_pwm']):
                    raise ValueError('Seqlet outside native bounds or misaligned with PWM')
                if not 0 <= a < b <= end-start: raise ValueError('Invalid trimmed core')
                sequence = np.eye(4)[metadata['sequence'][i, start:end]]
                if rc: sequence = sequence[::-1, ::-1]
                np.testing.assert_array_equal(sequence[a:b], node['sequence'][j, a:b])
                left, right = (end-b, end-a) if rc else (start+a, start+b)
                masks.setdefault(i, np.zeros(length, bool))[left:right] = True
            selected = np.asarray(sorted(masks), dtype=int)
            values = np.asarray([actual[i, :, :len(masks[i])][:, masks[i]].mean(1, dtype=np.float64)
                                 for i in selected])
            if len(selected) != row['supporting_discovery_enhancers'] or not np.isfinite(values).all():
                raise ValueError('Core support/profile mismatch')
            profiles.append(dict(id=row['id'], pattern=row['pattern'], sign=row['sign'], rank=row['rank'],
                indices=selected, values=values, covered_bp=np.asarray([masks[i].sum() for i in selected]),
                mean_signed_ig_per_bp=values.mean(0).tolist(),
                positive_fraction=(values > 0).mean(0).tolist()))
    return profiles


def annotate(project, root, task):
    cumulative.annotate(project, root, task, spec=specs()[task['target']], composites=False)
    folder = cumulative.directory(root, task)
    audit = json.loads((folder/'report/audit.json').read_text()); group = audit['groups'][0]
    with np.load(folder/'examples.npz', allow_pickle=False) as z: indices = z['indices']
    with np.load(root/'native/metadata.npz', allow_pickle=False) as z: metadata = dict(z)
    actual = np.load(root/'native/actual.npy', mmap_mode='r', allow_pickle=False)
    profiles = core_profiles(folder/'motifs.h5', group['rows'], indices, metadata, actual)
    arrays = {}
    for j, (row, profile) in enumerate(zip(group['rows'], profiles)):
        for key in ('indices', 'values', 'covered_bp'): arrays[f'pattern_{j}_{key}'] = profile.pop(key)
        row['core_context_profile'] = dict(**profile, array_prefix=f'pattern_{j}',
            averaging='Union trimmed seqlet-core bases per enhancer; mean per base, then equal enhancer means. All eight outputs, no active mask.')
    np.savez_compressed(folder/'report/core_context_profiles.npz', **arrays)
    write_json(folder/'report/audit.json', audit)
    write_json(folder/'report_complete.json', dict(status='complete', task=task,
        files={str(p.relative_to(folder)):digest(p) for p in (folder/'report').rglob('*') if p.is_file()}))


def profile_report(groups, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none', 'font.size':9})
    page = page_start('De novo motif contributions by context and family', ('jaspar',))
    page.append('<p>Profiles use the actual trimmed cores of discovery seqlets, not rescanned JASPAR sites. '
        'Overlapping cores are united within each enhancer; signed IG is averaged per covered base and then '
        'equally across supporting enhancers. No observed-active mask is applied to this diagnostic. '
        'Each fit uses one symmetric color scale across both signs; compare numeric values across fits. '
        'A population mean is not proof that each motif occurrence contributes in every family.</p>')
    for group in groups:
        task = group['task']; rows = group['rows']
        page.append('<h2>'+html.escape(label(task['target'])+' / '+label(task['group']))+'</h2>')
        if not rows: page.append('<p>No retained patterns.</p>'); continue
        matrix = np.asarray([r['core_context_profile']['mean_signed_ig_per_bp'] for r in rows])
        families = FAMILIES[task['scheme']]
        family = np.column_stack([matrix[:, [CONTEXTS.index(c) for c in members]].mean(1)
                                  for members in families.values()])
        limit = max(float(np.abs(matrix).max()), 1e-12)
        fig, axes = plt.subplots(1, 2, figsize=(12, max(3, .33*len(rows))), layout='constrained',
                                 sharey=True, gridspec_kw={'width_ratios':[8, len(families)]})
        for ax, values, names in zip(axes, (matrix, family), (CONTEXTS, list(families))):
            im = ax.imshow(values, vmin=-limit, vmax=limit, cmap='RdBu_r', aspect='auto')
            ax.set_xticks(range(len(names)), names, rotation=45, ha='right')
        axes[0].set_yticks(range(len(rows)), [r['sign'][0]+' #'+str(r['rank'])+' '+r['consensus'] for r in rows])
        fig.colorbar(im, ax=axes, label='Mean signed IG per core base', shrink=.7)
        name=f'profiles_{task["task"]:02d}.svg'; fig.savefig(output/name); plt.close(fig)
        page.append(f'<img style="width:100%" src="{name}" alt="Context and family contributions">')
    (output/'context_profiles.html').write_text(''.join(page)+'</body></html>')


def finalize(root, config):
    native.finalize(root, task_list=tasks())
    output = root/'output'; output.mkdir(exist_ok=False)
    groups, best, provenance = [], {}, []
    for task in tasks():
        folder = cumulative.directory(root, task)
        receipt = json.loads((folder/'report_complete.json').read_text())
        if receipt['status'] != 'complete' or receipt['task'] != task: raise ValueError('Incomplete annotation')
        for name, sha in receipt['files'].items():
            if digest(folder/name) != sha: raise ValueError('Changed fit annotation')
        groups.extend(json.loads((folder/'report/audit.json').read_text())['groups'])
        matches = json.loads((folder/'report/jaspar/matches.json').read_text())
        if best.keys() & matches['best'].keys(): raise ValueError('Duplicate motif query')
        best.update(matches['best']); provenance.append(dict(task=task, **{k:v for k,v in matches.items() if k != 'best'}))
        shutil.copytree(folder/'report', output/'annotation'/f'fit_{task["task"]:02d}')
    audit = dict(groups=groups, rule=RULE, contexts=list(CONTEXTS), discovery_note=
        'Experimental family label = any observed active member. Groups are exactly one, ≥2, ≥3, ≥4 families; '
        'adult/larval-brain-separated sensitivity adds ≥5. One-family is not necessarily one-context. '
        'Family readout = mean member probability, NOT probability of any activity or calibrated family breadth. '
        'All-family branch averages all members. Corrected observed-active branch averages ONLY experimentally '
        'active members within each family, then sums family means; inactive contexts and inactive families '
        'have zero weight. Weights are fixed along the IG path. No new attribution, inference, scanning, '
        'or extra clustering. Nine unchanged all-family fits are reused; nine corrected fits are new. '
        'The former whole-family-mask branch is superseded and excluded. Cumulative groups overlap.')
    results = dict(jaspar=dict(database=DATABASES['jaspar'], best=best, fit_provenance=provenance))
    write_json(output/'audit.json', audit); write_json(output/'jaspar_matches.json', results['jaspar'])
    write_json(output/'config.json', config)
    shutil.copy2(root/'family_groups.json', output/'family_groups.json')
    shutil.copy2(root/'sharing_reuse.json', output/'sharing_reuse.json')
    (output/'queries_4of5.meme').write_text(meme_queries(audit))
    for database, name in ((None, 'index.html'), ('jaspar', 'jaspar.html')):
        content = render(audit, results, database, database_keys=('jaspar',))
        content = content.replace('<nav>', '<p><a href="context_profiles.html">Actual motif-core context profiles</a> · '
            '<a href="family_groups.json">Family definitions and counts</a> · '
            '<a href="sharing/index.html">Fixed-PWM comparisons and same-context-count checks</a></p><nav>', 1)
        (output/name).write_text(content)
    profile_report(groups, output)
    shutil.copytree(root/'sharing', output/'sharing')
    summary = dict(fits=len(groups), new_discoveries=9, reused=9, retained=len(best),
        positive=sum(r['sign']=='positive' for g in groups for r in g['rows']),
        negative=sum(r['sign']=='negative' for g in groups for r in g['rows']),
        jaspar_q_le_005=sum(m['q']<=.05 for m in best.values()))
    write_json(output/'complete.json', dict(status='complete', manifest_sha256=digest(root/'MANIFEST.sha256'),
        summary=summary, files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}))
    event('family_motifs_complete', summary=summary)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare','discover','finalize'))
    parser.add_argument('--project', type=Path, required=True); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=int, default=0); args=parser.parse_args(); require_allocation('cpu')
    if importlib.metadata.version('modisco') != '2.5.2': raise ValueError('Pinned TF-MoDISco required')
    config=json.loads((args.root/'config.json').read_text())
    if config['tasks'] != tasks() or config['specs'] != specs(): raise ValueError('Changed family design')
    if args.stage == 'prepare':
        cumulative.prepare(args.project, args.root, config, task_list=tasks())
        with np.load(args.root/'native/metadata.npz', allow_pickle=False) as z: metadata=dict(z)
        write_json(args.root/'family_groups.json', family_metadata(metadata))
        previous=args.project/PREVIOUS
        if digest(previous/'MANIFEST.sha256') != config['previous_manifest_sha256']:
            raise ValueError('Changed previous family package')
        native.verify_ready(previous)
        if json.loads((previous/'family_groups.json').read_text()) != family_metadata(metadata):
            raise ValueError('Family metadata changed')
        hashes={str(p.relative_to(previous/'sharing')):digest(p) for p in (previous/'sharing').rglob('*') if p.is_file()}
        if 'report.json' not in hashes or 'index.html' not in hashes: raise ValueError('Missing cached sharing report')
        shutil.copytree(previous/'sharing',args.root/'sharing')
        for name,sha in hashes.items():
            if digest(args.root/'sharing'/name) != sha: raise ValueError('Changed copied sharing output')
        write_json(args.root/'sharing_reuse.json',dict(source=str(previous/'sharing'),files=hashes))
        event('family_active_members_prepared',counts=family_metadata(metadata),no_new_attributions=True)
    elif args.stage == 'discover':
        task=tasks()[args.task]; source=reuse_task(task)
        if source: reuse(args.project, args.root, config, task, source)
        else: native.discover(args.root, config, args.task, task=task, spec=specs()[task['target']])
        annotate(args.project, args.root, task)
    else:
        for task in tasks():
            source=reuse_task(task)
            if source:
                reuse(args.project, args.root, config, task, source, copy_report=True)
        finalize(args.root, config)
