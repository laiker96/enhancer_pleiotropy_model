"""Sixteen observed-activity-masked discoveries from saved calibrated IG.

Eight individual contexts, four active-member family means, four EXACT family
breadths with a mean of active-family means. Never recompute attribution.
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
from .calibrated_tomtom import RULE, render
from .common import digest, event, require_allocation, write_json
from .family_breadth import CONTEXTS, FAMILIES
from .hierarchical_motifs import core_occurrences
from .motif_sharing import enhancer_means
from .tomtom_atlas import DATABASES, meme_queries

NAME = 'classifier_masked_motifs_20260929'
SCHEME = 'four_families'


def tasks():
    result = []
    for context in CONTEXTS:
        result.append(dict(task=len(result), target='active_context_'+context,
            group='enhancers_active_in_'+context, scheme='observed_contexts',
            contexts=[context], scope='context'))
    for family, members in FAMILIES[SCHEME].items():
        result.append(dict(task=len(result), target='active_member_mean_'+family,
            group='enhancers_active_in_'+family, scheme='observed_contexts',
            contexts=list(members), scope='family'))
    for degree in range(1,5):
        result.append(dict(task=len(result), target='mean_active_family_means',
            group='exactly_'+str(degree)+'_active_families', scheme=SCHEME,
            family_degree=True, low=degree, high=degree, scope='breadth'))
    return result


def specs():
    out = {}
    for task in tasks():
        if task['scope'] == 'breadth':
            out[task['target']] = dict(weights=[1.]*8, observed_mask=False,
                active_member_scheme=SCHEME, mean_active_families=True)
        else:
            out[task['target']] = dict(weights=[float(c in task['contexts']) for c in CONTEXTS],
                                      observed_mask=True, masked_mean=True)
    return out


def validate_task(metadata, task, seed):
    ix, eligible = native.selection(metadata, task, seed)
    weights = native.coefficients(metadata['labels'][ix], task['target'], spec=specs()[task['target']])
    if not len(ix): raise ValueError('Empty masked discovery cohort')
    np.testing.assert_allclose(weights.sum(1), 1, atol=1e-14, rtol=0)
    if (weights[metadata['labels'][ix] == 0] != 0).any():
        raise ValueError('An inactive context received weight')
    return ix, eligible, weights


def annotate(project, root, task):
    cumulative.annotate(project, root, task, spec=specs()[task['target']], composites=False)
    folder = cumulative.directory(root, task)
    with np.load(root/'native/metadata.npz', allow_pickle=False) as z: meta = dict(z)
    with np.load(folder/'examples.npz', allow_pickle=False) as z: selected = dict(z)
    config = json.loads((root/'config.json').read_text())
    ix, _, weights = validate_task(meta,task,config['discovery_parameters']['seed'])
    np.testing.assert_array_equal(ix, selected['indices'])
    np.testing.assert_allclose(weights, selected['weights'],atol=0,rtol=0)
    out = folder/'report'
    audit = json.loads((out/'audit.json').read_text())
    rows = audit['groups'][0]['rows']
    actual = np.load(root/'native/actual.npy',mmap_mode='r',allow_pickle=False)
    profiles = core_occurrences(folder/'motifs.h5',rows,ix,meta,actual)
    arrays = {}; summaries = []
    for j,(row,profile) in enumerate(zip(rows,profiles)):
        indices = profile['indices']; widths = profile['end']-profile['start']
        w = native.coefficients(meta['labels'][indices],task['target'],spec=specs()[task['target']])
        target = (profile['context_signed_sum']*w).sum(1)
        profile.update(discovery_weights=w,masked_target_signed_sum=target,
                       masked_target_signed_per_bp=target/widths)
        arrays.update({f'pattern_{j}_{k}':v for k,v in profile.items()})
        keep = profile['selected']
        per_enhancer,_ = enhancer_means(indices[keep],
            np.column_stack([target[keep],target[keep]/widths[keep],
                             profile['context_signed_sum'][keep]/widths[keep,None]]),len(meta['ids']))
        values = per_enhancer[np.unique(indices[keep])]
        summaries.append(dict(id=row['id'],pattern_index=j,sign=row['sign'],
            carriers=len(values),discovery_enhancers=len(ix),raw_seqlets=len(indices),
            nonoverlap_cores=int(keep.sum()),
            mean_masked_ig_per_core=float(values[:,0].mean()),
            mean_masked_ig_per_bp=float(values[:,1].mean()),
            mean_unmasked_context_ig_per_bp=values[:,2:].mean(0).tolist(),
            interpretation='Mean across nonoverlapping sites within enhancer, then equal enhancer weights; carriers only'))
    np.savez_compressed(out/'occurrences.npz',**arrays)
    np.savez_compressed(out/'enhancers.npz',indices=ix,ids=meta['ids'][ix],labels=meta['labels'][ix],
        split=meta['split'][ix],weights=weights,calibrated_probabilities=meta['calibrated_probabilities'][ix])
    write_json(out/'motif_summary.json',dict(task=task,patterns=summaries,contexts=list(CONTEXTS)))
    write_json(folder/'report_complete.json',dict(status='complete',task=task,
        files={str(p.relative_to(folder)):digest(p) for p in out.rglob('*') if p.is_file()}))
    event('masked_fit_annotated',task=task['task'],patterns=len(rows),enhancers=len(ix))


def finalize(root, *, task_list=None, cumulative_breadth=False, run_summary=None):
    task_list=tasks() if task_list is None else task_list
    native.finalize(root,task_list=task_list)
    output = root/'output'; output.mkdir(exist_ok=False)
    groups=[];best={};provenance=[];summaries=[]
    for task in task_list:
        folder=cumulative.directory(root,task)
        receipt=json.loads((folder/'report_complete.json').read_text())
        if receipt['status']!='complete' or receipt['task']!=task: raise ValueError('Incomplete masked fit')
        for name,sha in receipt['files'].items():
            if digest(folder/name)!=sha: raise ValueError('Changed fit output '+name)
        group=json.loads((folder/'report/audit.json').read_text())['groups'][0];groups.append(group)
        match=json.loads((folder/'report/jaspar/matches.json').read_text())
        if best.keys() & match['best'].keys(): raise ValueError('Duplicate motif identifiers')
        best.update(match['best'])
        provenance.append(dict(task=task,**{k:v for k,v in match.items() if k!='best'}))
        summaries.append(json.loads((folder/'report/motif_summary.json').read_text()))
        dest=output/'annotation'/f'fit_{task["task"]:02d}'
        shutil.copytree(folder/'report',dest)
        single=dict(jaspar=match)
        (dest/'jaspar.html').write_text(render(dict(groups=[group],rule=RULE),single,'jaspar',database_keys=('jaspar',)))
    breadth_note=('cumulative family-breadth groups: exactly 1, >=2, >=3, 4. The >= groups overlap and are not independent'
                  if cumulative_breadth else 'exact family-breadth groups')
    audit=dict(groups=groups,rule=RULE,contexts=list(CONTEXTS),
        discovery_note='Observed-activity-masked context and family discovery; '+breadth_note+'. '
                       'All selected weights sum to one. All splits exploratory. No new attribution or reclustering.')
    result=dict(database=DATABASES['jaspar'],best=best,fit_provenance=provenance)
    write_json(output/'audit.json',audit);write_json(output/'jaspar_matches.json',result)
    write_json(output/'motif_summaries.json',dict(fits=summaries))
    (output/'queries_4of5.meme').write_text(meme_queries(audit))
    (output/'jaspar.html').write_text(render(audit,dict(jaspar=result),'jaspar',database_keys=('jaspar',)))
    page=['<!doctype html><html><meta charset="utf-8"><title>Masked context and family motifs</title>',
          '<style>body{font:15px system-ui;max-width:1300px;margin:30px auto}table{border-collapse:collapse;width:100%}',
          'td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style>',
          '<h1>Observed-activity-masked motif discovery</h1><p>8 contexts; 4 active-member family means; '
          +breadth_note+'. No inactive output contributes. All maps have total weight one.</p>',
          '<p>All 40,309 common-QC enhancers are eligible; each fit uses its applicable subset. '
          'Exploratory all-split analysis. Counts are selected discovery-seqlet support, not exhaustive PWM prevalence. '
          'Independent fits and seqlet caps can affect support; raw counts are not comparable enrichment tests.</p>',
          '<p><a href="jaspar.html">All aligned JASPAR logos</a></p>',
          '<table><tr><th>Scope</th><th>Target / cohort</th><th>Enhancers</th><th>Positive / negative</th></tr>']
    for group in groups:
        task=group['task'];rows=group['rows']
        page.append(f'<tr><td>{html.escape(task["scope"])}</td><td><a href="annotation/fit_{task["task"]:02d}/jaspar.html">'
            f'{html.escape(task["target"])} / {html.escape(task["group"])}</a></td><td>{group["elements"]:,}</td>'
            f'<td>{sum(r["sign"]=="positive" for r in rows)} / {sum(r["sign"]=="negative" for r in rows)}</td></tr>')
    page.append('</table>')
    for group,summary in zip(groups,summaries):
        page.append('<h2>'+html.escape(group['task']['target']+' / '+group['task']['group'])+'</h2><table>'
            '<tr><th>Pattern</th><th>Sign</th><th>Best JASPAR match</th><th>q</th><th>Carriers</th>'
            '<th>Support %</th><th>Seqlets</th><th>Mean masked IG/base</th></tr>')
        stats={s['id']:s for s in summary['patterns']}
        for row in group['rows']:
            hit=best[row['id'].replace('/','__')];stat=stats[row['id']]
            page.append(f'<tr><td>{html.escape(row["pattern"])}</td><td>{row["sign"]}</td>'
                f'<td>{html.escape(hit["reference"]["name"])}'+(' (ns)' if hit['q']>.05 else '')+
                f'</td><td>{hit["q"]:.3g}</td><td>{stat["carriers"]}</td><td>{100*row["support_fraction"]:.2f}</td>'
                f'<td>{stat["raw_seqlets"]}</td><td>{stat["mean_masked_ig_per_bp"]:.5g}</td></tr>')
        page.append('</table>')
    (output/'index.html').write_text(''.join(page)+'</html>')
    summary=dict(fits=16,new_discoveries=15,reused_ovary=1,retained=len(best),positive=sum(r['sign']=='positive' for g in groups for r in g['rows']),
                 negative=sum(r['sign']=='negative' for g in groups for r in g['rows']),
                 jaspar_q_le_005=sum(m['q']<=.05 for m in best.values()))
    if run_summary is not None:summary.update(run_summary)
    write_json(output/'complete.json',dict(status='complete',summary=summary,
        manifest_sha256=digest(root/'MANIFEST.sha256'),
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}))
    event('masked_motifs_complete',summary=summary)


def reuse_ovary(project,root,config):
    source_task=next(t for t in tasks() if t['target']=='active_context_o')
    task=next(t for t in tasks() if t['target']=='active_member_mean_ovary')
    source=cumulative.directory(root,source_task);dest=cumulative.directory(root,task)
    done=json.loads((source/'complete.json').read_text())
    if done['status']!='complete' or done['task']!=source_task:raise ValueError('Incomplete ovary source')
    with np.load(root/'native/metadata.npz',allow_pickle=False) as z:meta=dict(z)
    ix,_,weights=validate_task(meta,task,config['discovery_parameters']['seed'])
    with np.load(source/'examples.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'],ix)
        np.testing.assert_array_equal(z['weights'],weights)
    dest.mkdir(parents=True,exist_ok=False)
    for name,sha in done['files'].items():
        if digest(source/name)!=sha:raise ValueError('Changed ovary source')
        if name!='selection.json':(dest/name).symlink_to(source/name)
    selection=json.loads((source/'selection.json').read_text())
    selection.update(task);selection.update(spec=specs()[task['target']],reused_from=str(source))
    write_json(dest/'selection.json',selection)
    write_json(dest/'complete.json',dict(status='complete',task=task,audit=done['audit'],
        modisco_version=done['modisco_version'],prepared_sha256=digest(root/'prepared.json'),
        reused_from=str(source),source_complete_sha256=digest(source/'complete.json'),
        files={p.name:digest(p) for p in dest.iterdir() if p.is_file()}))
    annotate(project,root,task)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','discover','finalize'))
    p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',type=int,default=0);args=p.parse_args();require_allocation('cpu')
    if importlib.metadata.version('modisco')!='2.5.2':raise ValueError('Use pinned TF-MoDISco 2.5.2')
    config=json.loads((args.root/'config.json').read_text())
    if config['tasks']!=tasks() or config['specs']!=specs():raise ValueError('Wrong masked analysis contract')
    if args.stage=='prepare':
        cumulative.prepare(args.project,args.root,config,task_list=tasks())
        with np.load(args.root/'native/metadata.npz',allow_pickle=False) as z: meta=dict(z)
        for task in tasks():validate_task(meta,task,config['discovery_parameters']['seed'])
    elif args.stage=='discover':
        task=tasks()[args.task]
        native.discover(args.root,config,args.task,task=task,spec=specs()[task['target']])
        annotate(args.project,args.root,task)
    else:
        reuse_ovary(args.project,args.root,config)
        finalize(args.root)
