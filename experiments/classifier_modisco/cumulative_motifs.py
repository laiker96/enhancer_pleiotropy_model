"""Cumulative observed-degree discovery from existing native eight-context IG."""
import argparse
from functools import partial
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

import numpy as np

from . import calibrated_motifs as native
from .calibrated_tomtom import ASSETS, RULE, render
from .common import digest, event, require_allocation, write_json
from .composite_motifs import analyze as analyze_composites, render as render_composites
from .dual_motif_pipeline import informative_core
from .simple_report import filter_native
from .tomtom_atlas import DATABASES, meme_queries, run as tomtom_run

NAME='classifier_cumulative_motifs_20260928'
PARENT='experiments/classifier_calibrated_motifs_20260927'
TARGETS=('observed_active_sum','all_context_sum')


def tasks():
    return [dict(task=i*8+k-1,target=target,group='degree_1' if k==1 else f'degree_ge_{k}',
        scheme='raw_degree',low=k,high=1 if k==1 else 8)
        for i,target in enumerate(TARGETS) for k in range(1,9)]


def directory(root,task):
    return root/'fits'/f'{task["task"]:02d}_{task["target"]}__{task["group"]}'


def prepare(project,root,config, *, task_list=None):
    parent=project/PARENT
    for name,sha in config['source_hashes'].items():
        if digest(project/name)!=sha:raise ValueError('Changed frozen source '+name)
    original=json.loads((parent/'config.json').read_text())
    if config['discovery_parameters']!=original['discovery_parameters']:
        raise ValueError('Discovery parameters changed')
    ready=native.verify_ready(parent)
    gate=json.loads((parent/'synthetic_passed.json').read_text())
    if gate['status']!='passed' or gate['manifest_sha256']!=digest(parent/'MANIFEST.sha256'):
        raise ValueError('Parent native-boundary gate invalid')
    for name,sha in ready['files'].items():
        if digest(parent/name)!=sha:raise ValueError('Native tensor changed '+name)
    (root/'native').symlink_to(parent/'native',target_is_directory=True)
    with np.load(parent/'native/metadata.npz',allow_pickle=False) as f:metadata=dict(f)
    if len(metadata['ids'])!=40338 or int(metadata['quality_pass'].all(1).sum())!=40309:
        raise ValueError('Unexpected common-QC cohort')
    counts=[]
    for task in tasks() if task_list is None else task_list:
        ix,eligible=native.selection(metadata,task,config['discovery_parameters']['seed'])
        counts.append(dict(**task,elements=len(ix),eligible=int(eligible.sum()),
            quality_excluded=int(eligible.sum()-len(ix))))
    signature=digest(root/'MANIFEST.sha256')
    write_json(root/'prepared.json',dict(status='complete',manifest_sha256=signature,files=ready['files'],
        counts=counts,parent_prepared_sha256=digest(parent/'prepared.json'),
        reuse='Original full tensor audit retained; native array checksums reverified. No new attribution.'))
    write_json(root/'synthetic_passed.json',dict(status='passed',manifest_sha256=signature,
        reused_from=str(parent/'synthetic_passed.json'),source_sha256=digest(parent/'synthetic_passed.json'),
        note='Same adapter and parameters; original two-sign synthetic recovery and boundary gate reused, not rerun.'))
    event('cumulative_prepared',counts=counts)


def reuse_exact_one(project,root,config,task, *, spec=None, source_task=None):
    if source_task is None:source_task=native.tasks()[0 if task['target']==TARGETS[0] else 3]
    source=directory(project/PARENT,source_task)
    done=json.loads((source/'complete.json').read_text())
    with np.load(root/'native/metadata.npz',allow_pickle=False) as f:metadata=dict(f)
    selected,_=native.selection(metadata,task,config['discovery_parameters']['seed'])
    with np.load(source/'examples.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['indices'],selected)
        np.testing.assert_array_equal(f['ids'],metadata['ids'][selected])
        np.testing.assert_array_equal(f['weights'],native.coefficients(metadata['labels'][selected],task['target'],spec=spec))
    if done['status']!='complete' or done['task']!=source_task:raise ValueError('Invalid reusable fit')
    dest=directory(root,task);dest.mkdir(parents=True,exist_ok=False)
    for name,sha in done['files'].items():
        if digest(source/name)!=sha:raise ValueError('Changed degree-1 fit')
        if name!='selection.json':(dest/name).symlink_to(source/name)
    selection=json.loads((source/'selection.json').read_text())
    selection.update(task);selection['reused_from']=str(source)
    if spec is not None:selection['spec']=spec
    write_json(dest/'selection.json',selection)
    write_json(dest/'complete.json',dict(status='complete',task=task,audit=done['audit'],
        files={p.name:digest(p) for p in dest.iterdir() if p.is_file()},
        modisco_version=done['modisco_version'],prepared_sha256=digest(root/'prepared.json'),
        reused_from=str(source),source_complete_sha256=digest(source/'complete.json')))
    event('cumulative_exact_one_reused',task=task['task'],source=str(source))


def annotate(project,root,task, *, spec=None, composites=True):
    folder=directory(root,task);done=json.loads((folder/'complete.json').read_text())
    selection=json.loads((folder/'selection.json').read_text())
    counts={s:done['audit'][s]['patterns'] for s in ('positive','negative')}
    rows,excluded=filter_native(folder/'motifs.h5',folder.name,counts,
        core_filter=partial(informative_core,flank_threshold=.2))
    if len(rows)+len(excluded)!=sum(counts.values()):raise ValueError('Pattern accounting mismatch')
    ordered=[]
    for sign in ('positive','negative'):
        selected=sorted((r for r in rows if r['sign']==sign),key=lambda r:(-r['supporting_discovery_enhancers'],r['id']))
        for rank,row in enumerate(selected,1):
            row.update(rank=rank,consensus=''.join('ACGT'[i] for i in np.argmax(row['trimmed_pwm'],axis=1)),
                support_fraction=row['supporting_discovery_enhancers']/selection['elements'])
        ordered.extend(selected)
    group=dict(task=task,elements=selection['elements'],rows=ordered,exclusions=excluded,
        attribution=native.targets()[task['target']] if spec is None else spec,raw_counts=counts)
    out=folder/'report';out.mkdir(exist_ok=False)
    write_json(out/'audit.json',dict(groups=[group]))
    refs=project/ASSETS/'references';binary=project/ASSETS/'bin/tomtom'
    if ordered:
        tomtom_run(out,out/'audit.json',binary,query_rule=RULE,references=refs,database_keys=('jaspar',))
    else:
        (out/'jaspar').mkdir()
        write_json(out/'jaspar/matches.json',dict(database=DATABASES['jaspar'],best={},
            reason='No passing patterns; Tomtom not run.'))
        (out/'jaspar/tomtom.tsv').write_text('# No passing patterns\n')
    if composites: analyze_composites(out/'composites',group,folder/'motifs.h5',binary,refs)
    write_json(folder/'report_complete.json',dict(status='complete',task=task,
        files={str(p.relative_to(folder)):digest(p) for p in out.rglob('*') if p.is_file()}))
    event('cumulative_fit_report_complete',task=task['task'],retained=len(ordered))


def finalize(root):
    native.finalize(root,task_list=tasks())
    output=root/'output';output.mkdir(exist_ok=False)
    groups=[];best={};pairs=[]
    provenance=[]
    for task in tasks():
        folder=directory(root,task);receipt=json.loads((folder/'report_complete.json').read_text())
        if receipt['status']!='complete' or receipt['task']!=task:raise ValueError('Incomplete report')
        for name,sha in receipt['files'].items():
            if digest(folder/name)!=sha:raise ValueError('Changed report')
        group=json.loads((folder/'report/audit.json').read_text())['groups'][0];groups.append(group)
        match=json.loads((folder/'report/jaspar/matches.json').read_text())
        if best.keys()&match['best'].keys():raise ValueError('Duplicate query')
        best.update(match['best'])
        provenance.append(dict(task=task,**{k:v for k,v in match.items() if k!='best'}))
        pairs.append(json.loads((folder/'report/composites/diagnostics.json').read_text()))
        dest=output/'annotation'/f'fit_{task["task"]:02d}'
        shutil.copytree(folder/'report',dest)
    audit=dict(groups=groups,rule=RULE,contexts=list(native.CONTEXTS),
        discovery_note='New cumulative fits reuse saved IG maps; identical degree-1 fits reused. '
            'Groups overlap (1, ≥2, …, ≥8); not independent replicates. No new gradients or extra clustering.')
    results=dict(jaspar=dict(database=DATABASES['jaspar'],best=best,fit_provenance=provenance))
    write_json(output/'audit.json',audit);write_json(output/'jaspar_matches.json',results['jaspar'])
    write_json(output/'composites.json',dict(groups=pairs))
    (output/'queries_4of5.meme').write_text(meme_queries(audit))
    for db,file in ((None,'index.html'),('jaspar','jaspar.html')):
        content=render(audit,results,db,database_keys=('jaspar',))
        content=content.replace('<nav>', '<p><a href="composites.html">Candidate paired-motif diagnostics</a></p><nav>',1)
        (output/file).write_text(content)
    (output/'composites.html').write_text(render_composites(pairs))
    summary=dict(fits=16,new_discoveries=14,reused=2,retained=len(best),
        positive=sum(r['sign']=='positive' for g in groups for r in g['rows']),
        negative=sum(r['sign']=='negative' for g in groups for r in g['rows']),
        jaspar_q_le_005=sum(m['q']<=.05 for m in best.values()),
        composite_candidates=sum(len(g['patterns']) for g in pairs))
    write_json(output/'complete.json',dict(status='complete',summary=summary,
        manifest_sha256=digest(root/'MANIFEST.sha256'),
        files={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    event('cumulative_complete',summary=summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','discover','finalize'))
    p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',type=int,default=0);args=p.parse_args();require_allocation('cpu')
    if importlib.metadata.version('modisco')!='2.5.2':raise ValueError('Use pinned TF-MoDISco 2.5.2')
    config=json.loads((args.root/'config.json').read_text())
    if config['tasks']!=tasks():raise ValueError('Wrong cumulative task contract')
    if args.stage=='prepare':prepare(args.project,args.root,config)
    elif args.stage=='discover':
        task=tasks()[args.task]
        if task['low']==1:reuse_exact_one(args.project,args.root,config,task)
        else:native.discover(args.root,config,args.task,task=task)
        annotate(args.project,args.root,task)
    else:finalize(args.root)
