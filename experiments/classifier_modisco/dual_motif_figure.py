"""New IG64/100-reference Figure B-E, including a supported Grh example.

Run selection and rendering in a CECAR CPU allocation. The local collector
renders the completed PDF to PNG for mandatory human visual inspection.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .common import digest, event, require_allocation, write_json
from .paper_figure3 import check_text_bounds
from .paper_figure3_examples import ExampleFigure, display_tracks, validate_site, zoom_interval
from .paper_figure3_nature import CONTEXTS
from .attribution_example_selection import choose_configuration, select_clear_example


def select_rows(audit, matches, limit=4):
    rows=[]
    for group in audit['groups']:
        positives=sorted([r for r in group['rows'] if r['sign']=='positive'],key=lambda r:r['rank'])[:limit]
        negatives=sorted([r for r in group['rows'] if r['sign']=='negative'],key=lambda r:r['rank'])
        for row in positives+negatives:
            support=row['attribution_support']
            if not (support['n']==group['discovery_elements'] and 0<=support['hits']<=support['n']
                    and np.isclose(support['fraction'],support['hits']/support['n'])):
                raise ValueError('Invalid motif support')
            rows.append(dict(group=group['name'],discovery_n=group['discovery_elements'],**row,
                match=matches['best'][row['id'].replace('/','__')]))
    return rows


def select_examples(metadata, actual, sites, audit):
    """Same contrast gates as previous examples, TF-neutral within each stratum."""
    breadth=metadata['labels'].sum(1)
    def candidates(kind, degree=None, motif=None, excluded_contexts=()):
        result=[]
        for index,annotations in sites.items():
            if (metadata['split'][index]!='train' or not metadata['quality_pass'][index,0]
                    or (degree is not None and breadth[index]!=degree)
                    or (kind=='grh' and breadth[index]<2)):
                continue
            contexts=[c for c,active in zip(CONTEXTS,metadata['labels'][index]) if active]
            if len(contexts)==1 and contexts[0] in excluded_contexts:continue
            length=int(metadata['length'][index]);values=np.asarray(actual[index,0,:length])
            if kind=='grh':
                chosen=[s for s in annotations if s['best_match']=='grh' and s['tomtom_q']<.05]
                configs=[choose_configuration(values,[s],s['motif_id'].split('/')[-1],zoom_interval) for s in chosen]
                configs=[c for c in configs if c]
                config=min(configs,key=lambda c:(-c['contrast']['score'],c['anchor']['motif_id'])) if configs else None
            elif kind=='specific':
                chosen=[s for s in annotations if s['motif_id']==motif]
                config=choose_configuration(values,chosen,motif.split('/')[-1],zoom_interval)
            else:
                config=choose_configuration(values,annotations,kind,zoom_interval)
            if config is None:continue
            sequence=''.join('ACGT'[c] for c in metadata['sequence'][index,:length])
            result.append(dict(index=index,id=str(metadata['ids'][index]),degree=int(breadth[index]),
                active_contexts=contexts,length=length,sequence=sequence,actual_ig=values.tolist(),**config))
        return result
    specific=[];used_contexts=set();used_names=set()
    group=next(g for g in audit['groups'] if g['name']=='exact_1')
    for motif in sorted([r for r in group['rows'] if r['sign']=='positive'],key=lambda r:r['rank']):
        pool=candidates('specific',degree=1,motif=motif['id'],excluded_contexts=used_contexts)
        if not pool:continue
        choice=select_clear_example(pool)
        label=choice['anchor']['display_label']
        if label in used_names:continue
        specific.append(choice);used_contexts.update(choice['active_contexts']);used_names.add(label)
        if len(specific)==3:break
    if len(specific)!=3:
        raise ValueError('Fewer than three clear, distinct context-specific illustrations; review candidates')
    pools={'Trl':candidates('Trl',degree=2),'grh':candidates('grh'),'Trl+cg':candidates('Trl+cg',degree=8)}
    if any(not pool for pool in pools.values()):
        raise ValueError('Missing requested high-contrast example: '+','.join(k for k,v in pools.items() if not v))
    pleiotropic=[select_clear_example(pools[k]) for k in ('Trl','grh','Trl+cg')]
    examples=[value for pair in zip(specific,pleiotropic) for value in pair]
    if len({e['id'] for e in examples})!=len(examples):raise ValueError('Repeated illustrative enhancer')
    return dict(examples=examples,references=100,steps=64,input_bp=2048,diverse_examples=True,
        method='Integrated Gradients',target='Mean forward/RC logit over observed-active contexts',
        selection=dict(specific='Three top-support positive native motifs admitting clear examples in distinct sole-active contexts',
            pleiotropic='Trl at exact degree2; Grh at any observed degree2-8; proximal nonoverlapping Trl/cg at exact degree8',
            gates='Same prior thresholds: site mean>=0.04 logit/base, >=70% positive bases, >=3x native and >=2x local absolute-background contrast',
            quantile=.9,site_p=1e-4,required_TF_q=.05,
            interpretation='Illustrative, motif-oriented examples; not representative sampling, TF occupancy or cooperativity'))


def run(project,root,extension):
    require_allocation('cpu')
    config=json.loads((root/'config.json').read_text())
    result=root/'mean_active_logit'
    complete=json.loads((result/'report_complete.json').read_text())
    for name,checksum in complete['files'].items():
        if digest(result/name)!=checksum:raise ValueError('Changed completed motif report')
    audit=json.loads((result/'report_audit.json').read_text())
    matches=json.loads((result/'annotation/jaspar/matches.json').read_text())
    original_audit=json.loads((root/'audit_complete.json').read_text())
    for name in ['metadata.npz','native_actual.npy']:
        if digest(root/name)!=original_audit['files'][name]:raise ValueError('Changed attribution assembly')
    with np.load(root/'metadata.npz',allow_pickle=False) as f:metadata=dict(f)
    actual=np.load(root/'native_actual.npy',mmap_mode='r',allow_pickle=False)
    rows=select_rows(audit,matches)
    motif_lookup={r['id'].replace('/','__'):r for g in audit['groups'] for r in g['rows'] if r['sign']=='positive'}
    sites={}
    for group in audit['groups']:
        path=result/'scans'/group['name']/'hits.tsv.gz'
        if not path.exists():continue
        with gzip.open(path,'rt') as stream:
            for hit in csv.DictReader((line for line in stream if not line.startswith('#')),delimiter='\t'):
                if hit['motif_id'] not in motif_lookup:continue
                index=int(hit['sequence_name'][1:])
                if metadata['split'][index]!='train' or not metadata['quality_pass'][index,0]:continue
                if float(hit['p-value'])>1e-4:raise ValueError('Unexpected scan threshold')
                row=motif_lookup[hit['motif_id']];match=matches['best'][hit['motif_id']]
                length=int(metadata['length'][index])
                sequence=''.join('ACGT'[c] for c in metadata['sequence'][index,:length])
                start,end=validate_site(sequence,hit)
                name=match['reference']['name']
                if match['q']<.05:label=('GAF / Trl' if name=='Trl' else 'Grh' if name=='grh' else name)+'-like'
                else:
                    pwm=np.asarray(row['trimmed_pwm']);consensus=''.join('ACGT'[c] for c in pwm.argmax(1))
                    label=consensus+' motif'
                sites.setdefault(index,[]).append(dict(motif_id=row['id'],start=start,end=end,strand=hit['strand'],
                    site_p=float(hit['p-value']),mean_ig=float(actual[index,0,start:end].mean()),
                    best_match=name,tomtom_q=match['q'],display_label=label))
    examples=select_examples(metadata,actual,sites,audit)
    for e in examples['examples']:event('figure_example',id=e['id'],degree=e['degree'],kind=e['example_kind'],motif=e['anchor']['display_label'])
    importance_root=result/'importance/flyfactorsurvey'
    importance_complete=json.loads((importance_root/'complete.json').read_text())
    if digest(importance_root/'summary.json')!=importance_complete['files']['summary.json']:
        raise ValueError('Importance checksum mismatch')
    importance=json.loads((importance_root/'summary.json').read_text())
    if importance['target']!='mean_active_logit' or importance['references']!=100:raise ValueError('Wrong curve target')
    previous=json.loads((extension/'previous_figure.source.json').read_text())
    activity=previous['panel_e']
    perturbation=project/'experiments/repeat_relationship_20260918/perturbations/predictions.npz'
    if digest(perturbation)!=activity['predictions_sha256']:raise ValueError('Preserved perturbation predictions changed')
    output=root/'output/pdf';output.mkdir(parents=True,exist_ok=True)
    path=output/'figure_3bcde_ig64_ref100_native_examples_grh.pdf'
    if path.exists():raise FileExistsError('Figure already exists; preserve it and use a new version')
    figure=ExampleFigure(audit,path,project/config['assets']/'fonts')
    figure.initialize(revised_layout=True,motif_count=len(rows));figure.height+=420
    figure.c.setPageSize((figure.width,figure.height))
    figure.c.setTitle('Figure 3b-e | IG64/100-reference enhancer motifs and Grh example')
    figure.c.setSubject('New native enhancer discovery, mean-active-logit target; no panel A; unchanged frozen-model perturbations')
    figure.panel_b(rows);figure.examples_panel(examples);figure.panel_c(importance,panel_label='d')
    figure.panel_d_odds_ratios(activity,axis='linear',panel_label='e')
    check_text_bounds(figure.text_bounds);figure.c.showPage();figure.c.save()
    source=dict(panel_a='omitted',panel_b=rows,panel_c=examples,panel_d=importance,panel_e=activity,
        references=100,steps=64,attribution_audit_sha256=digest(root/'audit_complete.json'),
        report_sha256=digest(result/'report_audit.json'),importance_sha256=digest(importance_root/'summary.json'),
        previous_figure_source_sha256=digest(extension/'previous_figure.source.json'),builder_sha256=digest(Path(__file__)),
        frozen_presentation_manifest=digest(extension/'MANIFEST.sha256'),pdf_sha256=digest(path),
        page_size_pt=[figure.width,figure.height],text_bounds=figure.text_bounds,visual_qa='pending local inspection',
        grh_note='Example must match a retained positive de novo PWM with best JASPAR Grh q<0.05 and pass unchanged contrast gates; no silent reference-PWM fallback')
    write_json(path.with_suffix('.source.json'),source)
    write_json(root/'figure_complete.json',dict(status='rendered_pending_visual_qa',files={
        str(p.relative_to(root)):digest(p) for p in (path,path.with_suffix('.source.json'))}))
    event('dual_figure_rendered',path=str(path),visual_qa='pending')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--extension',type=Path,required=True)
    args=parser.parse_args();run(args.project,args.root,args.extension)
