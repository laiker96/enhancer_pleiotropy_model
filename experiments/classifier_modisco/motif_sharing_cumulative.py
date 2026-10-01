"""Regroup cached motif-sharing values by 1, >=2, ..., >=8; never rescan DNA."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .common import digest, event, require_allocation, write_json
from .family_breadth import grouping, CONTEXTS
from .motif_sharing import (TRL, FLOORS, FAMILIES, THRESHOLDS, metric_key,
    describe, locus_metrics, paired_comparisons, render)
from .trl_family import table

NAME='classifier_motif_sharing_cumulative_20260928'
PARENT='experiments/classifier_motif_sharing_20260928/output'


def cumulative_groups(labels):
    degree=grouping(labels)['raw']
    groups={'degree_1':degree==1}
    groups.update({'degree_ge_'+str(d):degree>=d for d in range(2,9)})
    return groups


def label(group):
    return '1 (context-specific)' if group=='degree_1' else '≥'+group.rsplit('_',1)[1]


def summarize_cache(motif, cache, groups):
    rows=[]
    for t,threshold in enumerate(THRESHOLDS):
        for scheme in FAMILIES:
            for sign in ('positive','negative'):
                for f,floor in enumerate(FLOORS):
                    names,_=locus_metrics(np.zeros((0,8)),np.zeros(0),scheme,sign,floor)
                    values=cache[metric_key(t,scheme,sign,f)]
                    for name,selected in groups.items():
                        rows.append(dict(motif=motif['id'],name=motif['name'],threshold=float(threshold),
                            scheme=scheme,sign=sign,floor=floor,group=name,metrics=names,
                            **describe(values[selected])))
    return rows


def overview(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors={TRL:'#2166ac','MA2107.1':'#d95f02','MA1457.2':'#1b9e77','MA0460.1':'#984ea3'}
    groups=list(report['groups']);motifs=report['config']['motifs']
    primary={(r['motif'],r['sign'],r['group']):r for r in report['summary'] if
        r['threshold']==.8 and r['floor']==FLOORS[0] and r['scheme']=='four_families'}
    fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained',sharex=True,sharey=True)
    metrics=['within_embryo','within_CNS','within_imaginal_discs','between_sharing']
    titles=['Within embryo','Within CNS','Within imaginal discs','Between families']
    handles=[]
    for i,sign in enumerate(('positive','negative')):
        for j,(metric,title) in enumerate(zip(metrics,titles)):
            ax=axes[i,j]
            for motif in sorted(motifs,key=lambda m:m['id'] in colors):
                values=[]
                for group in groups:
                    r=primary[motif['id'],sign,group]
                    v=r['mean'][r['metrics'].index(metric)]
                    values.append(v if v is not None else np.nan)
                line,=ax.plot(range(8),values,color=colors.get(motif['id'],'#b9b9b9'),
                    linewidth=1.8 if motif['id'] in colors else .75,
                    alpha=1 if motif['id'] in colors else .6,label=motif['name'])
                if i==0 and j==0 and motif['id'] in colors:handles.append(line)
            ax.set_ylim(-.02,1.02);ax.set_xticks(range(8),['1']+['≥'+str(d) for d in range(2,9)])
            if i==0:ax.set_title(title)
            if j==0:ax.set_ylabel(sign.capitalize()+' contribution sharing')
            if i==1:ax.set_xlabel('Degree of pleiotropy')
    fig.legend(handles=handles,loc='outside upper center',ncol=4,frameon=False)
    fig.savefig(output/'cumulative_sharing.svg');plt.close(fig)
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Cumulative motif contribution sharing</title>',
        '<style>body{font:15px system-ui;max-width:1400px;margin:36px auto;padding:0 20px}p{max-width:1150px;line-height:1.5}table{border-collapse:collapse;margin:18px 0}th,td{padding:8px 14px;text-align:right;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}img{width:100%}</style>',
        '<h1>Motif contributions by increasing degree of pleiotropy</h1>',
        '<p>The primary enhancer groups are <strong>1 (context-specific), ≥2, ≥3, ≥4, ≥5, ≥6, ≥7 and ≥8</strong>, '
        'defined by the experimental activity table. Groups ≥2 through ≥8 overlap and are not independent replicates. '
        'Within-family and between-family contribution sharing are reported within each group, separately for positive and negative contributions.</p>',
        '<img src="cumulative_sharing.svg" alt="Within- and between-family motif sharing across eight cumulative pleiotropy groups">',
        '<p>Trl, cg, Grh and ttk are highlighted; the other fixed JASPAR profiles are grey. These curves are per-locus sharing scores averaged equally within each enhancer, then across available carrier enhancers. '
        'They are not motif prevalence or attribution recalculated from a group-average profile. Different motifs have different available carriers. '
        'Use the within-group paired comparisons below for Trl-versus-other-motif differences in the same enhancer.</p>',
        '<h2>Group reports</h2>']
    rows=[]
    for group,counts in report['groups'].items():
        rows.append([label(group),counts['all_enhancers'],counts['qc_enhancers'],counts['excluded_qc']])
    parts.append(table(['Degree of pleiotropy','All enhancers','QC-passing enhancers','QC excluded'],rows))
    parts.append('<ul>'+''.join('<li><a href="'+group+'/index.html">'+html.escape(label(group))+
        ' — contributions, sharing and paired motif comparisons</a></li>' for group in groups)+'</ul>')
    parts += ['<p>Each group page includes all 22 motifs, context/family magnitude profiles, both sharing signs, and Trl-versus-comparator statistics calculated inside that group. '
        'Its JSON also retains both family definitions, both strength floors and all three PWM thresholds. '
        'Primary figures use the original0.8 cutoff, four families and10⁻⁶ numerical floor. Nothing was rescanned and no new IG was computed.</p>',
        '<p>Sharing color scales are fixed0–1. Signed-magnitude color scales are shared within each group page, but can differ between pages; compare the numeric values when comparing magnitude between groups. '
        'Bootstrap intervals are descriptive and not multiplicity-adjusted; overlapping cumulative groups are not independent confirmations.</p>',
        '<p><a href="supplement/index.html">Supplement: original pooled report and exact-degree/activity-pattern metrics</a> · '
        '<a href="supplement/report.json">Original detailed metrics</a> · <a href="groups.json">Group definitions/counts</a> · '
        '<a href="config.json">Frozen settings and provenance</a></p>','</html>']
    (output/'index.html').write_text('\n'.join(parts))


def run(project,root):
    require_allocation('cpu')
    operation=json.loads((root/'config.json').read_text());source=project/PARENT
    if digest(source/'complete.json')!=operation['source_complete_sha256']:raise ValueError('Changed parent receipt')
    done=json.loads((source/'complete.json').read_text())
    if done['status']!='complete' or done['manifest_sha256']!=operation['source_manifest_sha256']:
        raise ValueError('Wrong parent package')
    for name,expected in done['files'].items():
        if digest(source/name)!=expected:raise ValueError('Changed cached output: '+name)
    with np.load(source/'enhancers.npz',allow_pickle=False) as z:data=dict(z)
    if tuple(data['contexts'])!=CONTEXTS or len(set(data['ids']))!=len(data['ids']):raise ValueError('Invalid cached metadata')
    groups=cumulative_groups(data['labels'])
    config=json.loads((source/'config.json').read_text())
    config.update(cumulative_reporting=operation,
        stratification='Primary experimental degree1,>=2,...,>=8; identical subsets for summaries, signed profiles and paired Trl comparisons. Original exact-degree/pattern results are supplementary.')
    output=root/'output';output.mkdir(exist_ok=False)
    write_json(output/'config.json',config);shutil.copy2(source/'enhancers.npz',output/'enhancers.npz')
    report=dict(config=config,summary=[],paired=[],groups={name:dict(
        all_enhancers=int(mask.sum()),qc_enhancers=int((mask&data['qc']).sum()),excluded_qc=int((mask&~data['qc']).sum()))
        for name,mask in groups.items()})
    for motif in config['motifs']:
        with np.load(source/(motif['id']+'.npz'),allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache['ids'],data['ids'])
            report['summary'].extend(summarize_cache(motif,cache,groups))
        event('cumulative_sharing_summary',motif=motif['id'])
    report['paired']=paired_comparisons(config['motifs'],data,source,config,groups=groups)
    for group,mask in groups.items():
        directory=output/group;directory.mkdir()
        sub=dict(config=config,group=group,counts=report['groups'][group],
            summary=[r for r in report['summary'] if r['group']==group],
            paired=[r for r in report['paired'] if r['group']==group])
        write_json(directory/'report.json',sub)
        title='Degree of pleiotropy '+label(group)+' — '+str(report['groups'][group]['qc_enhancers'])+' QC-passing enhancers'
        render(sub,directory,group=group,subset=mask,cache_directory=source,title=title,artifact_prefix='../')
        event('cumulative_sharing_group_report',group=group)
    supplementary=output/'supplement';supplementary.mkdir()
    for name in ('index.html','report.json','profiles.json','config.json','enhancers.npz','sharing.svg','contribution_profiles.svg'):
        shutil.copy2(source/name,supplementary/name)
    write_json(output/'groups.json',report['groups']);overview(report,output)
    files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'complete.json',dict(status='complete',manifest_sha256=digest(root/'MANIFEST.sha256'),
        source_complete_sha256=operation['source_complete_sha256'],files=files,job=os.environ['SLURM_JOB_ID'],
        groups=report['groups'],no_new_attribution=True,no_new_scanning=True))
    event('cumulative_sharing_complete',report=str(output/'index.html'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();run(args.project,args.root)
