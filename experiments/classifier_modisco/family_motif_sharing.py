"""Family-degree regrouping and raw-degree controls from immutable motif caches."""
import html
import json

import numpy as np

from .common import digest, write_json, event
from .family_breadth import FAMILIES, grouping
from .motif_sharing import FLOORS, TRL, describe, locus_metrics, metric_key, paired_comparisons
from .motif_sharing_cumulative import summarize_cache
from .trl_family import table

SOURCE='experiments/classifier_motif_sharing_20260928/output'


def groups(labels):
    result={}
    for scheme, families in FAMILIES.items():
        breadth=grouping(labels, scheme=scheme)['breadth']
        for d in range(1, len(families)+1):
            result[scheme+'__'+str(d)]=breadth==1 if d==1 else breadth>=d
    return result


def controlled_summaries(motif, cache, labels):
    rows=[]
    for scheme, families in FAMILIES.items():
        g=grouping(labels, scheme=scheme)
        for sign in ('positive','negative'):
            names,_=locus_metrics(np.empty((0,8)),np.empty(0),scheme,sign,FLOORS[0])
            values=cache[metric_key(1,scheme,sign,0)]
            for raw in range(1,9):
                for family in range(1,len(families)+1):
                    mask=(g['raw']==raw)&(g['breadth']==family)
                    rows.append(dict(motif=motif['id'],name=motif['name'],scheme=scheme,sign=sign,
                        raw_degree=raw,exact_family_degree=family,metrics=names,**describe(values[mask])))
    return rows


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={TRL:'#2166ac','MA1457.2':'#1b9e77','MA2107.1':'#d95f02','MA1700.1':'#984ea3'}
    motifs=report['config']['motifs']
    primary={(r['motif'],r['sign'],r['group']):r for r in report['summary'] if
             r['threshold']==.8 and r['floor']==FLOORS[0] and r['group'].startswith(r['scheme']+'__')}
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharey=True)
    handles=[]
    for col,(scheme,families) in enumerate(FAMILIES.items()):
        for row,sign in enumerate(('positive','negative')):
            ax=axes[row,col]
            for motif in sorted(motifs,key=lambda m:m['id'] in colors):
                ys=[]
                for degree in range(1,len(families)+1):
                    r=primary[motif['id'],sign,scheme+'__'+str(degree)]
                    y=r['mean'][r['metrics'].index('between_sharing')]
                    ys.append(np.nan if y is None else y)
                line,=ax.plot(range(1,len(families)+1),ys,color=colors.get(motif['id'],'#bdbdbd'),
                    lw=2 if motif['id'] in colors else .7,alpha=1 if motif['id'] in colors else .55,label=motif['name'])
                if row==0 and col==0 and motif['id'] in colors: handles.append(line)
            ax.set_ylim(-.02,1.02)
            ax.set_xticks(range(1,len(families)+1),['1']+['≥'+str(d) for d in range(2,len(families)+1)])
            ax.set_xlabel('Observed family degree of pleiotropy')
            ax.set_ylabel(sign.capitalize()+' between-family sharing')
            if row==0: ax.set_title('Four families' if col==0 else 'Adult and larval brain separated')
    fig.legend(handles=handles,loc='outside upper center',ncol=4,frameon=False)
    fig.savefig(output/'sharing.svg');plt.close(fig)
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Family-level motif comparisons</title>',
        '<style>body{font:15px system-ui;max-width:1350px;margin:30px auto;padding:0 20px}p{line-height:1.5}table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border-bottom:1px solid #ddd}img{width:100%}</style>',
        '<h1>Motif contributions by family degree of pleiotropy</h1>',
        '<p>Family labels use experimental OR, not model probabilities. Exactly one family includes enhancers active in '
        'several related contexts. Higher groups are cumulative and overlap. Fixed JASPAR scans and saved per-locus values '
        'are reused; no new scanning, inference or attribution. Curves are not de novo motif frequencies.</p>',
        '<img src="sharing.svg" alt="Positive and negative sharing by family breadth">',
        '<p>Positive/negative NET site contributions are split per context before family averaging, as in the previous '
        'sharing analysis. This differs from splitting the sign after averaging a family. Sharing is computed per locus, '
        'then averaged equally over eligible loci per enhancer and eligible enhancers. Noncarriers are missing, not zero. '
        'Strength is reported separately. Families are not assumed statistically independent.</p>',
        '<p><a href="report.json">All thresholds, signs, floors and paired results</a> · '
        '<a href="same_context_count.json">Exact-context-count stratification</a> · '
        '<a href="config.json">Unchanged scan and sharing protocol</a></p>']
    for group,count in report['counts'].items():
        scheme,d=group.split('__');name=('1' if d=='1' else '≥'+d)+' families — '+scheme.replace('_',' ')
        page.append('<h2>'+html.escape(name)+'</h2><p>'+str(count['qc'])+' QC-passing / '+str(count['all'])+' total enhancers</p>')
        for sign in ('positive','negative'):
            records=[]
            for motif in motifs:
                r=primary[motif['id'],sign,group];j=r['metrics'].index('between_sharing')
                records.append([motif['name'],r['n'][0],r['mean'][0],r['mean'][1],r['n'][j],r['mean'][j]])
            page.append('<h3>'+sign.capitalize()+'</h3>'+table(
                ['Motif','Carriers','Strength/locus','Strength/bp','Sharing carriers','Sharing'],records))
            pairs=[]
            for r in report['paired']:
                if r['group']!=group or r['scheme']!=scheme or r['sign']!=sign or r['threshold']!=.8 or r['floor']!=FLOORS[0] or r['policy']!='disjoint':continue
                j=r['metrics'].index('between_sharing');delta=r['difference']
                pairs.append([r['name'],delta['n'][j],r['trl']['mean'][j],r['other_motif']['mean'][j],delta['mean'][j],
                              delta.get('descriptive_block_95ci',[None]*len(r['metrics']))[j]])
            page.append(table(['Paired comparator','Enhancers','Trl','Other','Trl − other','Descriptive block95%CI'],pairs))
    page.append('<h2>Same original context count</h2><p>These descriptive strata compare exact family degrees '
        'inside each exact eight-context degree. They are not matched for length, GC or individual context identities. '
        'The separate same-enhancer motif comparisons above control enhancer identity. No multiplicity-adjusted claims.</p>')
    for raw in range(2,8):
        records=[]
        for r in report['controlled']:
            if r['scheme']=='four_families' and r['raw_degree']==raw and r['sign']=='positive' and r['motif'] in colors:
                j=r['metrics'].index('between_sharing')
                if r['n'][j]:records.append([r['name'],r['exact_family_degree'],r['n'][j],r['mean'][j],r['mean'][1]])
        page.append('<h3>Exactly '+str(raw)+' active contexts</h3>'+table(
            ['Motif','Exact active families','Carriers','Positive sharing','Strength/bp'],records))
    (output/'index.html').write_text('\n'.join(page)+'</html>')


def run(project,root,config):
    source=project/SOURCE;receipt=json.loads((source/'complete.json').read_text())
    if digest(source/'complete.json')!=config['sharing_receipt_sha256']: raise ValueError('Changed motif cache receipt')
    for name,sha in receipt['files'].items():
        if digest(source/name)!=sha: raise ValueError('Changed motif cache '+name)
    with np.load(source/'enhancers.npz',allow_pickle=False) as z:data=dict(z)
    with np.load(root/'native/metadata.npz',allow_pickle=False) as z:
        for key in ('ids','labels'):np.testing.assert_array_equal(data[key],z[key])
        np.testing.assert_array_equal(data['qc'],z['quality_pass'].all(1))
    settings=json.loads((source/'config.json').read_text());masks=groups(data['labels'])
    report=dict(config=settings,counts={g:dict(all=int(mask.sum()),qc=int((mask&data['qc']).sum())) for g,mask in masks.items()},
                summary=[],paired=[],controlled=[])
    for motif in settings['motifs']:
        with np.load(source/(motif['id']+'.npz'),allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache['ids'],data['ids'])
            report['summary'].extend(summarize_cache(motif,cache,masks))
            report['controlled'].extend(controlled_summaries(motif,cache,data['labels']))
    report['paired']=paired_comparisons(settings['motifs'],data,source,settings,groups=masks)
    out=root/'sharing';out.mkdir(exist_ok=False)
    write_json(out/'config.json',settings);write_json(out/'report.json',report)
    write_json(out/'same_context_count.json',report['controlled']);render(report,out)
    event('family_sharing_complete',groups=report['counts'])
