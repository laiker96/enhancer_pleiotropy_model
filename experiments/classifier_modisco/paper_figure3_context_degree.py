"""Cumulative context-degree Figure 3 from verified saved native discoveries.

No discovery or attribution is run by `prepare` or `render`. `examples` is a
separately authorized CUDA step for the two changed illustrative scalar targets.
"""
import argparse
import copy
import html
import json
from pathlib import Path
import shutil

import numpy as np

from . import paper_figure3_hierarchical as draw
from . import paper_figure3_masked_examples as preview
from . import paper_figure3_mutant_ig as mutant
from .paper_figure3 import orient_pair

PROJECT = draw.PROJECT
SOURCE = PROJECT/'results/classifier_cumulative_mean_motifs_20260929/cecar_results'
ROOT = PROJECT/'results/figure3_context_degree_20260929'
EXAMPLES = PROJECT/'results/figure3_mutant_ig_20260929'
PREFIX = 'figure_3_context_degree_motifs_examples_20260929'
ORDER = np.asarray(preview.ORDER)
COLORS = {'positive':'#2166AC', 'negative':'#B2182B'}
VARIANT_COLORS = {'WT':'#444444','GAF mut.':'#D55E00','Grh mut.':'#009E73',
                  'Cg mut.':'#0072B2','Double mut.':'#8863A9','Control':'#BBBBBB'}


def verify(root):
    receipt = json.loads((root/'complete.json').read_text())
    if receipt['status'] != 'complete':
        raise ValueError('Incomplete input '+str(root))
    for name, sha in receipt['files'].items():
        if draw.sha(root/name) != sha:
            raise ValueError('Changed input '+name)
    return receipt


def contributions(values, labels):
    """Exact additive partition; family-balanced diagnostic is a different score."""
    v = np.asarray(values, float); y = np.asarray(labels)
    if (v.ndim != 2 or v.shape != y.shape or v.shape[1] != 8 or not len(v)
            or not np.isfinite(v).all() or not np.isin(y, (0,1)).all()
            or (y.sum(1) == 0).any()):
        raise ValueError('Finite carrier-by-eight profiles and nonempty binary masks required')
    weighted = v*y/y.sum(1,keepdims=True)
    family = np.column_stack([weighted[:,list(f)].sum(1) for f in draw.FAMILIES])
    active = np.column_stack([y[:,list(f)].any(1) for f in draw.FAMILIES])
    family_means = np.column_stack([(v[:,list(f)]*y[:,list(f)]).sum(1)/
        np.maximum(y[:,list(f)].sum(1),1) for f in draw.FAMILIES])
    balanced = family_means/active.sum(1,keepdims=True)
    np.testing.assert_allclose(weighted.sum(1),family.sum(1),atol=1e-15)
    return dict(context=weighted, family=family, family_balanced=balanced,
                family_active=active, discovery=weighted.sum(1))


def select(rows, sign, maximum=3):
    # Keep the original unique-carrier rank. No TF-name/q-value selection.
    return sorted((r for r in rows if r['sign']==sign), key=lambda r:r['rank'])[:maximum]


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    receipt = verify(SOURCE)
    audit = json.loads((SOURCE/'audit.json').read_text())
    matches = json.loads((SOURCE/'jaspar_matches.json').read_text())['best']
    config = json.loads((SOURCE/'config.json').read_text())
    if config['spec'] != dict(weights=[1.]*8, observed_mask=True, observed_mean=True):
        raise ValueError('Discovery must explain mean observed-active calibrated probabilities')
    groups = copy.deepcopy(audit['groups']); arrays = {}
    expected = [23080,17229,9587,5041,2865,1714,1006,507]
    for k,g in enumerate(groups):
        if (g['elements'] != expected[k] or g['task']['low'] != k+1
                or g['task']['high'] != (1 if k==0 else 8)):
            raise ValueError('Unexpected context-degree grouping')
        with np.load(SOURCE/f'annotation/fit_{k:02d}/core_context_profiles.npz',allow_pickle=False) as z:
            for row in g['rows']:
                pre = row['contribution_profile']['array_prefix']
                values, labels = z[pre+'_values'], z[pre+'_labels']
                if len(values) != row['supporting_discovery_enhancers']:
                    raise ValueError('Wrong carrier denominator')
                degree = labels.sum(1)
                if not ((degree==1) if k==0 else (degree>=k+1)).all():
                    raise ValueError('Carrier outside its observed-degree group')
                c = contributions(values,labels)
                np.testing.assert_allclose(c['discovery'],z[pre+'_discovery_scalar'],atol=1e-14)
                row['match'] = matches[row['id'].replace('/','__')]
                row['masked_profile'] = dict(
                    context=c['context'].mean(0).tolist(),family=c['family'].mean(0).tolist(),
                    family_balanced=c['family_balanced'].mean(0).tolist(),
                    mean_discovery=float(c['discovery'].mean()),
                    active_carriers_per_context=labels.sum(0).astype(int).tolist(),
                    active_carriers_per_family=c['family_active'].sum(0).astype(int).tolist())
                key=f'group_{k}_{pre}'
                for name in ('indices','ids','labels','covered_bp'):
                    arrays[key+'_'+name]=z[pre+'_'+name]
                for name in ('context','family','family_balanced','discovery'):
                    arrays[key+'_'+name]=c[name]
                arrays[key+'_unmasked_context']=values
                row['derived_array_prefix']=key
    np.savez_compressed(ROOT/'contributions.npz',**arrays)
    draw.save(ROOT/'analysis.json',dict(groups=groups,rule=audit['rule'],contexts=list(draw.CONTEXTS),
        families=list(draw.FAMILY_NAMES),source_complete_sha256=draw.sha(SOURCE/'complete.json'),
        verified_source_files=len(receipt['files']),source_manifest=receipt['manifest_sha256'],
        attribution='Calibrated-probability IG64/100 WT dinucleotide references; native enhancer intervals.',
        contribution='Union trimmed discovery-seqlet cores within each carrier; signed IG per covered base. '
        'Multiply each context by observed activity / degree, then equally average carriers. '
        'Family totals sum member contributions, so both partitions add to the discovery mean. '
        'Inactive contexts contribute zero. Family-balanced scores separately average active members '
        'within active families, then divide by active-family count; they explain a different scalar.',
        support='Unique discovery-seqlet carriers / all QC enhancers in that group, not scan prevalence.',
        caveats='Cumulative groups overlap. Motifs are separately discovered per group. Database matches '
        'are not TF-binding evidence. Carrier-average family profiles do not prove same-instance sharing.'))
    prepare_examples()
    print(json.dumps(dict(stage='context_degree_prepared',motifs=sum(len(g['rows']) for g in groups),
        displayed=sum(len(select(g['rows'],s)) for g in groups for s in COLORS))),flush=True)


def prepare_examples():
    """Freeze identical DNA/edits/references; reuse only identical scalar weights."""
    verify(EXAMPLES)
    source = json.loads((EXAMPLES/'plan.json').read_text())
    if draw.sha(EXAMPLES/'plan.json') != json.loads((EXAMPLES/'complete.json').read_text())['plan_sha256']:
        raise ValueError('Changed source mutation plan')
    dest = ROOT/'examples'
    if (dest/'reuse.json').exists():
        return
    dest.mkdir(exist_ok=True)
    plan = copy.deepcopy(source)
    reuse=[]; recompute=[]
    for j,e in enumerate(plan['examples']):
        mask=np.asarray(e['labels'],float); new=mask/mask.sum()
        (reuse if np.allclose(new,e['weights'],atol=1e-15,rtol=0) else recompute).append(j)
        e['weights']=new.tolist()
    plan.update(provenance='Same frozen accurate examples and edits; observed-active CONTEXT mean target.',
        attribution='IG64/100 shared WT references; unchanged variants/masks; only changed scalar targets recomputed.',
        source_plan_sha256=draw.sha(EXAMPLES/'plan.json'))
    shutil.copy2(EXAMPLES/'sequences.npz',dest/'sequences.npz')
    if (dest/'plan.json').exists():
        if json.loads((dest/'plan.json').read_text()) != plan:
            raise ValueError('Changed prepared mutation plan')
    else:
        draw.save(dest/'plan.json',plan)
    signature=draw.sha(dest/'plan.json')
    for j in reuse:
        with np.load(EXAMPLES/f'attribution_{j}.npz',allow_pickle=False) as z:data=dict(z)
        np.testing.assert_array_equal(data['weights'],np.asarray(plan['examples'][j]['weights'],np.float32))
        data['signature']=np.asarray(signature)
        np.savez_compressed(dest/f'attribution_{j}.npz',**data)
        r=json.loads((EXAMPLES/f'attribution_{j}.json').read_text())
        r['reused_from_sha256']=draw.sha(EXAMPLES/f'attribution_{j}.npz')
        draw.save(dest/f'attribution_{j}.json',r)
    draw.save(dest/'reuse.json',dict(reused=reuse,recompute=recompute,source=str(EXAMPLES.relative_to(PROJECT))))


def run_examples():
    # Reuse the already tested, resumable IG engine; isolate all writes.
    mutant.ROOT=ROOT/'examples'
    mutant.run()


def color_value(value,limit):
    from matplotlib import colormaps
    return colormaps['RdBu_r'](.5+.5*np.clip(value/limit,-1,1))


def heat_cells(fig,values,x,y,width,limit):
    from matplotlib.patches import Rectangle
    fw,fh=fig.get_size_inches()*72
    for j,v in enumerate(values):
        fig.add_artist(Rectangle(((x+j*width/len(values))/fw,y/fh),
            width/len(values)/fw,10/fh,transform=fig.transFigure,
            facecolor=color_value(v,limit),edgecolor='white',lw=.3))


def motif_row(fig,row,x,y,width,limit):
    """Aligned logos, explicit q, non-truncated support axis and family partition."""
    from matplotlib.patches import Rectangle
    fw,fh=fig.get_size_inches()*72
    color=COLORS[row['sign']]
    fig.add_artist(Rectangle((x/fw,(y+7)/fh),10/fw,10/fh,transform=fig.transFigure,color=color,lw=0))
    draw.label(fig,x+5,y+9,'+' if row['sign']=='positive' else '-',7,'white','bold',ha='center')
    draw.label(fig,x+17,y+10,str(row['rank']),6,ha='center')
    q,t,qs,ts,span=orient_pair(row)
    for pwm,start,yy in ((q,qs,y+13),(t,ts,y+1)):
        ax=draw.make_axes(fig,(x+27,yy,113,11))
        draw.logo(ax,pwm,start=start)
        ax.set(xlim=(0,span),ylim=(0,2.05));ax.axis('off')
    match=row['match'];name=match['reference']['name']
    draw.label(fig,x+149,y+15,name,6.2,weight='bold')
    draw.label(fig,x+149,y+5,f"q={match['q']:.2g}"+(' ns' if match['q']>.05 else ''),5.6,'#666666')
    fraction=row['supporting_discovery_enhancers']/row['group_n']
    if not np.isclose(fraction,row['support_fraction']):raise ValueError('Wrong support fraction')
    ax=draw.make_axes(fig,(x+207,y+10,52,5));ax.barh(0,100,color='#E5E5E5',height=.8)
    ax.barh(0,100*fraction,color='#777777',height=.8);ax.set(xlim=(0,100),ylim=(-.5,.5));ax.axis('off')
    draw.label(fig,x+263,y+9,f'{100*fraction:.1f}',5.8)
    heat_cells(fig,row['masked_profile']['family'],x+291,y+8,width-291,limit)


def setup_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none'})
    return plt


def example_card(fig,e,j,seq,probs,actual,x,top,width,height,with_mutants):
    from matplotlib.patches import Patch
    dna,_,coord,sites,lo,reverse=preview.display(e,seq)
    ix=[v['index'] for v in e['variants']];names=[v['name'] for v in e['variants']]
    if with_mutants:
        native=actual[:,e['native_offset']:e['native_offset']+len(e['sequence'])]
    else:
        native=(np.asarray(e['weights'])@np.asarray(e['actual']))[None,:]
    if reverse:native=native[:,::-1]
    values=native[:,lo:lo+dna.shape[1]]
    high=max(float(values.max())*1.15,1e-4);low=min(float(values.min())*1.15,-high*.05)
    title=' + '.join(preview.NAMES[s['pattern']] for s in e['sites'])
    draw.label(fig,x,top,f"{title} | degree {sum(e['labels'])} | {e['id']}"+(' | RC' if reverse else ''),7,weight='bold')
    draw.label(fig,x,top-11,f"{e['split']} | Brier {e['brier']:.3f}",5.5,'#666666')
    drawn=names if with_mutants else ['WT']
    step=19 if len(drawn)>3 else 27
    for v,name in enumerate(drawn):
        y=top-33-v*step;ax=draw.make_axes(fig,(x+38,y,width-38,step-8))
        for site in sites:ax.axvspan(site['start']-lo,site['end']-lo,color='#EFF2F4',zorder=-1)
        for k,(base,value) in enumerate(zip(dna[v],values[v])):
            draw.glyph(ax,draw.BASES[base],k+.03,0,.94,float(value),draw.DNA_COLORS[base])
        ax.axhline(0,color='#777777',lw=.3);ax.set(xlim=(0,dna.shape[1]),ylim=(low,high));ax.axis('off')
        if v==0:
            for site in sites:
                ax.text((site['start']+site['end'])/2-lo,high*1.08,site['name'],
                        fontsize=5,ha='center',va='bottom')
        draw.label(fig,x+34,y+4,name,5.8,VARIANT_COLORS[name],ha='right')
        letters=draw.make_axes(fig,(x+38,y-5,width-38,5))
        for k,base in enumerate(dna[v]):
            letters.text(k+.5,.5,draw.BASES[base],ha='center',va='center',fontsize=4,
                fontfamily='DejaVu Sans Mono',color='#B2182B' if dna[v,k]!=dna[0,k] else '#666666')
        letters.set(xlim=(0,dna.shape[1]),ylim=(0,1));letters.axis('off')
    if not with_mutants:
        # DNA edit rows are NOT mutant attribution logos. No gradients inferred.
        draw.label(fig,x+38,top-57,'Variant DNA (edits in red)',5.5,'#666666')
        for v,name in enumerate(names[1:],1):
            y=top-62-v*12
            letters=draw.make_axes(fig,(x+38,y,width-38,7))
            for k,base in enumerate(dna[v]):
                letters.text(k+.5,.5,draw.BASES[base],ha='center',va='center',fontsize=4.1,
                    fontfamily='DejaVu Sans Mono',color='#B2182B' if dna[v,k]!=dna[0,k] else '#888888')
            letters.set(xlim=(0,dna.shape[1]),ylim=(0,1));letters.axis('off')
            draw.label(fig,x+34,y+2,name,5.5,VARIANT_COLORS[name],ha='right')
    draw.label(fig,x+38,top-127,f'IG/base: {low:.3f} to {high:.3f}',5.3,'#666666')
    draw.label(fig,x+width,top-127,f'{coord[0]}-{coord[-1]} bp',5.3,'#666666',ha='right')
    ax=draw.make_axes(fig,(x+38,top-height+32,width-38,94));w=.8/len(ix)
    for v,name in enumerate(names):
        ax.bar(np.arange(8)-.4+w*(v+.5),probs[ix[v],ORDER],w,color=VARIANT_COLORS[name],lw=0)
    active=np.asarray(e['labels'])[ORDER]
    ax.set(xlim=(-.6,7.6),ylim=(0,1.07),xticks=np.arange(8),
        xticklabels=[draw.CONTEXTS[i] for i in ORDER],yticks=[0,.5,1])
    for tick,on in zip(ax.get_xticklabels(),active):tick.set_fontweight('bold' if on else 'normal')
    ax.set_ylabel('Predicted probability',fontsize=6);draw.clean(ax)
    ax.legend(handles=[Patch(facecolor=VARIANT_COLORS[n],label=n) for n in names],frameon=False,
        fontsize=5.6,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,handlelength=.9,columnspacing=.8)


def render():
    plt=setup_plot()
    analysis=json.loads((ROOT/'analysis.json').read_text());groups=analysis['groups']
    selected=[select(g['rows'],'positive')+select(g['rows'],'negative') for g in groups]
    limit=max(abs(v) for rows in selected for r in rows for v in r['masked_profile']['family'])
    fig=plt.figure(figsize=(800/72,1440/72))
    draw.label(fig,12,1415,'b',13,weight='bold')
    draw.label(fig,38,1415,'Degree of pleiotropy',8,weight='bold')
    draw.label(fig,280,1415,'Positive',6.5,COLORS['positive'])
    draw.label(fig,333,1415,'Negative',6.5,COLORS['negative'])
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    cax=draw.make_axes(fig,(600,1415,150,5))
    bar=fig.colorbar(ScalarMappable(norm=Normalize(-limit*1000,limit*1000),cmap='RdBu_r'),cax=cax,orientation='horizontal')
    bar.set_ticks([-limit*1000,0,limit*1000]);bar.ax.tick_params(labelsize=5.5,pad=1,length=2)
    draw.label(fig,600,1430,'Family contribution (IG/base x 1000)',6)
    for k,g in enumerate(groups):
        x=30+(k//4)*393;top=1370-(k%4)*190;width=347
        title='1' if k==0 else ('8' if k==7 else f'>={k+1}')
        draw.label(fig,x,top,title,10,weight='bold')
        draw.label(fig,x+32,top,f"n={g['elements']:,}",6.5)
        draw.label(fig,x+80,top,'De novo / JASPAR',5.8)
        draw.label(fig,x+207,top,'Support (0-100%)',5.8)
        for pos,name in enumerate(('Emb','Disc','CNS','Ova')):
            draw.label(fig,x+291+(pos+.5)*(width-291)/4,top,name,5,ha='center')
        for sign,offset in (('positive',0),('negative',3)):
            rows=select(g['rows'],sign)
            for j in range(3):
                y=top-32-(j+offset)*26
                if j<len(rows):motif_row(fig,dict(rows[j],group_n=g['elements']),x,y,width,limit)
                elif j==len(rows):
                    draw.label(fig,x+28,y+10,'No '+('further ' if rows else '')+sign+' motifs retained',6,'#888888')
    draw.label(fig,12,586,'c',13,weight='bold')
    draw.label(fig,38,586,'Mean observed-active-context attribution',8,weight='bold')
    plan=json.loads((ROOT/'examples/plan.json').read_text())
    with np.load(EXAMPLES/'sequences.npz',allow_pickle=False) as z:seq=z['sequence']
    with np.load(EXAMPLES/'predictions.npz',allow_pickle=False) as z:probs=z['calibrated_probabilities']
    with_mutants=(ROOT/'examples/complete.json').exists()
    if with_mutants:verify(ROOT/'examples')
    for j,e in enumerate(plan['examples']):
        actual=None
        if with_mutants:
            with np.load(ROOT/f'examples/attribution_{j}.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['weights'],np.asarray(e['weights'],np.float32))
                actual=z['actual']
        example_card(fig,e,j,seq,probs,actual,30+(j%2)*393,557-(j//2)*275,347,250,with_mutants)
    out=PROJECT/'output/pdf'/f'{PREFIX}.pdf'
    fig.savefig(out,metadata=dict(Title='Figure 3: cumulative context degree, motifs and native mutation examples',
        Subject='Masked mean calibrated-probability IG64/100; discovery-seqlet support; JASPAR matches.'))
    fig.savefig(out.with_suffix('.svg'));fig.savefig(out.with_suffix('.png'),dpi=110);plt.close(fig)
    render_profiles(analysis)
    draw.save(ROOT/'figure_source.json',dict(status='awaiting_visual_QA',mutant_logos=with_mutants,
        code_sha256=draw.sha(__file__),analysis_sha256=draw.sha(ROOT/'analysis.json'),
        contributions_sha256=draw.sha(ROOT/'contributions.npz'),
        source_complete_sha256=draw.sha(SOURCE/'complete.json'),
        examples_complete_sha256=draw.sha(EXAMPLES/'complete.json'),
        selected_ids=[[r['id'] for r in rows] for rows in selected],
        outputs={str(p.relative_to(PROJECT)):draw.sha(p) for p in
            (out,out.with_suffix('.svg'),PROJECT/'output/pdf'/f'{PREFIX}_context_profiles.pdf')}))
    print(json.dumps(dict(stage='figure_rendered',pdf=str(out),mutant_logos=with_mutants)),flush=True)


def render_profiles(analysis):
    plt=setup_plot()
    from matplotlib.backends.backend_pdf import PdfPages
    out=PROJECT/'output/pdf'/f'{PREFIX}_context_profiles.pdf'
    groups=analysis['groups']
    limit=max(abs(v) for g in groups for r in g['rows'] for key in ('context','family','family_balanced')
              for v in r['masked_profile'][key])*1000
    page=['<!doctype html><meta charset="utf-8"><title>Context-degree motif contributions</title>',
        '<style>body{font:14px system-ui;margin:35px}table{border-collapse:collapse}td,th{padding:6px;border-bottom:1px solid #ddd}th{position:sticky;top:0;background:white}</style>',
        '<h1>Context-degree motifs</h1><p>'+html.escape(analysis['contribution'])+'</p><p>'+html.escape(analysis['support'])+'</p>',
        '<p>'+html.escape(analysis['caveats'])+'</p><p>Values below: signed IG per base x 1000. '
        'Family-balanced values are available in analysis.json and contributions.npz.</p>']
    with PdfPages(out) as pdf:
        for k,g in enumerate(groups):
            rows=select(g['rows'],'positive',999)+select(g['rows'],'negative',999)
            labels=[('+' if r['sign']=='positive' else '-')+str(r['rank'])+' '+r['match']['reference']['name']+
                    (' (ns)' if r['match']['q']>.05 else '') for r in rows]
            fig,axes=plt.subplots(1,3,figsize=(11.7,max(5.5,.29*len(rows)+2)),sharey=True,
                layout='constrained',gridspec_kw={'width_ratios':[8,4,4]})
            titles=['Context contribution','Family total (same target)','Family-balanced (different target)']
            for ax,key,names,title in zip(axes,('context','family','family_balanced'),
                    ([draw.CONTEXTS[i] for i in ORDER],draw.FAMILY_NAMES,draw.FAMILY_NAMES),titles):
                values=np.asarray([r['masked_profile'][key] for r in rows])*1000
                if key=='context':values=values[:,ORDER]
                im=ax.imshow(values,aspect='auto',vmin=-limit,vmax=limit,cmap='RdBu_r')
                ax.set_xticks(range(len(names)),names,rotation=45,ha='right');ax.set_title(title,fontsize=8)
                ax.set_yticks(range(len(rows)),labels)
                for i in range(len(rows)):
                    for j,v in enumerate(values[i]):
                        ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=5.7,
                            color='white' if abs(v)>.6*limit else '#222222')
            fig.colorbar(im,ax=axes,label='Signed contribution (IG/base x 1000)',shrink=.7)
            fig.suptitle(('Degree 1' if k==0 else f'Degree >= {k+1}')+f" | {g['elements']:,} enhancers",fontsize=11)
            pdf.savefig(fig);plt.close(fig)
            page.append('<h2>'+('1' if k==0 else f'&ge;{k+1}')+' contexts</h2><table><tr><th>Motif</th><th>q</th><th>Carriers</th><th>Seqlets</th><th>Support %</th>'+''.join('<th>'+c+'</th>' for c in list(draw.CONTEXTS)+list(draw.FAMILY_NAMES))+'</tr>')
            for r,label in zip(rows,labels):
                values=r['masked_profile']['context']+r['masked_profile']['family']
                page.append('<tr><td>'+html.escape(label)+'</td><td>'+f"{r['match']['q']:.3g}"+'</td><td>'+str(r['supporting_discovery_enhancers'])+'</td><td>'+str(r['seqlets'])+'</td><td>'+f"{100*r['support_fraction']:.2f}"+'</td>'+''.join(f'<td>{v*1000:.3f}</td>' for v in values)+'</tr>')
            page.append('</table>')
    (ROOT/'index.html').write_text(''.join(page))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','examples','render','examples_render'))
    a=p.parse_args()
    if a.stage=='examples_render':run_examples();render()
    else:{'prepare':prepare,'examples':run_examples,'render':render}[a.stage]()
