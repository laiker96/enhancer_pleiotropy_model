"""Matched context/family Figure 3 panels, without mixing discovery populations."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from . import paper_figure3_context_degree as previous
from . import paper_figure3_hierarchical as draw
from . import paper_figure3_masked_examples as preview
from .paper_figure3 import orient_pair

PROJECT=draw.PROJECT
ROOT=PROJECT/'results/figure3_matched_20260929'
EXPORT=PROJECT/'results/figure3_matched_profiles_20260929/cecar_results'
FAMILY=PROJECT/'results/classifier_masked_cumulative_motifs_20260929/cecar_results'
ORDER=previous.ORDER
WIDTH=800


def family_partition(values,labels):
    """Active-member means / active-family count: exact family discovery partition."""
    c=previous.contributions(values,labels)
    result=c['family_balanced']
    weights=np.zeros_like(values,dtype=float)
    active=c['family_active'];count=active.sum(1)
    for f in draw.FAMILIES:
        members=list(f);n=labels[:,members].sum(1)
        weights[:,members]=labels[:,members]/np.maximum(n[:,None],1)/count[:,None]
    np.testing.assert_allclose(weights.sum(1),1,rtol=0,atol=1e-14)
    np.testing.assert_allclose(result.sum(1),(values*weights).sum(1),atol=1e-14)
    if (weights[labels==0]!=0).any():raise ValueError('Inactive context has weight')
    return result,weights


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    previous.verify(previous.SOURCE);previous.verify(FAMILY);previous.verify(EXPORT)
    previous.verify(previous.EXAMPLES)
    ctx=copy.deepcopy(json.loads((previous.SOURCE/'audit.json').read_text())['groups'])
    family=copy.deepcopy([g for g in json.loads((FAMILY/'audit.json').read_text())['groups']
                         if g['task']['scope']=='breadth'])
    arrays={};sources={str(r.relative_to(PROJECT)):draw.sha(r/'complete.json')
                      for r in (previous.SOURCE,FAMILY,EXPORT,previous.EXAMPLES)}
    for scheme,groups,source in (('context',ctx,previous.SOURCE),('family',family,FAMILY)):
        matches=json.loads((source/'jaspar_matches.json').read_text())['best']
        for k,g in enumerate(groups):
            task=g['task']['task']
            path=(source/f'annotation/fit_{task:02d}/core_context_profiles.npz' if scheme=='context'
                  else EXPORT/f'family_{task:02d}.npz')
            with np.load(path,allow_pickle=False) as z:
                for j,row in enumerate(g['rows']):
                    prefix=row['contribution_profile']['array_prefix'] if scheme=='context' else f'pattern_{j}'
                    values=z[prefix+'_values'];labels=z[prefix+'_labels']
                    if len(values)!=row['supporting_discovery_enhancers']:raise ValueError('Carrier mismatch')
                    if scheme=='context':
                        part=previous.contributions(values,labels)['context']
                        np.testing.assert_allclose(part.sum(1),z[prefix+'_discovery_scalar'],atol=1e-14)
                        degree=labels.sum(1)
                    else:
                        part,weights=family_partition(values,labels)
                        degree=np.stack([labels[:,list(f)].any(1) for f in draw.FAMILIES],1).sum(1)
                    if not ((degree==1) if k==0 else (degree>=k+1)).all():raise ValueError('Wrong cumulative cohort')
                    row['match']=matches[row['id'].replace('/','__')]
                    row['profile']=part.mean(0).tolist()
                    row['mean_ig']=float(part.sum(1).mean())
                    row['group_n']=g['elements']
                    key=f'{scheme}_{k}_{j}'
                    row['array_prefix']=key
                    for name in ('ids','indices','labels','covered_bp'):arrays[key+'_'+name]=z[prefix+'_'+name]
                    arrays[key+'_raw_context_ig']=values
                    arrays[key+'_discovery_partition']=part
    np.savez_compressed(ROOT/'motif_profiles.npz',**arrays)
    plan=json.loads((previous.EXAMPLES/'plan.json').read_text())
    extras=json.loads((EXPORT/'native_examples.json').read_text())['examples']
    base=[]
    for e in plan['examples']:
        e=copy.deepcopy(e)
        for site in e['sites']:site['name']=preview.NAMES[site['pattern']]
        e['motif']=' + '.join(site['name'] for site in e['sites'])
        base.append(e)
    # Left column: two context-specific examples and one disc-restricted example.
    # Right column: two-family GAF, four-family GAF, then GAF+Cg.
    examples=[extras[0],base[0],extras[1],base[1],base[2],base[3]]
    for e in examples:
        values=np.asarray(e['actual']);labels=np.asarray(e['labels'])
        if values.shape!=(8,len(e['sequence'])) or not np.isfinite(values).all():raise ValueError('Bad native maps')
        e['mean_active_context_ig']=(labels@values/labels.sum()).tolist()
    draw.save(ROOT/'analysis.json',dict(context=ctx,family=family,native_examples=examples,sources=sources,
        context_definition='Raw eight-context degree. Tile c = mean_carriers(label_c / degree * mean_core_base_IG_c).',
        family_definition='Family degree. Tile f = mean_carriers(mean_active_member_context_IG_f / active_family_count).',
        core_definition='Union of trimmed discovery-seqlet cores per enhancer; signed IG per covered base, then equal enhancer mean.',
        support_definition='Unique discovery-seqlet carriers / all QC enhancers in discovery group; not exhaustive PWM occurrence.',
        profiles='Both signed partitions reconstruct their OWN discovery mean. Inactive contexts/families are zero. '
        'Not row-normalized; not percentages. Raw unweighted context values are also saved separately.',
        caveats='Cumulative groups overlap; independent discoveries, no cross-group motif merging; TF matches are not binding evidence. '
        'Population mean profiles do not prove same-instance cross-family activity. Native examples are selected successful predictions.'))
    print(json.dumps(dict(stage='matched_figure_prepared',context_motifs=sum(len(g['rows']) for g in ctx),
        family_motifs=sum(len(g['rows']) for g in family),native_examples=len(examples))),flush=True)


def label(fig,x,y,text,size=7,color='#222222',weight='normal',**kw):
    return draw.label(fig,x,y,text,size,color,weight,**kw)


def rule(fig,x,y,width):
    from matplotlib.lines import Line2D
    fw,fh=fig.get_size_inches()*72
    fig.add_artist(Line2D([x/fw,(x+width)/fw],[y/fh,y/fh],transform=fig.transFigure,color='#DADDE1',lw=.5))


def nice_limit(values):
    high=float(np.max(np.abs(values)))
    if high<=0:return .001
    power=10**math.floor(math.log10(high))
    return next(v*power for v in (1,2,2.5,5,10) if v*power>=high)


def motif_row(fig,row,x,y,columns,limit):
    from matplotlib.patches import Rectangle
    fw,fh=fig.get_size_inches()*72
    sign=row['sign'];color=previous.COLORS[sign]
    fig.add_artist(Rectangle((x/fw,(y+8)/fh),9/fw,10/fh,transform=fig.transFigure,color=color,lw=0))
    label(fig,x+4.5,y+10,'+' if sign=='positive' else '-',7,'white','bold',ha='center')
    label(fig,x+17,y+11,str(row['rank']),5.8,ha='center')
    q,t,qs,ts,span=orient_pair(row)
    for matrix,start,yy in ((q,qs,y+14),(t,ts,y+1)):
        ax=draw.make_axes(fig,(x+25,yy,104,11));draw.logo(ax,matrix,start=start)
        ax.set(xlim=(0,span),ylim=(0,2.05));ax.axis('off')
    hit=row['match'];label(fig,x+138,y+16,hit['reference']['name'],6.2,weight='bold')
    label(fig,x+138,y+5,f"q={hit['q']:.2g}"+(' ns' if hit['q']>.05 else ''),5.3,'#555555')
    fraction=row['supporting_discovery_enhancers']/row['group_n']
    if not np.isclose(fraction,row['support_fraction']):raise ValueError('Incorrect support')
    ax=draw.make_axes(fig,(x+191,y+11,36,5));ax.barh(0,100,color='#E3E5E8',height=.8)
    ax.barh(0,fraction*100,color='#7F858E',height=.8);ax.set(xlim=(0,100),ylim=(-.5,.5));ax.axis('off')
    label(fig,x+230,y+10,f'{fraction*100:.1f}',5.8)
    v=np.asarray(row['profile'])
    if len(v)==8:v=v[ORDER]
    previous.heat_cells(fig,v,x+261,y+8,96,limit)


def motif_panel(fig,groups,scheme,top):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    selected=[previous.select(g['rows'],'positive')+previous.select(g['rows'],'negative') for g in groups]
    limit=nice_limit([v for rows in selected for r in rows for v in r['profile']])
    label(fig,12,top,'b',13,weight='bold')
    label(fig,32,top,'Degree of pleiotropy' if scheme=='context' else 'Number of active families',8,weight='bold')
    label(fig,275,top,'+ positive',6.3,previous.COLORS['positive'])
    label(fig,335,top,'- negative',6.3,previous.COLORS['negative'])
    cax=draw.make_axes(fig,(595,top+2,153,5))
    bar=fig.colorbar(ScalarMappable(norm=Normalize(-limit*1000,limit*1000),cmap='RdBu_r'),cax=cax,orientation='horizontal')
    bar.set_ticks([-limit*1000,0,limit*1000]);bar.ax.tick_params(labelsize=5.5,pad=1,length=2)
    label(fig,595,top+17,('Context' if scheme=='context' else 'Family')+' contribution (IG/base x 1000)',6)
    half=len(groups)//2
    columns=[draw.CONTEXTS[i] for i in ORDER] if scheme=='context' else ['Embryo','Discs','CNS','Ovary']
    for k,g in enumerate(groups):
        x=32+(k//half)*393;upper=top-45-(k%half)*187
        title='1' if k==0 else (str(len(groups)) if k==len(groups)-1 else '>='+str(k+1))
        label(fig,x,upper,title,10,weight='bold');label(fig,x+30,upper,f"n={g['elements']:,}",6.3)
        label(fig,x+96,upper,'De novo / JASPAR',5.5,ha='center')
        label(fig,x+191,upper,'Support (%)',5.7)
        for j,c in enumerate(columns):label(fig,x+261+(j+.5)*96/len(columns),upper,c,5.2,ha='center')
        label(fig,x+191,upper-9,'0',4.5,'#666666');label(fig,x+227,upper-9,'100',4.5,'#666666',ha='right')
        for sign,offset in (('positive',0),('negative',3)):
            rows=previous.select(g['rows'],sign)
            for j in range(3):
                y=upper-37-(offset+j)*26
                if j<len(rows):motif_row(fig,rows[j],x,y,columns,limit)
                elif j==len(rows):label(fig,x+25,y+11,'No '+('further ' if rows else '')+sign+' motifs retained',6,'#838993')
        rule(fig,x,upper-172,357)
    return [[r['id'] for r in rows] for rows in selected]


def native_view(e):
    seq=np.asarray(e['sequence']);values=np.asarray(e['mean_active_context_ig']);sites=copy.deepcopy(e['sites'])
    # Orient GAF-rich examples towards GAG; other motifs follow their discovered PWM.
    anchor=sites[0];reverse=bool(anchor['reverse']) ^ (anchor['name']=='GAF')
    coord=np.arange(1,len(seq)+1)
    if reverse:
        seq=3-seq[::-1];values=values[::-1];coord=coord[::-1]
        sites=[dict(s,start=len(seq)-s['end'],end=len(seq)-s['start']) for s in sites]
    span=max(s['end'] for s in sites)-min(s['start'] for s in sites)
    width=min(len(seq),max(60,span+18));center=(min(s['start'] for s in sites)+max(s['end'] for s in sites))//2
    lo=max(0,min(len(seq)-width,center-width//2));hi=lo+width
    return seq,values,sites,coord,lo,hi,reverse


def native_card(fig,e,x,top,width):
    seq,values,sites,coord,lo,hi,reverse=native_view(e)
    k=sum(e['labels']);family_count=sum(any(e['labels'][i] for i in f) for f in draw.FAMILIES)
    active=', '.join(c for c,on in zip(draw.CONTEXTS,e['labels']) if on) if k<8 else 'all contexts'
    label(fig,x,top,f"{e['motif']} | degree {k} | {family_count} "+('family' if family_count==1 else 'families'),7.2,weight='bold')
    label(fig,x,top-11,f"{e['id']} | {e['split']} | {active}"+(' | RC' if reverse else ''),5.7,'#555555')
    ax=draw.make_axes(fig,(x+36,top-39,width-36,16))
    ax.plot(np.arange(len(values))+.5,values,color='#656D77',lw=.45);ax.axhline(0,color='#AAAAAA',lw=.3)
    ax.axvspan(lo,hi,color='#E5E9ED',zorder=-1)
    ax.set(xlim=(0,len(seq)),xticks=[.5,len(seq)-.5],xticklabels=[str(coord[0]),str(coord[-1])],yticks=[])
    ax.tick_params(axis='x',labelsize=4.6,pad=1,length=1.5);ax.spines[['left','top','right']].set_visible(False)
    ax.spines['bottom'].set_linewidth(.35)
    ax=draw.make_axes(fig,(x+36,top-103,width-36,38));tick=nice_limit(values[lo:hi]);limit=tick*1.1
    for s in sites:
        ax.axvspan(s['start']-lo,s['end']-lo,color='#EEF1F4',zorder=-1)
        ax.text((s['start']+s['end'])/2-lo,limit*1.03,s['name'],ha='center',va='bottom',fontsize=5.4)
    for j,(base,value) in enumerate(zip(seq[lo:hi],values[lo:hi])):
        draw.glyph(ax,draw.BASES[base],j+.03,0,.94,float(value),draw.DNA_COLORS[base])
    ax.axhline(0,color='#777777',lw=.3)
    ax.set(xlim=(0,hi-lo),ylim=(-limit,limit),xticks=[.5,(hi-lo)/2,hi-lo-.5],
        xticklabels=[str(coord[lo]),str(coord[(lo+hi)//2]),str(coord[hi-1])],yticks=[-tick,0,tick],
        yticklabels=[f'{-tick:.2g}','0',f'{tick:.2g}'])
    ax.set_ylabel('IG/base',fontsize=5.5,labelpad=2);draw.clean(ax);ax.tick_params(labelsize=5,length=1.5)
    ax.set_xlabel('Position in native enhancer (bp)',fontsize=5.5,labelpad=2)


def render(scheme):
    plt=previous.setup_plot()
    analysis=json.loads((ROOT/'analysis.json').read_text());groups=analysis[scheme]
    height=1900 if scheme=='context' else 1526
    fig=plt.figure(figsize=(WIDTH/72,height/72))
    ids=motif_panel(fig,groups,scheme,height-40)
    ctop=height-70-(len(groups)//2)*187-18
    label(fig,12,ctop,'c',13,weight='bold');label(fig,32,ctop,'Native enhancer attribution',8,weight='bold')
    label(fig,770,ctop,'Mean across observed-active contexts',6,'#555555',ha='right')
    for j,e in enumerate(analysis['native_examples']):native_card(fig,e,32+(j%2)*393,ctop-28-(j//2)*139,347)
    dtop=ctop-456
    label(fig,12,dtop,'d',13,weight='bold');label(fig,32,dtop,'Motif disruption',8,weight='bold')
    context_ready=(previous.ROOT/'examples/complete.json').exists()
    # Never disguise old family-target mutant maps as the new context target.
    eroot=previous.ROOT/'examples' if scheme=='context' and context_ready else previous.EXAMPLES
    previous.verify(eroot)
    target='Mean active-context attribution' if eroot!=previous.EXAMPLES else 'Mean active-family attribution'
    label(fig,770,dtop,target,6,'#555555',ha='right')
    plan=json.loads((eroot/'plan.json').read_text())
    done=json.loads((eroot/'complete.json').read_text())
    if draw.sha(eroot/'plan.json')!=done['plan_sha256']:raise ValueError('Changed example plan')
    with np.load(eroot/'sequences.npz',allow_pickle=False) as z:sequence=z['sequence']
    with np.load(eroot/'predictions.npz',allow_pickle=False) as z:prob=z['calibrated_probabilities']
    for j,e in enumerate(plan['examples']):
        with np.load(eroot/f'attribution_{j}.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['weights'],np.asarray(e['weights'],np.float32))
            actual=z['actual']
        previous.example_card(fig,e,j,sequence,prob,actual,32+(j%2)*393,dtop-28-(j//2)*268,347,250,True)
    out=PROJECT/'output/pdf'/f'figure_3_{scheme}_cumulative_native_mutation_20260929.pdf'
    fig.savefig(out,metadata=dict(Title='Figure 3 | '+scheme+' cumulative motifs, native attribution and mutation examples',
        Subject='Calibrated IG64/100; activity-masked discovery. Mutation-logo target: '+target))
    fig.savefig(out.with_suffix('.svg'));fig.savefig(out.with_suffix('.png'),dpi=125)
    # Page-boundary audit complements visual inspection; axis clips are intentional for logo glyphs.
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    for artist in fig.texts:
        b=artist.get_window_extent(renderer)
        if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:outside.append(artist.get_text())
    plt.close(fig)
    if outside:raise ValueError('Text outside figure: '+str(outside))
    draw.save(ROOT/f'{scheme}_figure.json',dict(status='awaiting_visual_QA',scheme=scheme,
        native_target='Mean observed-active calibrated context probabilities',mutation_target=target,
        reused_mutant_target=eroot==previous.EXAMPLES,mutant_source=str(eroot.relative_to(PROJECT)),
        mutant_complete_sha256=draw.sha(eroot/'complete.json'),
        analysis_sha256=draw.sha(ROOT/'analysis.json'),source_sha256=draw.sha(__file__),
        selected_motifs=ids,outputs={str(p.relative_to(PROJECT)):draw.sha(p) for p in (out,out.with_suffix('.svg'))}))
    print(json.dumps(dict(stage='matched_figure_rendered',scheme=scheme,path=str(out),mutation_target=target)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','render'))
    p.add_argument('--scheme',choices=('context','family','both'),default='both');a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:
        for scheme in (('context','family') if a.scheme=='both' else (a.scheme,)):render(scheme)
