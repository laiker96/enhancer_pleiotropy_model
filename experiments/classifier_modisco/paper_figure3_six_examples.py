"""Display-only Figure 3 revision: six native examples and 100-bp mutation logos."""
import copy
import json

import numpy as np

from . import paper_figure3_flyfactor as previous

base=previous.base
ROOT=base.PROJECT/'results/figure3_six_examples_20260930'
OUTPUT=base.PROJECT/'output/pdf/figure_3_context_positive_six_examples_wide_20260930.pdf'


def select_examples(examples):
    """Keep exact degrees 1/2; choose four remaining saved cases by core contrast."""
    if len({e['id'] for e in examples})!=len(examples):raise ValueError('Duplicate native example')
    for e in examples:
        if sum(e['labels'])!=e['degree'] or e['core_strength']<=0 or e['contrast']<2:
            raise ValueError('Invalid native example')
        mean=np.asarray(e['labels'])@np.asarray(e['actual'])/e['degree']
        np.testing.assert_allclose(mean,e['mean_active_context_ig'],rtol=0,atol=1e-14)
    def ranked(items):return sorted(items,key=lambda e:(-e['contrast'],-e['core_strength'],e['id']))
    chosen=[ranked([e for e in examples if e['degree']==degree])[0] for degree in (1,2)]
    other=ranked([e for e in examples if e['degree']>2])[:4]
    if len(other)!=4:raise ValueError('Four additional native examples required')
    return chosen+sorted(other,key=lambda e:(e['degree'],e['id']))


def native_panel(fig,examples,top):
    base.text(fig,16,top,'c',25,weight='bold')
    for k,source in enumerate(examples):
        e=copy.deepcopy(source);e['motif']=base.compact.display_name(e['motif'])
        for site in e['sites']:site['name']=base.compact.display_name(site['name'])
        first_text,first_axis=len(fig.texts),len(fig.axes)
        base.compact.native_card(fig,e,55+(k%3)*421,top-28-(k//3)*183,389,'context',8)
        # These are selected illustrations, no longer one example per nested group.
        unit='context' if e['degree']==1 else 'contexts'
        fig.texts[first_text].set_text(f"{e['degree']} {unit} · {e['motif']}"+
                                       (' (ns)' if e['match_q']>.05 else ''))
        reverse=base.compact.native_view(e)[-1]
        fig.texts[first_text+1].set_text(e['id']+(' · RC' if reverse else ''))
        for label in fig.texts[first_text:]:label.set_fontsize(max(label.get_fontsize()*1.13,10.5))
        for ax in fig.axes[first_axis:]:
            ax.tick_params(labelsize=9.5)
            ax.xaxis.label.set_fontsize(10.5);ax.yaxis.label.set_fontsize(10.5)


def mutation_view(e,sequences,actual,width=100):
    """Same native interval, orientation and absolute attribution scale for all variants."""
    if width<1:raise ValueError('Positive window width required')
    n=len(e['sequence']);offset=e['native_offset'];ix=[v['index'] for v in e['variants']]
    dna=sequences[ix,offset:offset+n].copy();values=actual[:,offset:offset+n].copy()
    if dna.shape!=values.shape or dna.shape[0]!=3:raise ValueError('Unaligned mutant maps')
    np.testing.assert_array_equal(dna[0],e['sequence'])
    reverse=bool(e['sites'][0]['reverse']) ^ (e['sites'][0]['pattern']==0)
    coordinates=np.arange(1,n+1)
    def interval(site):
        a,b=site['start'],site['end']
        if not 0<=a<b<=n:raise ValueError('Annotation outside native enhancer')
        return (n-b,n-a) if reverse else (a,b)
    motifs=[interval(s) for s in e['sites']]
    edited=[[interval(s) for s in v['edits']] for v in e['variants']]
    if reverse:dna=3-dna[:,::-1];values=values[:,::-1];coordinates=coordinates[::-1]
    width=min(n,width);center=sum(motifs[0])//2
    lo=max(0,min(n-width,center-width//2));hi=lo+width
    spans=motifs+[s for row in edited for s in row]
    if any(a<lo or b>hi for a,b in spans):raise ValueError('Requested window omits motif or edit')
    return dict(dna=dna[:,lo:hi],values=values[:,lo:hi],coordinates=coordinates[lo:hi],
        motifs=[(a-lo,b-lo) for a,b in motifs],
        edited=[[(a-lo,b-lo) for a,b in row] for row in edited],reverse=reverse,
        native_interval=(n-hi,n-lo) if reverse else (lo,hi))


def mutation_card(fig,e,sequences,probabilities,actual,x,top,width):
    from matplotlib.patches import Patch,Rectangle
    view=mutation_view(e,sequences,actual);values=view['values'];dna=view['dna'];coord=view['coordinates']
    names=[v['name'] for v in e['variants']];ix=[v['index'] for v in e['variants']]
    if names!=['WT','GAF mut.','Control']:raise ValueError('Expected one GAF disruption')
    high=base.matched.nice_limit(values);upper=high*1.1
    lower=min(float(values.min())*1.1,-high*.08)
    base.text(fig,x,top,'GAF',15,weight='bold')
    base.text(fig,x+width,top,f"{e['id']} · {sum(e['labels'])} contexts",11.5,color='#555555',ha='right')
    for v,name in enumerate(names):
        yy=top-71-v*66
        ax=base.draw.make_axes(fig,(x+85,yy,width-92,51))
        for left,right in view['edited'][v]:
            ax.axvspan(left,right,color='#D62728',alpha=.18,lw=0,zorder=-2)
        for left,right in view['motifs']:
            ax.add_patch(Rectangle((left,lower),right-left,upper-lower,facecolor='none',
                                   edgecolor='#333333',lw=.9,zorder=5))
        for j,(nucleotide,value) in enumerate(zip(dna[v],values[v])):
            base.draw.glyph(ax,base.draw.BASES[nucleotide],j+.03,0,.94,float(value),
                            base.draw.DNA_COLORS[nucleotide])
        ax.axhline(0,color='#777777',lw=.4)
        ax.set(xlim=(0,dna.shape[1]),ylim=(lower,upper));ax.axis('off')
        base.text(fig,x+77,yy+20,name,11,ha='right',
                  color='#666666' if name=='Control' else base.previous.VARIANT_COLORS[name])
    base.text(fig,x+85,top-227,f'IG / bp: {lower:.2g} to {upper:.2g}',10.5,color='#555555')
    base.text(fig,x+width,top-227,f'{coord[0]}-{coord[-1]} bp'+(' · RC' if view['reverse'] else ''),
              10.5,color='#555555',ha='right')
    ax=base.draw.make_axes(fig,(x+85,top-383,width-92,134));w=.8/3
    for v,name in enumerate(names):
        ax.bar(np.arange(8)-.4+w*(v+.5),probabilities[ix[v],base.previous.ORDER],w,
               color=base.previous.VARIANT_COLORS[name],lw=0)
    ax.set(xlim=(-.6,7.6),ylim=(0,1.05),xticks=np.arange(8),
           xticklabels=[base.draw.CONTEXTS[i].upper() for i in base.previous.ORDER],yticks=[0,.5,1])
    ax.set_ylabel('Predicted probability',fontsize=12,labelpad=8)
    base.draw.clean(ax);ax.tick_params(labelsize=11,length=3,pad=4)
    for tick,active in zip(ax.get_xticklabels(),np.asarray(e['labels'])[base.previous.ORDER]):
        tick.set_fontweight('bold' if active else 'normal')
    ax.legend(handles=[Patch(facecolor=base.previous.VARIANT_COLORS[n],label=n) for n in names],
              frameon=False,fontsize=11,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,
              handlelength=1,columnspacing=1.2)
    return dict(id=e['id'],window_bp=len(coord),native_interval=list(view['native_interval']),
                reverse=view['reverse'],ylim=[lower,upper],motif_boxes=view['motifs'],edit_shading=view['edited'])


def render():
    for root in (base.DATA,previous.DATA,previous.screen.SCREEN):base.previous.verify(root)
    analysis=json.loads((base.DATA/'analysis.json').read_text())
    originals=json.loads((base.DATA/'native_examples.json').read_text())['examples']
    examples=select_examples(originals)
    summary=json.loads((previous.DATA/'importance/summary.json').read_text())
    plt=base.previous.setup_plot();plt.rcParams.update({'font.size':11})
    fig=plt.figure(figsize=(base.WIDTH/72,base.HEIGHT/72))
    selected=base.motif_panel(fig,analysis['context'],base.HEIGHT-38)
    native_panel(fig,examples,1190);curve=previous.importance_panel(fig,summary,795)
    base.text(fig,16,464,'e',25,weight='bold');mutations,seq,prob,actual=previous.mutation_data()
    windows=[mutation_card(fig,e,seq,prob,actual[k],55+k*644,436,598) for k,e in enumerate(mutations)]
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    from matplotlib.text import Text
    for label in fig.findobj(Text):
        if not label.get_visible() or not label.get_text():continue
        b=label.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1>fig.bbox.width+1 or b.y1>fig.bbox.height+1:
            outside.append(label.get_text())
    if outside:raise ValueError('Text outside page: '+str(outside))
    note=('B/D unchanged. C: exact-degree-1 and -2 cases plus four highest-contrast remaining '
          'saved rank-one native examples; headings give actual enhancer degree. E: unchanged '
          'outcome-selected GAF mutations in 100-bp native windows; no sequence strips; '
          'GAF sites outlined in every variant, shuffled intervals shaded red only where edited. '
          'Absolute IG scales shared across WT/mutant/control within each example. '
          'D averages observed-active-context maps, then union-of-hit-base scores per enhancer, '
          'then equally weights motif-containing enhancers within exact degree groups.')
    fig.savefig(OUTPUT,metadata=dict(Title='Figure 3: context motifs and GAF',Subject=note))
    fig.savefig(OUTPUT.with_suffix('.svg'));fig.savefig(OUTPUT.with_suffix('.png'),dpi=90)
    ROOT.mkdir(exist_ok=True)
    base.draw.save(ROOT/'figure_receipt.json',dict(status='awaiting_visual_QA',selected_motifs=selected,
        importance=curve,native_examples=[{k:e[k] for k in ('id','degree','group_index','motif_id',
            'contrast','core_strength','match_q')} for e in examples],mutation_windows=windows,
        new_inference=False,new_attributions=False,caption_note=note,
        input_receipts={str(p.relative_to(base.PROJECT)):base.draw.sha(p) for p in (
            base.DATA/'complete.json',previous.DATA/'complete.json',previous.screen.SELECTED/'complete.json')},
        source_sha256=base.draw.sha(__file__),
        outputs={str(p.relative_to(base.PROJECT)):base.draw.sha(p) for p in
                 (OUTPUT,OUTPUT.with_suffix('.svg'),OUTPUT.with_suffix('.png'))}))
    plt.close(fig);print(json.dumps(dict(stage='six_example_figure_rendered',path=str(OUTPUT))),flush=True)


if __name__=='__main__':render()
