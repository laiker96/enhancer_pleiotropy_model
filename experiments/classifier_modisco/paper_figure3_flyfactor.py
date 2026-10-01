"""Preserved 0.5-bit B/C panels; current FlyFactorSurvey curves and stronger GAF examples."""
import json

import numpy as np

from . import paper_figure3_flank05 as base
from . import figure3_gaf_screen as screen

ROOT=screen.ROOT
DATA=ROOT/'cecar_results'
GAF_ID='FBgn0013263'
OUTPUT=base.PROJECT/'output/pdf/figure_3_context_positive_flyfactor_gaf_examples_20260929.pdf'


def importance_panel(fig,summary,top):
    from matplotlib.lines import Line2D
    base.text(fig,16,top,'d',25,weight='bold')
    if summary['profiles']!=656 or len(summary['exact'])!=656:
        raise ValueError('All 656 FlyFactorSurvey reference profiles required')
    ax=base.draw.make_axes(fig,(138,top-242,1120,214));gaf=None;count=0
    for row in summary['exact']:
        values=np.asarray([np.nan if v is None else v for v in row['means']])*1000
        if row['id']==GAF_ID:gaf=values;continue
        if np.isfinite(values).any():
            count+=1
            ax.plot(np.arange(1,9),values,color='#B3B6BA',alpha=.55,lw=.7,zorder=1)
    if gaf is None or not np.isfinite(gaf).all():raise ValueError('Incomplete GAF curve')
    ax.plot(np.arange(1,9),gaf,color=base.GAF_COLOR,lw=2.8,zorder=5)
    ax.axhline(0,color='#666666',lw=.5,zorder=0)
    ax.set(xlim=(.9,8.1),xticks=np.arange(1,9),xlabel='Degree of pleiotropy',
           ylabel='Mean motif importance\n(IG / bp × 10³)')
    base.draw.clean(ax);ax.tick_params(labelsize=12,length=3,pad=5)
    ax.xaxis.label.set_size(14);ax.yaxis.label.set_size(14)
    ax.xaxis.labelpad=8;ax.yaxis.labelpad=12
    ax.legend(handles=[Line2D([],[],color=base.GAF_COLOR,lw=2.8,label='GAF')],
              loc='upper left',fontsize=12,frameon=False)
    return dict(database='FlyFactorSurvey',total_profiles=656,visible_profiles=count+1,
                grouping='exact',highlighted=[GAF_ID],legend=['GAF'])


def mutation_data():
    root=screen.SELECTED;done=base.previous.verify(root)
    plan=json.loads((root/'plan.json').read_text())
    assert base.draw.sha(root/'plan.json')==done['plan_sha256']
    assert base.draw.sha(root/'sequences.npz')==plan['sequence_sha256']
    with np.load(root/'sequences.npz') as z:seq=z['sequence']
    with np.load(root/'predictions.npz') as z:prob=z['calibrated_probabilities']
    actual=[]
    if len(plan['examples'])!=2:raise ValueError('Exactly two GAF examples required')
    for j,e in enumerate(plan['examples']):
        w=np.asarray(e['labels'],float);w/=w.sum()
        np.testing.assert_allclose(e['weights'],w)
        with np.load(root/f'attribution_{j}.npz') as z:
            np.testing.assert_allclose(z['weights'],w)
            assert int(z['references'])==100 and int(z['steps'])==64
            actual.append(z['actual'])
    return plan['examples'],seq,prob,actual


def render():
    base.previous.verify(base.DATA);base.previous.verify(DATA)
    base.previous.verify(screen.SCREEN)
    analysis=json.loads((base.DATA/'analysis.json').read_text())
    examples=json.loads((base.DATA/'native_examples.json').read_text())['examples']
    summary=json.loads((DATA/'importance/summary.json').read_text())
    plt=base.previous.setup_plot();plt.rcParams.update({'font.size':11})
    fig=plt.figure(figsize=(base.WIDTH/72,base.HEIGHT/72))
    selected=base.motif_panel(fig,analysis['context'],base.HEIGHT-38)
    base.native_panel(fig,examples,1190)
    curve=importance_panel(fig,summary,795)
    base.text(fig,16,464,'e',25,weight='bold')
    mutations,seq,prob,actual=mutation_data()
    for k,e in enumerate(mutations):
        base.mutation_card(fig,e,seq,prob,actual[k],55+k*644,436,598)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    from matplotlib.text import Text
    for label in fig.findobj(Text):
        if not label.get_visible() or not label.get_text():continue
        b=label.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1>fig.bbox.width+1 or b.y1>fig.bbox.height+1:
            outside.append(label.get_text())
    if outside:raise ValueError('Text outside page: '+str(outside))
    note=('B/C unchanged from the 0.5-bit positive cumulative-context figure. '
          'D: all 656 FlyFactorSurvey profiles scanned on the current IG64/100 maps, '
          'motif-carrier conditional means by exact degree, only GAF highlighted. '
          'E: two outcome-selected illustrative GAF disruptions, one typical shuffle '
          'from ten pairs, not population effect estimates. All attribution panels '
          'use the fixed observed-active-context mean of calibrated probabilities. '
          'FlyFactorSurvey is a separate database check, not independent biological validation.')
    fig.savefig(OUTPUT,metadata=dict(Title='Figure 3: context motifs and GAF',Subject=note))
    fig.savefig(OUTPUT.with_suffix('.svg'));fig.savefig(OUTPUT.with_suffix('.png'),dpi=90)
    base.draw.save(ROOT/'figure_receipt.json',dict(status='awaiting_visual_QA',selected_motifs=selected,
        flank_bits=.5,importance=curve,mutation_target='mean_active_context',
        mutant_ids=[e['id'] for e in mutations],native_ids=[e['id'] for e in examples],
        input_receipts={str(p.relative_to(base.PROJECT)):base.draw.sha(p) for p in (
            base.DATA/'complete.json',DATA/'complete.json',screen.SELECTED/'complete.json',
            screen.SCREEN/'complete.json')},
        source_sha256=base.draw.sha(__file__),caption_note=note,
        outputs={str(p.relative_to(base.PROJECT)):base.draw.sha(p) for p in
                 (OUTPUT,OUTPUT.with_suffix('.svg'),OUTPUT.with_suffix('.png'))}))
    plt.close(fig)
    print(json.dumps(dict(stage='flyfactor_figure_rendered',path=str(OUTPUT))),flush=True)


if __name__=='__main__':render()
