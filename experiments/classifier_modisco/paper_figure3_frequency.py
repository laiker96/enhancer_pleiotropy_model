"""Clearer seven-context GAF example and complementary motif-prevalence panel."""
import json

import h5py
import numpy as np

from . import paper_figure3_six_examples as previous

base=previous.base
ROOT=base.PROJECT/'results/figure3_frequency_20260930'
OUTPUT=base.PROJECT/'output/pdf/figure_3_context_positive_frequency_20260930.pdf'
POINTS_ROOT=base.PROJECT/'results/figure3_importance_points_20260930'
POINTS_OUTPUT=base.PROJECT/'output/pdf/figure_3_context_positive_importance_points_20260930.pdf'
CI_ROOT=base.PROJECT/'results/figure3_flyfactor_uncertainty_20260930'
CI_OUTPUT=base.PROJECT/'output/pdf/figure_3_context_positive_flyfactor_uncertainty_20260930.pdf'


def replacement_example(analysis):
    """Reuse saved maps; require a genuine rank-one >=7 seqlet and literal GAGAG."""
    fit=ROOT/'fit_07';done=json.loads((fit/'complete.json').read_text())
    exports=json.loads((base.PROJECT/'results/figure3_ranked_examples_20260929/cecar_results/native_examples.json').read_text())
    source_receipt=next(r for r in exports['selection'] if r['scheme']=='context' and r['group_index']==6)
    assert base.draw.sha(fit/'complete.json')==source_receipt['fit_receipt_sha256']
    assert done['status']=='complete'
    # Only these frozen fit inputs are needed; do not download the full discovery tensors.
    for name in ('motifs.h5','examples.npz'):assert base.draw.sha(fit/name)==done['files'][name]
    source=previous.previous.ROOT/'candidate_export'
    receipt=json.loads((source/'candidates_complete.json').read_text())
    assert base.draw.sha(source/'candidates.json')==receipt['sha256']
    candidates=json.loads((source/'candidates.json').read_text())['examples']
    row=next(r for r in analysis['context'][6]['rows'] if r['sign']=='positive' and r['rank']==1)
    assert row['match']['reference']['name']=='Trl'
    left_trim,right_trim=row['quality']['start'],row['quality']['end']
    with np.load(fit/'examples.npz') as z:metadata=dict(z)
    thresholds=np.asarray(json.loads((base.PROJECT/'results/classifier_binary_breadth_20260921/thresholds.json').read_text())['thresholds'])
    pool=[];checked=[]
    with h5py.File(fit/'motifs.h5','r') as h:
        node=h[row['pattern']+'/seqlets'];indices=node['example_idx'][:]
        for e in candidates:
            if e['degree']!=7:continue
            local=np.flatnonzero(metadata['ids']==e['id'])
            if len(local)!=1:continue
            loc=int(local[0]);np.testing.assert_array_equal(metadata['labels'][loc],e['labels'])
            assert int(metadata['indices'][loc])==e['index']
            p=np.asarray(e['expected_WT']);labels=np.asarray(e['labels'])
            assert ((p>=thresholds)==labels).all() and ((p>=.5)==labels).all()
            values=labels@np.asarray(e['actual'])/7;dna=np.asarray(e['sequence'])
            matches=np.flatnonzero(indices==loc)
            checked.append(dict(id=e['id'],assigned_seqlets=len(matches)))
            for j in matches:
                a,b=int(node['start'][j]),int(node['end'][j]);rc=bool(node['is_revcomp'][j])
                left,right=(b-right_trim,b-left_trim) if rc else (a+left_trim,a+right_trim)
                sequence=np.eye(4)[dna[a:b]]
                if rc:sequence=sequence[::-1,::-1]
                np.testing.assert_array_equal(sequence,node['sequence'][j])
                core=''.join(base.draw.BASES[dna[left:right]])
                hits=[i for i in range(len(core)-4) if core[i:i+5] in ('GAGAG','CTCTC')]
                if not hits:continue
                outside=np.ones(len(dna),bool);outside[max(0,left-2):right+2]=False
                strength=float(values[left:right].mean())
                contrast=strength/max(float(np.abs(values[outside]).mean()),1e-6)
                if contrast<2 or strength<=0:continue
                hit=max(hits,key=lambda i:values[left+i:left+i+5].mean())
                if not (values[left+hit:left+hit+5]>0).all():continue
                record=dict(e,start=left,end=right,reverse=rc,seqlet=int(j),untrimmed_start=a,untrimmed_end=b,
                    native_genomic_start=e['start'],native_genomic_end=e['end'],
                    scheme='context',group_index=6,task=analysis['context'][6]['task'],rank=1,
                    motif_id=row['id'],motif='Trl',match_q=row['match']['q'],core_strength=strength,
                    contrast=contrast,probabilities=p.tolist(),brier=float(((p-labels)**2).mean()),
                    sites=[dict(start=left,end=right,reverse=rc,name='Trl')],
                    mean_active_context_ig=values.tolist(),mean_discovery_ig=values.tolist(),
                    literal_core=core,literal_GAGAG_mean=float(values[left+hit:left+hit+5].mean()))
                pool.append(record)
    if not pool:raise ValueError('No verified clearer literal GAGAG example in saved candidates')
    selected=sorted(pool,key=lambda e:(-e['contrast'],-e['literal_GAGAG_mean'],e['id']))[0]
    audit=dict(checked=checked,candidates=len(pool),selected_id=selected['id'],
        policy='Saved accurately predicted degree7 candidates only; assigned rank-one >=7 motif seqlet; '
               'literal GAGAG/CTCTC with five positive base attributions; rank by native-background contrast.',
        files={str(p.relative_to(base.PROJECT)):base.draw.sha(p) for p in (
            fit/'complete.json',fit/'motifs.h5',fit/'examples.npz',source/'candidates.json')})
    return selected,audit


def frequencies(summary):
    """Percentage of all eligible enhancers with >=1 hit; unlike importance, zero is defined."""
    rows=summary['exact'];hits=np.asarray([r['motif_containing_enhancers'] for r in rows])
    totals=np.asarray([r['group_enhancers'] for r in rows])
    if hits.shape!=(656,8) or (totals<=0).any() or (hits<0).any() or (hits>totals).any():
        raise ValueError('Invalid carrier counts')
    np.testing.assert_array_equal(totals,np.broadcast_to(totals[0],totals.shape))
    return 100*hits/totals


def importance_frequency_panel(fig,summary,top):
    receipt=previous.previous.importance_panel(fig,summary,top)
    fig.axes[-1].set_position([138/base.WIDTH,(top-242)/base.HEIGHT,495/base.WIDTH,214/base.HEIGHT])
    percent=frequencies(summary);gaf=None
    ax=base.draw.make_axes(fig,(790,top-242,468,214))
    for j,row in enumerate(summary['exact']):
        if row['id']==previous.previous.GAF_ID:gaf=j;continue
        ax.plot(np.arange(1,9),percent[j],color='#B3B6BA',alpha=.55,lw=.7,zorder=1)
    if gaf is None:raise ValueError('Missing GAF reference')
    ax.plot(np.arange(1,9),percent[gaf],color=base.GAF_COLOR,lw=2.8,zorder=5)
    ax.set(xlim=(.9,8.1),ylim=(0,60),xticks=np.arange(1,9),yticks=[0,20,40,60],
           xlabel='Degree of pleiotropy',ylabel='Enhancers with motif (%)')
    if percent.max()>60:raise ValueError('Frequency data would be clipped')
    base.draw.clean(ax);ax.tick_params(labelsize=12,length=3,pad=5)
    ax.xaxis.label.set_size(14);ax.yaxis.label.set_size(14)
    ax.xaxis.labelpad=8;ax.yaxis.labelpad=12
    return dict(**receipt,frequency_profiles=656,gaf_frequency_percent=percent[gaf].tolist(),
                frequency_denominator=summary['exact'][gaf]['group_enhancers'],
                frequency_carriers=summary['exact'][gaf]['motif_containing_enhancers'],
                frequency_definition='At least one FIMO p<=1e-4 native-enhancer hit; all-QC denominator; raw unadjusted prevalence')


def mutation_card(fig,*args):
    first=len(fig.texts);record=previous.mutation_card(fig,*args)
    titles=[label for label in fig.texts[first:] if label.get_text()=='GAF']
    if len(titles)!=1:raise ValueError('Expected one removable GAF heading')
    titles[0].remove()
    return record


def importance_points_panel(fig,summary,top):
    """Display the same signed importance values, with points and no prevalence panel."""
    from matplotlib.lines import Line2D
    receipt=previous.previous.importance_panel(fig,summary,top)
    ax=fig.axes[-1]
    for line in list(ax.lines):
        if line.get_transform()!=ax.transData:
            line.remove()  # The original axes-wide zero line extended beyond degrees 1--8.
            continue
        highlighted=line.get_color()==base.GAF_COLOR
        line.set_marker('o');line.set_markersize(5 if highlighted else 2.2)
        line.set_markeredgewidth(0);line.set_clip_on(True)
        line.set_solid_capstyle('butt')
    ax.plot([1,8],[0,0],color='#666666',lw=.5,zorder=0,solid_capstyle='butt',clip_on=True)
    # Small margins preserve complete endpoint dots; no line extends past the data range.
    ax.set_xlim(.95,8.05);ax.spines['bottom'].set_bounds(1,8)
    ax.legend(handles=[Line2D([],[],color=base.GAF_COLOR,lw=2.8,marker='o',
                             markersize=5,markeredgewidth=0,label='GAF')],
              loc='upper left',fontsize=12,frameon=False)
    return dict(**receipt,frequency_shown=False,connected_points=True,
                line_domain=[1,8],xlim=[.95,8.05],values_changed=False)


def importance_ci_panel(fig,summary,top,intervals):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    receipt=importance_points_panel(fig,summary,top);ax=fig.axes[-1]
    if intervals['motif_id']!=previous.previous.GAF_ID:raise ValueError('Wrong band motif')
    row=next(r for r in summary['exact'] if r['id']==intervals['motif_id'])
    if [g['degree'] for g in intervals['groups']]!=list(range(1,9)):
        raise ValueError('Wrong band degree ordering')
    np.testing.assert_array_equal([g['mean'] for g in intervals['groups']],row['means'])
    low=np.asarray([np.nan if g['lower'] is None else g['lower'] for g in intervals['groups']])*1000
    high=np.asarray([np.nan if g['upper'] is None else g['upper'] for g in intervals['groups']])*1000
    ax.fill_between(np.arange(1,9),low,high,color=base.GAF_COLOR,alpha=.20,
                    linewidth=0,zorder=3,clip_on=True)
    handle=(Patch(facecolor=base.GAF_COLOR,alpha=.20,edgecolor='none'),
            Line2D([],[],color=base.GAF_COLOR,lw=2.8,marker='o',markersize=5,markeredgewidth=0))
    ax.legend(handles=[handle],labels=['GAF (95% CI)'],handler_map={tuple:HandlerTuple(ndivide=1)},
              loc='upper left',fontsize=12,frameon=False)
    return dict(**receipt,uncertainty=intervals,grey_profiles_show_means_only=True)


def render(*,importance_only=False,denovo=False,gaf_ci=False):
    from matplotlib.patches import Patch
    if gaf_ci and denovo:raise ValueError('GAF uncertainty requested only for FlyFactorSurvey')
    if gaf_ci:importance_only=True
    output=POINTS_OUTPUT if importance_only else OUTPUT
    result_root=POINTS_ROOT if importance_only else ROOT
    if gaf_ci:output=CI_OUTPUT;result_root=CI_ROOT
    extra_receipts={}
    if denovo:
        from . import paper_figure3_denovo as denovo_plot
        base.previous.verify(denovo_plot.DATA)
        output=denovo_plot.OUTPUT;result_root=denovo_plot.ROOT
        extra_receipts={str(p.relative_to(base.PROJECT)):base.draw.sha(p)
                        for p in (denovo_plot.DATA/'complete.json',base.PROJECT/'experiments/classifier_modisco/paper_figure3_denovo.py')}
    result_root.mkdir(parents=True,exist_ok=True)
    for root in (base.DATA,previous.previous.DATA,previous.previous.screen.SCREEN):base.previous.verify(root)
    analysis=json.loads((base.DATA/'analysis.json').read_text())
    original=json.loads((base.DATA/'native_examples.json').read_text())['examples']
    examples=previous.select_examples(original);replacement,audit=replacement_example(analysis)
    old=next(e for e in examples if e['degree']==7)
    assert replacement['contrast']>old['contrast'] and replacement['core_strength']>old['core_strength']
    examples=[replacement if e['degree']==7 else e for e in examples]
    summary=json.loads((previous.previous.DATA/'importance/summary.json').read_text())
    if gaf_ci:
        from .figure3_importance_uncertainty import gaf_intervals
        intervals=gaf_intervals(summary,previous.previous.DATA/'importance/scores.npz')
        base.draw.save(result_root/'gaf_bootstrap.json',intervals)
        extra_receipts.update({str(p.relative_to(base.PROJECT)):base.draw.sha(p) for p in (
            result_root/'gaf_bootstrap.json',base.PROJECT/'experiments/classifier_modisco/figure3_importance_uncertainty.py')})
    if denovo:
        reference=summary
        summary=json.loads((denovo_plot.DATA/'importance/summary.json').read_text())
        base.draw.save(result_root/'comparison.json',denovo_plot.comparison(summary,reference))
    plt=base.previous.setup_plot();plt.rcParams.update({'font.size':11})
    fig=plt.figure(figsize=(base.WIDTH/72,base.HEIGHT/72))
    selected=base.motif_panel(fig,analysis['context'],base.HEIGHT-38)
    title=[label for label in fig.texts if label.get_text().startswith('Context contribution (IG')]
    if len(title)!=1:raise ValueError('Expected one context-contribution heading')
    title[0].remove()
    previous.native_panel(fig,examples,1190)
    panel=denovo_plot.importance_panel if denovo else (importance_points_panel if importance_only else importance_frequency_panel)
    curves=importance_ci_panel(fig,summary,795,intervals) if gaf_ci else panel(fig,summary,795)
    base.text(fig,16,464,'e',25,weight='bold')
    fig.legend(handles=[Patch(facecolor='#D62728',alpha=.18,label='Edited region'),
                        Patch(facecolor='none',edgecolor='#333333',lw=.9,label='Motif of interest')],
               loc='center left',bbox_to_anchor=(138/base.WIDTH,469/base.HEIGHT),
               ncol=2,frameon=False,fontsize=11,handlelength=1.5,columnspacing=2)
    mutations,seq,prob,actual=previous.previous.mutation_data()
    windows=[mutation_card(fig,e,seq,prob,actual[k],55+k*644,436,598) for k,e in enumerate(mutations)]
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    from matplotlib.text import Text
    for label in fig.findobj(Text):
        if not label.get_visible() or not label.get_text():continue
        b=label.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1>fig.bbox.width+1 or b.y1>fig.bbox.height+1:outside.append(label.get_text())
    if outside:raise ValueError('Text outside page: '+str(outside))
    d_note=('D: unchanged conditional mean importance, connected points, full width; no frequency panel. '
            'Data curves and zero-reference line bounded to degrees 1--8; small margins preserve endpoint markers. '
            if importance_only else
            'D: conditional mean importance and all-enhancer PWM prevalence shown separately; '
            'frequency is raw sequence prevalence, not occupancy or length/GC-adjusted enrichment. ')
    database_note='same FlyFactorSurvey profiles, exact degrees, GAF highlighted, no rescaling or new scans. '
    if gaf_ci:
        d_note+=('GAF shading: pointwise 95% percentile-bootstrap confidence intervals of the carrier mean, '
                 '10000 resamples of whole enhancer scores within each exact degree, seed20260930. '
                 'Grey profiles show means only. Conditional on fixed model and scans, assuming independent '
                 'enhancers; not biological-replicate/model uncertainty or a simultaneous band. ')
    if denovo:
        d_note=('D: all 66 positive information-filtered TF-MoDISco PWMs rescanned on all native QC enhancers '
                'at FIMO p<=1e-4, fixed across exact degrees. Same signed union-of-hit-bases conditional '
                'importance; no frequency panel or new attribution. Separate GAF-matching curves require '
                'best JASPAR match Trl and q<=0.05. Missing hit groups remain missing. ')
        database_note='Same-map discovery makes this descriptive, not independent validation; no reclustering. '
    note=('C: seven-context example replaced with verified current rank-one seqlet containing literal GAGAG '
          'and stronger positive attribution/contrast; native coordinates retained, decreasing on reverse-complement '
          'displays marked RC. '+d_note+
          database_note+
          'E: redundant GAF headings removed; legend defines red edited intervals and outlined motifs. '
          'Existing outcome-selected mutations and all attribution values unchanged.')
    title='Figure 3: de novo motif importance' if denovo else ('Figure 3: GAF importance' if importance_only else 'Figure 3: GAF importance and motif frequency')
    fig.savefig(output,metadata=dict(Title=title,Subject=note))
    fig.savefig(output.with_suffix('.svg'));fig.savefig(output.with_suffix('.png'),dpi=90)
    base.draw.save(result_root/'example_selection.json',dict(example=replacement,audit=audit))
    base.draw.save(result_root/'figure_receipt.json',dict(status='awaiting_visual_QA',selected_motifs=selected,
        importance=curves,native_ids=[e['id'] for e in examples],replacement_id=replacement['id'],
        replacement_contrast=replacement['contrast'],replacement_core_strength=replacement['core_strength'],
        previous_contrast=old['contrast'],previous_core_strength=old['core_strength'],mutation_windows=windows,
        new_inference=False,new_attributions=False,new_scanning=denovo,caption_note=note,
        input_receipts=extra_receipts|{str(p.relative_to(base.PROJECT)):base.draw.sha(p) for p in (
            base.DATA/'complete.json',previous.previous.DATA/'complete.json',
            previous.previous.screen.SELECTED/'complete.json',result_root/'example_selection.json')},
        source_sha256=base.draw.sha(__file__),outputs={str(p.relative_to(base.PROJECT)):base.draw.sha(p)
            for p in (output,output.with_suffix('.svg'),output.with_suffix('.png'))}))
    plt.close(fig);print(json.dumps(dict(stage='figure_rendered',path=str(output))),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--importance-only',action='store_true',help='Connected importance points, no frequency panel')
    parser.add_argument('--gaf-ci',action='store_true',help='FlyFactorSurvey GAF mean with pointwise 95% bootstrap shading')
    args=parser.parse_args()
    render(importance_only=args.importance_only,gaf_ci=args.gaf_ci)
