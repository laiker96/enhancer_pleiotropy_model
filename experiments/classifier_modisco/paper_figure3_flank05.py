"""Context-positive Figure 3: 0.5-bit flanks, JASPAR trends, two GAF mutations."""
import copy
import json

import numpy as np

from . import paper_figure3_compact as compact
from . import paper_figure3_context_degree as previous
from . import paper_figure3_hierarchical as draw
from . import paper_figure3_matched as matched
from . import paper_figure3_masked_examples as preview
from .paper_figure3 import orient_pair

PROJECT = draw.PROJECT
ROOT = PROJECT/'results/figure3_context_flank05_20260929'
DATA = ROOT/'cecar_results'
WIDTH = 1320
HEIGHT = 2050
GAF_ID = 'MA0205.3'
GAF_COLOR = '#D55E00'


def text(fig, x, y, value, size=12, **kwargs):
    return draw.label(fig, x, y, value, size=size, **kwargs)


def motif_panel(fig, groups, top):
    from matplotlib.patches import Patch
    text(fig, 16, top, 'b', 25, weight='bold')
    legend = [Patch(facecolor=c, label=n) for c,n in zip(compact.FAMILY_COLORS, draw.FAMILY_NAMES)]
    fig.legend(handles=legend, frameon=False, fontsize=12, loc='lower left',
               bbox_to_anchor=(132/WIDTH, (top-4)/HEIGHT), ncol=4,
               handlelength=1, columnspacing=1.5)
    text(fig, WIDTH-34, top+1, 'Context contribution (IG / bp × 10³)', 12, ha='right')
    chosen = []
    for k, group in enumerate(groups):
        x = 132+(k%4)*294; upper = top-48-(k//4)*390
        text(fig, x, upper, compact.category(k, 8), 17, weight='bold')
        text(fig, x+264, upper, f"n={group['elements']:,}", 11, color='#555555', ha='right')
        if k == 0:
            text(fig, 118, upper, 'Degree of\npleiotropy', 11, ha='right', va='center')
        rows = previous.select(group['rows'], 'positive'); chosen.append([r['id'] for r in rows])
        for j, row in enumerate(rows):
            y = upper-111-j*112
            match = row['match']; name = compact.display_name(match['reference']['name'])
            text(fig, x, y+88, f"{row['rank']}  {name}", 13, weight='bold')
            q = match['q']
            text(fig, x+145, y+88, f'q={q:.2g}'+(' ns' if q>.05 else ''),
                 10.5, color='#555555', ha='right')
            query, target, qs, ts, span = orient_pair(row)
            for matrix, start, yy, title in ((query, qs, y+56, 'De novo'),
                                            (target, ts, y+30, 'JASPAR')):
                ax = draw.make_axes(fig, (x, yy, 145, 24))
                draw.logo(ax, matrix, start=start)
                ax.set(xlim=(0, span), ylim=(0, 2.05)); ax.axis('off')
                if k%4 == 0:
                    text(fig, 118, yy+7, title, 11.5, ha='right')
            support = row['support_fraction']*100
            ax = draw.make_axes(fig, (x, y+10, 94, 10))
            ax.barh(0, 100, height=.8, color='#E3E4E6')
            ax.barh(0, support, height=.8, color='#555A60')
            ax.set(xlim=(0,100), ylim=(-.5,.5)); ax.axis('off')
            text(fig, x+101, y+10, f'{support:.1f}%', 11)
            if k%4 == 0:
                text(fig, 118, y+9, 'Seqlet carriers', 10.5, ha='right')
            values = np.asarray(row['profile'])[previous.ORDER]*1000
            limits = compact.bar_limits([dict(rows=[row])], 'positive')
            ax = draw.make_axes(fig, (x+179, y+40, 85, 61))
            ax.bar(np.arange(8), values, width=.76, color=compact.CONTEXT_COLORS, linewidth=0)
            ax.axhline(0, color='#444444', lw=.5)
            ax.set(xlim=(-.6,7.6), ylim=limits, xticks=np.arange(8),
                   xticklabels=[draw.CONTEXTS[i].upper() for i in previous.ORDER],
                   yticks=compact.bar_ticks(limits))
            draw.clean(ax); ax.spines['bottom'].set_visible(False)
            ax.tick_params(labelsize=10, length=2, pad=1)
            ax.tick_params(axis='x', rotation=90, length=0, pad=4)
        matched.rule(fig, x, upper-348, 264)
    return chosen


def native_panel(fig, examples, top):
    text(fig, 16, top, 'c', 25, weight='bold')
    for k, source in enumerate(examples):
        e = copy.deepcopy(source); e['motif'] = compact.display_name(e['motif'])
        for s in e['sites']:
            s['name'] = compact.display_name(s['name'])
        first_text, first_axis = len(fig.texts), len(fig.axes)
        compact.native_card(fig, e, 55+(k%4)*316, top-28-(k//4)*183, 284, 'context', 8)
        for label in fig.texts[first_text:]:
            label.set_fontsize(max(label.get_fontsize()*1.13, 10.5))
        for ax in fig.axes[first_axis:]:
            ax.tick_params(labelsize=9.5)
            ax.xaxis.label.set_fontsize(10.5); ax.yaxis.label.set_fontsize(10.5)


def importance_panel(fig, summary, top):
    from matplotlib.lines import Line2D
    text(fig, 16, top, 'd', 25, weight='bold')
    if summary['profiles'] != 296 or len(summary['exact']) != 296:
        raise ValueError('All 296 reference profiles required')
    ax = draw.make_axes(fig, (138, top-242, 1120, 214))
    gaf = None; count = 0
    for row in summary['exact']:
        values = np.asarray([np.nan if v is None else v for v in row['means']])*1000
        if row['id'] == GAF_ID:
            gaf = values; continue
        if np.isfinite(values).any():
            count += 1
            ax.plot(np.arange(1,9), values, color='#B3B6BA', alpha=.65, lw=.8, zorder=1)
    if gaf is None or not np.isfinite(gaf).all():
        raise ValueError('Missing GAF curve')
    ax.plot(np.arange(1,9), gaf, color=GAF_COLOR, lw=2.8, zorder=5)
    ax.axhline(0, color='#666666', lw=.5, zorder=0)
    ax.set(xlim=(.9,8.1), xticks=np.arange(1,9), xlabel='Degree of pleiotropy',
           ylabel='Mean motif importance\n(IG / bp × 10³)')
    draw.clean(ax); ax.tick_params(labelsize=12, length=3, pad=5)
    ax.xaxis.label.set_size(14); ax.yaxis.label.set_size(14)
    ax.xaxis.labelpad=8; ax.yaxis.labelpad=12
    ax.legend(handles=[Line2D([],[],color=GAF_COLOR,lw=2.8,label='GAF'),
                       Line2D([],[],color='#B3B6BA',lw=1.2,label='Other JASPAR profiles')],
              loc='upper left', fontsize=12, frameon=False, ncol=2)
    return dict(visible_profiles=count+1, total_profiles=296, grouping='exact', highlighted=[GAF_ID])


def mutation_data():
    # A separately authorized recalculation, if available, contains only the two GAF examples.
    recalculated = ROOT/'gaf_mutations'
    if (recalculated/'complete.json').exists():
        previous.verify(recalculated)
        plan = json.loads((recalculated/'plan.json').read_text())
        with np.load(recalculated/'sequences.npz', allow_pickle=False) as z:
            seq = z['sequence']
        with np.load(recalculated/'predictions.npz', allow_pickle=False) as z:
            p = z['calibrated_probabilities']
        actual = []
        for j, e in enumerate(plan['examples']):
            with np.load(recalculated/f'attribution_{j}.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['weights'], np.asarray(e['weights'], np.float32))
                actual.append(z['actual'])
        return recalculated, plan['examples'], seq, p, actual, 'mean_active_context'
    root, plan, seq, probs, actual = compact.mutation_inputs('family')
    return root, plan['examples'][:2], seq, probs, actual[:2], 'mean_active_family'


def mutation_card(fig, e, seq, probs, actual, x, top, width):
    from matplotlib.patches import Patch
    if any(s['pattern'] != 0 for s in e['sites']):
        raise ValueError('GAF-only mutation examples required')
    dna, _, coord, sites, lo, reverse = preview.display(e, seq)
    ix = [v['index'] for v in e['variants']]; names = [v['name'] for v in e['variants']]
    if names != ['WT','GAF mut.','Control']:
        raise ValueError('Expected WT, single GAF disruption and matched control')
    native = actual[:, e['native_offset']:e['native_offset']+len(e['sequence'])]
    if reverse:
        native = native[:,::-1]
    values = native[:,lo:lo+dna.shape[1]]
    high = matched.nice_limit(values); upper = high*1.1
    lower = min(float(values.min())*1.1, -high*.08)
    text(fig, x, top, 'GAF', 15, weight='bold')
    text(fig, x+width, top, f"{e['id']} · {sum(e['labels'])} contexts", 11.5,
         color='#555555', ha='right')
    for v,name in enumerate(names):
        yy = top-65-v*61
        ax = draw.make_axes(fig, (x+85, yy, width-92, 42))
        for site in sites:
            ax.axvspan(site['start']-lo, site['end']-lo, color='#E8EDF1', zorder=-1)
        for j,(base,value) in enumerate(zip(dna[v], values[v])):
            draw.glyph(ax, draw.BASES[base], j+.03, 0, .94, float(value), draw.DNA_COLORS[base])
        ax.axhline(0,color='#777777',lw=.4)
        ax.set(xlim=(0,dna.shape[1]),ylim=(lower,upper)); ax.axis('off')
        text(fig,x+77,yy+15,name,11, color=previous.VARIANT_COLORS[name],ha='right')
        letters = draw.make_axes(fig,(x+85,yy-12,width-92,10))
        for j,base in enumerate(dna[v]):
            symbol = draw.BASES[base] if v==0 or base!=dna[0,j] else '·'
            letters.text(j+.5,.5,symbol,ha='center',va='center',fontsize=9,
                         fontfamily='DejaVu Sans Mono', color='#B2182B' if base!=dna[0,j] else '#777777')
        letters.set(xlim=(0,dna.shape[1]),ylim=(0,1)); letters.axis('off')
    text(fig,x+85,top-227,f'IG / bp: {lower:.2g} to {upper:.2g}',10.5,color='#555555')
    text(fig,x+width,top-227,f'{coord[0]}-{coord[-1]} bp'+(' · RC' if reverse else ''),
         10.5,color='#555555',ha='right')
    ax = draw.make_axes(fig,(x+85,top-383,width-92,134)); w=.8/3
    for v,name in enumerate(names):
        ax.bar(np.arange(8)-.4+w*(v+.5), probs[ix[v],previous.ORDER], w,
               color=previous.VARIANT_COLORS[name],lw=0)
    ax.set(xlim=(-.6,7.6),ylim=(0,1.05),xticks=np.arange(8),
           xticklabels=[draw.CONTEXTS[i].upper() for i in previous.ORDER],yticks=[0,.5,1])
    ax.set_ylabel('Predicted probability',fontsize=12,labelpad=8)
    draw.clean(ax); ax.tick_params(labelsize=11,length=3,pad=4)
    for tick,active in zip(ax.get_xticklabels(),np.asarray(e['labels'])[previous.ORDER]):
        tick.set_fontweight('bold' if active else 'normal')
    ax.legend(handles=[Patch(facecolor=previous.VARIANT_COLORS[n],label=n) for n in names],
              frameon=False,fontsize=11,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,
              handlelength=1,columnspacing=1.2)


def render():
    receipt = previous.verify(DATA)
    analysis = json.loads((DATA/'analysis.json').read_text())
    examples = json.loads((DATA/'native_examples.json').read_text())['examples']
    importance = json.loads((DATA/'importance/summary.json').read_text())
    plt = previous.setup_plot(); plt.rcParams.update({'font.size':11})
    fig = plt.figure(figsize=(WIDTH/72,HEIGHT/72))
    selected = motif_panel(fig,analysis['context'],HEIGHT-38)
    native_panel(fig,examples,1190)
    curve = importance_panel(fig,importance,795)
    text(fig,16,464,'e',25,weight='bold')
    root,mutations,seq,probabilities,actual,target = mutation_data()
    for k,e in enumerate(mutations):
        mutation_card(fig,e,seq,probabilities,actual[k],55+k*644,436,598)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside=[]
    from matplotlib.text import Text
    for label in fig.findobj(Text):
        if not label.get_visible() or not label.get_text():
            continue
        bounds=label.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.width+1 or bounds.y1 > fig.bbox.height+1:
            outside.append(label.get_text())
    if outside:
        raise ValueError('Text outside page: '+str(outside))
    out=PROJECT/'output/pdf/figure_3_context_positive_compact_flank05_20260929.pdf'
    note=('B: 0.5-bit flanks, top three positive motifs; seqlet carrier percentages; signed masked '
          'context contributions, individual labeled axes. C: native mean-active-context IG. '
          'D: all 296 JASPAR profiles scanned; conditional importance by exact degree, GAF highlighted. '
          'E: two frozen GAF examples, real WT/mutant/control maps; target '+target+'.')
    fig.savefig(out,metadata=dict(Title='Figure 3: context motifs and GAF',Subject=note))
    fig.savefig(out.with_suffix('.svg'));fig.savefig(out.with_suffix('.png'),dpi=90)
    draw.save(ROOT/'figure_receipt.json',dict(status='awaiting_visual_QA',selected_motifs=selected,
        new_filter=.5,importance=curve,mutant_ids=[e['id'] for e in mutations],mutation_target=target,
        mutation_source=str(root.relative_to(PROJECT)),native_ids=[e['id'] for e in examples],
        data_receipt_sha256=draw.sha(DATA/'complete.json'),source_sha256=draw.sha(__file__),
        caption_note=note,outputs={str(p.relative_to(PROJECT)):draw.sha(p) for p in
                                  (out,out.with_suffix('.svg'),out.with_suffix('.png'))}))
    plt.close(fig)
    print(json.dumps(dict(stage='flank05_figure_rendered',path=str(out),mutation_target=target)),flush=True)


if __name__=='__main__':
    render()
