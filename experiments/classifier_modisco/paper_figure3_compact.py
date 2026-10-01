"""Positive-only Figure 3, signed contribution bars and ranked native examples.

Negative discoveries move to scheme-matched supplements. All numerical inputs,
motif ranks, support denominators, mutation inputs and saved IG are unchanged.
"""
import argparse
import copy
import json
import math

import numpy as np

from . import paper_figure3_matched as matched
from . import paper_figure3_context_degree as previous
from . import paper_figure3_hierarchical as draw
from . import paper_figure3_masked_examples as preview
from .paper_figure3 import orient_pair

PROJECT = draw.PROJECT
ROOT = PROJECT / 'results/figure3_compact_20260929'
EXAMPLES = PROJECT / 'results/figure3_ranked_examples_20260929/cecar_results'
WIDTH = 1200
FAMILY_COLORS = ('#8E44AD', '#218C54', '#2468AE', '#D47A16')
CONTEXT_COLORS = [FAMILY_COLORS[next(j for j, f in enumerate(draw.FAMILIES) if c in f)]
                  for c in previous.ORDER]
NAMES = {'Trl': 'GAF', 'grh': 'Grh', 'cg': 'Cg'}


def display_name(name):
    return NAMES.get(name, name)


def category(index, count):
    return '1' if index == 0 else str(count) if index == count-1 else '≥'+str(index+1)


def label(fig, x, y, text, size=9, color='#222222', weight='normal', **kw):
    return draw.label(fig, x, y, text, size, color, weight, **kw)


def select_examples(analysis, scheme):
    previous.verify(EXAMPLES)
    items = json.loads((EXAMPLES/'native_examples.json').read_text())['examples']
    selected = sorted((copy.deepcopy(e) for e in items if e['scheme'] == scheme),
                      key=lambda e: e['group_index'])
    groups = analysis[scheme]
    if [e['group_index'] for e in selected] != list(range(len(groups))):
        raise ValueError('Missing or duplicated native example category')
    for e, g in zip(selected, groups):
        rows = previous.select(g['rows'], 'positive')
        if e['motif_id'] not in {r['id'] for r in rows}:
            raise ValueError('Native example is not a displayed top-three motif')
        if e['rank'] != 1:
            raise ValueError('Expected the verified rank-one exports')
        degree = sum(e['labels']) if scheme == 'context' else sum(
            any(e['labels'][i] for i in f) for f in draw.FAMILIES)
        if not g['task']['low'] <= degree <= g['task']['high'] or degree != e['degree']:
            raise ValueError('Native example outside its stated category')
        scalar = np.asarray(e['labels']) @ np.asarray(e['actual']) / sum(e['labels'])
        np.testing.assert_allclose(scalar, e['mean_active_context_ig'], rtol=0, atol=1e-14)
        if e['core_strength'] <= 0 or e['contrast'] < 2:
            raise ValueError('Example fails the fixed positive-core contrast gate')
        e['motif'] = display_name(e['motif'])
        for site in e['sites']:
            site['name'] = display_name(site['name'])
    return selected


def bar_limits(groups, sign):
    values = np.asarray([v*1000 for g in groups for r in previous.select(g['rows'], sign)
                         for v in r['profile']])
    # Labeled linear limits, including small opposite-sign effects; no data normalization.
    if not len(values):
        return (-1, 1)
    low = -matched.nice_limit([-values.min()]) if values.min() < 0 else 0
    high = matched.nice_limit([values.max()]) if values.max() > 0 else 0
    return (min(low, -.06*high), max(high, -.06*low))


def bar_ticks(limits):
    low, high = limits
    # Avoid colliding labels for a small opposite-sign range next to zero.
    ticks = [0, high] if high >= -low else [low, 0]
    if min(-low, high) >= .3*max(-low, high):
        ticks = [low, 0, high]
    return ticks


def motif_card(fig, row, x, y, scheme, limits):
    name = display_name(row['match']['reference']['name'])
    label(fig, x, y+77, f"{row['rank']}  {name}", 10.5, weight='bold')
    qvalue = row['match']['q']
    label(fig, x+146, y+77, f'q={qvalue:.1g}'+(' ns' if qvalue > .05 else ''),
          8, '#555555', ha='right')
    q, t, qs, ts, span = orient_pair(row)
    for matrix, start, yy in ((q, qs, y+47), (t, ts, y+25)):
        ax = draw.make_axes(fig, (x, yy, 146, 20))
        draw.logo(ax, matrix, start=start)
        ax.set(xlim=(0, span), ylim=(0, 2.05)); ax.axis('off')
    support = 100*row['supporting_discovery_enhancers']/row['group_n']
    if not np.isclose(support, 100*row['support_fraction']):
        raise ValueError('Changed support denominator')
    ax = draw.make_axes(fig, (x, y+8, 93, 7))
    ax.barh(0, 100, height=.8, color='#E3E4E6')
    ax.barh(0, support, height=.8, color='#53585F')
    ax.set(xlim=(0, 100), ylim=(-.5, .5)); ax.axis('off')
    label(fig, x+99, y+8, f'{support:.1f}%', 8.5)
    values = np.asarray(row['profile'])*1000
    if scheme == 'context':
        values = values[previous.ORDER]
        names = [draw.CONTEXTS[i].upper() for i in previous.ORDER]
        colors = CONTEXT_COLORS
    else:
        names = ['Emb', 'Disc', 'CNS', 'O']; colors = FAMILY_COLORS
    ax = draw.make_axes(fig, (x+174, y+26, 87, 46))
    ax.bar(np.arange(len(values)), values, width=.75, color=colors, linewidth=0)
    ax.axhline(0, color='#454545', lw=.5)
    ax.set(xlim=(-.6, len(values)-.4), ylim=limits, xticks=np.arange(len(values)),
           xticklabels=names, yticks=bar_ticks(limits))
    draw.clean(ax)
    ax.tick_params(labelsize=7.5, length=2, pad=1)
    ax.tick_params(axis='x', rotation=90 if scheme == 'context' else 0, length=0, pad=3)
    ax.spines['bottom'].set_visible(False)


def motif_panel(fig, groups, scheme, top, sign, letter='b'):
    from matplotlib.patches import Patch
    if letter:
        label(fig, 12, top, letter, 20, weight='bold')
    legend = [Patch(facecolor=c, label=n) for c, n in zip(FAMILY_COLORS, draw.FAMILY_NAMES)]
    fw, fh = fig.get_size_inches()*72
    fig.legend(handles=legend, frameon=False, fontsize=9, loc='lower left',
               bbox_to_anchor=(36/fw, (top-4)/fh), ncol=4, handlelength=1, columnspacing=1.3)
    label(fig, 639, top, 'Logos: de novo / JASPAR', 9)
    label(fig, 884, top, 'Bars: IG/base × 10³', 9)
    limits = bar_limits(groups, sign); ids = []
    for k, group in enumerate(groups):
        x = 36+(k % 4)*292; upper = top-48-(k//4)*329
        unit = 'contexts' if scheme == 'context' else 'families'
        if k == 0:
            unit = 'context' if scheme == 'context' else 'family'
        label(fig, x, upper, category(k, len(groups))+' '+unit, 13, weight='bold')
        label(fig, x+261, upper, f"n={group['elements']:,}", 9, '#555555', ha='right')
        rows = previous.select(group['rows'], sign); ids.append([r['id'] for r in rows])
        for j, row in enumerate(rows):
            motif_card(fig, row, x, upper-105-j*96, scheme,
                       bar_limits([dict(rows=[row])], sign))
        if not rows:
            label(fig, x, upper-48, 'None retained', 10, '#777777')
        matched.rule(fig, x, upper-307, 263)
    return ids, limits


def native_view(e):
    """Choose a GA-rich display strand for GAF/Clamp, without changing the site."""
    oriented = copy.deepcopy(e)
    anchor = oriented['sites'][0]
    if anchor['name'] in ('GAF', 'Clamp'):
        core = np.asarray(e['sequence'])[anchor['start']:anchor['end']]
        def score(values):
            bases = ''.join(draw.BASES[values])
            return (bases.count('GAG'), bases.count('GA'), bases.count('G')+bases.count('A'))
        reverse = score(3-core[::-1]) > score(core)
        # The shared viewer toggles the supplied orientation once more for GAF.
        anchor['reverse'] = bool(reverse) ^ (anchor['name'] == 'GAF')
    return matched.native_view(oriented)


def native_card(fig, e, x, top, width, scheme, group_count):
    seq, values, sites, coord, lo, hi, reverse = native_view(e)
    group = category(e['group_index'], group_count)
    label(fig, x, top, f"{group} · {e['motif']}"+(' (ns)' if e['match_q'] > .05 else ''), 11.5, weight='bold')
    # Actual degree is retained to distinguish cumulative cohort from this enhancer.
    unit = 'contexts' if scheme == 'context' else 'families'
    if e['degree'] == 1:
        unit = 'context' if scheme == 'context' else 'family'
    label(fig, x, top-15, f"{e['id']} · {e['degree']} {unit}"+(' · RC' if reverse else ''), 8.5, '#555555')
    ax = draw.make_axes(fig, (x+35, top-48, width-38, 21))
    ax.plot(np.arange(len(values))+.5, values, color='#525861', lw=.55)
    ax.axvspan(lo, hi, color='#DDE3E9', zorder=-1)
    ax.axhline(0, color='#AAAAAA', lw=.3)
    ax.set(xlim=(0, len(seq)), xticks=[.5, len(seq)-.5],
           xticklabels=[str(coord[0]), str(coord[-1])], yticks=[])
    ax.tick_params(axis='x', labelsize=7.5, pad=1, length=2)
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.spines['bottom'].set_linewidth(.4)
    high = matched.nice_limit(values[lo:hi]); limit = high*1.1
    ax = draw.make_axes(fig, (x+35, top-121, width-38, 48))
    for s in sites:
        ax.axvspan(s['start']-lo, s['end']-lo, color='#E8EDF1', zorder=-1)
    for j, (base, value) in enumerate(zip(seq[lo:hi], values[lo:hi])):
        draw.glyph(ax, draw.BASES[base], j+.03, 0, .94, float(value), draw.DNA_COLORS[base])
    lower = min(float(values[lo:hi].min())*1.1, -high*.1)
    ax.axhline(0, color='#777777', lw=.35)
    ax.set(xlim=(0, hi-lo), ylim=(lower, limit), xticks=[.5, hi-lo-.5],
           xticklabels=[str(coord[lo]), str(coord[hi-1])], yticks=[0, high],
           yticklabels=['0', f'{high:.2g}'])
    ax.set_ylabel('IG/base', fontsize=8, labelpad=2)
    ax.set_xlabel('Position (bp)', fontsize=8, labelpad=2)
    draw.clean(ax); ax.tick_params(labelsize=7.5, length=2, pad=2)


def mutation_inputs(scheme):
    context_ready = (previous.ROOT/'examples/complete.json').exists()
    root = previous.ROOT/'examples' if scheme == 'context' and context_ready else previous.EXAMPLES
    receipt = previous.verify(root)
    plan = json.loads((root/'plan.json').read_text())
    if draw.sha(root/'plan.json') != receipt['plan_sha256']:
        raise ValueError('Changed mutant plan')
    with np.load(root/'sequences.npz', allow_pickle=False) as z:
        seq = z['sequence']
    with np.load(root/'predictions.npz', allow_pickle=False) as z:
        probabilities = z['calibrated_probabilities']
    actual = []
    for j, e in enumerate(plan['examples']):
        with np.load(root/f'attribution_{j}.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['weights'], np.asarray(e['weights'], np.float32))
            actual.append(z['actual'])
    return root, plan, seq, probabilities, actual


def mutation_card(fig, e, seq, probs, actual, x, top, width):
    from matplotlib.patches import Patch
    dna, _, coord, sites, lo, reverse = preview.display(e, seq)
    ix = [v['index'] for v in e['variants']]; names = [v['name'] for v in e['variants']]
    native = actual[:, e['native_offset']:e['native_offset']+len(e['sequence'])]
    if reverse:
        native = native[:, ::-1]
    values = native[:, lo:lo+dna.shape[1]]
    high = matched.nice_limit(values); upper = high*1.1
    lower = min(float(values.min())*1.1, -high*.08)
    motif = ' + '.join(preview.NAMES[s['pattern']] for s in e['sites'])
    label(fig, x, top, motif, 12, weight='bold')
    label(fig, x+width, top, e['id'], 9, '#555555', ha='right')
    step = 30 if len(names) > 3 else 46
    for v, name in enumerate(names):
        base_y = top-(42 if len(names) > 3 else 53)-v*step
        ax = draw.make_axes(fig, (x+73, base_y, width-80, step-15))
        for site in sites:
            ax.axvspan(site['start']-lo, site['end']-lo, color='#E8EDF1', zorder=-1)
        for k, (base, value) in enumerate(zip(dna[v], values[v])):
            draw.glyph(ax, draw.BASES[base], k+.03, 0, .94, float(value), draw.DNA_COLORS[base])
        ax.axhline(0, color='#777777', lw=.3)
        ax.set(xlim=(0, dna.shape[1]), ylim=(lower, upper)); ax.axis('off')
        label(fig, x+68, base_y+4, name, 8.5, previous.VARIANT_COLORS[name], ha='right')
        letters = draw.make_axes(fig, (x+73, base_y-9, width-80, 8))
        for k, base in enumerate(dna[v]):
            # WT letters plus only changed letters below, avoiding repeated DNA text.
            letter = draw.BASES[base] if v == 0 or dna[v, k] != dna[0, k] else '·'
            letters.text(k+.5, .5, letter, ha='center', va='center', fontsize=6.5,
                         fontfamily='DejaVu Sans Mono',
                         color='#B2182B' if dna[v, k] != dna[0, k] else '#777777')
        letters.set(xlim=(0, dna.shape[1]), ylim=(0, 1)); letters.axis('off')
    label(fig, x+73, top-191, f'IG/base: {lower:.2g} to {upper:.2g}', 8, '#555555')
    label(fig, x+width, top-191, f'{coord[0]}-{coord[-1]} bp'+(' · RC' if reverse else ''),
          8, '#555555', ha='right')
    ax = draw.make_axes(fig, (x+73, top-319, width-80, 110)); w = .8/len(ix)
    for v, name in enumerate(names):
        ax.bar(np.arange(8)-.4+w*(v+.5), probs[ix[v], previous.ORDER], w,
               color=previous.VARIANT_COLORS[name], lw=0)
    ax.set(xlim=(-.6, 7.6), ylim=(0, 1.05), xticks=np.arange(8),
           xticklabels=[draw.CONTEXTS[i].upper() for i in previous.ORDER], yticks=[0, .5, 1])
    ax.set_ylabel('Predicted probability', fontsize=9)
    draw.clean(ax); ax.tick_params(labelsize=8.5, length=2)
    for tick, active in zip(ax.get_xticklabels(), np.asarray(e['labels'])[previous.ORDER]):
        tick.set_fontweight('bold' if active else 'normal')
    ax.legend(handles=[Patch(facecolor=previous.VARIANT_COLORS[n], label=n) for n in names],
              frameon=False, fontsize=8.5, loc='upper center', bbox_to_anchor=(.5, -.19),
              ncol=3, handlelength=1, columnspacing=1)


def save_figure(fig, out, receipt):
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); outside = []
    artists = [*fig.texts, *fig.legends, *[ax.get_legend() for ax in fig.axes if ax.get_legend()]]
    for artist in artists:
        b = artist.get_window_extent(renderer)
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.width or b.y1 > fig.bbox.height:
            outside.append(artist.get_text() if hasattr(artist, 'get_text') else 'legend')
    if outside:
        raise ValueError('Text outside page: '+str(outside))
    text = '\n'.join(t.get_text() for t in fig.texts)
    if any(s in text.lower() for s in ('brier', '| test', '| train', '| validation')):
        raise ValueError('Removed example annotation leaked into figure')
    fig.savefig(out, metadata=dict(Title=receipt['title'], Subject=receipt['caption_note']))
    fig.savefig(out.with_suffix('.svg')); fig.savefig(out.with_suffix('.png'), dpi=105)
    receipt.update(status='awaiting_visual_QA', source_sha256=draw.sha(__file__),
        analysis_sha256=draw.sha(matched.ROOT/'analysis.json'),
        outputs={str(p.relative_to(PROJECT)): draw.sha(p) for p in (out, out.with_suffix('.svg'))})
    draw.save(ROOT/(out.stem+'.json'), receipt)


def render(scheme):
    plt = previous.setup_plot(); plt.rcParams.update({'font.size': 9})
    ROOT.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((matched.ROOT/'analysis.json').read_text()); groups = analysis[scheme]
    examples = select_examples(analysis, scheme)
    bheight = math.ceil(len(groups)/4)*329+65
    cheight = math.ceil(len(examples)/4)*170+35
    height = bheight+cheight+850
    fig = plt.figure(figsize=(WIDTH/72, height/72)); btop = height-33
    ids, limits = motif_panel(fig, groups, scheme, btop, 'positive')
    ctop = btop-bheight
    label(fig, 12, ctop, 'c', 20, weight='bold')
    for k, e in enumerate(examples):
        native_card(fig, e, 36+(k % 4)*292, ctop-28-(k//4)*170, 262, scheme, len(groups))
    dtop = ctop-cheight
    label(fig, 12, dtop, 'd', 20, weight='bold')
    root, plan, seq, probs, actual = mutation_inputs(scheme)
    for k, e in enumerate(plan['examples']):
        mutation_card(fig, e, seq, probs, actual[k], 36+(k % 2)*584, dtop-28-(k//2)*385, 550)
    out = PROJECT/'output/pdf'/f'figure_3_{scheme}_positive_compact_20260929.pdf'
    mutation_target = 'mean_active_context' if root != previous.EXAMPLES else 'mean_active_family'
    save_figure(fig, out, dict(title=f'Figure 3: {scheme} cumulative positive motifs', scheme=scheme,
        selected_motifs=ids, full_panel_range_x1000=limits,
        contribution_axis='Individual labeled linear scales; signed raw IG/base x1000, no normalization.',
        per_motif_limits={r['id']: bar_limits([dict(rows=[r])], 'positive')
                          for g in groups for r in previous.select(g['rows'], 'positive')},
        native_examples=[dict(id=e['id'], group_index=e['group_index'], degree=e['degree'],
                              rank=e['rank'], motif_id=e['motif_id']) for e in examples],
        native_source_sha256=draw.sha(EXAMPLES/'complete.json'), native_target='mean_active_context',
        mutation_target=mutation_target, mutation_source=str(root.relative_to(PROJECT)),
        mutation_receipt_sha256=draw.sha(root/'complete.json'),
        caption_note='B: top three positive discoveries, signed additive context/family contributions; '
                     'support is unique discovery-seqlet carrier percentage, not PWM-scan prevalence. '
                     'C: genuine rank-one discovery seqlets, mean active-context calibrated-probability IG. '
                     'D: unchanged frozen mutation maps, target '+mutation_target+'. '
                     'Logos are de novo above aligned JASPAR; q>0.05 marked ns; illustrative accurate examples.'))
    plt.close(fig)
    # Negative motifs retain their own sign and the same contribution definition.
    fig = plt.figure(figsize=(WIDTH/72, (bheight+35)/72))
    ids, limits = motif_panel(fig, groups, scheme, bheight+2, 'negative', letter=None)
    out = PROJECT/'output/pdf'/f'figure_3_{scheme}_negative_supplement_20260929.pdf'
    save_figure(fig, out, dict(title=f'Supplement: {scheme} cumulative negative motifs', scheme=scheme,
        selected_motifs=ids, full_panel_range_x1000=limits,
        contribution_axis='Individual labeled linear scales; signed raw IG/base x1000, no normalization.',
        per_motif_limits={r['id']: bar_limits([dict(rows=[r])], 'negative')
                          for g in groups for r in previous.select(g['rows'], 'negative')},
        caption_note='Top three negative discoveries per cumulative group where available. Signed context/family '
                     'partition and discovery-carrier support unchanged; de novo above JASPAR, no motif reranking.'))
    plt.close(fig)
    print(json.dumps(dict(stage='compact_figure_ready', scheme=scheme, native_examples=len(examples),
                          mutation_target=mutation_target)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scheme', choices=('context', 'family', 'both'), default='both')
    a = p.parse_args()
    for scheme in ('context', 'family') if a.scheme == 'both' else (a.scheme,):
        render(scheme)
