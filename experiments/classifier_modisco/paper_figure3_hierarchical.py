"""Figure 3 from the frozen 2026-09-29 hierarchical motif catalogue.

Fetch only selected saved native maps; replay the saved repeat perturbations on
the CURRENT classifier on CPU; render without changing any analysis inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'results/figure3_hierarchical_20260929'
CATALOGUE = PROJECT / 'results/classifier_hierarchical_motifs_20260929/cecar_results'
PERTURBATIONS = PROJECT / 'results/repeat_relationship_20260918/cecar_results/perturbations'
CHECKPOINT = PROJECT / 'results/classifier_calibrated_context_20260921/inputs/best_model.pt'
CALIBRATION = PROJECT / 'results/classifier_calibration_20260921/calibrators.json'
CONTEXTS = ('ab', 'e13', 'e5', 'ead', 'hid', 'lb', 'o', 'wid')
FAMILIES = ((2, 1), (3, 4, 7), (0, 5), (6,))  # display: embryo, discs, CNS, ovary
FAMILY_NAMES = ('Embryo', 'Discs', 'CNS', 'Ovary')
BASES = np.asarray(list('ACGT'))
REMOTE_PROJECT = '/home/aaltamirano/ian_enhancer_pleiotropy'
REMOTE_NATIVE = REMOTE_PROJECT + '/experiments/classifier_calibrated_motifs_20260927/native'
REMOTE = 'aaltamirano@cecar.fcen.uba.ar'
CHECKPOINT_SHA = '7dc8eb721fbd4f292eba95ff2ea44399cfd1aae5a596e8d56e6d7ff25a95628a'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inputs():
    receipt = json.loads((CATALOGUE / 'complete.json').read_text())
    assert receipt['status'] == 'complete'
    for name in ('audit.json', 'summary.json', 'jaspar_matches.json', 'metadata.npz', 'occurrences.npz'):
        assert sha(CATALOGUE / name) == receipt['files'][name], name
    audit = json.loads((CATALOGUE / 'audit.json').read_text())
    summary = json.loads((CATALOGUE / 'summary.json').read_text())
    matches = json.loads((CATALOGUE / 'jaspar_matches.json').read_text())
    with np.load(CATALOGUE / 'metadata.npz', allow_pickle=False) as z:
        metadata = dict(z)
    with np.load(CATALOGUE / 'occurrences.npz', allow_pickle=False) as z:
        occurrences = dict(z)
    return audit['groups'][0]['rows'], summary, matches, metadata, occurrences


def family_maps(actual):
    return np.stack([actual[list(members)].mean(0) for members in FAMILIES])


def odds_ratios(logits):
    """One geometric-mean mutation/WT OR per enhancer, variant and context."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 4 or logits.shape[2:] != (5, 8) or not np.isfinite(logits).all():
        raise ValueError('Expected finite enhancer x shuffle x five variants x eight contexts')
    np.testing.assert_allclose(logits[:, :, 0], np.repeat(logits[:, :1, 0], logits.shape[1], axis=1), atol=1e-5)
    return np.exp((logits[:, :, 1:] - logits[:, :, :1]).mean(1))


def candidates(metadata, occurrences):
    """Outcome-aware illustrative selection, not an enrichment or validation test."""
    labels = metadata['labels']
    family_labels = np.stack([labels[:, list(f)].any(1) for f in FAMILIES], 1)
    breadth = family_labels.sum(1)
    proposals = []
    for title, pat, wanted_breadth in (('GAF - two families', 0, 2),
                                      ('GAF - four families', 0, 4),
                                      ('Grh - imaginal discs', 1, 1)):
        pre = f'pattern_{pat}_'
        ix = occurrences[pre + 'indices']
        keep = occurrences[pre + 'selected'] & (breadth[ix] == wanted_breadth)
        keep &= metadata['quality_pass'][ix].all(1)
        if pat == 1:
            keep &= family_labels[ix, 1]
        strength = occurrences[pre + 'positive_strength_per_bp']
        rows = np.flatnonzero(keep)
        rows = sorted(rows, key=lambda j: (-strength[j], str(metadata['ids'][ix[j]])))[:100]
        for j in rows:
            proposals.append(dict(category=title, index=int(ix[j]),
                sites=[dict(pattern=pat, start=int(occurrences[pre+'start'][j]),
                            end=int(occurrences[pre+'end'][j]),
                            reverse=bool(occurrences[pre+'is_revcomp'][j]))]))
    # Explicitly discovered, nonoverlapping GAF/Cg cores, not inferred PWM hits.
    for j in np.flatnonzero(occurrences['pattern_0_selected']):
        i = int(occurrences['pattern_0_indices'][j])
        if breadth[i] < 3:
            continue
        for k in np.flatnonzero(occurrences['pattern_2_selected'] & (occurrences['pattern_2_indices'] == i)):
            a, b = int(occurrences['pattern_0_start'][j]), int(occurrences['pattern_0_end'][j])
            c, d = int(occurrences['pattern_2_start'][k]), int(occurrences['pattern_2_end'][k])
            gap = max(a, c) - min(b, d)
            if not 0 <= gap <= 25 or max(b, d) - min(a, c) > 52:
                continue
            proposals.append(dict(category='GAF + Cg', index=i, sites=[
                dict(pattern=0, start=a, end=b, reverse=bool(occurrences['pattern_0_is_revcomp'][j])),
                dict(pattern=2, start=c, end=d, reverse=bool(occurrences['pattern_2_is_revcomp'][k]))]))
    if len({p['category'] for p in proposals}) != 4:
        raise ValueError('One of the four predeclared example categories has no candidates')
    return proposals


def fetch_examples():
    ROOT.mkdir(exist_ok=True, parents=True)
    if (ROOT / 'examples.json').exists():
        raise FileExistsError('Saved example selection already exists; do not silently change it')
    rows, summary, matches, meta, occ = inputs()
    proposals = candidates(meta, occ)
    indices = sorted({p['index'] for p in proposals})
    # Read-only file slicing on CECAR: no model, gradients, discovery or jobs.
    code = '''import json,numpy as np
from pathlib import Path
r=Path(REMOTE_NATIVE)
with np.load(r/'metadata.npz',allow_pickle=False) as z:
    ids=z['ids']; seq=z['sequence']; lengths=z['length']
actual=np.load(r/'actual.npy',mmap_mode='r',allow_pickle=False)
out=[dict(index=i,id=str(ids[i]),sequence=seq[i,:int(lengths[i])].tolist(),actual=actual[i,:,:int(lengths[i])].tolist()) for i in INDICES]
print(json.dumps(out,allow_nan=False))
'''.replace('REMOTE_NATIVE', repr(REMOTE_NATIVE)).replace('INDICES', repr(indices))
    command = 'OPENBLAS_NUM_THREADS=1 ' + REMOTE_PROJECT + '/runtime/venv_modisco_20260916/bin/python -c ' + shlex.quote(code)
    payload = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', REMOTE, command], text=True)
    records = {r['index']: r for r in json.loads(payload)}
    chosen = []
    for category in dict.fromkeys(p['category'] for p in proposals):
        qualified = []
        for p in (p for p in proposals if p['category'] == category):
            record = records[p['index']]
            assert record['id'] == str(meta['ids'][p['index']])
            actual = np.asarray(record['actual'])
            assert actual.shape == (8, int(meta['length'][p['index']]))
            maps = family_maps(actual)
            balanced = maps.mean(0)
            occupied = np.zeros(len(balanced), bool)
            for site in p['sites']:
                occupied[max(0, site['start']-2):site['end']+2] = True
                # Independent identity check against the locally archived core sums.
                pat = site['pattern']; pre = f'pattern_{pat}_'
                hit = (occ[pre+'indices'] == p['index']) & (occ[pre+'start'] == site['start']) & (occ[pre+'end'] == site['end'])
                np.testing.assert_allclose(actual[:, site['start']:site['end']].sum(1),
                                           occ[pre+'context_signed_sum'][np.flatnonzero(hit)[0]], atol=1e-7)
            base = np.abs(balanced[~occupied]).mean()
            strengths = [balanced[s['start']:s['end']].mean() for s in p['sites']]
            contrast = min(strengths) / max(base, 1e-6)
            if contrast < 2 or min(strengths) <= 0:
                continue
            qualified.append(dict(**p, contrast=float(contrast),
                                  selection_score=float(min(strengths)*min(contrast, 10))))
        if not qualified:
            raise ValueError('No clear example under fixed contrast criterion: ' + category)
        scores = np.asarray([p['selection_score'] for p in qualified])
        target = float(np.quantile(scores, .8))
        pick = min(qualified, key=lambda p: (abs(p['selection_score']-target), records[p['index']]['id']))
        record = records[pick['index']]
        i = pick['index']
        chosen.append(dict(**pick, **{k:v for k,v in record.items() if k!='index'},
            eligible=len(qualified), evaluated=sum(p['category']==category for p in proposals),
            active_contexts=[c for c,b in zip(CONTEXTS,meta['labels'][i]) if b],
            labels=meta['labels'][i].tolist(), split=str(meta['split'][i]),
            chrom=str(meta['chrom'][i]), start=int(meta['start'][i]), end=int(meta['end'][i])))
    save(ROOT / 'examples.json', dict(examples=chosen, fetched_enhancers=len(indices),
        remote_source=REMOTE_NATIVE, method='IG64 / 100 shuffled references; saved native actual maps only',
        selection='Illustrative, outcome-aware: top 100 site-strength candidates per single-site category; '
                  'all close GAF/Cg pairs; >=2x native-background contrast; nearest 80th percentile of '
                  'site-strength x capped contrast; no claim of random/representative sampling',
        local_core_sum_crosscheck=True, metadata_sha256=sha(CATALOGUE/'metadata.npz'),
        occurrences_sha256=sha(CATALOGUE/'occurrences.npz')))
    print(json.dumps(dict(stage='examples_complete', ids=[r['id'] for r in chosen])), flush=True)


def infer_cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This replay is CPU-only; set CUDA_VISIBLE_DEVICES to empty')
    ROOT.mkdir(exist_ok=True, parents=True)
    if (ROOT/'perturbations_current.npz').exists():
        raise FileExistsError('Perturbations already replayed')
    import torch
    from classifier_motifs.attribution import one_hot
    from classifier_motifs.calibrated_attribution import CalibratedTargets, load_classifier
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    old_done = json.loads((PERTURBATIONS/'complete.json').read_text())
    assert sha(PERTURBATIONS/'predictions.npz') == old_done['files']['predictions.npz']
    with np.load(PERTURBATIONS/'predictions.npz', allow_pickle=False) as z:
        old = dict(z)
    expected = ['WT','GA_disrupted','CA_GT_disrupted','both_disrupted','nonrepeat_sham']
    assert old['variants'].tolist() == expected
    wrapper = CalibratedTargets(load_classifier(CHECKPOINT, 'cpu'),
                                json.loads(CALIBRATION.read_text())['enhancers_only'])
    flat = old['sequence_codes'].reshape(-1, 2048)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    result = {k:[] for k in ('logits','probabilities','calibrated_probabilities')}
    started = time.monotonic()
    print(json.dumps(dict(stage='cpu_replay_start', enhancers=len(old['ids']), unique_sequences=len(unique))), flush=True)
    with torch.inference_mode():
        for start in range(0, len(unique), 16):
            values = wrapper.endpoints(one_hot(unique[start:start+16], 'cpu'))
            for key in result:
                result[key].append(values[key].numpy())
            if start % 160 == 0:
                print(json.dumps(dict(stage='cpu_replay', complete=min(start+16,len(unique)), total=len(unique), seconds=time.monotonic()-started)), flush=True)
    arrays = {k:np.concatenate(v)[inverse].reshape(30,10,5,8) for k,v in result.items()}
    for values in arrays.values():
        assert np.isfinite(values).all()
    ratios = odds_ratios(arrays['logits'])
    # Replay WT endpoints against the independently saved IG metadata.
    with np.load(CATALOGUE/'metadata.npz', allow_pickle=False) as z:
        meta = dict(z)
    lookup = {v:i for i,v in enumerate(meta['ids'])}
    ix = np.asarray([lookup[v] for v in old['ids']])
    np.testing.assert_array_equal(meta['labels'][ix], old['labels'])
    assert np.all(meta['split'][ix] == 'validation')
    difference = np.abs(arrays['calibrated_probabilities'][:,0,0] - meta['calibrated_probabilities'][ix])
    assert difference.max() < .005, float(difference.max())
    np.savez_compressed(ROOT/'perturbations_current.npz', **arrays, odds_ratios=ratios,
        ids=old['ids'], labels=old['labels'], variants=old['variants'], changed_bases=old['changed_bases'])
    save(ROOT/'perturbation_receipt.json', dict(status='complete', device='cpu', torch=torch.__version__,
        threads=4, enhancers=30, shuffles=10, unique_sequences=len(unique),
        seconds=time.monotonic()-started, checkpoint_sha256=CHECKPOINT_SHA,
        calibration_sha256=sha(CALIBRATION), input_sha256=sha(PERTURBATIONS/'predictions.npz'),
        result_sha256=sha(ROOT/'perturbations_current.npz'),
        maximum_WT_probability_replay_error=float(difference.max()),
        readout='exp(mean_shuffle(mean_forward_RC_mutant_logit - mean_forward_RC_WT_logit))',
        probability_readouts_saved_separately=True, no_new_attributions=True, no_training=True,
        selection='Same 30 validation enhancers and saved perturbation sequences as the template; no reselection by current-model effect'))
    print(json.dumps(dict(stage='cpu_replay_complete', seconds=time.monotonic()-started,
                         max_endpoint_error=float(difference.max()), max_OR=float(ratios.max()))), flush=True)


MOTIF_COLORS = {0:'#D55E00', 1:'#009E73', 2:'#0072B2', 12:'#8B395A'}
DNA_COLORS = ('#238B45', '#2878B5', '#E9A825', '#D64A44')
NAMES = {0:'GAF / Trl', 1:'Grh', 2:'Cg', 12:'TTK-like', 13:'PHDP-like', 14:'ATCTAT'}


def aligned_pwm(row, match):
    """Preserve Tomtom offset/strand; rotate the complete alignment for GA/CA display."""
    query = np.asarray(row['trimmed_pwm'])
    target = np.asarray(match['reference']['pwm'])
    if match['orientation'] == '-': target = target[::-1, ::-1]
    qs, ts = max(0, match['offset']), max(0, -match['offset'])
    span = max(qs+len(query), ts+len(target))
    overlap = min(qs+len(query), ts+len(target))-max(qs,ts)
    assert overlap == match['overlap']
    q = np.zeros((span,4)); t = q.copy()
    q[qs:qs+len(query)] = query; t[ts:ts+len(target)] = target
    left, right = max(qs,ts), min(qs+len(query),ts+len(target))
    if row['consensus'] in ('CTCTCTCTC', 'TGTGTGTGTGTGTGTGT'):
        q, t = q[::-1,::-1], t[::-1,::-1]
        left, right = span-right, span-left
    return q, t, (left,right)


def information(pwm):
    pwm = np.asarray(pwm, float)
    return pwm*np.maximum(0, 2+np.sum(pwm*np.log2(np.maximum(pwm,1e-30)),axis=1))[:,None]


def glyph(ax, letter, x, y, width, height, color):
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D
    if abs(height) < 1e-8: return
    path = TextPath((0,0), letter, size=1,
                    prop=FontProperties(family='DejaVu Sans', weight='bold'))
    box = path.get_extents()
    trans = Affine2D().translate(-box.x0,-box.y0).scale(width/box.width, abs(height)/box.height)
    # Negative observed contributions are upright letters below their zero line.
    trans = trans.translate(x, y if height > 0 else y-abs(height))
    ax.add_patch(PathPatch(path, transform=trans+ax.transData, facecolor=color,
                          edgecolor='none', clip_on=True))


def logo(ax, pwm, start=0, height=1):
    for j, values in enumerate(information(pwm)):
        bottom = 0
        for i in np.argsort(values):
            h = float(values[i])*height
            glyph(ax, BASES[i], start+j+.03, bottom, .94, h, DNA_COLORS[i])
            bottom += h


def make_axes(fig, rectangle):
    w,h = fig.get_size_inches()*72
    x,y,dx,dy = rectangle
    return fig.add_axes([x/w,y/h,dx/w,dy/h])


def label(fig, x, y, text, size=7.5, color='#222222', weight='normal', **kwargs):
    w,h = fig.get_size_inches()*72
    return fig.text(x/w,y/h,text,fontsize=size,color=color,weight=weight,**kwargs)


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2, width=.5, pad=2, labelsize=7)
    ax.spines[['left','bottom']].set_linewidth(.5)


def motif_card(fig, row, match, pattern, rect):
    x,y,w,h = rect
    sign_color = '#2878B5' if row['sign']=='positive' else '#C94444'
    from matplotlib.patches import Rectangle
    fw,fh = fig.get_size_inches()*72
    fig.add_artist(Rectangle((x/fw,(y+h-12)/fh),11/fw,11/fh,
                             transform=fig.transFigure,color=sign_color,lw=0))
    label(fig,x+5.5,y+h-9,'+' if row['sign']=='positive' else '-',8,'white','bold',ha='center')
    name = NAMES.get(pattern, match['reference']['name'] if match['q']<=.05 else row['consensus'])
    label(fig,x+16,y+h-9,name,8,MOTIF_COLORS.get(pattern,'#222222'),'bold')
    label(fig,x+w,y+h-9,f"q={match['q']:.2g}" + (' (ns)' if match['q']>.05 else ''),6.4,ha='right')
    q,t,overlap = aligned_pwm(row,match)
    for matrix,yy,tag in ((q,y+33,'De novo'),(t,y+7,'JASPAR')):
        ax = make_axes(fig,(x+31,yy,w-33,20))
        ax.axvspan(*overlap, color='#EFF3F6', zorder=-1)
        logo(ax,matrix)
        ax.set(xlim=(0,len(matrix)),ylim=(0,2.05)); ax.axis('off')
        label(fig,x,yy+6,tag,6.3,'#666666')
        if tag == 'JASPAR':
            label(fig,x,yy-1,match['reference']['name'],5.5,'#666666')
    return dict(pattern_index=pattern,query=q.tolist(),reference=t.tolist(),overlap=list(overlap))


def display_example(example):
    seq = np.asarray(example['sequence'])
    actual = np.asarray(example['actual'])
    anchor = example['sites'][0]
    reverse = bool(anchor['reverse']) ^ (anchor['pattern']==0)
    sites = [dict(s) for s in example['sites']]
    if reverse:
        seq = 3-seq[::-1]
        actual = actual[:,::-1]
        sites = [dict(s,start=len(seq)-s['end'],end=len(seq)-s['start']) for s in sites]
    width = max(40,max(s['end'] for s in sites)-min(s['start'] for s in sites)+12)
    mid = (min(s['start'] for s in sites)+max(s['end'] for s in sites))//2
    left = max(0,min(len(seq)-width,mid-width//2)); right=left+width
    return seq, family_maps(actual), sites, (left,right), reverse


def example_card(fig, e, rect):
    x,y,w,h = rect
    seq,maps,sites,(left,right),reverse = display_example(e)
    family_labels = [bool(np.asarray(e['labels'])[list(f)].any()) for f in FAMILIES]
    breadth = sum(family_labels)
    name = e['category'].replace(' - two families','').replace(' - four families','').replace(' - imaginal discs','')
    title = f'{name} | {breadth} ' + ('family' if breadth==1 else 'families')
    label(fig,x+29,y+h-7,title,8,weight='bold')
    label(fig,x+w,y+h-7,e['id']+(' RC' if reverse else ''),6.5,'#666666',ha='right')
    whole = make_axes(fig,(x+29,y+h-26,w-33,12))
    whole.plot(np.arange(len(seq))+.5,maps.mean(0),lw=.5,color='#636363')
    whole.axvspan(left,right,color='#DDE4EA',zorder=-1)
    whole.axhline(0,lw=.35,color='#999999')
    whole.set_xlim(0,len(seq)); whole.axis('off')
    # Fixed scale within an example; an explicit 0.05 probability-unit bar in every card.
    maximum = float(np.max(np.abs(maps[:,left:right])))
    scale = max(.06, np.ceil(maximum/.02)*.02)
    ax = make_axes(fig,(x+29,y+14,w-33,h-49))
    offsets = np.asarray([6.,4.,2.,0.])*scale
    for f,baseline in enumerate(offsets):
        ax.axhline(baseline,color='#BFC5C9',lw=.3,zorder=-1)
        for pos in range(left,right):
            glyph(ax,BASES[seq[pos]],pos+.025,baseline,.95,float(maps[f,pos]),DNA_COLORS[seq[pos]])
    for site in sites:
        ax.axvspan(site['start'],site['end'],color=MOTIF_COLORS[site['pattern']],alpha=.08,zorder=-2)
        ax.plot([site['start'],site['end']],[offsets[0]+1.02*scale]*2,
                color=MOTIF_COLORS[site['pattern']],lw=1.4,clip_on=False)
        ax.text((site['start']+site['end'])/2,offsets[0]+1.06*scale,
                'GAF' if site['pattern']==0 else NAMES[site['pattern']],
                color=MOTIF_COLORS[site['pattern']],fontsize=6,ha='center',va='bottom')
    ax.set(xlim=(left,right),ylim=(-1.02*scale,7.25*scale))
    ax.set_yticks(offsets,FAMILY_NAMES,fontsize=6.7)
    for text,active in zip(ax.get_yticklabels(),family_labels):
        text.set_color('#222222' if active else '#929292')
        text.set_weight('bold' if active else 'normal')
    ticks = np.unique(np.linspace(left,right,4).round().astype(int))
    ax.set_xticks(ticks,[str(len(seq)-i if reverse else i) for i in ticks])
    ax.tick_params(axis='y',length=0,pad=3)
    ax.tick_params(axis='x',length=2,width=.4,labelsize=6.5,pad=1)
    ax.spines[['top','left','right']].set_visible(False); ax.spines['bottom'].set_linewidth(.4)
    ax.plot([right-1,right-1],[-.9*scale,-.9*scale+.05],color='#222222',lw=.7)
    ax.text(right-2,-.9*scale+.025,'0.05',ha='right',va='center',fontsize=5.8,
            bbox=dict(facecolor='white',edgecolor='none',pad=.2))
    return dict(id=e['id'],reverse_display=reverse,zoom=[left,right],
                scale_per_track=scale,family_maps=maps.tolist(),display_sequence=''.join(BASES[seq]),
                sites=sites,native_coordinates=[e['chrom'],e['start'],e['end']],
                family_labels=family_labels,selection_contrast=e['contrast'])


def metric(summary, pattern, group, name, sign='positive'):
    values = summary['patterns'][pattern]['signs'][sign]
    j = values['metrics'].index(name)
    row = values['groups'][group]
    return float(row['mean'][j]), np.asarray(row['descriptive_block_95ci'][j],float), int(row['n'][j])


def frequency_panel(fig, summary, rows, rect):
    ax = make_axes(fig,rect)
    groups = ('family_1','family_ge_2','family_ge_3','family_ge_4')
    x = np.arange(4)
    source = []
    for j,row in enumerate(rows):
        if row['sign']!='positive': continue
        vals = [metric(summary,j,g,'selected_seqlet_carrier_fraction') for g in groups]
        means = np.asarray([v[0] for v in vals])*100
        ci = np.asarray([v[1] for v in vals])*100
        color = MOTIF_COLORS.get(j,'#C4C4C4')
        ax.plot(x,means,color=color,lw=1.3 if j in (0,1,2) else .6,zorder=5 if j in (0,1,2) else 1)
        if j in (0,1,2):
            ax.fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.12,lw=0)
            ax.plot(x,means,'o',ms=2.5,color=color,zorder=6)
            ax.text(3.12,means[-1],NAMES[j],color=color,fontsize=7,va='center')
        source.append(dict(pattern_index=j,percent=means.tolist(),ci95_percent=ci.tolist()))
    ax.set(xticks=x,xticklabels=['1','≥2','≥3','4'],xlim=(-.15,3.85),ylim=(0,36),yticks=[0,10,20,30])
    ax.set_xlabel('Degree of pleiotropy (context families)',fontsize=7.3,labelpad=4)
    ax.set_ylabel('Enhancers with a contributing seqlet (%)',fontsize=7.3,labelpad=4)
    clean(ax)
    return source


def sharing_panel(fig,summary,rect):
    ax = make_axes(fig,rect)
    source=[]
    for j,y in zip((0,1,2),(2,1,0)):
        mean,ci,n = metric(summary,j,'all','effective_families_supported')
        color=MOTIF_COLORS[j]
        ax.errorbar(mean,y,xerr=[[mean-ci[0]],[ci[1]-mean]],fmt='o',ms=4,
                    color=color,lw=1,capsize=2)
        ax.text(1.05,y+.20,f'n={n:,}',fontsize=6.4,color='#666666')
        source.append(dict(pattern_index=j,mean=mean,ci95=ci.tolist(),n=n))
    ax.set(xlim=(1,4.05),ylim=(-.55,2.55),xticks=[1,2,3,4],yticks=[2,1,0],yticklabels=[NAMES[j] for j in (0,1,2)])
    ax.set_xlabel('Effective contributing families',fontsize=7.3,labelpad=4)
    clean(ax);ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
    return source


def perturbation_panel(fig, values, rect):
    from matplotlib.lines import Line2D
    ax=make_axes(fig,rect)
    colors=('#D55E00','#0072B2','#7B5AA6','#A7A7A7')
    labels=('GA-repeat disruption','CA/GT-repeat disruption','Both','Matched control')
    # Same box-summary convention as the template: quartiles/whiskers in log2 OR.
    from matplotlib.cbook import boxplot_stats
    source=[]
    for v in range(4):
        stats=boxplot_stats(np.log2(values[:,v,:]),whis=1.5)
        drawn=[]
        for c,stat in enumerate(stats):
            item={k:2**np.asarray(stat[k]) for k in ('med','q1','q3','whislo','whishi','fliers')}
            drawn.append(item)
            source.append(dict(context=CONTEXTS[c],variant=labels[v],n=len(values),
                               **{k:np.asarray(a).tolist() for k,a in item.items()}))
        bp=ax.bxp(drawn,positions=np.arange(8)+(v-1.5)*.19,widths=.15,patch_artist=True,
                  showfliers=True,manage_ticks=False,
                  boxprops=dict(facecolor=colors[v],edgecolor=colors[v],linewidth=.6,alpha=.75),
                  medianprops=dict(color='#222222',linewidth=.6),
                  whiskerprops=dict(color=colors[v],linewidth=.6),capprops=dict(color=colors[v],linewidth=.6),
                  flierprops=dict(marker='o',markersize=1.3,markerfacecolor=colors[v],markeredgewidth=0,alpha=.65))
    upper=max(3,float(np.ceil(values.max()*1.02/.5)*.5))
    ax.axhline(1,ls=(0,(3,2)),color='#555555',lw=.65,zorder=0)
    ax.set(xticks=np.arange(8),xticklabels=CONTEXTS,xlim=(-.6,7.6),ylim=(0,upper))
    ax.set_ylabel('Predicted odds ratio\n(mutant / WT)',fontsize=7.5)
    ax.set_xlabel('Context',fontsize=7.5,labelpad=3)
    ax.legend([Line2D([],[],lw=5,color=c,alpha=.75) for c in colors],labels,
              ncol=4,loc='lower center',bbox_to_anchor=(.5,1.05),frameon=False,
              fontsize=6.7,columnspacing=1.4,handlelength=1)
    clean(ax)
    return source


def supplementary(path,rows,matches,summary):
    """Full, unselected positive and negative discovery catalogues, 12 per page."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        for sign in ('positive','negative'):
            fig=plt.figure(figsize=(595/72,842/72))
            label(fig,32,811,sign.capitalize()+' family-balanced motifs',12,weight='bold')
            label(fig,32,795,'De novo and best JASPAR match; common catalogue, without post-hoc reclustering',8,'#555555')
            label(fig,423,773,'Seqlet carriers (%)',8,weight='bold')
            label(fig,414,758,'1 family',7);label(fig,472,758,'4 families',7)
            label(fig,533,765,'Seqlets',7,ha='center')
            indices=[j for j,row in enumerate(rows) if row['sign']==sign]
            for k,j in enumerate(indices):
                row=rows[j]; match=matches['best'][row['id'].replace('/','__')]
                y=701-k*56
                name=NAMES.get(j,match['reference']['name'] if match['q']<=.05 else 'Unassigned')
                label(fig,32,y+33,f'{k+1}. {name}',7.5,weight='bold')
                label(fig,32,y+21,f"Best: {match['reference']['name']}; q={match['q']:.2g}"+
                      (' (ns)' if match['q']>.05 else ''),6,'#555555')
                label(fig,32,y+9,row['pattern'],5.8,'#666666')
                q,t,overlap=aligned_pwm(row,match)
                for mat,yy,tag in ((q,y+25,'De novo'),(t,y+3,'JASPAR')):
                    ax=make_axes(fig,(181,yy,207,19)); ax.axvspan(*overlap,color='#EFF3F6',zorder=-1)
                    logo(ax,mat); ax.set(xlim=(0,len(mat)),ylim=(0,2.05)); ax.axis('off')
                    label(fig,152,yy+6,tag,5.8,'#666666')
                for xx,group in ((435,'family_1'),(490,'family_ge_4')):
                    v,_,_=metric(summary,j,group,'selected_seqlet_carrier_fraction',sign)
                    label(fig,xx,y+23,f'{100*v:.1f}',8,ha='center')
                label(fig,533,y+23,f"{row['seqlets']:,}",8,ha='center')
            label(fig,32,45,'Carrier percentages count retained discovery seqlets, not all genomic PWM matches. Logos: 0-2 bits.',7,'#555555')
            label(fig,32,32,'Tomtom q values test motif similarity; ns denotes q > 0.05. Best matches do not establish TF binding.',7,'#555555')
            pdf.savefig(fig);plt.close(fig)


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'pdf.fonttype':42,
                         'svg.fonttype':'none','axes.linewidth':.5,'savefig.facecolor':'white'})
    rows,summary,matches,meta,occ=inputs()
    examples=json.loads((ROOT/'examples.json').read_text())
    receipt=json.loads((ROOT/'perturbation_receipt.json').read_text())
    assert receipt['checkpoint_sha256']==CHECKPOINT_SHA
    assert receipt['result_sha256']==sha(ROOT/'perturbations_current.npz')
    with np.load(ROOT/'perturbations_current.npz',allow_pickle=False) as z:
        ratios=z['odds_ratios']; ids=z['ids'].tolist()
        np.testing.assert_allclose(odds_ratios(z['logits']),ratios,rtol=1e-12)
    out=PROJECT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    main=out/'figure_3bcde_hierarchical_20260929.pdf'
    sup=out/'figure_3_hierarchical_motif_catalogue_20260929.pdf'
    fig=plt.figure(figsize=(520/72,800/72))
    for letter,y in (('B',782),('C',603),('D',313),('E',145)):
        label(fig,8,y,letter,13,weight='bold')
    # Six highest-support patterns: top three for EACH discovery sign, unchanged order.
    source_b=[]
    for k,j in enumerate((0,1,2,12,13,14)):
        row=rows[j];match=matches['best'][row['id'].replace('/','__')]
        source_b.append(motif_card(fig,row,match,j,(43+(k%3)*157,695-(k//3)*80,144,74)))
    source_c=[]
    for k,e in enumerate(examples['examples']):
        source_c.append(example_card(fig,e,(18+(k%2)*256,460-(k//2)*139,233,133)))
    label(fig,269,315,'Position within native enhancer (bp)',6.5,'#666666',ha='center')
    source_d=dict(frequencies=frequency_panel(fig,summary,rows,(49,201,253,98)),
                  sharing=sharing_panel(fig,summary,(374,201,130,98)))
    source_e=perturbation_panel(fig,ratios,(49,36,455,91))
    fig.savefig(main,metadata=dict(Title='Figure 3: family-balanced motif contributions',
        Subject='Current retained-head background-trained CNN; IG64/ref100; CPU-replayed perturbations'))
    fig.savefig(main.with_suffix('.svg'))
    fig.savefig(main.with_suffix('.png'),dpi=220)
    plt.close(fig)
    supplementary(sup,rows,matches,summary)
    source=dict(panels=dict(B=source_b,C=source_c,D=source_d,E=source_e),
        checkpoint_sha256=CHECKPOINT_SHA,contexts=list(CONTEXTS),families=list(FAMILY_NAMES),
        family_members=[list(m) for m in FAMILIES],discovery='Unmasked mean of four all-member family means',
        population='40,309 QC-passing native enhancers; exploratory all-split discovery',
        perturbation_enhancer_ids=ids,perturbation_receipt=receipt,
        figure_E_scale='Raw mean-F/RC-logit odds ratios; not calibrated-probability ratios',
        inputs={str(p.relative_to(PROJECT)):sha(p) for p in [CATALOGUE/'complete.json',
                CATALOGUE/'audit.json',CATALOGUE/'summary.json',CATALOGUE/'jaspar_matches.json',
                ROOT/'examples.json',ROOT/'perturbations_current.npz',CALIBRATION,CHECKPOINT]},
        code_sha256=sha(__file__),outputs={str(p.relative_to(PROJECT)):sha(p) for p in [main,sup]},
        visual_qa='Pending raster inspection; see docs/figure3_hierarchical_20260929.md')
    save(ROOT/'figure_source.json',source)
    print(json.dumps(dict(stage='render_complete',main=str(main),supplement=str(sup))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('examples','infer','render'))
    args = p.parse_args()
    if args.stage == 'examples': fetch_examples()
    elif args.stage == 'infer': infer_cpu()
    else: render()
