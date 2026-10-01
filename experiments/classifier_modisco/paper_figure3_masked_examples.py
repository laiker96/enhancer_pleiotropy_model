"""Native motif-disruption preview; CPU only, no new attribution or discovery.

The four illustrative WT examples are frozen from the previous catalogue, not
selected by mutant predictions. They are NOT results of the ongoing masked fits.
"""
import argparse
import json
import os
import time

import numpy as np

from . import paper_figure3_hierarchical as old
from .calibrated_motifs import coefficients
from .masked_motifs import specs

ROOT = old.PROJECT / 'results/figure3_masked_examples_20260929'
COHORT = old.PROJECT / 'results/classifier_modisco_20260916/package/cohort.npz'
ORDER = [2, 1, 3, 4, 7, 0, 5, 6]
NAMES = {0: 'GAF', 1: 'Grh', 2: 'Cg'}
SEED = 20260929


def pwm_score(sequence, pwm):
    """Maximum fixed-site log2 odds over the two orientations; uniform background."""
    x = np.asarray(sequence, int)
    p = np.asarray(pwm, float)
    if p.shape != (len(x), 4) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Bad motif or sequence shape')
    p = (p + .001) / (p + .001).sum(1, keepdims=True)
    score = np.log2(p / .25)
    return max(float(score[np.arange(len(x)), x].sum()),
               float(score[np.arange(len(x)), 3-x[::-1]].sum()))


def disruption(sequence, pwm, rng):
    """First composition-preserving shuffle reducing core score by >=3 bits."""
    x = np.asarray(sequence, dtype=np.uint8)
    before = pwm_score(x, pwm)
    for _ in range(20000):
        candidate = rng.permutation(x)
        changed = int((candidate != x).sum())
        after = pwm_score(candidate, pwm)
        if changed >= 2 and after <= before - 3:
            return candidate, dict(changed=changed, score_before=before, score_after=after)
    raise ValueError('No score-reducing shuffle found; do not select by prediction')


def control(sequence, site, excluded, edits, rng):
    """Nearest disjoint native window; same width, base composition and edit count.

    Not selected by attribution or model effect; not assumed biologically neutral.
    """
    width = site['end'] - site['start']
    positions = sorted(range(len(sequence)-width+1),
        key=lambda start: (min(abs(start+width-site['start']), abs(start-site['end'])), start))
    for start in positions:
        end = start+width
        if any(start < b and end > a for a,b in excluded):
            continue
        original = sequence[start:end]
        for _ in range(300):
            candidate = rng.permutation(original)
            if int((candidate != original).sum()) == edits:
                return start, end, candidate
    raise ValueError('No matched flank shuffle found')


def prepare(root=ROOT, examples_path=None):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists():
        raise FileExistsError('Frozen mutation plan already exists')
    examples_path=old.ROOT/'examples.json' if examples_path is None else examples_path
    source=json.loads(examples_path.read_text())
    examples = source['examples']
    rows, _, _, meta, _ = old.inputs()
    with np.load(COHORT, allow_pickle=False) as z:
        cohort = dict(z)
    lookup = {value:i for i,value in enumerate(cohort['ids'])}
    rng = np.random.default_rng(SEED)
    sequences = []; records = []
    for example in examples:
        i = lookup[example['id']]
        full = cohort['sequence'][i].copy()
        offset = int(example['start'] - (cohort['summit'][i]-1024))
        native = np.asarray(example['sequence'], dtype=np.uint8)
        np.testing.assert_array_equal(full[offset:offset+len(native)], native)
        np.testing.assert_array_equal(cohort['labels'][i], example['labels'])
        j = int(example['index'])
        assert str(meta['ids'][j]) == example['id']
        weights = coefficients(np.asarray(example['labels'])[None], 'mean_active_family_means',
                               spec=specs()['mean_active_family_means'])[0]
        changes = []; exclusions = [(s['start'],s['end']) for s in example['sites']]
        for site in example['sites']:
            a,b = site['start'],site['end']
            mutant, scores = disruption(native[a:b], rows[site['pattern']]['trimmed_pwm'], rng)
            changes.append(dict(**site, name=NAMES[site['pattern']], mutation=mutant.tolist(), **scores))
        controls = []
        for change in changes:
            a,b,mutant = control(native, change, exclusions, change['changed'], rng)
            controls.append(dict(start=a,end=b,mutation=mutant.tolist(),changed=change['changed']))
            exclusions.append((a,b))
        variants = [('WT', [])] + [(s['name']+' mut.', [s]) for s in changes]
        if len(changes)>1:
            variants.append(('Double mut.', changes))
        variants.append(('Control', controls))
        variant_records = []
        for name, edits in variants:
            x = full.copy(); allowed = np.zeros(2048, bool)
            for edit in edits:
                a,b = offset+edit['start'],offset+edit['end']
                x[a:b] = edit['mutation']; allowed[a:b] = True
                np.testing.assert_array_equal(np.sort(x[a:b]),np.sort(full[a:b]))
            np.testing.assert_array_equal(x[~allowed],full[~allowed])
            assert np.count_nonzero(x != full) == sum(s['changed'] for s in edits)
            variant_records.append(dict(name=name,index=len(sequences),edits=edits,
                                        changed_bases=int((x!=full).sum())))
            sequences.append(x)
        records.append(dict(**example, variants=variant_records, changes=changes, controls=controls,
            native_offset=offset, weights=weights.tolist(), expected_WT=meta['calibrated_probabilities'][j].tolist()))
    np.savez_compressed(root/'sequences.npz', sequence=np.stack(sequences))
    old.save(root/'plan.json', dict(examples=records, seed=SEED, sequences=len(sequences),
        selection=source.get('selection','Same four WT examples chosen previously from WT IG and labels; no mutant-effect selection'),
        provenance='PREVIEW: prior unmasked catalogue candidates; require crosscheck against new masked discoveries',
        mutations='First shuffled core reducing maximum two-orientation fixed-site log2 PWM odds by >=3 bits; '
                  'uniform background, 0.001 pseudocount; nucleotide composition preserved',
        controls='Nearest disjoint native flanks with same width and edit count; composition-preserving shuffles, '
                 'not selected by attribution or prediction; not guaranteed neutral',
        attribution='IG64/100 saved WT maps; mean of active-member family means, observed labels fixed',
        inputs={str(p.relative_to(old.PROJECT)):old.sha(p) for p in (
            COHORT,examples_path,old.CATALOGUE/'audit.json',old.CATALOGUE/'metadata.npz')},
        sequence_sha256=old.sha(root/'sequences.npz'), no_new_attributions=True))
    print(json.dumps(dict(stage='mutation_plan_frozen',sequences=len(sequences))),flush=True)


def infer():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU only: explicitly hide CUDA')
    if (ROOT/'predictions.npz').exists():
        raise FileExistsError('Predictions already exist')
    import torch
    from classifier_motifs.attribution import one_hot
    from classifier_motifs.calibrated_attribution import CalibratedTargets,load_classifier
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((ROOT/'plan.json').read_text())
    assert old.sha(ROOT/'sequences.npz') == plan['sequence_sha256']
    assert old.sha(old.CHECKPOINT) == old.CHECKPOINT_SHA
    for path,digest in plan['inputs'].items():
        assert old.sha(old.PROJECT/path) == digest
    with np.load(ROOT/'sequences.npz',allow_pickle=False) as z: sequences=z['sequence']
    wrapper=CalibratedTargets(load_classifier(old.CHECKPOINT,'cpu'),
                             json.loads(old.CALIBRATION.read_text())['enhancers_only'])
    started=time.monotonic(); result={key:[] for key in ('logits','probabilities','calibrated_probabilities')}
    with torch.inference_mode():
        for begin in range(0,len(sequences),8):
            out=wrapper.endpoints(one_hot(sequences[begin:begin+8],'cpu'))
            for key in result: result[key].append(out[key].numpy())
    result={key:np.concatenate(values) for key,values in result.items()}
    assert all(np.isfinite(v).all() for v in result.values())
    differences=[]
    for example in plan['examples']:
        wt=result['calibrated_probabilities'][example['variants'][0]['index']]
        differences.extend(np.abs(wt-example['expected_WT']))
    assert max(differences)<.005, max(differences)
    np.savez_compressed(ROOT/'predictions.npz',**result)
    old.save(ROOT/'inference_receipt.json',dict(status='complete',device='cpu',threads=4,
        seconds=time.monotonic()-started,sequences=len(sequences),maximum_WT_replay_error=float(max(differences)),
        plan_sha256=old.sha(ROOT/'plan.json'),predictions_sha256=old.sha(ROOT/'predictions.npz'),
        checkpoint_sha256=old.CHECKPOINT_SHA,calibration_sha256=old.sha(old.CALIBRATION),
        model_code_sha256=old.sha(__file__),torch=torch.__version__,
        readout='Mean forward/RC probability, then saved enhancer-only sigmoid calibration; no recalibration',
        no_new_attributions=True,no_training=True))
    print(json.dumps(dict(stage='example_inference_complete',seconds=time.monotonic()-started,
                         maximum_WT_replay_error=float(max(differences)))),flush=True)


def display(example, full_sequences):
    native=np.asarray(example['sequence']); actual=np.asarray(example['actual'])
    lo=max(0,min(e['start'] for v in example['variants'] for e in v['edits'])-3)
    hi=min(len(native),max(e['end'] for v in example['variants'] for e in v['edits'])+3)
    variants=full_sequences[[v['index'] for v in example['variants']],
                           example['native_offset']:example['native_offset']+len(native)]
    reverse=bool(example['sites'][0]['reverse']) ^ (example['sites'][0]['pattern']==0)
    sites=[dict(name=s['name'],start=s['start'],end=s['end']) for s in example['changes']]
    coordinates=np.arange(1,len(native)+1)
    if reverse:
        native=3-native[::-1]; actual=actual[:,::-1]; variants=3-variants[:,::-1]
        coordinates=coordinates[::-1]; lo,hi=len(native)-hi,len(native)-lo
        sites=[dict(name=s['name'],start=len(native)-s['end'],end=len(native)-s['start']) for s in sites]
    return variants[:,lo:hi],np.asarray(example['weights'])@actual[:,lo:hi],coordinates[lo:hi],sites,lo,reverse


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.cm import ScalarMappable
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,
                         'svg.fonttype':'none','axes.linewidth':.5,'savefig.facecolor':'white'})
    plan=json.loads((ROOT/'plan.json').read_text()); receipt=json.loads((ROOT/'inference_receipt.json').read_text())
    assert receipt['plan_sha256']==old.sha(ROOT/'plan.json')
    assert receipt['predictions_sha256']==old.sha(ROOT/'predictions.npz')
    with np.load(ROOT/'predictions.npz',allow_pickle=False) as z: probabilities=z['calibrated_probabilities']
    with np.load(ROOT/'sequences.npz',allow_pickle=False) as z: sequences=z['sequence']
    out=old.PROJECT/'output/pdf';out.mkdir(exist_ok=True)
    main=out/'figure_3_masked_native_mutations_preview_20260929.pdf'
    sup=out/'figure_3_supplementary_mutation_boxplots_20260929.pdf'
    fig=plt.figure(figsize=(650/72,760/72)); sources=[]
    old.label(fig,62,745,'WT attribution and sequence edits',8)
    old.label(fig,510,745,'Predicted activity probability',8,ha='center')
    for n,example in enumerate(plan['examples']):
        base=569-n*178
        dna,importance,coordinates,sites,lo,reverse=display(example,sequences)
        variants=example['variants']; names=[v['name'] for v in variants]; length=dna.shape[1]
        old.label(fig,10,base+150,'abcd'[n],11,weight='bold')
        families=sum(any(example['labels'][i] for i in members) for members in old.FAMILIES)
        heading=' + '.join(NAMES[s['pattern']] for s in example['sites'])
        old.label(fig,62,base+151,f"{heading} | {example['id']} | {families} active "+('family' if families==1 else 'families')+
                  (' | RC' if reverse else ''),7.5,weight='bold')
        ax=old.make_axes(fig,(62,base+88,323,45))
        bound=max(float(np.abs(importance).max())*1.12,.001)
        for site in sites:
            a,b=site['start']-lo,site['end']-lo
            ax.axvspan(a,b,color='#E9EDF2',zorder=-1)
            ax.text((a+b)/2,bound*1.08,site['name'],ha='center',va='bottom',fontsize=6,clip_on=False)
        for j,(code,value) in enumerate(zip(dna[0],importance)):
            old.glyph(ax,old.BASES[code],j+.03,0,.94,float(value),old.DNA_COLORS[code])
        ax.axhline(0,color='#777777',lw=.4)
        lower=min(-bound*.15,float(importance.min())*1.12)
        ax.set(xlim=(0,length),ylim=(lower,bound),xticks=[],yticks=[0,round(bound*.7,3)])
        ax.set_ylabel('IG',fontsize=6,labelpad=3);old.clean(ax);ax.spines['bottom'].set_visible(False)
        text=old.make_axes(fig,(62,base+14,323,62))
        for v,name in enumerate(names):
            text.text(-.9,v,name,fontsize=5.7,ha='right',va='center')
            for j,code in enumerate(dna[v]):
                changed=code!=dna[0,j]
                letter=old.BASES[code] if v==0 or changed else '.'
                text.text(j+.5,v,letter,fontsize=min(6.2,323/length*1.2),fontfamily='DejaVu Sans Mono',
                          color='#B32933' if changed else '#777777',ha='center',va='center')
        text.set(xlim=(0,length),ylim=(len(names)-.25,-.6),xticks=[],yticks=[]);text.axis('off')
        text.text(0,len(names)+.25,str(coordinates[0]),fontsize=5.7,color='#555555')
        text.text(length,len(names)+.25,str(coordinates[-1]),fontsize=5.7,color='#555555',ha='right')
        p=probabilities[[v['index'] for v in variants]][:,ORDER]
        activity=old.make_axes(fig,(451,base+121,188,8))
        activity.imshow(np.asarray(example['labels'])[None,ORDER],aspect='auto',vmin=0,vmax=1,
                        cmap=ListedColormap(['white','#333333']))
        activity.set(xticks=np.arange(8),xticklabels=np.asarray(old.CONTEXTS)[ORDER],yticks=[])
        activity.xaxis.tick_top();activity.tick_params(axis='x',length=0,pad=3,labelsize=6)
        for spine in activity.spines.values():spine.set_linewidth(.3)
        activity.text(-.85,0,'Observed',fontsize=6,ha='right',va='center')
        heat=old.make_axes(fig,(451,base+20,188,90))
        heat.imshow(p,aspect='auto',vmin=0,vmax=1,cmap='Blues')
        heat.set(xticks=[],yticks=np.arange(len(names)),yticklabels=names)
        heat.tick_params(axis='y',length=0,labelsize=6,pad=4)
        for i in range(len(names)):
            for j in range(8):
                heat.text(j,i,f'{p[i,j]:.2f}',ha='center',va='center',fontsize=5.8,
                          color='white' if p[i,j]>.6 else '#222222')
        for spine in heat.spines.values():spine.set_visible(False)
        for b in (1.5,4.5,6.5):heat.axvline(b,color='white',lw=1.3)
        sources.append(dict(id=example['id'],variants=names,probabilities=p.tolist(),
                            delta_vs_WT=(p-p[:1]).tolist(),weights=example['weights'],
                            dna=dna.tolist(),WT_IG=importance.tolist(),native_positions=coordinates.tolist()))
    colorbar=fig.colorbar(ScalarMappable(norm=plt.Normalize(0,1),cmap='Blues'),
                         cax=old.make_axes(fig,(473,15,140,6)),orientation='horizontal',ticks=[0,.5,1])
    colorbar.ax.tick_params(length=2,labelsize=6,pad=2)
    fig.savefig(main,metadata=dict(Title='Preview: masked native-enhancer mutation examples',
        Subject='Previously selected motifs; new masked TF-MoDISco results pending. Observed strip: black=active.'))
    fig.savefig(main.with_suffix('.svg'));fig.savefig(main.with_suffix('.png'),dpi=180);plt.close(fig)
    old_receipt=json.loads((old.ROOT/'perturbation_receipt.json').read_text())
    assert old_receipt['result_sha256']==old.sha(old.ROOT/'perturbations_current.npz')
    with np.load(old.ROOT/'perturbations_current.npz',allow_pickle=False) as z:ratios=z['odds_ratios']
    fig=plt.figure(figsize=(540/72,230/72))
    box_source=old.perturbation_panel(fig,ratios,(52,39,472,142))
    fig.savefig(sup,metadata=dict(Title='Supplementary: repeat-disruption odds ratios',
        Subject='Same 30 validation enhancers, ten shuffles and current frozen model as prior Figure 3.'))
    fig.savefig(sup.with_suffix('.png'),dpi=180);plt.close(fig)
    old.save(ROOT/'figure_source.json',dict(status='preview_new_masked_discovery_pending',
        examples=sources,contexts=[old.CONTEXTS[c] for c in ORDER],supplement=box_source,
        plan_sha256=old.sha(ROOT/'plan.json'),receipt_sha256=old.sha(ROOT/'inference_receipt.json'),
        code_sha256=old.sha(__file__),outputs={str(p.relative_to(old.PROJECT)):old.sha(p) for p in (main,sup)},
        no_mutant_attributions=True,visual_qa='Pending final PDF raster review'))
    print(json.dumps(dict(stage='preview_rendered',main=str(main),supplement=str(sup))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','infer','render'))
    args=p.parse_args();{'prepare':prepare,'infer':infer,'render':render}[args.stage]()
