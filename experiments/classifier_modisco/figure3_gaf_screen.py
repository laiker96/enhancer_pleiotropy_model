"""Bounded frozen-model GAF screen; effect-selected illustrations, not validation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .common import digest,write_json,event
from . import paper_figure3_hierarchical as model
from . import paper_figure3_masked_examples as edits
from . import paper_figure3_mutant_ig as attribution

ROOT=model.PROJECT/'results/figure3_flyfactor_gaf_20260929'
SCREEN=ROOT/'screen'
SELECTED=ROOT/'gaf_mutations'
REPLICATES=10


def seed(ident):
    return int.from_bytes(hashlib.sha256(('figure3_gaf_20260929|'+ident).encode()).digest()[:8],'little')


def mutation_variants(full,native,offset,site,pwm,rng):
    records=[];sequences=[full.copy()];seen=set()
    for replicate in range(REPLICATES):
        for _ in range(200):
            mutant,stats=edits.disruption(native[site['start']:site['end']],pwm,rng)
            if tuple(mutant) not in seen:break
        else:raise ValueError('Insufficient distinct core shuffles')
        seen.add(tuple(mutant))
        change=dict(site,name='GAF',mutation=mutant.tolist(),**stats)
        a,b,control=edits.control(native,site,[(site['start'],site['end'])],stats['changed'],rng)
        ctrl=dict(start=a,end=b,mutation=control.tolist(),changed=stats['changed'])
        for edit in (change,ctrl):
            x=full.copy();left,right=offset+edit['start'],offset+edit['end']
            x[left:right]=edit['mutation'];outside=np.ones(len(full),bool);outside[left:right]=False
            np.testing.assert_array_equal(x[outside],full[outside])
            np.testing.assert_array_equal(np.sort(x[left:right]),np.sort(full[left:right]))
            assert np.count_nonzero(x!=full)==edit['changed']
            sequences.append(x)
        records.append(dict(replicate=replicate,change=change,control=ctrl))
    return sequences,records


def prepare():
    source=ROOT/'candidate_export/candidates.json'
    receipt=json.loads((source.with_name('candidates_complete.json')).read_text())
    assert digest(source)==receipt['sha256']
    data=json.loads(source.read_text());assert 0<len(data['examples'])<=100
    with np.load(edits.COHORT,allow_pickle=False) as z:cohort=dict(z)
    lookup={str(v):i for i,v in enumerate(cohort['ids'])}
    SCREEN.mkdir(exist_ok=False);examples=[];seq=[];excluded=[]
    for e in data['examples']:
        i=lookup[e['id']];full=cohort['sequence'][i].copy()
        offset=int(e['start']-(cohort['summit'][i]-1024));native=np.asarray(e['sequence'],np.uint8)
        np.testing.assert_array_equal(native,full[offset:offset+len(native)])
        np.testing.assert_array_equal(e['labels'],cohort['labels'][i])
        try:variants,records=mutation_variants(full,native,offset,e['sites'][0],data['pwm'],np.random.default_rng(seed(e['id'])))
        except ValueError as error:
            excluded.append(dict(id=e['id'],reason=str(error)));continue
        weights=np.asarray(e['labels'],float);weights/=weights.sum()
        examples.append(dict(e,weights=weights.tolist(),native_offset=offset,
                             base_index=len(seq),replicates=records))
        seq.extend(variants)
    if {e['category'] for e in examples}!={'moderate','broad'}:raise ValueError('Missing candidate category')
    np.savez_compressed(SCREEN/'sequences.npz',sequence=np.stack(seq))
    write_json(SCREEN/'plan.json',dict(examples=examples,excluded=excluded,replicates=REPLICATES,
        selection=data['selection'],source_sha256=digest(source),pwm=data['pwm'],source_motif=data['source_motif'],
        mutations='Ten distinct composition-preserving core shuffles, each reducing two-strand PWM score >=3 bits; '
                  'nearest disjoint native control shuffle matched in width and edit count, preserving its own composition.',
        sequence_sha256=digest(SCREEN/'sequences.npz')))
    event('gaf_screen_prepared',enhancers=len(examples),sequences=len(seq),excluded=excluded)


def effect_metrics(probabilities,weights):
    """A WT plus ten paired mutant/control predictions; positive means activity drop."""
    p=np.asarray(probabilities);w=np.asarray(weights)
    if p.shape!=(1+2*REPLICATES,8) or w.shape!=(8,) or not np.isclose(w.sum(),1):
        raise ValueError('Invalid paired screen inputs')
    mutant_drop=(p[0]-p[1::2])@w
    control_drop=(p[0]-p[2::2])@w
    specific=mutant_drop-control_drop
    return dict(mutant_drop=mutant_drop,control_drop=control_drop,specific_drop=specific,
                median_drop=float(np.median(mutant_drop)),median_control=float(np.median(control_drop)),
                median_specific=float(np.median(specific)),fraction_positive=float((mutant_drop>0).mean()))


def select(plan,predictions):
    previous=json.loads((model.PROJECT/'results/figure3_mutant_ig_20260929/plan.json').read_text())
    with np.load(model.PROJECT/'results/figure3_mutant_ig_20260929/predictions.npz') as z:old_probs=z['calibrated_probabilities']
    baselines={}
    for category,e in zip(('moderate','broad'),previous['examples'][:2]):
        w=np.asarray(e['labels'],float);w/=w.sum();s=e['sites'][0]
        strength=float((w@np.asarray(e['actual']))[s['start']:s['end']].mean())
        wt,mut,ctrl=[v['index'] for v in e['variants']]
        baselines[category]=dict(id=e['id'],strength=strength,
            drop=float((old_probs[wt]-old_probs[mut])@w),control=float((old_probs[wt]-old_probs[ctrl])@w))
    audit=[]
    for j,e in enumerate(plan['examples']):
        ix=e['base_index'];m=effect_metrics(predictions[ix:ix+21],e['weights']);base=baselines[e['category']]
        m={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in m.items()}
        audit.append(dict(index=j,id=e['id'],category=e['category'],split=e['split'],degree=e['degree'],
            core_strength=e['core_strength'],contrast=e['contrast'],**m,
            stronger_attribution=e['core_strength']>base['strength'],
            larger_drop=m['median_drop']>base['drop']))
    chosen=[]
    for category in ('moderate','broad'):
        candidates=[r for r in audit if r['category']==category and r['stronger_attribution'] and
                    r['larger_drop'] and r['median_specific']>0 and r['fraction_positive']>=.8 and
                    abs(r['median_control'])<=.05]
        if not candidates:
            write_json(SCREEN/'selection_audit.json',dict(all_candidates=audit,baselines=baselines,status='no_joint_improvement'))
            raise ValueError('No candidate improves attribution AND median effect for '+category)
        best=sorted(candidates,key=lambda r:(-r['median_specific'],-r['core_strength'],r['id']))[0]
        # Show a typical replicate, never the largest individual drop.
        typical=min(range(REPLICATES),key=lambda r:(abs(best['specific_drop'][r]-best['median_specific']),r))
        chosen.append(dict(best,representative_replicate=typical,eligible=len(candidates)))
    return chosen,dict(all_candidates=audit,baselines=baselines,selected=chosen,
        policy='Outcome-selected illustrations: stronger WT core IG and larger median active-context probability '
               'drop than previous examples; >=80% positive shuffles; absolute median control effect <=0.05. '
               'Rank by median control-adjusted drop; display shuffle closest to median adjusted effect. '
               'All 100 candidates and all ten paired edits retained. Not an unbiased or held-out validation.')


def screen():
    import torch
    from classifier_motifs.attribution import one_hot
    from classifier_motifs.calibrated_attribution import CalibratedTargets,load_classifier
    if not torch.cuda.is_available():raise RuntimeError('Authorized local CUDA required')
    assert digest(model.CHECKPOINT)==model.CHECKPOINT_SHA
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads((SCREEN/'plan.json').read_text());assert digest(SCREEN/'sequences.npz')==plan['sequence_sha256']
    with np.load(SCREEN/'sequences.npz') as z:seq=z['sequence']
    target=CalibratedTargets(load_classifier(model.CHECKPOINT,'cuda'),
        json.loads(model.CALIBRATION.read_text())['enhancers_only']).to('cuda').eval()
    parts=[];start=time.monotonic()
    with torch.inference_mode():
        for begin in range(0,len(seq),32):
            parts.append(target.endpoints(one_hot(seq[begin:begin+32],'cuda'))['calibrated_probabilities'].cpu().numpy())
            if begin%320==0:event('gaf_screen_progress',sequences=min(begin+32,len(seq)),total=len(seq),seconds=time.monotonic()-start)
    probabilities=np.concatenate(parts);assert np.isfinite(probabilities).all()
    replay=max(float(np.abs(probabilities[e['base_index']]-e['expected_WT']).max()) for e in plan['examples'])
    assert replay<.005,replay
    np.savez_compressed(SCREEN/'predictions.npz',calibrated_probabilities=probabilities)
    selected,audit=select(plan,probabilities);write_json(SCREEN/'selection_audit.json',audit)
    SELECTED.mkdir(exist_ok=False);records=[];selected_seq=[]
    for choice in selected:
        e=copy.deepcopy(plan['examples'][choice['index']]);rep=choice['representative_replicate']
        edit=e['replicates'][rep];source_indices=[e['base_index'],e['base_index']+1+2*rep,e['base_index']+2+2*rep]
        variants=[]
        for name,mut,source_index in zip(('WT','GAF mut.','Control'),([], [edit['change']], [edit['control']]),source_indices):
            variants.append(dict(name=name,index=len(selected_seq),edits=mut,
                                 changed_bases=0 if not mut else mut[0]['changed']))
            selected_seq.append(seq[source_index])
        e.update(variants=variants,changes=[edit['change']],controls=[edit['control']],screen_selection=choice)
        records.append(e)
    np.savez_compressed(SELECTED/'sequences.npz',sequence=np.stack(selected_seq))
    write_json(SELECTED/'plan.json',dict(examples=records,sequences=6,seed=20260929,
        selection=audit['policy'],attribution='IG64/100 shared WT dinucleotide references; mean observed-active '
            'calibrated CONTEXT probabilities, same fixed WT labels in every variant.',
        sequence_sha256=digest(SELECTED/'sequences.npz'),screen_plan_sha256=digest(SCREEN/'plan.json'),
        screen_predictions_sha256=digest(SCREEN/'predictions.npz'),selection_sha256=digest(SCREEN/'selection_audit.json')))
    write_json(SCREEN/'complete.json',dict(status='complete',seconds=time.monotonic()-start,
        maximum_WT_replay_error=replay,checkpoint_sha256=model.CHECKPOINT_SHA,
        calibration_sha256=digest(model.CALIBRATION),selected=selected,
        files={name:digest(SCREEN/name) for name in ('plan.json','sequences.npz','predictions.npz','selection_audit.json')}))
    event('gaf_screen_complete',seconds=time.monotonic()-start,selected=selected)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','screen','attribute','run'))
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='screen':screen()
    elif a.stage=='attribute':attribution.ROOT=SELECTED;attribution.run()
    else:
        start=time.monotonic();screen()
        subprocess.run([sys.executable,'-u','-m','classifier_modisco.figure3_gaf_screen','attribute'],
                       check=True,timeout=max(1,1800-(time.monotonic()-start)))
