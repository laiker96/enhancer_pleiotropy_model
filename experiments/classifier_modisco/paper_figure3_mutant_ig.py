"""Accuracy-selected native examples with paired WT/mutant/control IG and bars."""
import argparse
import json
import os
import shlex
import signal
import subprocess
import time

import numpy as np

from . import paper_figure3_hierarchical as old
from . import paper_figure3_masked_examples as preview
from .calibrated_motifs import coefficients
from .masked_motifs import specs

ROOT=old.PROJECT/'results/figure3_mutant_ig_20260929'
THRESHOLDS=old.PROJECT/'results/classifier_binary_breadth_20260921/thresholds.json'
STOP=False


def weights(labels):
    return coefficients(labels,'mean_active_family_means',spec=specs()['mean_active_family_means'])


def candidates(meta,occ,thresholds):
    y=meta['labels'];p=meta['calibrated_probabilities'];w=weights(y)
    family=np.stack([y[:,list(f)].any(1) for f in old.FAMILIES],1)
    correct=((p>=thresholds)==y).all(1)&((p>=.5)==y).all(1)&meta['quality_pass'].all(1)
    error=((p-y)**2).mean(1)
    def site(pat,j):
        pre=f'pattern_{pat}_'
        return dict(pattern=pat,start=int(occ[pre+'start'][j]),end=int(occ[pre+'end'][j]),
                    reverse=bool(occ[pre+'is_revcomp'][j]))
    by_pattern={};result=[]
    for pat in (0,1,2):
        pre=f'pattern_{pat}_';indices=occ[pre+'indices'];width=occ[pre+'end']-occ[pre+'start']
        strength=(occ[pre+'context_signed_sum']*w[indices]).sum(1)/width
        keep=occ[pre+'selected']&correct[indices]&(strength>0)
        by_pattern[pat]=[(int(indices[j]),site(pat,j),float(strength[j])) for j in np.flatnonzero(keep)]
    for category,pat,breadth in (('GAF - two families',0,2),('GAF - four families',0,4),('Grh - discs',1,1)):
        for i,s,strength in by_pattern[pat]:
            if family[i].sum()==breadth and (pat!=1 or family[i,1]):
                result.append(dict(category=category,index=i,sites=[s],brier=float(error[i]),core_strength=strength))
    cg={}
    for i,s,strength in by_pattern[2]:cg.setdefault(i,[]).append((s,strength))
    for i,a,strength in by_pattern[0]:
        if family[i].sum()<3:continue
        for b,other in cg.get(i,[]):
            gap=max(a['start'],b['start'])-min(a['end'],b['end'])
            if 0<=gap<=25 and max(a['end'],b['end'])-min(a['start'],b['start'])<=52:
                result.append(dict(category='GAF + Cg',index=i,sites=[a,b],brier=float(error[i]),
                                   core_strength=min(strength,other)))
    # At most 20 candidate enhancers per category, preferring test over validation
    # over train, then lowest endpoint Brier error. No mutant predictions exist yet.
    selected=[];counts={}
    for category in ('GAF - two families','GAF - four families','Grh - discs','GAF + Cg'):
        pool=[r for r in result if r['category']==category]
        counts[category]=dict(candidates=len(pool),enhancers=len({r['index'] for r in pool}),
            by_split={s:len({r['index'] for r in pool if meta['split'][r['index']]==s}) for s in ('test','validation','train')})
        pool.sort(key=lambda r:({'test':0,'validation':1,'train':2}[str(meta['split'][r['index']])],
                                r['brier'],-r['core_strength'],str(meta['ids'][r['index']])))
        seen=set()
        for r in pool:
            if r['index'] in seen:continue
            seen.add(r['index']);selected.append(r)
            if len(seen)==20:break
        if not seen:raise ValueError('No accurately predicted example: '+category)
    return selected,counts


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'examples.json').exists():raise FileExistsError('Examples already frozen')
    _,_,_,meta,occ=old.inputs()
    threshold=json.loads(THRESHOLDS.read_text())
    assert threshold['checkpoint_sha256']==old.CHECKPOINT_SHA
    assert threshold['calibrators_sha256']==old.sha(old.CALIBRATION)
    np.testing.assert_array_equal(threshold['contexts'],old.CONTEXTS)
    proposals,counts=candidates(meta,occ,np.asarray(threshold['thresholds']))
    old.save(ROOT/'candidate_audit.json',dict(counts=counts,candidates=proposals,thresholds=threshold,
        stage='Before any mutant inference',threshold_file_sha256=old.sha(THRESHOLDS)))
    indices=sorted({r['index'] for r in proposals})
    code='''import json,numpy as np
from pathlib import Path
r=Path(REMOTE_NATIVE)
with np.load(r/'metadata.npz',allow_pickle=False) as z:
    ids=z['ids'];seq=z['sequence'];lengths=z['length']
a=np.load(r/'actual.npy',mmap_mode='r',allow_pickle=False)
print(json.dumps([dict(index=i,id=str(ids[i]),sequence=seq[i,:int(lengths[i])].tolist(),actual=a[i,:,:int(lengths[i])].tolist()) for i in INDICES],allow_nan=False))
'''.replace('REMOTE_NATIVE',repr(old.REMOTE_NATIVE)).replace('INDICES',repr(indices))
    payload=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',old.REMOTE,
        'OPENBLAS_NUM_THREADS=1 '+old.REMOTE_PROJECT+'/runtime/venv_modisco_20260916/bin/python -c '+shlex.quote(code)],text=True)
    records={r['index']:r for r in json.loads(payload)};chosen=[]
    for category in counts:
        qualified=[]
        for p in (v for v in proposals if v['category']==category):
            i=p['index'];r=records[i];assert r['id']==meta['ids'][i]
            actual=np.asarray(r['actual']);w=weights(meta['labels'][i:i+1])[0]
            combined=w@actual;occupied=np.zeros(len(combined),bool)
            for site in p['sites']:
                a,b=site['start'],site['end'];occupied[max(0,a-2):b+2]=True
                pre=f'pattern_{site["pattern"]}_'
                hit=np.flatnonzero((occ[pre+'indices']==i)&(occ[pre+'start']==a)&(occ[pre+'end']==b))[0]
                np.testing.assert_allclose(actual[:,a:b].sum(1),occ[pre+'context_signed_sum'][hit],atol=1e-7)
            strengths=[float(combined[s['start']:s['end']].mean()) for s in p['sites']]
            contrast=min(strengths)/max(float(np.abs(combined[~occupied]).mean()),1e-6)
            if contrast>=2 and min(strengths)>0:
                qualified.append(dict(**p,contrast=contrast))
        if not qualified:raise ValueError('No high-contrast accurate candidate for '+category)
        pick=qualified[0];i=pick['index'];r=records[i]
        chosen.append(dict(**pick,**{k:v for k,v in r.items() if k!='index'},
            labels=meta['labels'][i].tolist(),split=str(meta['split'][i]),chrom=str(meta['chrom'][i]),
            start=int(meta['start'][i]),end=int(meta['end'][i]),eligible=len(qualified),
            active_contexts=[c for c,v in zip(old.CONTEXTS,meta['labels'][i]) if v]))
    old.save(ROOT/'examples.json',dict(examples=chosen,selection='All eight labels correct at both frozen F1 '
        'thresholds and 0.5; prefer test/validation/train, lowest WT Brier, >=2x masked-IG motif contrast. '
        'Illustrative success cases, not unbiased performance estimates. No mutant-effect selection.',
        provenance='Existing native seqlet catalogue; new masked discoveries still pending',
        candidate_audit_sha256=old.sha(ROOT/'candidate_audit.json')))
    preview.prepare(root=ROOT,examples_path=ROOT/'examples.json')
    plan=json.loads((ROOT/'plan.json').read_text())
    plan.update(no_new_attributions=False,attribution='New IG64, 100 WT-derived dinucleotide references shared '
        'by all variants; fixed observed-WT family mask; one weighted calibrated-probability target')
    old.save(ROOT/'plan.json',plan)
    print(json.dumps(dict(stage='accurate_examples_frozen',examples=[dict(id=e['id'],category=e['category'],
        split=e['split'],brier=e['brier'],contrast=e['contrast']) for e in chosen])),flush=True)


def integrate_scalar(target,x,baseline,w,steps=64,internal_batch=32):
    """One VJP of a fixed weighted probability mean; full-input hypothetical IG."""
    import torch
    if (x.shape!=baseline.shape or w.shape!=(len(x),8) or internal_batch<len(x)
            or not torch.isfinite(w).all() or (w<0).any()):raise ValueError('Unaligned scalar IG inputs')
    torch.testing.assert_close(w.sum(1),torch.ones(len(x),device=x.device))
    nodes,quad=np.polynomial.legendre.leggauss(steps)
    nodes=torch.as_tensor((nodes+1)/2,device=x.device,dtype=x.dtype)
    quad=torch.as_tensor(quad/2,device=x.device,dtype=x.dtype)
    integrated=torch.zeros_like(x);group=internal_batch//len(x)
    for first in range(0,steps,group):
        if STOP:raise InterruptedError('Requested checkpoint stop')
        alpha=nodes[first:first+group]
        points=(baseline[None]+alpha[:,None,None,None]*(x-baseline)[None]).flatten(0,1).requires_grad_(True)
        q=target.endpoints(points)['calibrated_probabilities']
        gradient=torch.autograd.grad((q*w.repeat(len(alpha),1)).sum(),points)[0].reshape(len(alpha),*x.shape)
        integrated+=(gradient*quad[first:first+len(alpha),None,None,None]).sum(0)
    hyp=integrated-(integrated*baseline).sum(1,keepdim=True)
    actual=(hyp*x).sum(1)
    with torch.no_grad():difference=((target.endpoints(x)['calibrated_probabilities']-
        target.endpoints(baseline)['calibrated_probabilities'])*w).sum(1)
    return {k:v.detach() for k,v in dict(actual=actual,hypothetical=hyp,
        delta=actual.sum(1)-difference,target_difference=difference).items()}


def run():
    import torch
    from classifier_motifs.attribution import one_hot,dinucleotide_shuffle,seed_for
    from classifier_motifs.calibrated_attribution import CalibratedTargets,load_classifier
    if not torch.cuda.is_available():raise RuntimeError('Authorized local CUDA device required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    plan=json.loads((ROOT/'plan.json').read_text());signature=old.sha(ROOT/'plan.json')
    assert old.sha(ROOT/'sequences.npz')==plan['sequence_sha256']
    with np.load(ROOT/'sequences.npz',allow_pickle=False) as z:seq=z['sequence']
    target=CalibratedTargets(load_classifier(old.CHECKPOINT,'cuda'),
        json.loads(old.CALIBRATION.read_text())['enhancers_only']).to('cuda').eval()
    with torch.inference_mode():
        predictions={k:v.cpu().numpy() for k,v in target.endpoints(one_hot(seq,'cuda')).items()}
    errors=[float(np.abs(predictions['calibrated_probabilities'][e['variants'][0]['index']]-e['expected_WT']).max())
            for e in plan['examples']]
    assert max(errors)<.005
    np.savez_compressed(ROOT/'predictions.npz',**predictions)
    receipts=[];overall=time.monotonic()
    print(json.dumps(dict(stage='local_cuda_IG_start',gpu=torch.cuda.get_device_name(),
        sequences=len(seq),references=100,steps=64,scalar_targets=1,maximum_WT_endpoint_error=max(errors))),flush=True)
    for j,e in enumerate(plan['examples']):
        path=ROOT/f'attribution_{j}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if str(z['signature'])!=signature:raise ValueError('Incompatible saved attribution')
            continue
        ix=[v['index'] for v in e['variants']];n=len(ix);w=np.asarray(e['weights'],np.float32)
        refs=np.stack([dinucleotide_shuffle(seq[ix[0]],seed_for(20260916,e['id'],r)) for r in range(100)])
        state_path=ROOT/f'state_{j}.npz'
        if state_path.exists():
            with np.load(state_path,allow_pickle=False) as z:state=dict(z)
            if str(state['signature'])!=signature:raise ValueError('Incompatible resume')
        else:state=dict(signature=np.asarray(signature),count=np.asarray(0),sum_hyp=np.zeros((n,4,2048)),
            half_actual=np.zeros((2,n,2048)),delta=np.zeros((n,100)),difference=np.zeros((n,100)))
        started=time.monotonic();initial=int(state['count'])
        for r in range(initial,100,2):
            rows=np.repeat(np.arange(n),2);refnums=np.tile(np.arange(r,r+2),n)
            result=integrate_scalar(target,one_hot(seq[np.asarray(ix)[rows]],'cuda'),one_hot(refs[refnums],'cuda'),
                torch.as_tensor(np.broadcast_to(w,(n*2,8)).copy(),device='cuda'))
            values={k:v.cpu().numpy().reshape(n,2,*v.shape[1:]) for k,v in result.items()}
            if not all(np.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite IG result')
            state['sum_hyp']+=values['hypothetical'].sum(1)
            state['half_actual'][r//50]+=values['actual'].sum(1)
            state['delta'][:,r:r+2]=values['delta'];state['difference'][:,r:r+2]=values['target_difference']
            state['count']=np.asarray(r+2)
            partial=state_path.with_suffix('.partial.npz');np.savez_compressed(partial,**state);partial.replace(state_path)
            if r==initial or (r+2)%10==0:
                seconds=time.monotonic()-started
                print(json.dumps(dict(stage='reference_progress',example=j,id=e['id'],references=r+2,
                    elapsed_seconds=seconds,seconds_per_reference=seconds/(r+2-initial),
                    peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)),flush=True)
        hyp=state['sum_hyp']/100
        actual=np.take_along_axis(hyp,seq[ix,None].astype(int),axis=1)[:,0]
        quality=np.abs(state['delta'])<=.002+.05*np.abs(state['difference'])
        native=actual[0,e['native_offset']:e['native_offset']+len(e['sequence'])]
        expected=w@np.asarray(e['actual'])
        cosine=float(native@expected/(np.linalg.norm(native)*np.linalg.norm(expected)))
        assert cosine>.995,cosine
        mean_error=np.abs(state['delta'].mean(1)); tolerance=.002+.05*np.abs(state['difference'].mean(1))
        if not (mean_error<=tolerance).all():raise ValueError('Mean-reference completeness check failed')
        np.savez_compressed(path,signature=np.asarray(signature),actual=actual,hypothetical=hyp,
            actual_reference_halves=state['half_actual']/50,reference_sequence=refs,
            delta=state['delta'],target_difference=state['difference'],quality_pass=quality,
            indices=np.asarray(ix),weights=w,references=np.asarray(100),steps=np.asarray(64))
        receipt=dict(example=j,id=e['id'],variants=n,WT_saved_map_cosine=cosine,
            max_abs_mean_delta=float(mean_error.max()),failed_reference_checks=(~quality).sum(1).tolist())
        old.save(ROOT/f'attribution_{j}.json',receipt);receipts.append(receipt)
        print(json.dumps(dict(stage='example_IG_complete',**receipt)),flush=True)
    old.save(ROOT/'complete.json',dict(status='complete',plan_sha256=signature,gpu=torch.cuda.get_device_name(),
        seconds_this_run=time.monotonic()-overall,checkpoint_sha256=old.CHECKPOINT_SHA,
        calibration_sha256=old.sha(old.CALIBRATION),code_sha256=old.sha(__file__),
        attributions=[json.loads((ROOT/f'attribution_{j}.json').read_text()) for j in range(len(plan['examples']))],
        files={p.name:old.sha(p) for p in [ROOT/'predictions.npz']+
            [ROOT/f'attribution_{j}.npz' for j in range(len(plan['examples']))]}))


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none'})
    done=json.loads((ROOT/'complete.json').read_text());plan=json.loads((ROOT/'plan.json').read_text())
    assert done['plan_sha256']==old.sha(ROOT/'plan.json')
    for name,sha in done['files'].items():assert old.sha(ROOT/name)==sha
    with np.load(ROOT/'sequences.npz',allow_pickle=False) as z:seq=z['sequence']
    with np.load(ROOT/'predictions.npz',allow_pickle=False) as z:probs=z['calibrated_probabilities']
    colors={'WT':'#444444','GAF mut.':'#D55E00','Grh mut.':'#009E73','Cg mut.':'#0072B2',
            'Double mut.':'#8863A9','Control':'#BBBBBB'}
    fig=plt.figure(figsize=(680/72,930/72));sources=[]
    for j,e in enumerate(plan['examples']):
        top=892-j*210
        ix=[v['index'] for v in e['variants']];names=[v['name'] for v in e['variants']]
        dna,_,coord,sites,lo,reverse=preview.display(e,seq)
        with np.load(ROOT/f'attribution_{j}.npz',allow_pickle=False) as z:actual=z['actual']
        native=actual[:,e['native_offset']:e['native_offset']+len(e['sequence'])]
        if reverse:native=native[:,::-1]
        values=native[:,lo:lo+dna.shape[1]]
        upper=max(float(values.max())*1.1,.0001);lower=min(float(values.min())*1.1,-upper*.05)
        old.label(fig,10,top,'abcd'[j],12,weight='bold')
        family_count=sum(any(e['labels'][i] for i in f) for f in old.FAMILIES)
        heading=' + '.join(preview.NAMES[s['pattern']] for s in e['sites'])
        old.label(fig,65,top,f"{heading} | {e['id']} | {family_count} "+('family' if family_count==1 else 'families')+
                  (' | RC' if reverse else ''),7.5,weight='bold')
        old.label(fig,65,top-11,f'IG scale: {lower:.3f} to {upper:.3f}',5.4,'#666666')
        height=min(41,155/len(ix));bottom=top-20-len(ix)*height
        for v,name in enumerate(names):
            y=top-20-(v+1)*height;ax=old.make_axes(fig,(65,y,315,height-12))
            for site in sites:ax.axvspan(site['start']-lo,site['end']-lo,color='#EDF0F3',zorder=-1)
            for k,(base,value) in enumerate(zip(dna[v],values[v])):
                old.glyph(ax,old.BASES[base],k+.03,0,.94,float(value),old.DNA_COLORS[base])
                if dna[v,k]!=dna[0,k]:ax.plot([k+.12,k+.88],[lower*.92]*2,color='#B32933',lw=.8)
            ax.axhline(0,lw=.3,color='#777777');ax.set(xlim=(0,dna.shape[1]),ylim=(lower,upper),xticks=[],yticks=[])
            for spine in ax.spines.values():spine.set_visible(False)
            ax.text(-.012,.52,name,transform=ax.transAxes,fontsize=6,ha='right',va='center',color=colors[name])
            letters=old.make_axes(fig,(65,y-6,315,6))
            for k,base in enumerate(dna[v]):
                letters.text(k+.5,.5,old.BASES[base],ha='center',va='center',
                    fontfamily='DejaVu Sans Mono',fontsize=min(4.8,315/dna.shape[1]*.85),
                    color='#B32933' if dna[v,k]!=dna[0,k] else '#777777')
            letters.set(xlim=(0,dna.shape[1]),ylim=(0,1));letters.axis('off')
            if v==0:
                for site in sites:ax.text((site['start']+site['end'])/2-lo,upper*1.08,site['name'],
                                        fontsize=5.5,ha='center',va='bottom')
        old.label(fig,65,bottom-16,str(coord[0]),5.7,'#555555')
        old.label(fig,380,bottom-16,str(coord[-1]),5.7,'#555555',ha='right')
        p=probs[ix][:,preview.ORDER];ax=old.make_axes(fig,(443,top-140,225,104))
        width=.8/len(ix)
        for v,name in enumerate(names):
            ax.bar(np.arange(8)-.4+width*(v+.5),p[v],width,color=colors[name],linewidth=0,label=name)
        ax.set(xlim=(-.6,7.6),ylim=(0,1),xticks=np.arange(8),xticklabels=np.asarray(old.CONTEXTS)[preview.ORDER],
               yticks=[0,.5,1]);ax.set_ylabel('Predicted probability',fontsize=7);old.clean(ax)
        ax.legend(handles=[Patch(facecolor=colors[n],label=n) for n in names],frameon=False,fontsize=5.7,
                  loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,handlelength=1,columnspacing=.8)
        observed=old.make_axes(fig,(443,top-22,225,7))
        observed.imshow(np.asarray(e['labels'])[None,preview.ORDER],aspect='auto',vmin=0,vmax=1,
                        cmap=ListedColormap(['white','#444444']))
        observed.set(xticks=[],yticks=[])
        for spine in observed.spines.values():spine.set_linewidth(.3)
        observed.text(-.045,.5,'Observed',transform=observed.transAxes,ha='right',va='center',fontsize=5.7)
        sources.append(dict(id=e['id'],names=names,probabilities=p.tolist(),delta_vs_WT=(p-p[:1]).tolist(),
                            native_positions=coord.tolist(),DNA=dna.tolist(),actual=values.tolist(),weights=e['weights']))
    out=old.PROJECT/'output/pdf/figure_3_accurate_examples_mutant_control_IG_20260929.pdf'
    fig.savefig(out,metadata=dict(Title='Accuracy-selected native motif mutations: WT, mutant and control IG',
        Subject='IG64/100 common WT references. Fixed observed-active family means. Illustrative cases.'))
    fig.savefig(out.with_suffix('.svg'));fig.savefig(out.with_suffix('.png'),dpi=180);plt.close(fig)
    old.save(ROOT/'figure_source.json',dict(status='awaiting_visual_QA',examples=sources,
        context_order=[old.CONTEXTS[i] for i in preview.ORDER],output_sha256=old.sha(out),
        code_sha256=old.sha(__file__),note='New masked discovery results pending; prior catalogue identifies candidate sites'))
    print(json.dumps(dict(stage='mutant_IG_figure_rendered',path=str(out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','run','render','run_render'))
    a=p.parse_args()
    if a.stage=='run_render':run();render()
    else:{'prepare':prepare,'run':run,'render':render}[a.stage]()
