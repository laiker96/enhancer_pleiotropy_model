"""Gated hardware pilots and resumable full-length calibrated context IG."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import time

import numpy as np
import torch

from classifier_motifs.attribution import active_weights, dinucleotide_shuffle, one_hot, seed_for
from classifier_motifs.calibrated_attribution import (CalibratedTargets, CONTEXTS,
    TARGETS, integrate, load_classifier, quality)
from classifier_motifs.context_attribution import map_agreement, pilot_indices
from .common import digest, event, write_json
from .original_intervals import load_intervals


def require_gpu():
    host = socket.gethostname().split('.')[0]
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST')
            or host not in ('a100', 'xg05', 'xg07', 'xg08', 'xg09')
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Requires one approved CECAR Slurm GPU; never login or local GPU')
    name = torch.cuda.get_device_name()
    if host == 'a100' and 'A100' in name: return 'A100'
    if host != 'a100' and 'RTX 2080' in name: return 'rtx2080'
    raise RuntimeError('Unvalidated hardware: '+host+' / '+name)


def save_npz(path, **arrays):
    # Uncompressed checkpoints avoid repeated deflate work on the critical path.
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def shuffled(task):
    codes, identifier, reference, seed = task
    value = dinucleotide_shuffle(codes, seed_for(seed, identifier, reference))
    return value, hashlib.sha256(value.tobytes()).hexdigest()


def make_references(data, indices, config, pool):
    tasks = [(data['sequence'][i], str(data['ids'][i]), r, config['seed'])
             for i in indices for r in range(config['references'])]
    values = list(map(shuffled, tasks) if pool is None else pool.map(shuffled, tasks, chunksize=16))
    n, r, length = len(indices), config['references'], data['sequence'].shape[1]
    return (np.stack([v[0] for v in values]).reshape(n,r,length),
            np.asarray([v[1] for v in values], dtype='U64').reshape(n,r))


def new_state(n, length, references, signature, indices, target_count=9):
    if target_count not in (8, 9): raise ValueError('Expected eight or nine targets')
    return dict(signature=np.asarray(signature), indices=indices, count=np.asarray(0),
        sum_hyp=np.zeros((n,target_count,4,length), np.float64),
        sum_sq_hyp=np.zeros((n,target_count,4,length), np.float64),
        sum_half_actual=np.zeros((n,2,target_count,length), np.float64),
        sum_sq_breadth=np.zeros((n,length), np.float64),
        delta=np.zeros((n,references,target_count), np.float32),
        target_difference=np.zeros((n,references,target_count), np.float32),
        reference_orientation_logits=np.zeros((n,references,2,8), np.float32),
        reference_probabilities=np.zeros((n,references,8), np.float32),
        reference_calibrated_probabilities=np.zeros((n,references,8), np.float32))


def validate_state(state, template, signature, indices, maximum):
    if (str(state['signature']) != signature or not np.array_equal(state['indices'], indices)
            or not 0 <= int(state['count']) <= maximum or set(state) != set(template)):
        raise ValueError('Incompatible reference checkpoint')
    for key in template:
        if state[key].shape != template[key].shape or state[key].dtype != template[key].dtype:
            raise ValueError('Malformed checkpoint: '+key)
        if state[key].dtype.kind in 'fiu' and not np.isfinite(state[key]).all():
            raise ValueError('Nonfinite checkpoint: '+key)


def update_state(state, values, local_rows, references, half):
    # Fixed pair ordering and FP64 accumulation make batching reproducible.
    for pair, (row, reference) in enumerate(zip(local_rows, references)):
        hyp = values['hypothetical'][pair].astype(np.float64)
        actual = values['actual'][pair].astype(np.float64)
        state['sum_hyp'][row] += hyp
        state['sum_sq_hyp'][row] += hyp*hyp
        state['sum_half_actual'][row, int(reference >= half)] += actual
        state['sum_sq_breadth'][row] += actual[:8].sum(0)**2
        for key in ('delta', 'target_difference'):
            state[key][row, reference] = values[key][pair]
        for key in ('orientation_logits', 'probabilities', 'calibrated_probabilities'):
            state['reference_'+key][row, reference] = values['reference_'+key][pair]


def finish_state(state, data, intervals, indices, reference_hashes, target, device, signature, config):
    count = config['references']
    if int(state['count']) != count or count < 2 or count % 2:
        raise ValueError('Incomplete or odd reference set')
    mean = state['sum_hyp']/count
    variance = np.maximum(0, (state['sum_sq_hyp']-state['sum_hyp']**2/count)/(count-1))
    hyp, hyp_se = mean.astype(np.float32), np.sqrt(variance/count).astype(np.float32)
    codes = data['sequence'][indices]
    def project(value):
        return np.take_along_axis(value, codes[:,None,None,:], axis=2)[:,:,0]
    actual, actual_se = project(hyp), project(hyp_se)
    breadth = project(mean)[:, :8].sum(1)
    breadth_se = np.sqrt(np.maximum(0, (state['sum_sq_breadth']-count*breadth**2)/(count-1))/count)
    with torch.no_grad():
        endpoints = {k:v.cpu().numpy() for k,v in target.endpoints(one_hot(codes, device)).items()}
    delta, difference = state['delta'], state['target_difference']
    per_reference_pass = quality(delta, difference)
    breadth_pass = np.abs(delta[:,:,:8].sum(2)) <= .02+.05*np.abs(difference[:,:,:8].sum(2))
    result = dict(signature=np.asarray(signature), indices=indices,
        targets=np.asarray(config.get('targets', TARGETS)),
        references=np.asarray(count), steps=np.asarray(config['steps']),
        hypothetical=hyp, actual=actual, hypothetical_reference_se=hyp_se,
        actual_reference_se=actual_se, breadth_reference_se=breadth_se.astype(np.float32),
        actual_reference_halves=(state['sum_half_actual']/(count//2)).astype(np.float32),
        delta=delta, target_difference=difference, reference_quality_pass=per_reference_pass,
        quality_pass=per_reference_pass.all(1), breadth_quality_pass=breadth_pass.all(1),
        reference_hashes=reference_hashes, **endpoints)
    for key in ('ids','labels','split','chrom','summit','sequence'):
        result[key] = data[key][indices]
    for key in ('start','end','offset','length'):
        result['native_'+key] = intervals[key][indices]
    for key in ('reference_orientation_logits','reference_probabilities','reference_calibrated_probabilities'):
        result[key] = state[key]
    return result


def validate_output(result, indices, signature, config):
    n = len(indices)
    targets = list(config.get('targets', TARGETS)); t = len(targets)
    if t not in (8, 9) or targets != list(TARGETS[:t]):
        raise ValueError('Unexpected attribution target order')
    if (str(result['signature']) != signature or not np.array_equal(result['indices'],indices)
            or list(result['targets']) != targets or int(result['references']) != config['references']
            or int(result['steps']) != config['steps']):
        raise ValueError('Wrong completed attribution chunk')
    shapes = dict(hypothetical=(n,t,4,2048), actual=(n,t,2048),
        hypothetical_reference_se=(n,t,4,2048), actual_reference_se=(n,t,2048),
        actual_reference_halves=(n,2,t,2048), delta=(n,config['references'],t),
        target_difference=(n,config['references'],t), quality_pass=(n,t))
    for key, shape in shapes.items():
        if result[key].shape != shape or not np.isfinite(result[key]).all():
            raise ValueError('Malformed output '+key)
    np.testing.assert_array_equal(result['quality_pass'], quality(result['delta'],result['target_difference']).all(1))
    projection = np.take_along_axis(result['hypothetical'], result['sequence'][:,None,None,:], axis=2)[:,:,0]
    np.testing.assert_allclose(projection, result['actual'], atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(result['actual_reference_halves'].mean(1), result['actual'], atol=2e-7, rtol=1e-4)


def score_chunk(target, data, intervals, indices, config, settings, path, signature, pool, stop, device='cuda'):
    if path.exists():
        with np.load(path, allow_pickle=False) as saved: result = dict(saved)
        validate_output(result,indices,signature,config)
        return result
    count, length = config['references'], data['sequence'].shape[1]
    target_count = len(config.get('targets', TARGETS))
    template = new_state(len(indices), length, count, signature, indices, target_count)
    state_path = path.with_suffix('.state.npz')
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as saved: state = dict(saved)
        validate_state(state, template, signature, indices, count)
    else: state = template
    references, hashes = make_references(data,indices,config,pool)
    x = one_hot(data['sequence'][indices], device)
    labels = torch.as_tensor(data['labels'][indices],device=device)
    for begin in range(int(state['count']), count, config['reference_block']):
        try:
            end = min(begin+config['reference_block'],count)
            rows = np.repeat(np.arange(len(indices)),end-begin)
            refs = np.tile(np.arange(begin,end),len(indices))
            # Publish the reference block only when ALL its pairs have finished.
            blocks = []
            for first in range(0,len(rows),settings['pair_batch']):
                if stop(): raise TimeoutError('Stopped at pair batch boundary')
                rr, rf = rows[first:first+settings['pair_batch']], refs[first:first+settings['pair_batch']]
                baseline = one_hot(references[rr,rf],device)
                out = integrate(target,x[rr],baseline,labels[rr],config['steps'],
                                settings['internal_batch'],settings['target_batch'],stop,
                                target_count=target_count)
                with torch.no_grad():
                    out.update({'reference_'+k:v for k,v in target.endpoints(baseline).items()
                                if k != 'logits'})
                blocks.append((rr,rf,{k:v.cpu().numpy() for k,v in out.items()}))
            for rr,rf,values in blocks: update_state(state,values,rr,rf,count//2)
            state['count'] = np.asarray(end)
        except TimeoutError:
            save_npz(state_path,**state)
            raise
        if end % 20 == 0 or end == count:
            save_npz(state_path,**state)
            event('calibrated_reference_progress',first_index=int(indices[0]),references=end,total=count)
    result = finish_state(state,data,intervals,indices,hashes,target,device,signature,config)
    validate_output(result,indices,signature,config)
    save_npz(path,**result)
    return result


def compare_grid(low, high, data, intervals, indices):
    rows = []
    for j,i in enumerate(indices):
        lo, length = int(intervals['offset'][i]), int(intervals['length'][i])
        for region, sl in [('native',slice(lo,lo+length)),('full',slice(None))]:
            for t, metrics in enumerate(map_agreement(low[j,:,sl],high[j,:,sl])):
                tiny = max(np.linalg.norm(low[j,t,sl]),np.linalg.norm(high[j,t,sl])) < 1e-8
                rows.append(dict(index=int(i),target=TARGETS[t],region=region,
                    tiny=bool(tiny),**metrics))
    return rows


def benchmark(target, data, indices, config, family, stop):
    # 32 distinct pairs: sixteen enhancers, two references each, same in every test.
    chosen = np.repeat(indices,2)
    refs = np.tile([0,1],len(indices))
    x = one_hot(data['sequence'][chosen],'cuda')
    b = one_hot(np.stack([shuffled((data['sequence'][i],str(data['ids'][i]),r,config['seed']))[0]
                         for i,r in zip(chosen,refs)]),'cuda')
    labels = torch.as_tensor(data['labels'][chosen],device='cuda')
    anchor = {k:v.cpu() for k,v in integrate(target,x,b,labels,64,32,1,stop).items()}
    variants = []
    candidates = [(4,128,1),(8,256,1),(8,512,1),(16,1024,1),
                  (8,512,3),(8,512,9),(16,1024,3),(16,1024,9),
                  (32,2048,1)] if family == 'A100' else [(2,32,1),(4,64,1),(8,128,1),(8,128,3)]
    for pairs, internal, directions in candidates:
        settings = dict(pair_batch=pairs,internal_batch=internal,target_batch=directions)
        event('batch_candidate_started',**settings)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        times = []
        try:
            for repeat in range(3):  # first pass is warm-up, next two are timed
                torch.cuda.synchronize(); began=time.monotonic(); outputs=[]
                for first in range(0,len(x),pairs):
                    value=integrate(target,x[first:first+pairs],b[first:first+pairs],labels[first:first+pairs],
                                    64,internal,directions,stop)
                    outputs.append(value)
                torch.cuda.synchronize(); elapsed=time.monotonic()-began
                for key in anchor:
                    torch.testing.assert_close(torch.cat([v[key] for v in outputs]).cpu(),anchor[key],atol=3e-5,rtol=2e-3)
                if repeat: times.append(elapsed/len(x))
                del outputs,value
            peak=torch.cuda.max_memory_reserved()
            headroom=peak <= .85*torch.cuda.get_device_properties(0).total_memory
            row=dict(status='passed' if headroom else 'excluded_memory_headroom',settings=settings,
                seconds_per_pair=float(np.median(times)),timings=times,peak_reserved_bytes=peak)
        except torch.cuda.OutOfMemoryError:
            row=dict(status='oom',settings=settings)
            gc.collect(); torch.cuda.empty_cache()
        variants.append(row); event('batch_candidate_result',**row)
    passing=[v for v in variants if v['status']=='passed']
    if not passing: raise ValueError('No numerically equivalent batch with memory headroom')
    return min(passing,key=lambda v:v['seconds_per_pair'])['settings'],variants


def mutation_check(target,data,intervals,result,indices):
    """Small diagnostic only: baseline IG is not an exact mutation predictor."""
    rows=[]
    for j,i in enumerate(indices[:4]):
        codes=data['sequence'][i]; lo=int(intervals['offset'][i]); hi=lo+int(intervals['length'][i])
        native=np.arange(lo,hi); flank=np.r_[np.arange(lo),np.arange(hi,2048)]
        magnitude=np.abs(result['actual'][j,:8].sum(0))
        rng=np.random.default_rng(seed_for('calibrated_ism',str(data['ids'][i])))
        top=native[np.argsort(-magnitude[native],kind='stable')[:2]]
        positions=np.unique(np.r_[top,rng.choice(np.setdiff1d(native,top),2,replace=False),
                                  rng.choice(flank,2,replace=False)])
        x=one_hot(codes[None],'cuda')
        weights=active_weights(torch.as_tensor(data['labels'][i:i+1],device='cuda'))
        mutations=[(int(p),a) for p in positions for a in range(4) if a!=codes[p]]
        batch=x.repeat(len(mutations),1,1)
        for k,(p,a) in enumerate(mutations): batch[k,:,p]=0; batch[k,a,p]=1
        with torch.no_grad():
            delta=(target(batch,weights.repeat(len(batch),1))-target(x,weights)).cpu().numpy()
        for k,(p,a) in enumerate(mutations):
            approximate=result['hypothetical'][j,:,a,p]-result['actual'][j,:,p]
            rows.append(dict(index=int(i),position=p,alternate=a,native=bool(lo<=p<hi),
                model_delta=delta[k].tolist(),ig_hypothetical_difference=approximate.tolist()))
    exact=np.asarray([r['model_delta'] for r in rows]); estimate=np.asarray([r['ig_hypothetical_difference'] for r in rows])
    return dict(rows=rows,pearson=[float(np.corrcoef(exact[:,i],estimate[:,i])[0,1])
        if np.std(exact[:,i])>0 and np.std(estimate[:,i])>0 else None for i in range(9)],
        interpretation='Diagnostic, not a pass/fail accuracy guarantee: IG integrates from shuffled references; ISM changes the observed sequence.')


def pilot(target,data,intervals,config,root,signature,family,pool,stop):
    directory=root/('pilot_'+family); directory.mkdir(exist_ok=True)
    indices=pilot_indices(data,intervals,2,config['seed'])
    settings,timings=benchmark(target,data,indices,config,family,stop)
    write_json(directory/'batch_benchmark.json',dict(signature=signature,settings=settings,variants=timings))
    # Replay held-out cached endpoints only; no fitting or selection here.
    with np.load(root/'endpoint_replay.npz',allow_pickle=False) as saved: replay=dict(saved)
    with torch.no_grad(): prediction=target.endpoints(one_hot(replay['sequence'],'cuda'))['probabilities'].cpu().numpy()
    endpoint_error=float(np.abs(prediction-replay['probabilities']).max())
    if endpoint_error > .005: raise ValueError('FP32 endpoint replay differs from locked FP16 cache')
    # One matched reference per training enhancer, fixed64 versus128 nodes.
    x=one_hot(data['sequence'][indices],'cuda')
    b=one_hot(np.stack([shuffled((data['sequence'][i],str(data['ids'][i]),0,config['seed']))[0]
                       for i in indices]),'cuda')
    labels=torch.as_tensor(data['labels'][indices],device='cuda')
    low=integrate(target,x,b,labels,64,settings['internal_batch'],settings['target_batch'],stop)
    high=integrate(target,x,b,labels,128,settings['internal_batch'],settings['target_batch'],stop)
    grid=compare_grid(low['actual'].cpu().numpy(),high['actual'].cpu().numpy(),data,intervals,indices)
    eligible=[r for r in grid if r['region']=='native' and not r['tiny']]
    grid_fraction=sum(r['cosine'] is not None and r['cosine']>=.99 for r in eligible)/max(1,len(eligible))
    save_npz(directory/'grid_check.npz',indices=indices,actual64=low['actual'].cpu().numpy(),
             actual128=high['actual'].cpu().numpy())
    del low,high,x,b,labels
    began=time.monotonic()
    result=score_chunk(target,data,intervals,indices,config,settings,directory/'references100.npz',signature,pool,stop)
    elapsed=time.monotonic()-began
    quality_fraction=result['quality_pass'].mean(0)
    breadth_fraction=float(result['breadth_quality_pass'].mean())
    # Same-device direct summed-target identity is tested on CPU; cross-hardware
    # maps are independently anchored to the small-batch kernel on each device.
    reference_stability=compare_grid(result['actual_reference_halves'][:,0],
        result['actual_reference_halves'][:,1],data,intervals,indices)
    ism=mutation_check(target,data,intervals,result,indices)
    passed=bool(grid_fraction>=.95 and (quality_fraction>=.95).all() and breadth_fraction>=.95)
    report=dict(status='passed' if passed else 'failed',signature=signature,family=family,
        settings=settings,endpoint_replay_max_error=endpoint_error,
        quality_pass_fraction=quality_fraction.tolist(),breadth_pass_fraction=breadth_fraction,
        integration64_vs128=grid,integration_native_fraction_cosine_ge_099=grid_fraction,
        reference_half_agreement=reference_stability,mutation_diagnostic=ism,
        seconds_for_16_enhancers_100_references=elapsed,
        full_serial_hours=elapsed/len(indices)*len(data['ids'])/3600,
        conservative_hours_per_shard=2*elapsed/len(indices)*np.ceil(len(data['ids'])/config['shards'])/3600,
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        files={p.name:digest(p) for p in directory.glob('*.npz') if '.state.' not in p.name})
    # Fail closed if the estimated shard cannot fit its24h allocation.
    if report['conservative_hours_per_shard']>23: report['status']='failed'
    write_json(directory/'report.json',report)
    event('calibrated_pilot_complete',family=family,status=report['status'],
          settings=settings,quality=quality_fraction.tolist(),serial_hours=report['full_serial_hours'])
    if report['status']!='passed': raise ValueError('Pilot numerical/runtime gate failed; full run blocked')


def verify_gate(root,signature,families=('A100','rtx2080')):
    if tuple(families) not in (('A100','rtx2080'),('rtx2080',)):
        raise ValueError('Unapproved pilot hardware set')
    reports={}
    for family in families:
        path=root/('pilot_'+family)/'report.json'
        report=json.loads(path.read_text())
        if report['status']!='passed' or report['signature']!=signature or report['family']!=family:
            raise ValueError('Missing or incompatible passing hardware pilot')
        for name,sha in report['files'].items():
            if digest(path.parent/name)!=sha: raise ValueError('Pilot result changed')
        reports[family]=report
    if len(families)==2:
        with np.load(root/'pilot_A100/references100.npz',allow_pickle=False) as a, np.load(
                root/'pilot_rtx2080/references100.npz',allow_pickle=False) as b:
            np.testing.assert_array_equal(a['indices'],b['indices'])
            np.testing.assert_array_equal(a['reference_hashes'],b['reference_hashes'])
            for key in ('hypothetical','actual','logits','probabilities','calibrated_probabilities'):
                np.testing.assert_allclose(a[key],b[key],atol=3e-5,rtol=2e-3)
    return reports


def resolve_pilots(root,signature,config):
    if 'pilot_parent' not in config:
        return verify_gate(root,signature)
    parent_name=config['pilot_parent']
    if Path(parent_name).name!=parent_name or config.get('required_pilot_families')!=['rtx2080']:
        raise ValueError('Invalid imported pilot contract')
    parent=root.parent/parent_name
    parent_signature=digest(parent/'MANIFEST.sha256')
    if parent_signature!=config['pilot_parent_manifest_sha256']:
        raise ValueError('Imported pilot package changed')
    original=json.loads((parent/'config.json').read_text())
    scientific_keys=('cohort','original','checkpoint','targets','references','steps',
        'refinement_steps','seed','input_bp','enhancer_batch','reference_block','shards',
        'calibration_population','source_hashes')
    if any(config[k]!=original[k] for k in scientific_keys):
        raise ValueError('Cannot reuse a pilot for different scientific settings')
    for name in ('calibrators.json','endpoint_replay.npz'):
        if digest(root/name)!=digest(parent/name):
            raise ValueError('Imported calibration or endpoint replay changed')
    return verify_gate(parent,parent_signature,('rtx2080',))


def production(target,data,intervals,config,root,signature,family,shard,pool,stop):
    reports=resolve_pilots(root,signature,config)
    settings=reports[family]['settings']
    if not 0<=shard<config['shards']: raise ValueError('Wrong shard index')
    directory=root/'chunks'; directory.mkdir(exist_ok=True)
    processed=0; failures=np.zeros(9,int); hashes={}
    for batch,begin in enumerate(range(0,len(data['ids']),config['enhancer_batch'])):
        if batch%config['shards']!=shard: continue
        if stop(): raise TimeoutError('Stopped at chunk boundary')
        indices=np.arange(begin,min(begin+config['enhancer_batch'],len(data['ids'])))
        path=directory/('chunk_%06d.npz'%begin)
        result=score_chunk(target,data,intervals,indices,config,settings,path,signature,pool,stop)
        processed+=len(indices); failures+=(~result['quality_pass']).sum(0)
        hashes[path.name]=digest(path)
        progress=dict(signature=signature,shard=shard,elements=processed,
            quality_failures=failures.tolist(),last_chunk=path.name,job_id=os.environ['SLURM_JOB_ID'])
        write_json(root/f'progress_{shard}.json',progress); event('calibrated_progress',**progress)
        if processed>=256 and (1-failures/processed<.95).any():
            raise ValueError('Full-run completeness below95%; outputs retained, no silent filtering')
    write_json(root/f'shard_{shard}.json',dict(status='complete',signature=signature,
        shard=shard,elements=processed,quality_failures=failures.tolist(),chunks=hashes))


def run(project,root,stage,shard):
    family=require_gpu()
    config=json.loads((root/'config.json').read_text())
    rtx_only=config.get('required_pilot_families')==['rtx2080']
    if (config['references']!=100 or config['steps']!=64 or config['refinement_steps']!=[]
            or config['shards']!=80 or config['maximum_concurrent_gpus']!=(4 if rtx_only else 5)
            or config['targets']!=list(TARGETS) or config['seed']!=20260916):
        raise ValueError('Changed scientific/resource contract')
    if rtx_only and family!='rtx2080':
        raise RuntimeError('This continuation is restricted to RTX2080 GPUs')
    for name,sha in config['source_hashes'].items():
        if digest(project/name)!=sha: raise ValueError('Changed input: '+name)
    signature=digest(root/'MANIFEST.sha256')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    with np.load(project/config['cohort'],allow_pickle=False) as saved: data=dict(saved)
    if (len(data['ids'])!=40338 or len(np.unique(data['ids']))!=40338
            or data['sequence'].shape!=(40338,2048) or not np.isin(data['sequence'],range(4)).all()
            or data['labels'].shape!=(40338,8) or not np.isin(data['labels'],[0,1]).all()
            or not data['labels'].any(1).all()):
        raise ValueError('Invalid full cohort')
    intervals=load_intervals(project/config['original'],data)
    target=CalibratedTargets(load_classifier(project/config['checkpoint'],'cuda'),
        json.loads((root/'calibrators.json').read_text())['enhancers_only']).to('cuda').eval()
    began=time.monotonic(); stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGUSR1): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    limit=1680 if stage=='pilot' else 85800
    def stop(): return stopped[0] or (root/'STOP').exists() or time.monotonic()-began>=limit
    identity=f'{stage}_{shard}_{os.environ["SLURM_JOB_ID"]}'
    runtime=dict(signature=signature,family=family,stage=stage,shard=shard,
        gpu=torch.cuda.get_device_name(),job_id=os.environ['SLURM_JOB_ID'],
        node=socket.gethostname(),torch=torch.__version__,cuda=torch.version.cuda,
        max_seconds=limit,targets=list(TARGETS),no_training=True)
    write_json(root/f'runtime_{identity}.json',runtime); event('calibrated_started',**runtime)
    try:
        with ProcessPoolExecutor(4,mp_context=multiprocessing.get_context('spawn')) as pool:
            if stage=='pilot': pilot(target,data,intervals,config,root,signature,family,pool,stop)
            else: production(target,data,intervals,config,root,signature,family,shard,pool,stop)
    except Exception as error:
        write_json(root/f'stopped_{identity}.json',dict(status='stopped' if isinstance(error,TimeoutError) else 'failed',
            reason=type(error).__name__+': '+str(error),elapsed_seconds=time.monotonic()-began,**runtime))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('pilot','attribute'))
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0)
    args=p.parse_args(); run(args.project.resolve(),args.root.resolve(),args.stage,args.shard)
