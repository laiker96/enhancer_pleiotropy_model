"""Frozen two-target attribution: gated pilot and resumable four-shard run."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import time

import numpy as np
import torch

from classifier_motifs.attribution import active_weights, dinucleotide_shuffle, ensemble, integrated_gradients, load_model, one_hot, seed_for
from classifier_motifs.context_attribution import map_agreement, region_masks, REGIONS
from classifier_motifs.dual_attribution import TARGETS, dual_integrated_gradients, refine_dual
from .common import digest, event, write_json
from .context_pilot import save_npz
from .original_intervals import load_intervals


def require_gpu():
    host = socket.gethostname().split('.')[0]
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST')
            or not (host == 'a100' or (host.startswith('xg') and host[2:].isdigit()))
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('One CECAR Slurm CUDA compute allocation required; never login/local GPU')


def validate_config(config):
    required = dict(references=50, steps=32, internal_batch=64, batch_size=32,
        absolute_tolerance=.02, relative_tolerance=.05, minimum_pass_fraction=.95,
        seed=20260916, shards=4, targets=list(TARGETS), input_bp=2048)
    if any(config.get(k) != v for k,v in required.items()):
        raise ValueError('Original 50-reference protocol or two-target contract changed')


def validate_state(state, signature, indices, length, maximum):
    n = len(indices); count = int(state['count'])
    if (str(state['signature']) != signature or not np.array_equal(state['indices'], indices)
            or not 0 <= count <= maximum):
        raise ValueError('Incompatible reference checkpoint')
    shapes = dict(sum_hypothetical=(n,2,4,length), sum_actual=(n,2,length),
        delta=(n,count,2), target_difference=(n,count,2), steps=(n,count,2), passed=(n,count,2))
    if any(state[k].shape != shape or not np.isfinite(state[k]).all() for k,shape in shapes.items()):
        raise ValueError('Malformed/nonfinite reference checkpoint')
    if not np.isin(state['steps'], [32,64,128]).all() or state['passed'].dtype != bool:
        raise ValueError('Invalid checkpoint convergence metadata')


def score_chunk(model, data, indices, config, path, signature, maximum, device, stop):
    """Reference-boundary resume; each accepted reference contributes once."""
    n, length = len(indices), data['sequence'].shape[1]
    state_path = path.with_suffix('.state.npz')
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as f: state = dict(f)
        validate_state(state, signature, indices, length, maximum)
    else:
        state = dict(signature=np.asarray(signature), indices=indices, count=np.asarray(0),
            sum_hypothetical=np.zeros((n,2,4,length), np.float32), sum_actual=np.zeros((n,2,length), np.float32),
            delta=np.zeros((n,0,2), np.float32), target_difference=np.zeros((n,0,2), np.float32),
            steps=np.zeros((n,0,2), np.int32), passed=np.zeros((n,0,2), bool))
    x = one_hot(data['sequence'][indices], device)
    labels = torch.as_tensor(data['labels'][indices], device=device)
    for r in range(int(state['count']), maximum):
        if stop():
            save_npz(state_path, **state)
            raise TimeoutError('Stopped with reference checkpoint')
        codes = np.stack([dinucleotide_shuffle(data['sequence'][i], seed_for(config['seed'], data['ids'][i], r)) for i in indices])
        try:
            value = refine_dual(model, x, one_hot(codes, device), labels, config, stop)
        except TimeoutError:
            save_npz(state_path, **state)
            raise
        state['sum_hypothetical'] += value['hypothetical']
        state['sum_actual'] += value['actual']
        for key in ('delta','target_difference','steps','passed'):
            source = 'quality_pass' if key == 'passed' else key
            state[key] = np.concatenate([state[key], value[source][:,None]], axis=1)
        state['count'] = np.asarray(r+1)
        if (r+1) % 10 == 0 or r+1 == maximum:
            save_npz(state_path, **state)
            event('dual_reference_progress', first_index=int(indices[0]), references=r+1, maximum=maximum)
    result = dict(indices=indices, signature=np.asarray(signature), references=np.asarray(maximum),
        targets=np.asarray(TARGETS), actual=state['sum_actual']/maximum,
        hypothetical=state['sum_hypothetical']/maximum,
        delta=state['delta'], target_difference=state['target_difference'], steps=state['steps'],
        quality_pass=state['passed'].all(1))
    with torch.no_grad(): logits, probabilities = ensemble(model, x)
    result.update(logits=logits.cpu().numpy(), probabilities=probabilities.cpu().numpy(),
                  ids=data['ids'][indices], labels=data['labels'][indices], split=data['split'][indices])
    validate_output(result, data, indices, signature, maximum)
    save_npz(path, **result)
    return result


def validate_output(result, data, indices, signature, references):
    n, length = len(indices), data['sequence'].shape[1]
    if (str(result['signature']) != signature or int(result['references']) != references
            or not np.array_equal(result['indices'], indices) or list(result['targets']) != list(TARGETS)
            or not np.array_equal(result['ids'], data['ids'][indices])
            or not np.array_equal(result['labels'], data['labels'][indices])):
        raise ValueError('Incompatible completed chunk')
    for key, shape in dict(actual=(n,2,length), hypothetical=(n,2,4,length), delta=(n,references,2),
        target_difference=(n,references,2), steps=(n,references,2), quality_pass=(n,2), logits=(n,8), probabilities=(n,8)).items():
        if result[key].shape != shape or not np.isfinite(result[key]).all():
            raise ValueError('Malformed/nonfinite completed output: '+key)
    observed = np.take_along_axis(result['hypothetical'], data['sequence'][indices,None,None,:], axis=2)[:,:,0]
    np.testing.assert_allclose(observed, result['actual'], atol=1e-7, rtol=1e-5)
    tolerance = .02+.05*np.abs(result['target_difference'])
    np.testing.assert_array_equal(result['quality_pass'], (np.abs(result['delta']) <= tolerance).all(1))
    if not np.isin(result['steps'], [32,64,128]).all() or not ((result['probabilities'] >= 0) & (result['probabilities'] <= 1)).all():
        raise ValueError('Invalid output steps/probabilities')


def agreement(a, b, intervals, indices):
    rows = []
    for j,i in enumerate(indices):
        masks = region_masks(int(intervals['offset'][i]), int(intervals['length'][i]))
        for region, mask in [('full',np.ones(a.shape[-1],bool)), *zip(REGIONS,masks)]:
            if not mask.any(): continue
            for target, values in zip(TARGETS, map_agreement(a[j][:,mask], b[j][:,mask])):
                rows.append(dict(index=int(i), region=region, target=target, **values))
    return rows


def pilot(model, data, intervals, config, project, root, signature, stop, device):
    directory = root/'pilot'; directory.mkdir(exist_ok=True)
    indices = np.asarray(config['pilot_indices'])
    if (len(indices) != 16 or len(np.unique(indices)) != 16 or not (data['split'][indices] == 'train').all()
            or not np.array_equal(data['labels'][indices].sum(1), np.tile(np.arange(1,9),2))):
        raise ValueError('Pilot must be two training enhancers per exact degree')
    x = one_hot(data['sequence'][indices], device)
    labels = torch.as_tensor(data['labels'][indices], device=device)
    baseline = one_hot(np.stack([dinucleotide_shuffle(data['sequence'][i], seed_for(config['seed'],data['ids'][i],0)) for i in indices]), device)
    value = dual_integrated_gradients(model,x,baseline,labels,32,64,stop)
    old = integrated_gradients(model,x,baseline,active_weights(labels),32,64)
    for key in old:
        torch.testing.assert_close(value[key][:,0],old[key],atol=3e-5,rtol=2e-3)
    high = dual_integrated_gradients(model,x,baseline,labels,64,64,stop)
    anchor = agreement(value['actual'].cpu().numpy(),high['actual'].cpu().numpy(),intervals,indices)
    del value, old, high
    first = score_chunk(model,data,indices,config,directory/'references50.npz',signature,50,device,stop)
    # Extend a separate state; preserve the fixed50 snapshot and its resumability.
    if not (directory/'references100.state.npz').exists():
        shutil.copy2(directory/'references50.state.npz',directory/'references100.state.npz')
    extended = score_chunk(model,data,indices,config,directory/'references100.npz',signature,100,device,stop)
    complete = json.loads((project/config['original']/'attribution_complete.json').read_text())
    original_max = 0.
    for row,index in enumerate(indices):
        name = 'chunk_%06d.npz' % (int(index)//32*32)
        path = project/config['original']/'chunks'/name
        if digest(path) != complete['chunks'][name]:
            raise ValueError('Original 50-reference output changed')
        with np.load(path,allow_pickle=False) as f:
            position = int(np.flatnonzero(f['indices'] == index)[0])
            np.testing.assert_allclose(first['actual'][row,0],f['actual'][position],atol=3e-5,rtol=2e-3)
            start,n = int(intervals['offset'][index]),int(intervals['length'][index])
            np.testing.assert_allclose(first['hypothetical'][row,0,:,start:start+n],f['hypothetical'][position,:,:n],atol=3e-5,rtol=2e-3)
            np.testing.assert_array_equal(first['steps'][row,:,0],f['steps'][position])
            original_max = max(original_max,float(np.abs(first['actual'][row,0]-f['actual'][position]).max()))
    quality = first['quality_pass'].mean(0)
    grid_pass = all(r['cosine'] is not None and r['cosine'] >= .99 for r in anchor if r['region'] == 'native')
    report = dict(status='passed' if grid_pass and (quality >= .95).all() else 'failed',
        signature=signature, indices=indices.tolist(), target_order=list(TARGETS),
        original50_reproduction=dict(status='passed',max_absolute_actual_difference=original_max),
        quality_pass_fraction=quality.tolist(), integration32_vs64_anchor=anchor,
        independent50_vs50=agreement(first['actual'],2*extended['actual']-first['actual'],intervals,indices),
        nested50_vs100=agreement(first['actual'],extended['actual'],intervals,indices),
        reference_policy='Fixed50 production. Reference comparisons are diagnostic; nested agreement is optimistic. No adaptive reference-count claim.',
        files={p.name:digest(p) for p in (directory/'references50.npz',directory/'references100.npz')})
    write_json(directory/'report.json',report)
    if report['status'] != 'passed':
        raise ValueError('Pilot numerical gate failed; production remains blocked')
    write_json(root/'pilot_passed.json',dict(signature=signature, report_sha256=digest(directory/'report.json')))
    event('dual_pilot_passed',quality_pass_fraction=quality.tolist(),original_max_difference=original_max)


def production(model,data,intervals,config,root,signature,shard,stop,device):
    gate = json.loads((root/'pilot_passed.json').read_text())
    if (gate['signature'] != signature or gate['report_sha256'] != digest(root/'pilot/report.json')
            or json.loads((root/'pilot/report.json').read_text())['status'] != 'passed'):
        raise ValueError('Missing or changed passing pilot')
    shards, references = config.get('shards', 4), config.get('references', 50)
    if not 0 <= shard < shards: raise ValueError('Unexpected shard index')
    directory = root/'chunks'; directory.mkdir(exist_ok=True)
    processed, failed, hashes = 0, np.zeros(2,int), {}
    for batch,begin in enumerate(range(0,len(data['ids']),config['batch_size'])):
        if batch % shards != shard: continue
        if stop(): raise TimeoutError('Stopped at completed chunk boundary')
        indices = np.arange(begin,min(begin+config['batch_size'],len(data['ids'])))
        path = directory/('chunk_%06d.npz' % begin)
        if path.exists():
            with np.load(path,allow_pickle=False) as f: result = dict(f)
            validate_output(result,data,indices,signature,references)
        else:
            result = score_chunk(model,data,indices,config,path,signature,references,device,stop)
        if config.get('refinement_steps') == [] and not (result['steps'] == config['steps']).all():
            raise ValueError('Fixed-grid output contains a different integration resolution')
        processed += len(indices); failed += (~result['quality_pass']).sum(0)
        hashes[path.name] = digest(path)
        write_json(root/f'progress_{shard}.json',dict(signature=signature,shard=shard,
            elements=processed,quality_failures=failed.tolist(),last_chunk=path.name,job_id=os.environ['SLURM_JOB_ID']))
        event('dual_attribution_progress',shard=shard,elements=processed,quality_failures=failed.tolist())
        if processed >= 512 and (1-failed/processed < config['minimum_pass_fraction']).any():
            raise ValueError('Attribution convergence below original minimum95%')
    write_json(root/f'attribution_shard_{shard}.json',dict(status='complete',signature=signature,
        shard=shard,elements=processed,quality_failures=failed.tolist(),chunks=hashes))
    event('dual_shard_complete',shard=shard,elements=processed)


def run(project,root,stage,shard=0):
    require_gpu()
    config = json.loads((root/'config.json').read_text()); validate_config(config)
    for relative,expected in config['source_hashes'].items():
        if digest(project/relative) != expected: raise ValueError('Frozen input changed: '+relative)
    signature = digest(root/'MANIFEST.sha256')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with np.load(project/config['parent']/'cohort.npz',allow_pickle=False) as f: data = dict(f)
    if len(data['ids']) != 40338: raise ValueError('Unexpected cohort size')
    intervals = load_intervals(project/config['original'],data)
    began = time.monotonic(); stopped = [False]
    for sig in (signal.SIGTERM,signal.SIGUSR1): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    limit = 1680 if stage == 'pilot' else 85800
    def stop(): return stopped[0] or (root/'STOP').exists() or time.monotonic()-began >= limit
    model = load_model(project/config['parent']/'best_model.pt','cuda')
    event('dual_started',stage=stage,shard=shard,gpu=torch.cuda.get_device_name(),signature=signature,
          references=50,steps=[32,64,128],targets=list(TARGETS),max_seconds=limit)
    try:
        if stage == 'pilot': pilot(model,data,intervals,config,project,root,signature,stop,'cuda')
        else: production(model,data,intervals,config,root,signature,shard,stop,'cuda')
    except Exception as error:
        write_json(root/f'{stage}_{shard}_stopped.json',dict(status='stopped' if isinstance(error,TimeoutError) else 'failed',
            reason=type(error).__name__+': '+str(error),elapsed_seconds=time.monotonic()-began,signature=signature))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('pilot','attribute'))
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    run(args.project.resolve(),args.root.resolve(),args.stage,args.shard)
