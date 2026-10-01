"""Fixed IG64/100 references, two hardware pilots, at most five CUDA shards."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time

import numpy as np
import torch

from classifier_motifs.attribution import active_weights, dinucleotide_shuffle, integrated_gradients, load_model, one_hot, seed_for
from classifier_motifs.dual_attribution import TARGETS, dual_integrated_gradients
from .common import digest, event, write_json
from .context_pilot import save_npz
from .dual_ig import agreement, production, require_gpu, score_chunk
from .original_intervals import load_intervals


def validate_config(config):
    expected = dict(protocol='fixed_ig64_ref100', steps=64, references=100,
        refinement_steps=[], batch_size=16, shards=40, maximum_concurrent_gpus=5,
        internal_batches={'A100':512, 'rtx2080':128}, seed=20260916, input_bp=2048,
        absolute_tolerance=.02, relative_tolerance=.05, minimum_pass_fraction=.95,
        targets=list(TARGETS))
    if any(config.get(k) != v for k,v in expected.items()):
        raise ValueError('Fixed IG64/100-reference five-GPU contract changed')


def hardware_family(host, gpu_name):
    if host == 'a100' and 'A100' in gpu_name:
        return 'A100'
    if host in ('xg05', 'xg07', 'xg08', 'xg09') and 'RTX 2080' in gpu_name:
        return 'rtx2080'
    raise ValueError('Unapproved node or GPU type: '+host+' / '+gpu_name)


def anchor_check(model, data, indices, config, device, stop):
    """Same-device batching check against the old smaller computation batch."""
    x = one_hot(data['sequence'][indices], device)
    labels = torch.as_tensor(data['labels'][indices], device=device)
    baseline = one_hot(np.stack([dinucleotide_shuffle(data['sequence'][i],
        seed_for(config['seed'], data['ids'][i], 0)) for i in indices]), device)
    got = dual_integrated_gradients(model, x, baseline, labels, 64, config['internal_batch'], stop)
    small = dual_integrated_gradients(model, x, baseline, labels, 64, 64, stop)
    maximum = {}
    for key in got:
        torch.testing.assert_close(got[key], small[key], atol=3e-5, rtol=2e-3)
        maximum[key] = float((got[key]-small[key]).abs().max())
    return got, x, baseline, labels, maximum


def publish_gate(root, signature):
    """Both immutable hardware reports must pass before the full array starts."""
    reports = {}
    for family in ('A100', 'rtx2080'):
        path = root/('pilot_'+family)/'report.json'
        report = json.loads(path.read_text())
        if report['status'] != 'passed' or report['signature'] != signature or report['family'] != family:
            raise ValueError('Missing or incompatible hardware pilot')
        for name, checksum in report['files'].items():
            if digest(path.parent/name) != checksum:
                raise ValueError('Hardware pilot maps changed')
        reports[family] = digest(path)
    (root/'pilot').mkdir(exist_ok=True)
    write_json(root/'pilot/report.json', dict(status='passed', signature=signature, hardware_reports=reports))
    write_json(root/'pilot_passed.json', dict(signature=signature, report_sha256=digest(root/'pilot/report.json')))


def verify_gate(root, signature):
    report = json.loads((root/'pilot/report.json').read_text())
    gate = json.loads((root/'pilot_passed.json').read_text())
    if (report['status'] != 'passed' or report['signature'] != signature
            or gate['signature'] != signature or gate['report_sha256'] != digest(root/'pilot/report.json')
            or set(report['hardware_reports']) != {'A100', 'rtx2080'}):
        raise ValueError('Invalid production gate')
    for family, checksum in report['hardware_reports'].items():
        if digest(root/('pilot_'+family)/'report.json') != checksum:
            raise ValueError('Hardware pilot report changed')


def pilot(model, data, intervals, config, root, signature, family, stop, device):
    directory = root/('pilot_'+family); directory.mkdir(exist_ok=True)
    indices = np.asarray(config['pilot_indices'])
    if (len(indices) != 16 or len(np.unique(indices)) != 16 or not (data['split'][indices] == 'train').all()
            or not np.array_equal(data['labels'][indices].sum(1), np.tile(np.arange(1,9),2))):
        raise ValueError('Pilot must use the unchanged balanced training examples')
    got, x, baseline, labels, batch_error = anchor_check(model, data, indices, config, device, stop)
    old = integrated_gradients(model, x, baseline, active_weights(labels), 64, 64)
    for key in old:
        torch.testing.assert_close(got[key][:,0], old[key], atol=3e-5, rtol=2e-3)
    high = dual_integrated_gradients(model, x, baseline, labels, 128, config['internal_batch'], stop)
    grid = agreement(got['actual'].cpu().numpy(), high['actual'].cpu().numpy(), intervals, indices)
    grid_pass = all(r['cosine'] is not None and r['cosine'] >= .99 for r in grid if r['region'] == 'native')
    save_npz(directory/'anchor64.npz', **{k:v.cpu().numpy() for k,v in got.items()}, indices=indices)
    del got, old, high, x, baseline, labels
    began = time.monotonic()
    value = score_chunk(model, data, indices, config, directory/'references100.npz',
                        signature, 100, device, stop)
    quality = value['quality_pass'].mean(0)
    if not (value['steps'] == 64).all():
        raise ValueError('IG64 pilot changed integration resolution')
    cross_gpu = None
    if family == 'rtx2080':
        prior = root/'pilot_A100'
        previous = json.loads((prior/'report.json').read_text())
        if previous['status'] != 'passed' or previous['signature'] != signature:
            raise ValueError('A100 pilot must pass first')
        if digest(prior/'references100.npz') != previous['files']['references100.npz']:
            raise ValueError('A100 pilot maps changed')
        with np.load(prior/'references100.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['indices'], indices)
            cross_gpu = {}
            for key in ('actual', 'hypothetical', 'logits', 'probabilities', 'delta', 'target_difference'):
                np.testing.assert_allclose(value[key], saved[key], atol=3e-5, rtol=2e-3)
                cross_gpu[key] = float(np.max(np.abs(value[key]-saved[key])))
            np.testing.assert_array_equal(value['quality_pass'], saved['quality_pass'])
    report = dict(status='passed' if grid_pass and (quality >= .95).all() else 'failed',
        signature=signature, family=family, job_id=os.environ.get('SLURM_JOB_ID'),
        internal_batch=config['internal_batch'], indices=indices.tolist(), quality_pass_fraction=quality.tolist(),
        integration64_vs128_anchor=grid, batch_equivalence_maximum=batch_error, cross_gpu_maximum=cross_gpu,
        references100_seconds_including_shuffle_and_checkpoints=time.monotonic()-began,
        files={name:digest(directory/name) for name in ('anchor64.npz', 'references100.npz')})
    write_json(directory/'report.json', report)
    if report['status'] != 'passed':
        raise ValueError('IG64 pilot failed convergence checks; no full array')
    if family == 'rtx2080':
        publish_gate(root, signature)
    event('ig64_pilot_passed', family=family, quality=quality.tolist(), cross_gpu=cross_gpu)


def run(project, root, stage, shard=0):
    require_gpu()
    config = json.loads((root/'config.json').read_text()); validate_config(config)
    family = hardware_family(socket.gethostname().split('.')[0], torch.cuda.get_device_name())
    if stage == 'pilot_A100' and family != 'A100' or stage == 'pilot_rtx2080' and family != 'rtx2080':
        raise ValueError('Wrong hardware for this pilot')
    for name, expected in config['source_hashes'].items():
        if digest(project/name) != expected:
            raise ValueError('Frozen input changed: '+name)
    signature = digest(root/'MANIFEST.sha256')
    config = dict(config, internal_batch=config['internal_batches'][family])
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with np.load(project/config['parent']/'cohort.npz', allow_pickle=False) as saved:
        data = dict(saved)
    if (len(data['ids']) != 40338 or len(np.unique(data['ids'])) != 40338
            or data['sequence'].shape != (40338,2048) or not np.isin(data['sequence'], range(4)).all()
            or data['labels'].shape != (40338,8) or not np.isin(data['labels'], [0,1]).all()
            or not data['labels'].any(1).all()):
        raise ValueError('Invalid full enhancer cohort')
    intervals = load_intervals(project/config['original'], data)
    began = time.monotonic(); stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    limit = 1680 if stage.startswith('pilot') else 85800
    def stop():
        return stopped[0] or (root/'STOP').exists() or time.monotonic()-began >= limit
    identity = f'{stage}_{shard}_{os.environ["SLURM_JOB_ID"]}'
    runtime = dict(stage=stage, shard=shard, family=family, node=socket.gethostname(),
        gpu=torch.cuda.get_device_name(), job_id=os.environ['SLURM_JOB_ID'], signature=signature,
        internal_batch=config['internal_batch'], references=100, steps=64, targets=list(TARGETS),
        torch_version=torch.__version__, cuda_version=torch.version.cuda, max_seconds=limit)
    write_json(root/f'runtime_{identity}.json', runtime); event('ig64_started', **runtime)
    try:
        model = load_model(project/config['parent']/'best_model.pt', 'cuda')
        if stage.startswith('pilot'):
            pilot(model,data,intervals,config,root,signature,family,stop,'cuda')
        else:
            verify_gate(root, signature)
            # Every allocated node checks its smaller/larger batch equivalence.
            check = anchor_check(model,data,np.asarray(config['pilot_indices']),config,'cuda',stop)
            runtime['anchor_batch_maximum'] = check[-1]; del check
            write_json(root/f'runtime_{identity}.json',runtime)
            production(model,data,intervals,config,root,signature,shard,stop,'cuda')
    except Exception as error:
        write_json(root/f'stopped_{identity}.json',dict(status='stopped' if isinstance(error,TimeoutError) else 'failed',
            reason=type(error).__name__+': '+str(error), elapsed_seconds=time.monotonic()-began, **runtime))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('pilot_A100', 'pilot_rtx2080', 'attribute'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    run(args.project.resolve(), args.root.resolve(), args.stage, args.shard)
