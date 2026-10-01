"""Bounded 100-to-200 reference continuation in a new, immutable-parent experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from classifier_motifs.attribution import dinucleotide_shuffle, load_model, one_hot, seed_for
from classifier_motifs.context_attribution import CONTEXTS, REGIONS, map_agreement
from .common import digest, event, require_allocation, write_json
from .context_pilot import check_config, refine_reference, save_npz, validate_state


def read_json(path):
    return json.loads(path.read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def select_cases(rows):
    """One low-stability and one median-stability completed case per exact degree."""
    selected = []
    if len({row['index'] for row in rows}) != len(rows):
        raise ValueError('Duplicate candidate indices')
    for degree in range(1, 9):
        group = sorted((r for r in rows if r['degree'] == degree),
                       key=lambda r: (r['stability_p10'], r['index']))
        if len(group) < 2 or not np.isfinite([r['stability_p10'] for r in group]).all():
            raise ValueError('Need at least two finite completed candidates per degree')
        median = float(np.median([r['stability_p10'] for r in group]))
        typical = min(group[1:], key=lambda r: (abs(r['stability_p10']-median), r['index']))
        selected.extend([dict(group[0], role='difficult'), dict(typical, role='typical')])
    return selected


def child_state(parent, parent_signature, parent_hash, signature):
    validate_state(parent, parent_signature, 100)
    if int(parent['count']) != 100 or len(json.loads(str(parent['diagnostics']))) != 100:
        raise ValueError('Only a complete 100-reference parent stream may be extended')
    state = {key: value.copy() for key, value in parent.items()}
    state.update(signature=np.asarray(signature), parent_signature=np.asarray(parent_signature),
                 parent_state_sha256=np.asarray(parent_hash), inherited_references=np.asarray(100))
    return state


def validate_child(state, parent, parent_signature, parent_hash, signature):
    validate_state(state, signature, 200)
    if (int(state['count']) < 100 or str(state['parent_signature']) != parent_signature
            or str(state['parent_state_sha256']) != parent_hash
            or int(state['inherited_references']) != 100):
        raise ValueError('Child provenance mismatch')
    for key in ('delta', 'difference', 'steps', 'passed', 'reference_hashes'):
        np.testing.assert_array_equal(state[key][:100], parent[key])
    diagnostics = json.loads(str(state['diagnostics']))
    if (len(diagnostics) != int(state['count'])
            or diagnostics[:100] != json.loads(str(parent['diagnostics']))):
        raise ValueError('Reference diagnostics/prefix mismatch')
    if int(state['count']) == 100:
        np.testing.assert_array_equal(state['sum_hypothetical'], parent['sum_hypothetical'])


def check_extension(config, parent):
    check_config(parent)
    if (config['references_per_block'] != [50, 100, 200]
            or config['max_seconds'] != 6600 or len(config['cases']) != 16
            or len({r['index'] for r in config['cases']}) != 16
            or sorted(r['degree'] for r in config['cases']) != sorted(list(range(1, 9))*2)):
        raise ValueError('Unexpected bounded extension contract')


def run(project, root):
    require_allocation('gpu')
    began = time.monotonic()
    config = read_json(root/'config.json')
    parent_root = project/config['pilot_parent']
    parent_config = read_json(parent_root/'config.json')
    check_extension(config, parent_config)
    # Parent states, exact ISM, source data and model are read-only and pinned.
    for relative, expected in config['source_hashes'].items():
        if digest(project/relative) != expected:
            raise ValueError('Frozen source changed: '+relative)
    parent_signature = hashlib.sha256((digest(parent_root/'config.json')+
        digest(parent_root/'MANIFEST.sha256')).encode()).hexdigest()
    if parent_signature != config['parent_signature']:
        raise ValueError('Parent contract changed')
    signature = hashlib.sha256((digest(root/'config.json')+digest(root/'MANIFEST.sha256')).encode()).hexdigest()
    out = root/'pilot'
    out.mkdir(exist_ok=True)
    selection = dict(signature=signature, parent_signature=parent_signature, cases=config['cases'],
        train_only=True, numerical_pilot_not_prevalence=True, contexts=list(CONTEXTS), regions=list(REGIONS))
    if (out/'selection.json').exists() and read_json(out/'selection.json') != selection:
        raise ValueError('Changed extension selection or code')
    write_json(out/'selection.json', selection)
    if (out/'complete.json').exists():
        raise ValueError('Extension already complete')
    stopped = [False]
    for signum in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.__setitem__(0, True))
    def stop():
        return stopped[0] or (root/'STOP').exists() or time.monotonic()-began >= config['max_seconds']
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA GPU not available')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = load_model(project/parent_config['parent']/'best_model.pt', 'cuda')
    event('reference_extension_started', cases=16, inherited_per_block=100, target_per_block=200,
          max_seconds=config['max_seconds'], gpu=torch.cuda.get_device_name(), signature=signature)
    completed = 0
    for case in config['cases']:
        i = case['index']
        source = parent_root/'pilot'/f'enhancer_{i:06d}'
        folder = out/f'enhancer_{i:06d}'
        folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():
            receipt = read_json(folder/'complete.json')
            if receipt['signature'] != signature:
                raise ValueError('Completed child signature changed')
            for name, expected in receipt['outputs'].items():
                if digest(folder/name) != expected:
                    raise ValueError('Completed child output changed: '+name)
            completed += 1
            continue
        template = read_npz(source/'block_0_n100.npz')
        codes, masks = template['sequence'], template['region_masks']
        if int(template['labels'].sum()) != case['degree']:
            raise ValueError('Selected degree mismatch')
        x = one_hot(codes[None], 'cuda')
        for block in range(2):
            parent_path = source/f'block_{block}_state.npz'
            parent_state = read_npz(parent_path)
            parent_hash = digest(parent_path)
            path = folder/f'block_{block}_state.npz'
            state = read_npz(path) if path.exists() else child_state(
                parent_state, parent_signature, parent_hash, signature)
            validate_child(state, parent_state, parent_signature, parent_hash, signature)
            diagnostics = json.loads(str(state['diagnostics']))
            for reference in range(int(state['count']), 200):
                if stop():
                    save_npz(path, **state)
                    write_json(out/'progress.json', dict(status='paused', signature=signature,
                        completed=completed, total=16, index=i, block=block, references=int(state['count']),
                        elapsed_seconds=time.monotonic()-began, reason='wall_budget_signal_or_STOP',
                        job_id=os.environ['SLURM_JOB_ID']))
                    event('reference_extension_paused', completed=completed, index=i,
                          block=block, references=int(state['count']))
                    return
                reference_seed = seed_for(parent_config['seed'], case['id'], 'context_pilot', block, reference)
                shuffled = dinucleotide_shuffle(codes, reference_seed)
                result, steps, passed, comparisons = refine_reference(model, x,
                    one_hot(shuffled[None], 'cuda'), masks, parent_config, anchor=reference == 100)
                np.testing.assert_allclose(result['logits'][0].cpu().numpy(), template['logits'], atol=1e-5, rtol=1e-5)
                hyp = result['hypothetical'][0].cpu().numpy()
                np.testing.assert_allclose(hyp[:, codes, np.arange(2048)],
                    result['actual'][0].cpu().numpy(), atol=1e-7, rtol=1e-5)
                state['sum_hypothetical'] += hyp
                state['count'] = np.asarray(reference+1)
                state['delta'] = np.vstack([state['delta'], result['delta'].cpu().numpy()])
                state['difference'] = np.vstack([state['difference'], result['target_difference'].cpu().numpy()])
                state['steps'] = np.append(state['steps'], np.int32(steps))
                state['passed'] = np.vstack([state['passed'], passed.cpu().numpy()])
                state['reference_hashes'] = np.append(state['reference_hashes'], hashlib.sha256(shuffled.tobytes()).hexdigest())
                diagnostics.append(dict(reference=reference, seed=reference_seed,
                    unchanged=bool(np.array_equal(shuffled, codes)), integration_comparisons=comparisons))
                state['diagnostics'] = np.asarray(json.dumps(diagnostics, allow_nan=False))
                if reference == 199:
                    mean = (state['sum_hypothetical']/200).astype(np.float32)
                    snapshot = dict(template, hypothetical=mean, actual=mean[:, codes, np.arange(2048)],
                                    signature=np.asarray(signature))
                    save_npz(folder/f'block_{block}_n200.npz', **snapshot)
                if (reference+1) % 10 == 0:
                    save_npz(path, **state)
                    event('reference_extension_progress', index=i, block=block, references=reference+1,
                          completed=completed, elapsed_seconds=time.monotonic()-began)
            save_npz(path, **state)
        comparisons = {}
        for count in (50, 100, 200):
            base = source if count < 200 else folder
            a, b = [read_npz(base/f'block_{block}_n{count}.npz') for block in range(2)]
            comparisons[str(count)] = dict(full=map_agreement(a['actual'], b['actual']),
                regions={name: map_agreement(a['actual'][:, mask], b['actual'][:, mask])
                         for name, mask in zip(REGIONS, masks)})
        combined = (a['hypothetical']+b['hypothetical'])*.5
        mutations = read_npz(source/'ism.npz')
        mutations['ig_hypothetical_difference'] = np.stack([
            combined[:, base, p]-combined[:, int(codes[p]), p]
            for p, base in zip(mutations['positions'], mutations['alternate'])])
        save_npz(folder/'ism.npz', **mutations)
        write_json(folder/'agreement.json', dict(independent_reference_blocks=True, comparisons=comparisons,
            contexts=list(CONTEXTS), inherited_references_per_block=100,
            ism_note='Same exact mutations/sites as parent; no reselection. IG differences use pooled400, not an independent400 comparison.'))
        outputs = {p.name: digest(p) for p in sorted(folder.iterdir()) if p.suffix in ('.npz', '.json')}
        write_json(folder/'complete.json', dict(signature=signature, index=i, outputs=outputs,
            reference_blocks=2, references_per_block=200, parent_signature=parent_signature))
        completed += 1
        write_json(out/'progress.json', dict(status='in_progress', signature=signature,
            completed=completed, total=16, elapsed_seconds=time.monotonic()-began))
        event('reference_extension_enhancer_complete', completed=completed, total=16, index=i)
    receipt = dict(status='complete', signature=signature, examples=completed,
        job_id=os.environ['SLURM_JOB_ID'], elapsed_seconds_this_allocation=time.monotonic()-began,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        interpretation='Selected numerical follow-up, not a representative prevalence/biological comparison.')
    write_json(out/'complete.json', receipt)
    write_json(out/'progress.json', receipt)
    event('reference_extension_complete', examples=completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.project.resolve(), args.root.resolve())


if __name__ == '__main__':
    main()
