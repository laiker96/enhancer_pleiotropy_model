"""Pinned Fi-NeMo projected-CWM fits on native enhancer intervals.

Loads the upstream hitcaller directly, avoiding unused CLI/report dependencies.
The sole upstream patch adds native-boundary masks and native-length scaling.
"""
import importlib.util
import json
import time
from pathlib import Path

import h5py
import numpy as np

from .common import digest, event, write_json
from .motif_occurrence import TARGETS, GROUPS, read_inputs


def engine(path):
    spec = importlib.util.spec_from_file_location('pinned_finemo_hitcaller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fit_contribs


def motif_library(parent, target, group):
    """Unit-L2 retained CWMs, with upstream's FP16-to-FP32 convention.

    Explicit trim coordinates are the frozen 0.2-bit PWM boundaries, for both
    strands. Normalize AFTER retaining the core: normalization of the original
    50-bp CWM followed by masking makes four cores unable to pass lambda .7.
    There is no additional boundary trimming or motif reclustering.
    """
    audit = json.loads((parent/TARGETS[target]/'report_audit.json').read_text())
    rows = next(g['rows'] for g in audit['groups'] if g['name'] == group)
    cwms, masks, records = [], [], []
    with h5py.File(parent/TARGETS[target]/'groups'/group/'motifs.h5') as source:
        for row in rows:
            raw = source[row['pattern']+'/contrib_scores'][:].T
            pwm = source[row['pattern']+'/sequence'][:]
            np.testing.assert_allclose(pwm/pwm.sum(1, keepdims=True), row['full_pwm'], atol=1e-6)
            full_norm = np.sqrt((raw**2).sum())
            if not np.isfinite(raw).all() or full_norm <= 0:
                raise ValueError('Invalid CWM')
            start, end = row['quality']['start'], row['quality']['end']
            width = raw.shape[1]
            raw[:, :start] = 0; raw[:, end:] = 0
            norm = np.sqrt((raw**2).sum())
            if norm <= 0: raise ValueError('Zero retained CWM')
            maximum_projected_similarity = np.sqrt((raw**2).max(0).sum())/norm
            if maximum_projected_similarity <= .7:
                raise ValueError('Retained projected CWM cannot pass Fi-NeMo lambda .7')
            for strand in ('+', '-'):
                a, b = (start, end) if strand == '+' else (width-end, width-start)
                cwm = raw/norm if strand == '+' else (raw/norm)[::-1, ::-1]
                mask = np.zeros(width, np.int8); mask[a:b] = 1
                cwms.append(cwm); masks.append(mask)
                records.append(dict(id=row['id'], sign=row['sign'], strand=strand,
                    start=a, end=b, scale=float(norm), full_cwm_norm=float(full_norm),
                    retained_energy_fraction=float((norm/full_norm)**2),
                    maximum_projected_similarity=float(maximum_projected_similarity)))
    return np.stack(cwms).astype(np.float16), np.stack(masks), records


def padded_inputs(metadata, actual, indices, target, width, extra=0):
    lengths = metadata['length'][indices]
    pad = width-1+extra
    total = int(lengths.max())+2*pad
    sequence = np.zeros((len(indices), 4, total), np.int8)
    contribution = np.zeros((len(indices), total), np.float32)
    for j, index in enumerate(indices):
        length = int(lengths[j])
        codes = metadata['sequence'][index, :length]
        sequence[j, codes, np.arange(pad, pad+length)] = 1
        contribution[j, pad:pad+length] = actual[index, target, :length]
    if not np.isfinite(contribution).all():
        raise ValueError('Nonfinite saved attribution')
    return sequence, contribution, np.column_stack((np.full(len(indices), pad), pad+lengths)), pad


def invoke(fit, cwms, masks, sequence, contribution, bounds, device, batch=128):
    import torch
    return fit(cwms=cwms, contribs=contribution, sequences=sequence,
        cwm_trim_mask=masks, use_hypothetical=False,
        lambdas=np.full(len(cwms), .7, np.float32), device=torch.device(device),
        batch_size=min(batch, len(sequence)), max_steps=10000, convergence_tol=.0005,
        post_filter=True, compile_optimizer=False, region_bounds=bounds)


def synthetic_checks(fit, device):
    """Positive, negative and RC insertions at both native edges; zero decoy.

    Also compare the adapter to upstream behavior without padding, then to the
    same data padded on both ends. Coefficients compared with numerical tolerance.
    """
    codes = np.random.default_rng(71).integers(0, 4, (5, 64), dtype=np.uint8)
    core = np.array([0, 1, 2, 3, 0, 0, 1])
    cwm = np.zeros((4, 7), np.float32); cwm[core, np.arange(7)] = 1/np.sqrt(7)
    cwms = np.stack((cwm, cwm[::-1, ::-1], -cwm, -cwm[::-1, ::-1]))
    masks = np.ones((4, 7), np.int8)
    contributions = np.zeros((5, 64), np.float32)
    expected = []
    for i, (motif, start) in enumerate(((0, 0), (1, 57), (2, 0), (3, 57))):
        codes[i, start:start+7] = core if motif % 2 == 0 else 3-core[::-1]
        contributions[i, start:start+7] = 2 if motif < 2 else -2
        expected.append((i, motif, start))
    sequences = np.eye(4, dtype=np.int8)[codes].transpose(0, 2, 1).copy()
    base, qc = invoke(fit, cwms, masks, sequences, contributions, None, device, batch=3)
    exact, _ = invoke(fit, cwms, masks, sequences, contributions,
        np.tile([0, 64], (5, 1)), device, batch=3)
    key = ['peak_id', 'motif_id', 'hit_start']
    base, exact = base.sort(key), exact.sort(key)
    np.testing.assert_array_equal(base.select(key).to_numpy(), exact.select(key).to_numpy())
    np.testing.assert_allclose(base['hit_coefficient'], exact['hit_coefficient'], atol=2e-5, rtol=2e-4)
    pad = 19
    padded, pqc = invoke(fit, cwms, masks, np.pad(sequences, ((0, 0), (0, 0), (pad, pad))),
        np.pad(contributions, ((0, 0), (pad, pad))), np.tile([pad, pad+64], (5, 1)), device, batch=3)
    import polars as pl
    padded = padded.with_columns((pl.col('hit_start')-pad).alias('hit_start')).sort(key)
    np.testing.assert_array_equal(base.select(key).to_numpy(), padded.select(key).to_numpy())
    np.testing.assert_allclose(base['hit_coefficient'], padded['hit_coefficient'], atol=3e-4, rtol=3e-3)
    np.testing.assert_allclose(qc.sort('peak_id')['global_scale'], pqc.sort('peak_id')['global_scale'], atol=1e-7)
    keys = set(map(tuple, padded.select(key).to_numpy()))
    if set(expected) != keys:
        raise ValueError('Synthetic positive/negative/RC/edge/zero control failed: '+str(keys))
    trimmed, _ = invoke(fit, np.pad(cwms, ((0,0),(0,0),(6,6))),
        np.pad(masks, ((0,0),(6,6))),
        np.pad(sequences, ((0,0),(0,0),(pad,pad))),
        np.pad(contributions, ((0,0),(pad,pad))), np.tile([pad,pad+64], (5,1)), device, batch=3)
    trimmed = trimmed.with_columns((pl.col('hit_start')+6-pad).alias('hit_start')).sort(key)
    np.testing.assert_array_equal(base.select(key).to_numpy(), trimmed.select(key).to_numpy())
    np.testing.assert_allclose(base['hit_coefficient'], trimmed['hit_coefficient'], atol=3e-4, rtol=3e-3)
    if not np.isfinite(pqc.select(['dual_gap', 'nll', 'global_scale']).to_numpy()).all():
        raise ValueError('Nonfinite synthetic QC')
    return dict(status='passed', expected_hits=len(expected), observed_hits=len(keys),
        boundary_invariance=True, trimmed_cwm_edge_recovery=True,
        no_padding_upstream_equivalence=True, device=device)


def run_chunk(root, metadata, actual, target, group, indices, directory, fit, device):
    import polars as pl
    config, parent, _ = read_inputs(root)
    receipt = directory/'complete.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['config_sha256'] != digest(root/'config.json'):
            raise ValueError('Resume/config mismatch')
        for name, expected in saved['files'].items():
            if digest(directory/name) != expected:
                raise ValueError('Changed chunk')
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    cwms, masks, records = motif_library(parent, target, group)
    write_json(directory/'motif_library.json', records)
    sequence, contribution, bounds, pad = padded_inputs(metadata, actual, indices, target, cwms.shape[2])
    start = time.monotonic()
    hits, qc = invoke(fit, cwms, masks, sequence, contribution, bounds, device, config['batch_size'])
    seconds = time.monotonic()-start
    np.testing.assert_array_equal(np.sort(qc['peak_id'].to_numpy()), np.arange(len(indices)))
    if not np.isfinite(qc.select(['dual_gap', 'nll', 'global_scale']).to_numpy()).all():
        raise ValueError('Nonfinite real-data QC')
    if len(hits) and not np.isfinite(hits.select(['hit_coefficient', 'hit_similarity', 'hit_importance']).to_numpy()).all():
        raise ValueError('Nonfinite hit')
    mapping = pl.DataFrame(dict(peak_id=np.arange(len(indices), dtype=np.uint32),
        enhancer_index=indices, enhancer_id=metadata['ids'][indices], native_length=metadata['length'][indices]))
    motifs = pl.DataFrame(records).with_row_index('motif_id')
    qc = qc.join(mapping, on='peak_id').with_columns(
        ((pl.col('dual_gap') <= .0005) & (pl.col('num_steps') <= 10000)
         & (pl.col('step_size') >= .08)).alias('converged'))
    hits = hits.join(mapping, on='peak_id').join(motifs, on='motif_id').join(
        qc.select(['peak_id', 'global_scale', 'converged']), on='peak_id').with_columns(
        (pl.col('hit_start')+pl.col('start')-pad).alias('native_start'),
        (pl.col('hit_start')+pl.col('end')-pad).alias('native_end'),
        (pl.col('hit_coefficient')*pl.col('global_scale')**2).alias('hit_coefficient_global'),
        (pl.col('hit_importance')*pl.col('global_scale')).alias('hit_importance_global'))
    if len(hits.filter((pl.col('native_start') < 0) | (pl.col('native_end') > pl.col('native_length')))):
        raise ValueError('Hit extends outside native enhancer')
    # Retain failed-region hits for audit, but never count them as valid support.
    hits.write_parquet(directory/'hits.parquet'); qc.write_parquet(directory/'qc.parquet')
    np.save(directory/'indices.npy', indices, allow_pickle=False)
    saved = dict(status='complete', target=TARGETS[target], group=group, n=len(indices),
        converged=int(qc['converged'].sum()), hits=len(hits), seconds=seconds,
        config_sha256=digest(root/'config.json'), files={p.name:digest(p) for p in
        (directory/'hits.parquet', directory/'qc.parquet', directory/'indices.npy',directory/'motif_library.json')})
    write_json(receipt, saved)
    event('finemo_chunk_complete', **{k:v for k,v in saved.items() if k!='files'})
    return saved


def pilot(root, device):
    import torch
    config, parent, metadata = read_inputs(root)
    fit = engine(root/'vendor/hitcaller.py')
    synthetic = synthetic_checks(fit, device)
    actual = np.load(parent/'native_actual.npy', mmap_mode='r')
    degree = metadata['labels'].sum(1); rng = np.random.default_rng(config['seed'])
    selected = np.concatenate([rng.choice(np.flatnonzero((metadata['split']=='train')
        & metadata['quality_pass'].all(1) & (degree>=low) & (degree<=high)), 16, replace=False)
        for _,low,high in GROUPS])
    chunks = []
    for target in range(2):
        for group, _, _ in GROUPS:
            chunks.append(run_chunk(root, metadata, actual, target, group, selected,
                root/'pilot'/TARGETS[target]/group, fit, device))
    fraction = min(c['converged']/c['n'] for c in chunks)
    # Time projection is conservative for launch gating, not a promised ETA.
    estimate = sum(c['seconds']/c['n']*int(metadata['quality_pass'][:,t].sum())
        for t in range(2) for c in chunks if c['target']==TARGETS[t])
    passed = fraction >= .95 and estimate*1.5 < config['full_limit_hours']*3600
    write_json(root/'pilot_complete.json', dict(status='passed' if passed else 'failed',
        synthetic=synthetic, minimum_convergence=fraction, chunks=chunks,
        projected_full_seconds=estimate, runtime_safety_factor=1.5,
        device=torch.cuda.get_device_name() if device=='cuda' else device,
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
        config_sha256=digest(root/'config.json')))
    if not passed:
        raise RuntimeError('Pilot convergence/runtime gate failed; full scan prohibited')
    event('finemo_pilot_passed', minimum_convergence=fraction, projected_seconds=estimate)


def full(root):
    config, parent, metadata = read_inputs(root)
    gate = json.loads((root/'pilot_complete.json').read_text())
    if gate['status'] != 'passed' or gate['config_sha256'] != digest(root/'config.json'):
        raise ValueError('Full run requires passed matching pilot')
    actual = np.load(parent/'native_actual.npy', mmap_mode='r')
    fit = engine(root/'vendor/hitcaller.py')
    results = []
    for target in range(2):
        indices = np.flatnonzero(metadata['quality_pass'][:, target])
        for group, _, _ in GROUPS:
            for start in range(0, len(indices), config['chunk_size']):
                directory = root/TARGETS[target]/'finemo'/group/f'chunk_{start:06d}'
                result = run_chunk(root, metadata, actual, target, group,
                    indices[start:start+config['chunk_size']], directory, fit, 'cuda')
                results.append(dict(path=str(directory.relative_to(root)), **result))
    write_json(root/'finemo_complete.json', dict(status='complete', chunks=results,
        config_sha256=digest(root/'config.json'), pilot_sha256=digest(root/'pilot_complete.json')))
    event('finemo_full_complete', chunks=len(results))
