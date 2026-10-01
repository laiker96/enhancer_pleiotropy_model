"""Relative-PWM occurrence, matched dinucleotide nulls, and saved IG importance.

The cutoff is a fraction of maximum log-odds, not a p-value or percent identity.
No attribution or model inference is performed here.
"""
import json
from pathlib import Path

import numpy as np
from numba import njit, prange

from .common import digest, event, write_json

TARGETS = ('mean_active_logit', 'soft_breadth')
GROUPS = (('exact_1', 1, 1), ('between_2_5', 2, 5), ('ge_6', 6, 8))
THRESHOLDS = (.7, .8, .9)


def read_inputs(root):
    config = json.loads((root/'config.json').read_text())
    parent = root.parents[1]/config['parent']
    with np.load(parent/'metadata.npz', allow_pickle=False) as saved:
        metadata = dict(saved)
    return config, parent, metadata


def pssm(pwm, background):
    pwm = np.asarray(pwm, dtype=float)
    background = np.asarray(background, dtype=float)
    if (pwm.ndim != 2 or pwm.shape[1] != 4 or not np.isfinite(pwm).all()
            or np.any(pwm < 0) or not np.allclose(pwm.sum(1), 1)
            or background.shape != (4,) or np.any(background <= 0)
            or not np.isclose(background.sum(), 1) or not np.allclose(background, background[::-1])):
        raise ValueError('Invalid PWM/background probabilities')
    weights = np.log2(((pwm + .001*background)/(1+.001))/background)
    maximum = weights.max(1).sum()
    if maximum <= 0:
        raise ValueError('Uninformative motif')
    return weights, maximum


@njit(parallel=True)
def scan(codes, lengths, weights, maximum, actual, importance):
    """Both strands; complete native hits only; union of >=80% hit bases."""
    n, width = len(codes), len(weights)
    best = np.full(n, -np.inf, np.float32)
    means = np.full(n, np.nan, np.float32)
    for i in prange(n):
        covered = np.zeros(lengths[i], np.uint8)
        for start in range(lengths[i]-width+1):
            forward, reverse = 0., 0.
            for j in range(width):
                base = codes[i, start+j]
                forward += weights[j, base]
                reverse += weights[width-1-j, 3-base]
            score = max(forward, reverse)/maximum
            best[i] = max(best[i], score)
            if importance and score >= .8:
                covered[start:start+width] = 1
        if importance:
            total, count = 0., 0
            for j in range(lengths[i]):
                if covered[j]:
                    total += actual[i, j]
                    count += 1
            if count:
                means[i] = total/count
    return best, means


def setup(root):
    import importlib.metadata
    from .original_sequence_report import shuffle
    config, parent, metadata = read_inputs(root)
    # Check frozen inputs, including original tensors, before either analysis.
    ready = json.loads((parent/'setup_complete.json').read_text())
    if ready['status'] != 'complete':
        raise ValueError('Incomplete parent')
    for name, expected in ready['files'].items():
        if digest(parent/name) != expected:
            raise ValueError('Changed parent: '+name)
    for name, expected in config['source_hashes'].items():
        if digest(parent/name) != expected:
            raise ValueError('Changed motif source: '+name)
    lengths, codes = metadata['length'], metadata['sequence']
    if (len(np.unique(metadata['ids'])) != len(codes)
            or not np.isin(metadata['labels'], [0, 1]).all()
            or not np.all((metadata['labels'].sum(1) >= 1) & (metadata['labels'].sum(1) <= 8))):
        raise ValueError('Invalid enhancer metadata')
    nulls = np.lib.format.open_memmap(root/'null_sequences.npy', mode='w+',
        dtype=np.uint8, shape=(config['null_replicates'], *codes.shape))
    nulls[:] = 4
    unchanged, duplicates = 0, 0
    for i, length in enumerate(lengths):
        seq = codes[i, :length]
        if not np.isin(seq, range(4)).all() or np.any(codes[i, length:] != 4):
            raise ValueError('Unexpected native sequence/padding')
        expected = np.bincount(seq[:-1]*4+seq[1:], minlength=16)
        seen = set()
        for k in range(config['null_replicates']):
            shuffled = shuffle(seq, config['seed']+i*1009+k)
            np.testing.assert_array_equal(np.bincount(shuffled[:-1]*4+shuffled[1:], minlength=16), expected)
            np.testing.assert_array_equal(np.bincount(shuffled, minlength=4), np.bincount(seq, minlength=4))
            if not (shuffled[0] == seq[0] and shuffled[-1] == seq[-1]):
                raise ValueError('Shuffle endpoints changed')
            unchanged += int(np.array_equal(shuffled, seq))
            duplicates += int(shuffled.tobytes() in seen)
            seen.add(shuffled.tobytes())
            nulls[k, i, :length] = shuffled
        if i % 5000 == 0:
            event('null_progress', enhancers=i, total=len(codes))
    nulls.flush()
    write_json(root/'setup_complete.json', dict(status='complete', n=len(codes),
        config_sha256=digest(root/'config.json'), unchanged_nulls=unchanged,
        duplicate_null_replicates=duplicates, replicates=config['null_replicates'],
        null='Randomized Euler trails, exact mono/dinucleotide counts and endpoints; not uniform over all shuffles.',
        versions={name:importlib.metadata.version(name) for name in
                  ('torch','numpy','numba','h5py','polars-lts-cpu','jaxtyping','tqdm')},
        files={'null_sequences.npy':digest(root/'null_sequences.npy')}))
    event('occurrence_setup_complete', enhancers=len(codes), unchanged=unchanged, duplicates=duplicates)


def run_scan(root, target):
    config, parent, metadata = read_inputs(root)
    if json.loads((root/'setup_complete.json').read_text())['config_sha256'] != digest(root/'config.json'):
        raise ValueError('Setup/config mismatch')
    result = root/TARGETS[target]/'sequence'; result.mkdir(parents=True, exist_ok=True)
    audit = json.loads((parent/TARGETS[target]/'report_audit.json').read_text())
    actual = np.load(parent/'native_actual.npy', mmap_mode='r')[:, target]
    nulls = np.load(root/'null_sequences.npy', mmap_mode='r')
    lengths, codes = metadata['length'], metadata['sequence']
    paths = []
    for group in audit['groups']:
        for row in group['rows']:
            path = result/(row['id'].replace('/', '__')+'.npz')
            receipt = path.with_suffix('.complete.json')
            if receipt.exists():
                if digest(path) != json.loads(receipt.read_text())['sha256']:
                    raise ValueError('Changed saved motif scan')
                paths.append(path); continue
            weights, maximum = pssm(row['trimmed_pwm'], audit['background']['probabilities'])
            best, means = scan(codes, lengths, weights, maximum, actual, True)
            means[~metadata['quality_pass'][:, target]] = np.nan
            null_best = np.stack([scan(nulls[k], lengths, weights, maximum,
                np.empty((1, 1), np.float32), False)[0] for k in range(len(nulls))])
            np.savez_compressed(path, ids=metadata['ids'], best_fraction=best,
                mean_importance=means, null_best_fraction=null_best, maximum_logodds=maximum,
                weights=weights, thresholds=np.array(THRESHOLDS))
            write_json(receipt, dict(status='complete', sha256=digest(path), id=row['id']))
            paths.append(path)
            event('sequence_motif_complete', target=TARGETS[target], motif=row['id'],
                  carriers_80=int((best >= .8).sum()))
    write_json(result/'complete.json', dict(status='complete', files={p.name:digest(p) for p in paths}))


def group_summary(best, null_best, importance, metadata, target, low, high, split):
    degree = metadata['labels'].sum(1)
    selected = (degree >= low) & (degree <= high) & (metadata['split'] == split)
    n = int(selected.sum()); rows = []
    for threshold in THRESHOLDS:
        hits = int(np.sum(selected & (best >= threshold)))
        null_counts = (null_best[:, selected] >= threshold).sum(1)
        expected = float(null_counts.mean())
        rows.append(dict(threshold=threshold, n=n, sequence_carriers=hits,
            fraction=hits/n if n else None, null_counts=null_counts.tolist(),
            null_fraction=expected/n if n else None,
            excess_fraction=(hits-expected)/n if n else None,
            enrichment=hits/expected if expected else None))
    valid = selected & metadata['quality_pass'][:, target] & np.isfinite(importance)
    return dict(thresholds=rows, n=n, attribution_carriers=int(valid.sum()),
        quality_excluded=int((selected & ~metadata['quality_pass'][:, target]).sum()),
        mean_importance=float(importance[valid].mean()) if valid.any() else None)
