"""CPU-only, hash-verified review of a completed prefix of the context pilot.

No Torch import, model execution, job submission, or modification of raw outputs.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CONTEXTS = ('ab', 'e13', 'e5', 'ead', 'hid', 'lb', 'o', 'wid')
REGIONS = ('native', 'inner_flanks', 'middle_flanks', 'outer_flanks')
METRICS = ('cosine', 'weighted_sign_agreement', 'top10pct_jaccard', 'mean_absolute_difference')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def distribution(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(a):
        return dict(n=0, median=None, mean=None, p10=None, p90=None, minimum=None, maximum=None)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite summary input')
    return dict(n=len(a), median=float(np.median(a)), mean=float(a.mean()),
        p10=float(np.quantile(a, .1)), p90=float(np.quantile(a, .9)),
        minimum=float(a.min()), maximum=float(a.max()))


def agreement(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.shape != b.shape or not np.isfinite([a, b]).all():
        raise ValueError('Map shapes or values differ')
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    weight = np.maximum(np.abs(a), np.abs(b))
    k = max(1, int(np.ceil(.1 * len(a)))) if len(a) else 0
    ai = np.argsort(np.abs(a), kind='stable')[-k:] if k else []
    bi = np.argsort(np.abs(b), kind='stable')[-k:] if k else []
    union = np.union1d(ai, bi)
    return dict(cosine=float(a @ b / norm) if norm else None,
        weighted_sign_agreement=float(weight[np.sign(a) == np.sign(b)].sum() / weight.sum()) if weight.sum() else None,
        top10pct_jaccard=float(len(np.intersect1d(ai, bi)) / len(union)) if norm and len(union) else None,
        mean_absolute_difference=float(np.abs(a - b).mean()) if len(a) else None)


def compare_saved(actual, saved):
    for key in METRICS:
        if actual[key] is None or saved[key] is None:
            if actual[key] != saved[key]:
                raise ValueError('Saved agreement missing-value mismatch')
        elif not np.isclose(actual[key], saved[key], atol=1e-11, rtol=1e-10):
            raise ValueError('Saved agreement mismatch: ' + key)


def mutation_metrics(exact, approximate):
    exact, approximate = np.asarray(exact, float), np.asarray(approximate, float)
    if exact.shape != approximate.shape or not np.isfinite([exact, approximate]).all():
        raise ValueError('Invalid paired mutation predictions')
    n = len(exact)
    valid = n >= 3 and np.std(exact) > 0 and np.std(approximate) > 0
    large = np.abs(exact) >= .05
    return dict(n=n,
        pearson=float(np.corrcoef(exact, approximate)[0, 1]) if valid else None,
        spearman=float(spearmanr(exact, approximate).statistic) if valid else None,
        mae=float(np.abs(exact - approximate).mean()) if n else None,
        exact_abs_mean=float(np.abs(exact).mean()) if n else None,
        sign_agreement_abs_exact_ge_005=float((np.sign(exact[large]) == np.sign(approximate[large])).mean()) if large.any() else None,
        n_abs_exact_ge_005=int(large.sum()))


def paired_improvement(records, selection):
    """Enhancer-level averaging, then paired enhancer bootstrap (not bin resampling)."""
    pairs = defaultdict(dict)
    for row in records:
        if selection(row):
            pairs[(row['index'], row['target'])][row['references']] = row
    result = {}
    for metric in METRICS:
        by_enhancer = defaultdict(list)
        for (index, _), pair in pairs.items():
            if set(pair) != {50, 100}:
                raise ValueError('Unpaired reference comparison')
            a, b = pair[50][metric], pair[100][metric]
            if a is not None and b is not None:
                by_enhancer[index].append(b - a)
        values = np.asarray([np.mean(v) for _, v in sorted(by_enhancer.items())])
        if not len(values):
            result[metric] = dict(n_enhancers=0, mean_delta=None, ci95=None)
            continue
        rng = np.random.default_rng(20260919)
        samples = values[rng.integers(len(values), size=(3000, len(values)))].mean(1)
        result[metric] = dict(n_enhancers=len(values), mean_delta=float(values.mean()),
            median_delta=float(np.median(values)), ci95=np.quantile(samples, [.025, .975]).tolist())
    return result


def summarize_maps(records):
    groups = defaultdict(list)
    for row in records:
        if row['target'] == 'observed_active_mean':
            views = [('observed_active_mean', 'all', row['region']),
                     ('observed_active_mean', 'degree_' + str(row['degree']), row['region'])]
        else:
            views = [('all_contexts', 'all', row['region']),
                     ('per_context', row['target'], row['region'])]
            if row['active']:
                views += [('active_contexts', 'all', row['region'])]
        for view, subgroup, region in views:
            groups[(row['kind'], row['references'], view, subgroup, region)].append(row)
    summaries = []
    for (kind, refs, view, subgroup, region), rows in sorted(groups.items()):
        scores = {key: distribution([r[key] for r in rows]) for key in METRICS}
        finite = [r for r in rows if r['cosine'] is not None]
        summaries.append(dict(kind=kind, references=refs, view=view, subgroup=subgroup,
            region=region, n_enhancers=len({r['index'] for r in rows}), metrics=scores,
            fraction_cosine_ge_095=float(np.mean([r['cosine'] >= .95 for r in finite])) if finite else None,
            fraction_top10pct_jaccard_ge_070=float(np.mean([r['top10pct_jaccard'] >= .7 for r in finite])) if finite else None))
    return summaries


def analyze(root, output, package):
    progress, selection = read_json(root / 'pilot/progress.json'), read_json(root / 'pilot/selection.json')
    config = read_json(root / 'config.json')
    signature = hashlib.sha256((digest(root / 'config.json') + digest(root / 'MANIFEST.sha256')).encode()).hexdigest()
    if any(digest(root / name) != digest(package / name) for name in ('config.json', 'MANIFEST.sha256')):
        raise ValueError('Remote package/config differs from submitted local package')
    if (signature != progress['signature'] or signature != selection['signature']
            or selection['contexts'] != list(CONTEXTS) or selection['regions'] != list(REGIONS)
            or not selection['train_only'] or config['references_per_block'] != [50, 100]
            or config['reference_blocks'] != 2):
        raise ValueError('Pilot contract differs')
    lookup = {int(index): pos for pos, index in enumerate(selection['indices'])}
    receipts = sorted((root / 'pilot').glob('enhancer_*/complete.json'))
    if len(receipts) != progress['completed']:
        raise ValueError('Progress does not match completion receipts')
    completed_indices = {read_json(p)['index'] for p in receipts}
    if completed_indices != set(selection['indices'][:len(receipts)]):
        raise ValueError('Completed examples are not the recorded selection prefix')
    hashes = {str(p.relative_to(root)): digest(p) for p in
              (root / 'config.json', root / 'MANIFEST.sha256', root / 'pilot/progress.json', root / 'pilot/selection.json')}
    records, convergence, integration, ism_records, mutations = [], [], [], [], []
    reference_hashes, failures, degrees = [], [], Counter()
    verified_files = 0
    for receipt_path in receipts:
        receipt, folder = read_json(receipt_path), receipt_path.parent
        if receipt['signature'] != signature or receipt['references_per_block'] != 100 or receipt['reference_blocks'] != 2:
            raise ValueError('Invalid enhancer completion receipt')
        hashes[str(receipt_path.relative_to(root))] = digest(receipt_path)
        for name, expected in receipt['outputs'].items():
            if Path(name).name != name or digest(folder / name) != expected:
                raise ValueError('Output hash mismatch: ' + str(folder / name))
            hashes[str((folder / name).relative_to(root))] = expected
            verified_files += 1
        index = receipt['index']; pos = lookup[index]
        labels = np.asarray(selection['labels'][pos], bool)
        degree = int(labels.sum()); degrees[degree] += 1
        base = dict(index=index, id=selection['ids'][pos], degree=degree)
        saved = read_json(folder / 'agreement.json')
        snapshots = {}
        quality = []
        for block in range(2):
            with np.load(folder / f'block_{block}_state.npz', allow_pickle=False) as z:
                if int(z['count']) != 100 or str(z['signature']) != signature:
                    raise ValueError('Incomplete state in completed enhancer')
                delta, difference, passed = z['delta'], z['difference'], z['passed']
                tolerance = .01 + .01 * np.abs(difference)
                calculated_pass = np.abs(delta) <= tolerance
                mismatch = calculated_pass != passed
                if np.any(mismatch & (np.abs(np.abs(delta) - tolerance) > 1e-7)):
                    raise ValueError('Completeness flags disagree with saved residuals')
                if delta.shape != (100, 8) or not np.isfinite([delta, difference]).all():
                    raise ValueError('Invalid convergence arrays')
                quality.append(passed)
                reference_hashes.extend(z['reference_hashes'].tolist())
                for ref in range(100):
                    convergence.append(dict(**base, block=block, reference=ref, steps=int(z['steps'][ref]),
                        passed=passed[ref].tolist(), absolute_delta=np.abs(delta[ref]).tolist(),
                        tolerance_ratio=(np.abs(delta[ref]) / tolerance[ref]).tolist()))
                for item in json.loads(str(z['diagnostics'])):
                    for comparison in item['integration_comparisons']:
                        for region in ('full',) + REGIONS:
                            maps = comparison['full'] if region == 'full' else comparison['regions'][region]
                            for context, metrics in zip(CONTEXTS, maps):
                                integration.append(dict(**base, block=block, reference=item['reference'],
                                    anchor=item['reference'] == 0, from_steps=comparison['from_steps'],
                                    to_steps=comparison['to_steps'], region=region, target=context, **metrics))
            for n in (50, 100):
                with np.load(folder / f'block_{block}_n{n}.npz', allow_pickle=False) as z:
                    snapshot = {key: z[key] for key in z.files}
                if (str(snapshot['signature']) != signature or snapshot['hypothetical'].shape != (8, 4, 2048)
                        or snapshot['actual'].shape != (8, 2048) or snapshot['sequence'].shape != (2048,)
                        or not np.isfinite(snapshot['hypothetical']).all()):
                    raise ValueError('Snapshot contract mismatch')
                np.testing.assert_array_equal(snapshot['labels'], labels)
                np.testing.assert_allclose(snapshot['hypothetical'][:, snapshot['sequence'], np.arange(2048)],
                                           snapshot['actual'], atol=1e-7, rtol=1e-6)
                snapshots[(block, n)] = snapshot
        quality = np.concatenate(quality)
        if not quality.all():
            failures.append(dict(**base, failed_reference_contexts=int((~quality).sum())))
        codes = snapshots[(0, 50)]['sequence']
        masks = snapshots[(0, 50)]['region_masks'].astype(bool)
        if masks.shape != (4, 2048) or not np.all(masks.sum(0) == 1):
            raise ValueError('Region masks do not partition full input')
        offset, length = selection['native_offsets'][pos], selection['lengths'][pos]
        np.testing.assert_array_equal(masks[0], (np.arange(2048) >= offset) & (np.arange(2048) < offset + length))
        for snapshot in snapshots.values():
            np.testing.assert_array_equal(snapshot['sequence'], codes)
            np.testing.assert_array_equal(snapshot['region_masks'], masks)
            np.testing.assert_allclose(snapshot['logits'], snapshots[(0, 50)]['logits'], atol=1e-6)
        region_masks = dict(full=np.ones(2048, bool), **dict(zip(REGIONS, masks)))
        for n in (50, 100):
            a, b = snapshots[(0, n)], snapshots[(1, n)]
            for kind in ('actual', 'hypothetical'):
                for region, mask in region_masks.items():
                    for c, context in enumerate(CONTEXTS):
                        metrics = agreement(a[kind][c][..., mask], b[kind][c][..., mask])
                        if kind == 'actual':
                            original = saved['comparisons'][str(n)]
                            expected = original['full'][c] if region == 'full' else original['regions'][region][c]
                            compare_saved(metrics, expected)
                        records.append(dict(**base, target=context, active=bool(labels[c]), kind=kind,
                            references=n, region=region, all_references_pass=bool(quality.all()), **metrics))
                    metrics = agreement(a[kind][labels].mean(0)[..., mask], b[kind][labels].mean(0)[..., mask])
                    records.append(dict(**base, target='observed_active_mean', active=True, kind=kind,
                        references=n, region=region, all_references_pass=bool(quality.all()), **metrics))
        with np.load(folder / 'ism.npz', allow_pickle=False) as z:
            ism = {key: z[key] for key in z.files}
        positions, alternate, exact = ism['positions'], ism['alternate'], ism['delta_logits']
        if (exact.shape != (len(positions), 8) or not np.isfinite(exact).all()
                or not np.all(alternate != codes[positions])):
            raise ValueError('Invalid exact mutation outputs')
        position_counts = Counter(positions.tolist())
        if any(n != 3 for n in position_counts.values()) or len(set(zip(positions, alternate))) != len(positions):
            raise ValueError('Expected all three unique alternate bases at each ISM site')
        pooled_actual = (snapshots[(0, 100)]['actual'] + snapshots[(1, 100)]['actual']) * .5
        score = np.abs(pooled_actual).mean(0)
        tops = set()
        for mask in masks:
            valid = np.flatnonzero(mask)
            tops.update(valid[np.argsort(-score[valid], kind='stable')[:2]].tolist())
        for label, hyp in [('block0_50', snapshots[(0, 50)]['hypothetical']),
                           ('block1_50', snapshots[(1, 50)]['hypothetical']),
                           ('block0_100', snapshots[(0, 100)]['hypothetical']),
                           ('block1_100', snapshots[(1, 100)]['hypothetical']),
                           ('pooled200', (snapshots[(0, 100)]['hypothetical'] + snapshots[(1, 100)]['hypothetical']) * .5)]:
            predicted = np.stack([hyp[:, b, p] - hyp[:, codes[p], p] for p, b in zip(positions, alternate)])
            if label == 'pooled200':
                np.testing.assert_allclose(predicted, ism['ig_hypothetical_difference'], atol=1e-7, rtol=1e-6)
            for region, mask in region_masks.items():
                for site_type in ('all', 'top', 'random'):
                    keep = mask[positions] & np.asarray([site_type == 'all' or ((int(p) in tops) == (site_type == 'top')) for p in positions])
                    for c, context in enumerate(CONTEXTS):
                        ism_records.append(dict(**base, target=context, active=bool(labels[c]), references=label,
                            region=region, site_type=site_type, **mutation_metrics(exact[keep, c], predicted[keep, c])))
                    ism_records.append(dict(**base, target='observed_active_mean', active=True, references=label,
                        region=region, site_type=site_type,
                        **mutation_metrics(exact[keep][:, labels].mean(1), predicted[keep][:, labels].mean(1))))
            if label == 'pooled200':
                mutations.append(dict(**base, positions=positions.tolist(), alternate=alternate.tolist(),
                    labels=labels.tolist(), exact=exact.tolist(), approximate=predicted.tolist()))
        print(f'Verified and analyzed {base["id"]}: degree {degree}', flush=True)
    summaries = summarize_maps(records)
    integration_summary = []
    for anchor in (True, False):
        for stage in ((64, 128), (128, 256), (256, 512)):
            for region in ('full',) + REGIONS:
                subset = [r for r in integration if r['anchor'] == anchor and (r['from_steps'], r['to_steps']) == stage and r['region'] == region]
                if subset:
                    integration_summary.append(dict(anchor=anchor, from_steps=stage[0], to_steps=stage[1], region=region,
                        n_enhancers=len({r['index'] for r in subset}), metrics={m: distribution([r[m] for r in subset]) for m in METRICS}))
    flags = np.asarray([r['passed'] for r in convergence])
    ratios = np.asarray([r['tolerance_ratio'] for r in convergence])
    ism_summary = []
    for refs in ('block0_50', 'block1_50', 'block0_100', 'block1_100', 'pooled200'):
        for region in ('full',) + REGIONS:
            for site_type in ('all', 'top', 'random'):
                for view in ('all_contexts', 'active_contexts', 'observed_active_mean'):
                    rows = [r for r in ism_records if r['references'] == refs and r['region'] == region and r['site_type'] == site_type
                            and ((r['target'] == 'observed_active_mean') if view == 'observed_active_mean'
                                 else r['target'] != 'observed_active_mean' and (view != 'active_contexts' or r['active']))]
                    ism_summary.append(dict(references=refs, region=region, site_type=site_type, view=view,
                        n_enhancers=len({r['index'] for r in rows}),
                        metrics={key: distribution([r[key] for r in rows]) for key in
                                 ('pearson', 'spearman', 'mae', 'exact_abs_mean', 'sign_agreement_abs_exact_ge_005')}))
    result = dict(status='partial_pilot_review', signature=signature,
        completed_enhancers=len(receipts), planned_enhancers=progress['total'], degree_counts=dict(sorted(degrees.items())),
        contexts=list(CONTEXTS), verified_files=verified_files, hashes=hashes,
        config=config, progress=progress, reference_comparisons=summaries,
        paired_native_improvement={view: paired_improvement(records,
            lambda r, view=view: r['kind'] == 'actual' and r['region'] == 'native' and
            ((r['target'] == 'observed_active_mean') if view == 'observed_active_mean'
             else r['target'] != 'observed_active_mean' and (view != 'active_contexts' or r['active'])))
            for view in ('all_contexts', 'active_contexts', 'observed_active_mean')},
        convergence=dict(reference_evaluations=len(convergence), context_checks=int(flags.size),
            failed_context_checks=int((~flags).sum()), references_with_any_failure=int((~flags.all(1)).sum()),
            failed_enhancers=failures, steps=dict(sorted(Counter(r['steps'] for r in convergence).items())),
            per_context=[dict(context=context, failures=int((~flags[:, c]).sum()), tolerance_ratio=distribution(ratios[:, c])) for c, context in enumerate(CONTEXTS)],
            absolute_delta=distribution([v for r in convergence for v in r['absolute_delta']]),
            reference_hash_duplicates=len(reference_hashes) - len(set(reference_hashes))),
        integration_comparisons=integration_summary, ism_comparisons=ism_summary,
        builder_sha256=digest(Path(__file__)),
        limitations=['39-enhancer interleaved training prefix, not prevalence or an independent test set',
            '50 and 100 are compared using independent blocks at each count; counts are nested within blocks',
            'Pooled200 is not an independent200-vs200 reference stability test',
            'No universally optimal reference number or predeclared production acceptance threshold',
            'Integration anchor checks are first reference per stream; other refinements are failure-selected',
            'ISM covers up to16 selected positions per enhancer, not exhaustive mutations; positions selected using pooled200 maps',
            'IG differences along shuffled-reference paths are not guaranteed local mutation effects',
            'Bootstrap resamples enhancers; precision remains limited by five or four examples per degree',
            'No comparison to older32-node/50-reference Figure3 scores: pilot uses stricter integration settings'])
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in [('summary.json', result), ('reference_maps.json', records),
                           ('integration_maps.json', integration), ('convergence.json', convergence),
                           ('ism_metrics.json', ism_records), ('mutations.json', mutations)]:
        write_json(output / filename, data)
    write_json(output / 'complete.json', dict(status='complete', scope='analysis of completed pilot prefix, not pilot completion',
        files={p.name: digest(p) for p in output.glob('*.json') if p.name != 'complete.json'}))
    print(f'Analysis saved: {output}; {len(receipts)} enhancers, {verified_files} verified raw files', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.output.resolve(), args.package.resolve())
