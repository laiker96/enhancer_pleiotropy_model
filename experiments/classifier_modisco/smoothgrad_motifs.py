"""Native-only SmoothGrad motif discovery and context profiles; CPU only."""
import argparse
from functools import partial
import html
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .common import digest, event, require_allocation, write_json
from .motif_recovery import occurrence, receipt, save

NAME = 'classifier_smoothgrad_motifs_20260923'
CONTEXTS = ('ab', 'e13', 'e5', 'ead', 'hid', 'lb', 'o', 'wid')


def native_batch(values, width):
    """Put centered sensitivities in MoDISco's four-channel input slot, not IG units."""
    n = len(values['ids'])
    centered = values['centered_sensitivity']
    observed = values['observed_sensitivity']
    if centered.shape != (n, 8, 4, 2048) or observed.shape != (n, 8, 2048):
        raise ValueError('Expected eight full-input context maps')
    if not np.isfinite(centered).all() or not np.isfinite(observed).all():
        raise ValueError('Nonfinite sensitivities')
    sequence = np.zeros((n, width, 4), np.float32)
    summed = np.zeros_like(sequence)
    context = np.zeros((n, 8, width), np.float32)
    for i, (offset, length) in enumerate(zip(values['native_offset'], values['native_length'])):
        offset, length = int(offset), int(length)
        if not 30 <= length <= width or not 0 <= offset <= 2048-length:
            raise ValueError('Invalid native interval')
        codes = values['sequence'][i, offset:offset+length]
        if not np.isin(codes, range(4)).all():
            raise ValueError('Invalid DNA')
        sequence[i, :length] = np.eye(4, dtype=np.float32)[codes]
        cropped = centered[i, :, :, offset:offset+length]
        projected = np.take_along_axis(cropped, codes[None, None, :].astype(int), axis=1)[:, 0]
        np.testing.assert_allclose(projected, observed[i, :, offset:offset+length], atol=3e-6, rtol=2e-3)
        summed[i, :length] = cropped.sum(0).T
        context[i, :, :length] = projected
    np.testing.assert_allclose((sequence*summed).sum(2), context.sum(1), atol=3e-6, rtol=2e-3)
    return dict(sequence=sequence, summed_sensitivity=summed, observed_context=context)


def context_profile(observed, labels, lengths, occurrences):
    """Union overlapping motif cores; average bases, then equal-weight enhancers."""
    masks = {}
    for item in occurrences:
        i, start, end = item['example'], item['start'], item['end']
        if not 0 <= i < len(labels) or not 0 <= start < end <= lengths[i]:
            raise ValueError('Occurrence outside native enhancer')
        masks.setdefault(i, np.zeros(int(lengths[i]), bool))[start:end] = True
    members = np.asarray(sorted(masks), dtype=int)
    scores = np.asarray([observed[i, :, :int(lengths[i])][:, masks[i]].mean(1)
                         for i in members], dtype=float).reshape(-1, 8)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite context profile')
    degree = labels.sum(1)
    def summarize(mask, population):
        chosen = scores[mask]
        return dict(hits=len(chosen), n=int(population.sum()),
            support_fraction=float(len(chosen)/population.sum()) if population.any() else None,
            mean_signed_sensitivity_per_bp=chosen.mean(0).tolist() if len(chosen) else None,
            active_context_fraction=labels[members[mask]].mean(0).tolist() if len(chosen) else None)
    return dict(member_indices=members.tolist(), enhancer_mean_signed_sensitivity_per_bp=scores.tolist(),
        overall=summarize(np.ones(len(members), bool), np.ones(len(labels), bool)),
        by_degree=[dict(degree=k, **summarize(degree[members] == k, degree == k)) for k in range(1, 9)],
        by_group=[dict(group=name, **summarize((degree[members] >= lo) & (degree[members] <= hi),
                         (degree >= lo) & (degree <= hi)))
                  for name, lo, hi in [('1', 1, 1), ('2-5', 2, 5), ('6-8', 6, 8)]])


def report_html(rows, annotations, audit, *, examples=1000, description=None,
                population_note=None):
    from .report_breadth import sequence_logo
    from .tomtom_atlas import query_id
    values = [abs(v) for r in rows for g in r['context_profile']['by_degree']
              if g['mean_signed_sensitivity_per_bp'] is not None for v in g['mean_signed_sensitivity_per_bp']]
    scale = max(values, default=1.) or 1.
    cards = ['<!doctype html><html><head><meta charset="utf-8"><title>SmoothGrad motifs by context</title>',
        '<style>body{font:15px Arial,sans-serif;max-width:1120px;margin:30px auto;padding:0 16px;color:#222}'
        'article{border-top:1px solid #bbb;padding:20px 0}.logo{max-width:750px;width:100%}'
        'table{border-collapse:collapse;width:100%;margin:12px 0}td,th{padding:7px;text-align:center;border-bottom:1px solid #ddd}'
        '.support{display:flex;align-items:center;gap:12px;margin:6px 0}.bar{width:55%;height:12px;background:#eee}'
        '.bar i{display:block;height:100%}.note{color:#555}code{font-size:13px}</style></head><body>',
        '<h1>SmoothGrad motifs and context sensitivity</h1><p>'
        +html.escape(description or '1,000 training enhancers; 125 per exact degree. One pooled TF-MoDISco fit on the sum of eight '
        'calibrated-probability sensitivity maps. Native enhancer intervals only; no extra clustering.')+'</p>'
        '<p class="note">Support means distinct enhancers assigned a discovery seqlet, not sequence-scan prevalence. '
        'Context values are signed local sensitivity per base, averaged equally over supporting enhancers; '
        'they are not mutation effects or IG contributions. Positive is blue, negative red; '
        'heatmap saturation uses one common scale across all motifs. Missing means no supporting enhancer.</p>']
    for sign in ('positive', 'negative'):
        selected = sorted((r for r in rows if r['sign'] == sign), key=lambda r:r['rank'])
        cards.append(f'<h2>{sign.capitalize()} motifs</h2>')
        if not selected:
            cards.append('<p>No motifs passed the reporting filter. This does not establish biological absence.</p>')
        for row in selected:
            key = query_id(row)
            cards.append('<article><h3>'+html.escape(f"#{row['rank']} · {row['consensus']} · {row['id']}")+'</h3>')
            cards.append(sequence_logo(np.asarray(row['trimmed_pwm']), row['id']))
            cards.append(f"<p>{row['supporting_discovery_enhancers']}/{examples:,} supporting enhancers; {row['seqlets']} seqlets.</p>")
            cards.append('<table><tr><th>Database</th><th>Best match</th><th>Tomtom q</th></tr>')
            for db, matches in annotations.items():
                m = matches.get(key)
                if m is None:
                    cards.append(f'<tr><td>{html.escape(db)}</td><td>No reported match</td><td>—</td></tr>')
                    continue
                label = html.escape(m['reference_name']+' ('+m['target_id']+')')
                cards.append(f"<tr><td>{html.escape(db)}</td><td>{label}</td><td>{m['q']:.3g}"
                             +(' (not significant)' if m['q'] >= .05 else '')+'</td></tr>')
            cards.append('</table><h4>Discovery support by degree of pleiotropy</h4>')
            for group in row['context_profile']['by_degree']:
                fraction = group['support_fraction'] or 0.
                color = '#2466c4' if sign == 'positive' else '#ce3646'
                cards.append(f'<div class="support"><span>{group["degree"]}</span><span class="bar">'
                             f'<i style="width:{100*fraction:.4f}%;background:{color}"></i></span>'
                             f'<span>{group["hits"]}/{group["n"]} ({100*fraction:.1f}%)</span></div>')
            cards.append('<h4>Mean signed sensitivity per base at motif cores</h4><table><tr><th>Degree (support n)</th>'
                         +''.join('<th>'+c+'</th>' for c in CONTEXTS)+'</tr>')
            for group in row['context_profile']['by_degree']:
                cards.append(f'<tr><td>{group["degree"]} ({group["hits"]})</td>')
                for value in group['mean_signed_sensitivity_per_bp'] or [None]*8:
                    if value is None:
                        cards.append('<td>—</td>')
                    else:
                        rgb = '36,102,196' if value >= 0 else '206,54,70'
                        alpha = min(.6, .6*abs(value)/scale)
                        cards.append(f'<td style="background:rgba({rgb},{alpha:.4f})">{value:+.4g}</td>')
                cards.append('</tr>')
            overall = row['context_profile']['overall']['mean_signed_sensitivity_per_bp']
            cards.append('<tr><th>All supporters</th>'+''.join(f'<td>{v:+.4g}</td>' for v in overall)+'</tr></table></article>')
    cards.append('<details><summary>Discovery audit and interpretation</summary><pre>'+html.escape(json.dumps(audit, indent=2))
        +'</pre><p>Raw patterns and all filter exclusions are retained. Database matches are sequence similarities, '
        'not proof of TF binding. '+html.escape(population_note or
        'This balanced training subset is not a population prevalence estimate.')+' '
        'Pooled discovery may miss context-restricted motifs that cancel in the sum; this is not eight separate '
        'context-specific motif searches. No perturbation validation was run.</p></details></body></html>')
    return '\n'.join(cards)


def run(project, root):
    require_allocation('cpu')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU analysis must hide CUDA')
    import h5py
    from .original_intervals import discover_original
    from .dual_motif_pipeline import informative_core, core_rule
    from .simple_report import filter_native, rank_rows
    from .tomtom_atlas import DATABASES, meme_queries, parse_matches, read_meme
    config = json.loads((root/'config.json').read_text())
    output = root/'output'; output.mkdir(exist_ok=False)
    with np.load(root/'inputs.npz', allow_pickle=False) as saved:
        data = dict(saved)
    expected = config.get('examples', 1000)
    if (len(data['ids']) != expected or len(np.unique(data['ids'])) != expected
            or data['labels'].shape != (expected, 8) or not np.isin(data['labels'], [0, 1]).all()
            or not data['labels'].any(1).all()
            or not np.array_equal(np.bincount(data['labels'].sum(1).astype(int), minlength=9)[1:],
                                  config.get('degree_counts', [125]*8))):
        raise ValueError('Wrong discovery cohort')
    group_name = config.get('group_name', 'smoothgrad_sum8')
    versions = {name: importlib.metadata.version(name) for name in ('modisco', 'numpy', 'scipy', 'h5py', 'numba')}
    event('smoothgrad_motif_started', job=os.environ['SLURM_JOB_ID'], examples=len(data['ids']), versions=versions)
    audit = discover_original(data['sequence'], data['summed_sensitivity'], data['lengths'],
        config['discovery_parameters'], output/'motifs.h5', 20000)
    counts = {s:audit[s]['patterns'] for s in ('positive', 'negative')}
    rows, exclusions = filter_native(output/'motifs.h5', group_name, counts,
        core_filter=partial(informative_core, flank_threshold=.2))
    group = dict(name=group_name, rows=rows, rank_by='native_support', discovery_elements=expected)
    rank_rows(group)
    with h5py.File(output/'motifs.h5', 'r') as handle:
        for row in rows:
            seqlets = handle[row['pattern']]['seqlets']
            occurrences = []
            for i, start, end, reverse in zip(seqlets['example_idx'][:], seqlets['start'][:],
                                             seqlets['end'][:], seqlets['is_revcomp'][:]):
                if int(end-start) != len(row['full_pwm']):
                    raise ValueError('Seqlet width disagrees with aligned pattern')
                left, right = occurrence(int(start), int(end), bool(reverse), row['quality']['start'],
                    row['quality']['end'], 0, int(data['lengths'][i]))
                occurrences.append(dict(example=int(i), start=left, end=right, rc=bool(reverse)))
            row['occurrences'] = occurrences
            row['consensus'] = ''.join('ACGT'[np.argmax(p)] for p in row['trimmed_pwm'])
            row['context_profile'] = context_profile(data['observed_context'], data['labels'], data['lengths'], occurrences)
            row['member_ids'] = data['ids'][row['context_profile']['member_indices']].tolist()
            if len(row['member_ids']) != row['supporting_discovery_enhancers']:
                raise ValueError('Support count mismatch')
    cores = dict(groups=[group], exclusions=exclusions, rule=core_rule(.2), reclustered=False,
                 audit=audit, contexts=list(CONTEXTS), versions=versions,
                 interpretation='SmoothGrad sensitivity, not baseline-relative contribution or mutation effect.')
    write_json(output/'cores.json', cores)
    annotations = {}
    binary = project/config['tomtom']
    if digest(binary) != config['tomtom_sha256']:
        raise ValueError('Tomtom binary changed')
    if subprocess.check_output([str(binary), '-version'], text=True).strip() != '5.5.9':
        raise ValueError('Unexpected Tomtom version')
    if rows:
        query = output/'motifs.meme'; query.write_text(meme_queries(cores))
        for db, meta in DATABASES.items():
            path = project/config['database_root']/meta['file']
            if digest(path) != config['database_sha256'][db]:
                raise ValueError('Reference database changed: '+db)
            command = [str(binary), '-text', '-dist', 'pearson', '-min-overlap', '5',
                       '-motif-pseudo', '.1', '-thresh', '1', '-verbosity', '2', str(query), str(path)]
            done = subprocess.run(command, capture_output=True, text=True, check=True)
            (output/f'tomtom_{db}.tsv').write_text(done.stdout)
            (output/f'tomtom_{db}.stderr.log').write_text(done.stderr)
            targets = read_meme(path)
            matches, _ = parse_matches(done.stdout, read_meme(query), targets)
            annotations[db] = {key:dict(value, reference_name=targets[value['target_id']]['name']) for key,value in matches.items()}
    write_json(output/'matches.json', annotations)
    (output/'report.html').write_text(report_html(rows, annotations, audit, examples=expected,
        description=config.get('description'), population_note=config.get('population_note')))
    files = [str(p.relative_to(output)) for p in output.iterdir() if p.is_file()]
    write_json(output/'complete.json', receipt(output, files, examples=expected, raw=counts,
        retained={sign:sum(r['sign'] == sign for r in rows) for sign in ('positive', 'negative')},
        manifest_sha256=digest(root/'MANIFEST.sha256'), job=os.environ['SLURM_JOB_ID']))
    event('smoothgrad_motif_complete', raw=counts, retained=len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.root)
