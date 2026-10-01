"""Add a Grh illustration to the preserved 50-reference figure, CPU only.

Reuse archived native TF-MoDISco PWMs, FIMO sites and attribution tensors.
Only the middle-right example changes; no scans, discovery or inference.
"""
import argparse
import copy
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .attribution_example_selection import choose_configuration, select_clear_example
from .dual_motif_pipeline import informative_core
from .paper_figure3 import SUPPORT, check_text_bounds, digest
from .paper_figure3_examples import (BASES, COHORT, ORIGINAL, EXTRA_HEIGHT,
    ExampleFigure, project_actual, validate_site, zoom_interval)
from .paper_figure3_nature import CONTEXTS
from .tomtom_atlas import parse_matches, read_meme

PREVIOUS = 'output/pdf/figure_3bcde_native_enhancer_attribution_examples_diverse'
OUTPUT = 'output/pdf/figure_3bcde_native_enhancer_attribution_examples_diverse_grh50'
SIMPLE = 'results/classifier_modisco_simple_20260918/cecar_results'
PREVIOUS_SHA = 'b22895e3306cd98f25a1c44ed69a3440a9d4d2ac5f7029b8ade4c13038482397'


def unique_candidates(candidates):
    """Nested discovery groups must not give one enhancer multiple votes."""
    selected = {}
    for row in candidates:
        old = selected.get(row['id'])
        if old is not None:
            if row['sequence'] != old['sequence'] or row['degree'] != old['degree']:
                raise ValueError('Inconsistent enhancer across discovery groups')
            np.testing.assert_array_equal(row['actual_ig'], old['actual_ig'])
        key = lambda r: (-r['contrast']['score'], r['anchor']['motif_id'], r['anchor']['start'])
        if old is None or key(row) < key(old):
            selected[row['id']] = row
    return [selected[key] for key in sorted(selected)]


def prepare_grh(project):
    root, simple = project / ORIGINAL, project / SIMPLE
    hashes = {}

    def verified(path, expected=None):
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError(f'Changed source: {path}')
        hashes[str(path.relative_to(project))] = value
        return value

    completion = json.loads((simple / 'complete.json').read_text())
    if completion['status'] != 'complete':
        raise ValueError('Incomplete archived motif report')
    verified(simple / 'report_audit.json', completion['files']['report_audit.json'])
    audit = json.loads((simple / 'report_audit.json').read_text())
    matches = json.loads((simple / 'annotation/jaspar/matches.json').read_text())
    verified(simple / 'annotation/jaspar/matches.json')
    verified(simple / 'annotation/jaspar/tomtom.tsv', matches['raw_output_sha256'])
    verified(simple / 'annotation/quality_passing_queries.meme', matches['query_sha256'])
    verified(simple / 'references/jaspar2026_insects.meme', matches['database']['sha256'])
    references = read_meme(simple / 'references/jaspar2026_insects.meme')
    raw_matches, _ = parse_matches((simple / 'annotation/jaspar/tomtom.tsv').read_text(),
                                  set(matches['best']), references)
    scan_queries = read_meme(simple / 'annotation/quality_passing_queries.meme')
    with np.load(project / COHORT, allow_pickle=False) as f:
        cohort = dict(f)
    verified(project / COHORT, 'f1fa683f8a9224a702191a5d6776b3b0dd478546d873ad58f2604f4dc4fa98b1')
    with np.load(root / 'intervals.npz', allow_pickle=False) as f:
        intervals = dict(f)
    interval_sha = verified(root / 'intervals.npz')
    attribution_sha = verified(root / 'attribution_complete.json')
    attribution = json.loads((root / 'attribution_complete.json').read_text())
    if (attribution['status'] != 'complete' or attribution['references'] != 50
            or attribution['quality_failures'] != 0):
        raise ValueError('Expected quality-passing 50-reference attributions')
    np.testing.assert_array_equal(intervals['ids'], cohort['ids'])
    breadth = cohort['labels'].sum(1)
    candidates, motifs = [], []
    for group in audit['groups']:
        rows = [r for r in group['rows'] if r['sign'] == 'positive'
                and matches['best'][r['id'].replace('/', '__')]['reference']['name'] == 'grh'
                and matches['best'][r['id'].replace('/', '__')]['q'] < .05]
        if not rows:
            continue
        name = group['name']
        source = root / 'groups' / name
        receipt_path = root / 'refinement_v3/groups' / name / 'complete.json'
        verified(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        if receipt['status'] != 'complete':
            raise ValueError('Incomplete archived source verification')
        for file in ('motifs.h5', 'examples.npz', 'discovery_inputs.npz', 'selection.json'):
            verified(source / file, receipt['input_sha256'][file])
        for file in ('motifs.h5', 'selection.json'):
            verified(source / file, audit['inputs'][f'experiments/classifier_modisco_original_20260917/groups/{name}/{file}'])
        selection = json.loads((source / 'selection.json').read_text())
        if (not selection['train_only'] or selection['quality_excluded'] != 0
                or selection['intervals_sha256'] != interval_sha
                or selection['attribution_sha256'] != attribution_sha):
            raise ValueError('Mismatched attribution protocol')
        with np.load(source / 'examples.npz', allow_pickle=False) as f:
            indices, lengths = f['indices'], f['lengths']
            for key, expected in dict(ids=cohort['ids'][indices], chrom=cohort['chrom'][indices],
                    start=intervals['start'][indices], end=intervals['end'][indices]).items():
                np.testing.assert_array_equal(f[key], expected)
        if len(np.unique(indices)) != len(indices) or not np.all(cohort['split'][indices] == 'train'):
            raise ValueError('Repeated or nontraining examples')
        with np.load(source / 'discovery_inputs.npz', allow_pickle=False) as f:
            sequence, hypothetical = f['sequence'], f['hypothetical']
            np.testing.assert_array_equal(f['lengths'], lengths)
        lookup = {}
        for row in rows:
            ident = row['id'].replace('/', '__')
            match = matches['best'][ident]
            if any(match[k] != v for k, v in raw_matches[ident].items()):
                raise ValueError('Tomtom summary differs from raw result')
            if match['reference'] != references[match['target_id']]:
                raise ValueError('Changed reference PWM')
            # The checksum-verified report stores the full native PWM and pins
            # the original HDF5 hash, verified above against the archived file.
            full = np.asarray(row['full_pwm'])
            qc = informative_core(full)
            if not qc['passed']:
                raise ValueError('Grh candidate fails revised common information filter')
            # Keep the original archived scan/query core. Never attach its q/p
            # values to the differently trimmed core from the filter audit.
            np.testing.assert_allclose(scan_queries[ident]['pwm'], row['trimmed_pwm'], atol=1e-6)
            lookup[ident] = dict(row=row, match=match)
            motifs.append(dict(id=row['id'], archived_quality=row['quality'], revised_quality=qc,
                consensus=''.join(BASES[i] for i in np.asarray(row['trimmed_pwm']).argmax(1)), match=match))
        scan = simple / 'scans' / name
        verified(scan / 'complete.json')
        receipt = json.loads((scan / 'complete.json').read_text())
        verified(scan / 'hits.tsv.gz', receipt['hits_sha256'])
        verified(scan / 'cores.meme', receipt['motifs_sha256'])
        scan_pwms = read_meme(scan / 'cores.meme')
        for ident, data in lookup.items():
            np.testing.assert_allclose(scan_pwms[ident]['pwm'], data['row']['trimmed_pwm'], atol=1e-6)
        positions = {int(index): pos for pos, index in enumerate(indices)}
        sites = {}
        with gzip.open(scan / 'hits.tsv.gz', 'rt') as f:
            for hit in csv.DictReader((line for line in f if not line.startswith('#')), delimiter='\t'):
                index = int(hit['sequence_name'][1:])
                if hit['motif_id'] not in lookup or index not in positions:
                    continue
                if float(hit['p-value']) > 1e-4:
                    raise ValueError('Unexpected FIMO threshold')
                sites.setdefault(index, []).append(hit)
        for index, hits in sites.items():
            if breadth[index] < 2:
                continue
            pos, length = positions[index], int(intervals['length'][index])
            if length != lengths[pos]:
                raise ValueError('Native length mismatch')
            actual = project_actual(sequence[pos], hypothetical[pos], length)
            codes = sequence[pos, :length].argmax(1)
            offset = int(intervals['offset'][index])
            np.testing.assert_array_equal(codes, cohort['sequence'][index, offset:offset + length])
            bases = ''.join(BASES[c] for c in codes)
            for hit in hits:
                start, end = validate_site(bases, hit)
                row, match = lookup[hit['motif_id']]['row'], lookup[hit['motif_id']]['match']
                site = dict(motif_id=row['id'], start=start, end=end, strand=hit['strand'],
                    site_p=float(hit['p-value']), mean_ig=float(actual[start:end].mean()),
                    best_match='grh', tomtom_q=match['q'], display_label='Grh-like')
                config = choose_configuration(actual, [site], row['pattern'].split('/')[-1], zoom_interval)
                if config is None:
                    continue
                candidates.append(dict(index=index, id=str(cohort['ids'][index]), group=name,
                    degree=int(breadth[index]), active_contexts=[c for c, active in zip(CONTEXTS, cohort['labels'][index]) if active],
                    chrom=str(cohort['chrom'][index]), start=int(intervals['start'][index]),
                    end=int(intervals['end'][index]), length=length, sequence=bases,
                    actual_ig=actual.tolist(), sites=[site], **config))
    pool = unique_candidates(candidates)
    choice = select_clear_example(pool)
    choice['example_kind'] = 'Grh'
    return choice, dict(inputs=hashes, motifs=motifs, eligible_enhancers=len(pool),
        eligible_by_degree={str(d): sum(c['degree'] == d for c in pool) for d in range(1, 9)},
        filter_note='Grh PWMs pass the revised common 4-of-5 information filter. Archived 10/11-bp cores and their exact saved FIMO/Tomtom results are reused without retrimming or rescanning.',
        selection='Same contrast gates as previous examples; one best site/configuration per enhancer across both nested discovery groups, then closest to the 90th percentile. Training only. Illustrative, not population representative.')


def main(project):
    previous_pdf = project / (PREVIOUS + '.pdf')
    if digest(previous_pdf) != PREVIOUS_SHA:
        raise ValueError('Previous figure changed')
    previous = json.loads((project / (PREVIOUS + '.source.json')).read_text())
    if previous['pdf_sha256'] != PREVIOUS_SHA:
        raise ValueError('Previous source no longer matches PDF')
    choice, audit = prepare_grh(project)
    print(json.dumps(dict(selected=choice['id'], degree=choice['degree'], contexts=choice['active_contexts'],
        motif=choice['anchor'], contrast=choice['contrast'], eligible=audit['eligible_by_degree'])), flush=True)
    examples = copy.deepcopy(previous['panel_c'])
    replaced = examples['examples'][3]
    examples['examples'][3] = choice
    if len({e['id'] for e in examples['examples']}) != 6:
        raise ValueError('Repeated example')
    examples['selection']['pleiotropic'] = 'Preserved degree-2 Trl and degree-8 Trl/cg examples; middle-right replaced by an archived Grh-like native discovery under the same contrast gates.'
    examples['inputs'].update(audit['inputs'])
    examples['grh_audit'] = audit
    path = project / (OUTPUT + '.pdf')
    if path.exists():
        raise FileExistsError('Preserve existing output; use a new version')
    figure_data = json.loads((project / SUPPORT / 'report_audit.json').read_text())
    figure = ExampleFigure(figure_data, path, project / SUPPORT / 'fonts')
    figure.initialize(revised_layout=True, motif_count=len(previous['panel_b']))
    figure.height += EXTRA_HEIGHT
    figure.c.setPageSize((figure.width, figure.height))
    figure.c.setTitle('Figure 3b-e | 50-reference native enhancer examples including Grh')
    figure.c.setSubject('Archived 50-reference IG; only middle-right example changed; no new inference')
    figure.panel_b(previous['panel_b']); figure.examples_panel(examples)
    figure.panel_c(previous['panel_d'], panel_label='d')
    figure.panel_d_odds_ratios(previous['panel_e'], axis='linear', panel_label='e')
    check_text_bounds(figure.text_bounds)
    figure.c.showPage(); figure.c.save()
    source = dict(previous, panel_c=examples, pdf_sha256=digest(path),
        previous_pdf_sha256=PREVIOUS_SHA, previous_source_sha256=digest(project / (PREVIOUS + '.source.json')),
        builder_sha256=digest(Path(__file__)), text_bounds=figure.text_bounds, visual_qa='pending',
        replaced_example=replaced, change='Only panel C middle-right example replaced with Grh; B/D/E data unchanged',
        inputs={**previous['inputs'], **audit['inputs']})
    path.with_suffix('.source.json').write_text(json.dumps(source, indent=2, allow_nan=False) + '\n')
    if digest(previous_pdf) != PREVIOUS_SHA:
        raise ValueError('Previous figure was modified')
    print(f'Created {path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    main(parser.parse_args().project.resolve())
