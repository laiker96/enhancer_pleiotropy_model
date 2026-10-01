"""Insert saved native-enhancer IG examples into Figure 3; CPU rendering only.

No attribution, motif discovery, scans, or model inference are rerun. Example
selection is illustrative, not a test of enrichment or model generalization.
"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from .paper_figure3 import SUPPORT, digest, load_inputs, check_text_bounds, select_patterns
from .paper_figure3_nature import NatureFigure, CONTEXTS
from .attribution_example_selection import (DIVERSE_STRATA, choose_configuration,
    select_clear_example, selection_protocol)

ORIGINAL = 'results/classifier_modisco_original_20260917/cecar_results'
COHORT = 'results/classifier_modisco_20260916/package/cohort.npz'
PREVIOUS = 'output/pdf/figure_3bcd_flyfactorsurvey_exact_gaf_odds_ratio_wt_linear_control_expanded'
PREFIX = 'figure_3bcde_native_enhancer_attribution_examples'
EXTRA_HEIGHT = 420
BASES = 'ACGT'
# Display order: two columns, three rows. Exact degree, optional sole context.
STRATA = [('exact_1', 1, 'ab'), ('ge_2', 2, None),
          ('exact_1', 1, 'e5'), ('ge_4', 4, None),
          ('exact_1', 1, 'wid'), ('ge_8', 8, None)]


def project_actual(sequence, hypothetical, length):
    """Observed-base projection, in original logit units; exclude storage padding."""
    if sequence.shape != hypothetical.shape or sequence.shape[1] != 4:
        raise ValueError('Expected matching length-by-four tensors')
    if not 0 < length <= len(sequence):
        raise ValueError('Invalid native length')
    native = sequence[:length]
    if not np.all((native == 0) | (native == 1)) or not np.all(native.sum(1) == 1):
        raise ValueError('Native sequence must be one-hot A/C/G/T')
    if np.any(sequence[length:] != 0) or not np.isfinite(hypothetical).all():
        raise ValueError('Invalid padding or nonfinite attribution')
    return (native * hypothetical[:length]).sum(1).astype(float)


def validate_site(sequence, hit):
    """FIMO coordinates are 1-based inclusive; internal coordinates are half-open."""
    start, end = int(hit['start']) - 1, int(hit['stop'])
    if not 0 <= start < end <= len(sequence) or hit['strand'] not in ('+', '-'):
        raise ValueError('Invalid FIMO coordinates or strand')
    matched = sequence[start:end]
    if hit['strand'] == '-':
        matched = matched.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
    if matched != hit['matched_sequence'].upper():
        raise ValueError('FIMO sequence differs from the native attribution input')
    return start, end


def percentile_example(candidates):
    """Fixed upper-quartile illustration within the eligible motif-site pool."""
    if not candidates:
        raise ValueError('No eligible motif-bearing example in a requested stratum')
    target = float(np.quantile([c['anchor']['mean_ig'] for c in candidates], .75))
    chosen = min(candidates, key=lambda c: (abs(c['anchor']['mean_ig'] - target), c['id']))
    return dict(chosen, eligible_candidates=len(candidates), selection_target_ig=target)


def zoom_interval(start, end, length, width=60):
    width = min(width, length)
    left = min(max(0, (start + end - width) // 2), length - width)
    return left, left + width


def anchor_sites(annotations, degree, gaf_examples=False):
    """Optional user-requested GAF focus for the three pleiotropic examples."""
    if gaf_examples and degree > 1:
        return [s for s in annotations if s['best_match'] == 'Trl' and s['tomtom_q'] < .05]
    return annotations


def display_tracks(example, motif_oriented=False):
    sites = example.get('plot_sites', [example['anchor']])
    orientation_site = next((s for s in sites if s['best_match'] == 'Trl'), example['anchor'])
    reverse = motif_oriented and orientation_site['strand'] == '-'
    sequence = example['sequence']
    values = np.asarray(example['actual_ig'])
    left, right = example['zoom']
    if reverse:
        sequence = sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
        values = values[::-1]
        left, right = len(sequence) - right, len(sequence) - left
        sites = [dict(s, start=len(sequence) - s['end'], end=len(sequence) - s['start']) for s in sites]
    return dict(sequence=sequence, values=values, zoom=(left, right),
                sites=sites, reverse=reverse)


def prepare_examples(project, rows, gaf_examples=False, diverse_examples=False):
    root, support = project / ORIGINAL, project / SUPPORT
    hashes = {}

    def verified(path, expected=None):
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Input checksum mismatch: {path}')
        hashes[str(path.relative_to(project))] = actual
        return actual

    with np.load(project / COHORT, allow_pickle=False) as saved:
        cohort = {key: saved[key] for key in saved.files}
    with np.load(root / 'intervals.npz', allow_pickle=False) as saved:
        intervals = {key: saved[key] for key in saved.files}
    verified(project / COHORT, 'f1fa683f8a9224a702191a5d6776b3b0dd478546d873ad58f2604f4dc4fa98b1')
    interval_sha = verified(root / 'intervals.npz')
    attribution_sha = verified(root / 'attribution_complete.json')
    completion = json.loads((root / 'attribution_complete.json').read_text())
    if completion['status'] != 'complete' or completion['references'] != 50 or completion['quality_failures'] != 0:
        raise ValueError('Expected completed quality-passing 50-reference attributions')
    np.testing.assert_array_equal(intervals['ids'], cohort['ids'])
    breadth = cohort['labels'].sum(1)
    output = {}
    strata = DIVERSE_STRATA if diverse_examples else STRATA
    used_specific_contexts = set()
    for group in dict.fromkeys(s[0] for s in strata):
        source = root / 'groups' / group
        receipt_path = root / 'refinement_v3/groups' / group / 'complete.json'
        receipt = json.loads(receipt_path.read_text())
        verified(receipt_path)
        if receipt['status'] != 'complete':
            raise ValueError('Incomplete source verification receipt')
        for name in ('examples.npz', 'discovery_inputs.npz', 'selection.json'):
            verified(source / name, receipt['input_sha256'][name])
        selection = json.loads((source / 'selection.json').read_text())
        if (not selection['train_only'] or selection['quality_excluded'] != 0
                or selection['intervals_sha256'] != interval_sha
                or selection['attribution_sha256'] != attribution_sha):
            raise ValueError('Source group uses different attribution or intervals')
        with np.load(source / 'examples.npz', allow_pickle=False) as saved:
            indices, lengths = saved['indices'], saved['lengths']
            for key, expected in dict(ids=cohort['ids'][indices], chrom=cohort['chrom'][indices],
                    start=intervals['start'][indices], end=intervals['end'][indices]).items():
                np.testing.assert_array_equal(saved[key], expected)
        if len(np.unique(indices)) != len(indices) or not np.all(cohort['split'][indices] == 'train'):
            raise ValueError('Repeated indices or non-training discovery examples')
        with np.load(source / 'discovery_inputs.npz', allow_pickle=False) as saved:
            sequence, hypothetical = saved['sequence'], saved['hypothetical']
            np.testing.assert_array_equal(saved['lengths'], lengths)
        motifs = {row['id'].replace('/', '__'): row for row in rows
                  if row['group'] == group and row['sign'] == 'positive'}
        index_row = {int(index): pos for pos, index in enumerate(indices)}
        scan = support / 'scans' / group
        scan_complete = json.loads((scan / 'complete.json').read_text())
        verified(scan / 'complete.json')
        verified(scan / 'hits.tsv.gz', scan_complete['hits_sha256'])
        sites = {}
        with gzip.open(scan / 'hits.tsv.gz', 'rt') as handle:
            for hit in csv.DictReader((line for line in handle if not line.startswith('#')), delimiter='\t'):
                if hit['motif_id'] not in motifs:
                    continue
                index = int(hit['sequence_name'][1:])
                if index not in index_row:
                    continue
                if float(hit['p-value']) > 1e-4:
                    raise ValueError('Site exceeds the saved FIMO threshold')
                sites.setdefault(index, []).append(hit)
        for stratum in (s for s in strata if s[0] == group):
            _, degree, context = stratum
            candidates = []
            for index, hits in sites.items():
                if breadth[index] != degree or (not diverse_examples and context is not None and not cohort['labels'][index, CONTEXTS.index(context)]):
                    continue
                pos = index_row[index]
                length = int(lengths[pos])
                if length != intervals['length'][index]:
                    raise ValueError('Native length mismatch')
                actual = project_actual(sequence[pos], hypothetical[pos], length)
                codes = sequence[pos, :length].argmax(1)
                offset = int(intervals['offset'][index])
                np.testing.assert_array_equal(codes, cohort['sequence'][index, offset:offset + length])
                bases = ''.join(BASES[code] for code in codes)
                annotations = []
                for hit in hits:
                    start, end = validate_site(bases, hit)
                    row = motifs[hit['motif_id']]
                    match = row['match']
                    annotations.append(dict(motif_id=row['id'], start=start, end=end,
                        strand=hit['strand'], site_p=float(hit['p-value']),
                        mean_ig=float(actual[start:end].mean()),
                        best_match=match['reference']['name'], tomtom_q=match['q']))
                annotations.sort(key=lambda s: (-s['mean_ig'], s['site_p'], s['start'], s['motif_id']))
                eligible_sites = anchor_sites(annotations, degree, gaf_examples)
                if not eligible_sites or eligible_sites[0]['mean_ig'] <= 0:
                    continue
                configuration = choose_configuration(actual, annotations, context, zoom_interval) if diverse_examples else {}
                if configuration is None:
                    continue
                candidates.append(dict(index=index, id=str(cohort['ids'][index]), group=group,
                    degree=degree, active_contexts=[c for c, active in zip(CONTEXTS, cohort['labels'][index]) if active],
                    chrom=str(cohort['chrom'][index]), start=int(intervals['start'][index]),
                    end=int(intervals['end'][index]), length=length, sequence=bases,
                    actual_ig=actual.tolist(), sites=annotations,
                    **(configuration if diverse_examples else dict(anchor=eligible_sites[0]))))
            if diverse_examples:
                if degree == 1:
                    different_context = [c for c in candidates if c['active_contexts'][0] not in used_specific_contexts]
                    if not different_context:
                        raise ValueError('No distinct-context example passes the contrast criteria')
                    candidates = different_context
                chosen = select_clear_example(candidates)
                if degree == 1:
                    used_specific_contexts.add(chosen['active_contexts'][0])
            else:
                chosen = percentile_example(candidates)
                chosen['zoom'] = list(zoom_interval(chosen['anchor']['start'], chosen['anchor']['end'], chosen['length']))
            output[stratum] = chosen
            print(f"Selected {chosen['id']}: degree={degree}, contexts={chosen['active_contexts']}, "
                  f"motif={chosen['anchor']['best_match']}, candidates={len(candidates)}", flush=True)
        del sequence, hypothetical
    result = dict(examples=[output[s] for s in strata], inputs=hashes,
        references=50, method='Integrated Gradients', input_bp=2048, display_interval='original enhancer',
        target='Mean forward/RC logit over fixed observed-active contexts', split='train',
        selection='Within each stratum: enhancer closest to the 75th percentile of its strongest positive mean-IG eligible FIMO site; ties by ID. Sites restricted to positive panel-B motifs, p<=1e-4. ' +
                  ('At the user\'s request, degree-2/4/8 anchor sites are additionally restricted to motifs with best JASPAR match Trl and Tomtom q<0.05; context-specific selection is unchanged.' if gaf_examples else 'No selection by TF identity.'),
        gaf_examples=gaf_examples,
        annotation='FIMO sequence matches to discovered PWMs, not proof of TF occupancy or original seqlet assignment',
        coordinate_system='Native genomic bounds and site/zoom offsets: zero-based, half-open',
        scaling='Signed observed-base IG in logit units; common y scale across the six overviews and common y scale across the six zooms; no smoothing, normalization, or clipping')
    if diverse_examples:
        result['selection'] = selection_protocol()
        result['diverse_examples'] = True
    return result


class ExampleFigure(NatureFigure):
    def letter_track(self, sequence, scores, x, zero, width, scale):
        step = width / len(sequence)
        for i, (base, score) in enumerate(zip(sequence, scores)):
            height = abs(score) * scale
            if height < .015:
                continue
            self.c.saveState()
            self.c.translate(x + i * step, zero + max(score, 0) * scale)
            self.c.scale((step - .2) / 100, -height / 100)
            self.c.setFillColor(self.color(self.data['colors'][BASES.index(base)]))
            self.c.drawPath(self.glyphs[BASES.index(base)], stroke=0, fill=1, fillMode=0)
            self.c.restoreState()

    def examples_panel(self, data):
        examples = data['examples']
        # Fixed space between the moved panel B and unchanged bottom panels.
        self.text(20, 810, 'c', 8, bold=True)
        self.text(43, 810, 'Mean observed-active-context attribution', 7)
        self.text(490, 810, f"{data.get('references', 50)} references", 6, color=self.MUTED, align='right')
        self.text(63, 794, 'Context-specific', 7, bold=True)
        self.text(285, 794, 'Pleiotropic: GAF examples' if data.get('gaf_examples') else 'Pleiotropic', 7, bold=True)
        peak = max(abs(v) for e in examples for v in e['actual_ig'])
        zoom_peak = max(abs(v) for e in examples for v in e['actual_ig'][e['zoom'][0]:e['zoom'][1]])
        # Round shared symmetric limits upwards, preserving every base.
        limit = math.ceil(peak * 100) / 100
        zoom_limit = math.ceil(zoom_peak * 100) / 100
        data['overview_ylim'] = [-limit, limit]
        data['zoom_ylim'] = [-zoom_limit, zoom_limit]
        column_limits, column_zoom_limits = [limit] * 2, [zoom_limit] * 2
        if data.get('diverse_examples'):
            for col in (0, 1):
                column_limits[col] = math.ceil(max(abs(v) for e in examples[col::2]
                    for v in e['actual_ig']) / .05) * .05
                column_zoom_limits[col] = math.ceil(max(abs(v) for e in examples[col::2]
                    for v in e['actual_ig'][e['zoom'][0]:e['zoom'][1]]) / .05) * .05
            data['overview_ylim_by_column'] = [[-v, v] for v in column_limits]
            data['zoom_ylim_by_column'] = [[-v, v] for v in column_zoom_limits]
            data['scaling'] = 'Unnormalized signed actual IG (logit units); y limits shared within each column, explicitly ticked. No smoothing or clipping. Motif-oriented display reverses coordinates and complements bases together when needed.'
        self.y_axis_title(26, 620, 'Per-base contribution (logit)', size=6.5)
        for i, example in enumerate(examples):
            col, row = i % 2, i // 2
            limit, zoom_limit = column_limits[col], column_zoom_limits[col]
            x, top, width = 63 + col * 222, 775 - row * 118, 205
            view = display_tracks(example, motif_oriented=data.get('diverse_examples', False))
            values, (left, right) = view['values'], view['zoom']
            if data.get('diverse_examples'):
                example['display_orientation'] = 'reverse_complement' if view['reverse'] else 'forward'
            contexts = ', '.join(example['active_contexts']) if example['degree'] != 8 else 'all contexts'
            self.text(x, top, f"Degree {example['degree']} | {contexts}", 6.2, bold=True)
            orientation_label = '  |  RC' if view['reverse'] else ''
            self.text(x, top - 10, f"{example['id']}  |  {example['length']} bp" + orientation_label, 5.8, color=self.MUTED)
            # Entire native enhancer: signed individual-base bars, with zoom box.
            zero = top - 26
            self.c.setFillColor(self.color('#F0F0F0'))
            self.c.rect(x + width * left / len(values), zero - 8,
                        width * (right - left) / len(values), 16, stroke=0, fill=1)
            self.segment(x, zero, x + width, zero, self.LINE, .25)
            for j, value in enumerate(values):
                self.segment(x + width * (j + .5) / len(values), zero,
                             x + width * (j + .5) / len(values), zero + 7.5 * value / limit,
                             '#505050', .35)
            self.text(x - 4, zero + 5, f'{limit:g}', 4.6, align='right')
            self.text(x - 4, zero - 8, f'-{limit:g}', 4.6, align='right')
            self.text(x, zero - 15, str(example['length']) if view['reverse'] else '1', 4.8)
            self.text(x + width, zero - 15, '1' if view['reverse'] else str(example['length']), 4.8, align='right')
            for site in view['sites']:
                name = 'GAF / Trl' if site['best_match'] == 'Trl' else site['best_match']
                label = name + '-like' + (' (ns)' if site['tomtom_q'] >= .05 else '')
                if data.get('diverse_examples') and site['motif_id'] == 'exact_1/pos_patterns/pattern_0':
                    label = 'C-rich motif'
                if 'display_label' in site:
                    label = site['display_label']
                bracket_y = top - 54
                a, b = (x + width * (p - left) / (right - left) for p in (site['start'], site['end']))
                color = '#C46B22' if site['best_match'] == 'Trl' else '#167789'
                self.segment(a, bracket_y, b, bracket_y, color, 1)
                self.segment(a, bracket_y - 2, a, bracket_y, color, .6)
                self.segment(b, bracket_y - 2, b, bracket_y, color, .6)
                self.text((a + b) / 2, bracket_y + 3, label, 5.5, color=color, align='center')
            zero = top - 78
            self.segment(x, zero, x + width, zero, self.LINE, .35)
            self.letter_track(view['sequence'][left:right], values[left:right], x, zero, width, 19 / zoom_limit)
            for value in (-zoom_limit, 0, zoom_limit):
                y = zero + 19 * value / zoom_limit
                self.text(x - 4, y - 1.5, f'{value:g}', 4.8, align='right')
            start_label = example['zoom'][1] if view['reverse'] else left + 1
            end_label = example['zoom'][0] + 1 if view['reverse'] else right
            self.text(x, zero - 27, str(start_label), 5)
            self.text(x + width / 2, zero - 27, 'Position in enhancer (bp)', 5.5, align='center')
            self.text(x + width, zero - 27, str(end_label), 5, align='right')
        if not data.get('diverse_examples'):
            self.text(43, 412, 'Brackets: sequence matches to discovered motifs; labels: best JASPAR matches (ns, q >= 0.05).', 5.5, color=self.MUTED)


def main(project, gaf_examples=False, diverse_examples=False):
    previous_pdf = project / (PREVIOUS + '.pdf')
    if digest(previous_pdf) != 'f01e41e267494bbe0c04ad0cf0e7a64667b520ec81c43df80dddab76b0ad3243':
        raise ValueError('Previous figure has changed; review before inserting examples')
    previous = json.loads((project / (PREVIOUS + '.source.json')).read_text())
    audit, rows, _, _, _ = load_inputs(project, positive_limit=4)
    if rows != previous['panel_b']:
        raise ValueError('Motif panel differs from the preserved figure')
    example_rows = rows
    if diverse_examples:
        matches = json.loads((project / SUPPORT / 'annotation/jaspar/matches.json').read_text())
        example_rows = select_patterns(audit, matches, positive_limit=100)
    examples = prepare_examples(project, example_rows, gaf_examples=gaf_examples, diverse_examples=diverse_examples)
    prefix = PREFIX + ('_diverse' if diverse_examples else '_gaf' if gaf_examples else '')
    path = project / 'output/pdf' / (prefix + '.pdf')
    figure = ExampleFigure(audit, path, project / SUPPORT / 'fonts')
    figure.initialize(revised_layout=True, motif_count=len(rows))
    figure.height += EXTRA_HEIGHT
    figure.c.setPageSize((figure.width, figure.height))
    figure.c.setTitle('Figure 3b-e | Motifs, enhancer attribution examples and perturbations')
    figure.c.setSubject('Saved 50-reference original-enhancer IG; fixed observed-active logit target. No new inference.')
    figure.panel_b(rows)
    figure.examples_panel(examples)
    figure.panel_c(previous['panel_c'], panel_label='d')
    figure.panel_d_odds_ratios(previous['panel_d'], axis='linear', panel_label='e')
    check_text_bounds(figure.text_bounds)
    figure.c.showPage(); figure.c.save()
    result = dict(panel_a='omitted', panel_b=rows, panel_c=examples,
        panel_d=previous['panel_c'], panel_e=previous['panel_d'],
        inputs={**previous['inputs'], **examples['inputs'],
                PREVIOUS + '.source.json': digest(project / (PREVIOUS + '.source.json'))},
        previous_pdf_sha256=digest(previous_pdf),
        previous_source_sha256=digest(project / (PREVIOUS + '.source.json')),
        builder_sha256=digest(Path(__file__)),
        reused_builder_sha256=digest(Path(__file__).with_name('paper_figure3_nature.py')),
        selection_builder_sha256=digest(Path(__file__).with_name('attribution_example_selection.py')),
        pdf_sha256=digest(path), page_size_pt=[figure.width, figure.height],
        text_bounds=figure.text_bounds, text_overlap_check='passed', visual_qa='pending')
    path.with_suffix('.source.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Created {path.relative_to(project)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--gaf-examples', action='store_true',
                        help='Select GAF/Trl-like sites for degrees 2/4/8; preserve the original output')
    modes.add_argument('--diverse-examples', action='store_true',
                       help='Select diverse degree-1 motifs and high-contrast Trl-only / adjacent Trl-cg examples')
    args = parser.parse_args()
    main(args.project.resolve(), gaf_examples=args.gaf_examples, diverse_examples=args.diverse_examples)
