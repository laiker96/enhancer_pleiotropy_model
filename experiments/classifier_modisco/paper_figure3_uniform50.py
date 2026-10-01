"""Rebuild 50-reference panel B with the uniform revised information filter.

CPU-only: reuse all original native TF-MoDISco fits, recompute trimmed PWMs
and JASPAR Tomtom annotations, preserve the completed example/curve/box panels.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .dual_motif_pipeline import CORE_RULE, informative_core
from .dual_motif_figure import select_rows
from .original_intervals import GROUPS
from .paper_figure3 import SUPPORT, check_text_bounds, digest
from .paper_figure3_examples import EXTRA_HEIGHT, ORIGINAL, ExampleFigure
from .simple_report import filter_native, rank_rows
from .tomtom_atlas import meme_queries, parse_matches, read_meme

PREVIOUS = 'output/pdf/figure_3bcde_native_enhancer_attribution_examples_diverse_grh50'
PREVIOUS_SHA = 'bde0f8c3433c46f990b856d4f2e4238ae66a46c01ddfca3966357322b33bd67d'
OUTPUT = 'output/pdf/figure_3bcde_native_enhancer_uniform_filter_grh50'
RESULTS = 'results/figure3_uniform_filter_50ref_20260921'


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(project, root):
    support = project / SUPPORT
    receipt = json.loads((support / 'complete.json').read_text())
    old_path = support / 'report_audit.json'
    if receipt['status'] != 'complete' or digest(old_path) != receipt['files']['report_audit.json']:
        raise ValueError('Changed original report')
    old = json.loads(old_path.read_text())
    groups, exclusions, sensitivity, inputs = [], [], [], {str(old_path.relative_to(project)): digest(old_path)}
    for name, low, high in GROUPS:
        source = project / ORIGINAL / 'groups' / name
        for filename in ('motifs.h5', 'motifs.audit.json', 'selection.json'):
            path = source / filename
            expected = old['inputs'][f'experiments/classifier_modisco_original_20260917/groups/{name}/{filename}']
            value = digest(path)
            if value != expected:
                raise ValueError(f'Changed raw discovery: {path}')
            inputs[str(path.relative_to(project))] = value
        audit = json.loads((source / 'motifs.audit.json').read_text())
        counts = {s: audit[s]['patterns'] for s in ('positive', 'negative')}
        rows, failed = filter_native(source / 'motifs.h5', name, counts, core_filter=informative_core)
        selection = json.loads((source / 'selection.json').read_text())
        if not selection['train_only'] or selection['quality_excluded'] != 0:
            raise ValueError('Unexpected discovery selection')
        previous = next(g for g in old['groups'] if g['name'] == name)
        if selection['elements'] != previous['discovery_elements']:
            raise ValueError('Changed support denominator')
        group = dict(name=name, minimum_breadth=low, maximum_breadth=high,
            discovery_elements=selection['elements'], rank_by='native_support', rows=rows)
        rank_rows(group)
        groups.append(group); exclusions.extend(failed)
        before, after = {r['id'] for r in previous['rows']}, {r['id'] for r in rows}
        sensitivity.append(dict(group=name, raw=sum(counts.values()), old=len(before), revised=len(after),
            added=sorted(after-before), removed=sorted(before-after)))
    result = dict(groups=groups, exclusions=exclusions, colors=old['colors'], glyphs=old['glyphs'],
        inputs=inputs, rule=CORE_RULE, sensitivity=sensitivity,
        ranking='Original distinct discovery enhancers with assigned seqlets / original group N; separately by sign; ties by motif ID.',
        display='Top four positive motifs and every retained negative motif per group; no TF-specific exceptions.',
        interval='Native enhancer', references=50, rediscovered=False, reclustered=False)
    save(root / 'filter_audit.json', result)
    return result


def annotate(project, root, audit):
    assets = project / SUPPORT
    binary = (assets / 'bin/tomtom').resolve()
    database = (assets / 'references/jaspar2026_insects.meme').resolve()
    old_matches = json.loads((assets / 'annotation/jaspar/matches.json').read_text())
    if digest(database) != old_matches['database']['sha256']:
        raise ValueError('Changed pinned reference database')
    if digest(binary) != old_matches['tomtom_binary_sha256']:
        raise ValueError('Changed tested Tomtom binary')
    version = subprocess.check_output([str(binary), '-version'], text=True).strip()
    if version != '5.5.9':
        raise ValueError('Unexpected Tomtom version')
    query = root / 'queries.meme'
    with query.open('x') as stream:
        stream.write(meme_queries(audit))
    expected = {r['id'].replace('/', '__') for g in audit['groups'] for r in g['rows']}
    parsed = read_meme(query)
    if set(parsed) != expected:
        raise ValueError('Incomplete query export')
    for group in audit['groups']:
        for row in group['rows']:
            np.testing.assert_allclose(parsed[row['id'].replace('/', '__')]['pwm'], row['trimmed_pwm'], atol=1e-11)
    targets = read_meme(database)
    if len(targets) != 296:
        raise ValueError('Expected 296 JASPAR insect profiles')
    command = [str(binary), '-text', '-dist', 'pearson', '-min-overlap', '5',
        '-motif-pseudo', '0.1', '-thresh', '1', '-verbosity', '2', str(query), str(database)]
    print(json.dumps(dict(event='tomtom_start', queries=len(expected), targets=len(targets))), flush=True)
    started = time.monotonic()
    with (root / 'tomtom.tsv').open('x') as stdout, (root / 'tomtom.stderr.log').open('x') as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
    best, counts = parse_matches((root / 'tomtom.tsv').read_text(), expected, targets)
    for match in best.values():
        match['reference'] = targets[match['target_id']]
    matches = dict(best=best, returned_matches_per_query=counts, command=command,
        version=version, query_sha256=digest(query), database_sha256=digest(database),
        tomtom_binary_sha256=digest(binary), raw_output_sha256=digest(root / 'tomtom.tsv'),
        seconds=time.monotonic()-started, rules=old_matches['rules'])
    matches['rules']['query'] = CORE_RULE
    save(root / 'jaspar_matches.json', matches)
    print(json.dumps(dict(event='tomtom_complete', seconds=matches['seconds'])), flush=True)
    return matches


def render(project, root, audit, matches):
    previous_pdf = project / (PREVIOUS + '.pdf')
    if digest(previous_pdf) != PREVIOUS_SHA:
        raise ValueError('Previous figure changed')
    previous_path = project / (PREVIOUS + '.source.json')
    previous = json.loads(previous_path.read_text())
    if previous['pdf_sha256'] != PREVIOUS_SHA:
        raise ValueError('Previous figure source mismatch')
    rows = select_rows(audit, matches, limit=4)
    all_ids = {r['id'] for g in audit['groups'] for r in g['rows']}
    examples = copy.deepcopy(previous['panel_c'])
    for example in examples['examples']:
        if any(site['motif_id'] not in all_ids for site in example['plot_sites']):
            raise ValueError('Preserved example motif fails new uniform filter')
    path = project / (OUTPUT + '.pdf')
    if path.exists():
        raise FileExistsError('Do not overwrite an earlier figure')
    figure = ExampleFigure(audit, path, project / SUPPORT / 'fonts')
    figure.initialize(revised_layout=True, motif_count=len(rows)); figure.height += EXTRA_HEIGHT
    figure.c.setPageSize((figure.width, figure.height))
    figure.c.setTitle('Figure 3b-e | Native enhancer motifs with uniform information filter')
    figure.c.setSubject('Original 50-reference attribution; uniformly re-filtered raw motifs and new JASPAR matches')
    figure.panel_b(rows); figure.examples_panel(examples)
    figure.panel_c(previous['panel_d'], panel_label='d')
    figure.panel_d_odds_ratios(previous['panel_e'], axis='linear', panel_label='e')
    check_text_bounds(figure.text_bounds)
    figure.c.showPage(); figure.c.save()
    if examples != previous['panel_c']:
        raise ValueError('Example panel data changed')
    source = dict(panel_a='omitted', panel_b=rows, panel_c=examples,
        panel_d=previous['panel_d'], panel_e=previous['panel_e'],
        inputs={**audit['inputs'], str(previous_path.relative_to(project)): digest(previous_path)},
        rule=CORE_RULE, filter_audit_sha256=digest(root / 'filter_audit.json'),
        matches_sha256=digest(root / 'jaspar_matches.json'), previous_pdf_sha256=PREVIOUS_SHA,
        builder_sha256=digest(Path(__file__)), reused_builders={name:digest(Path(__file__).with_name(name+'.py'))
            for name in ('simple_report', 'dual_motif_pipeline', 'dual_motif_figure', 'paper_figure3_examples', 'paper_figure3_nature')},
        pdf_sha256=digest(path), page_size_pt=[figure.width, figure.height],
        text_bounds=figure.text_bounds, text_overlap_check='passed', visual_qa='pending',
        notes=['Panel B uses the same revised filter for all raw native motifs, both signs and all groups.',
            'Ranks and support retain original cluster-assigned enhancer counts, not scan prevalence.',
            'Tomtom is rerun on all revised cores; no old q value is attached to a new core.',
            'Panel C preserves archived site brackets and site statistics from their original core scans; all illustrated native patterns pass the revised filter.',
            'Panels C/D/E and original discovery/attribution are unchanged. No new scans, clustering or inference.'])
    save(path.with_suffix('.source.json'), source)
    save(root / 'complete.json', dict(status='complete', retained=sum(len(g['rows']) for g in audit['groups']),
        raw=sum(s['raw'] for s in audit['sensitivity']), displayed=len(rows), files={
            name:digest(root / name) for name in ('filter_audit.json','jaspar_matches.json','queries.meme','tomtom.tsv')},
        pdf_sha256=digest(path), visual_qa='pending'))
    if digest(previous_pdf) != PREVIOUS_SHA:
        raise ValueError('Previous PDF changed')
    print(json.dumps(dict(event='rendered', output=str(path), rows=len(rows),
        grh=[dict(id=r['id'], rank=r['rank'], q=r['match']['q'], support=r['attribution_support'])
             for r in rows if r['match']['reference']['name']=='grh'])), flush=True)


def main(project):
    root = project / RESULTS
    root.mkdir(exist_ok=False)
    audit = prepare(project, root)
    print(json.dumps(dict(event='filtered', groups=audit['sensitivity'])), flush=True)
    matches = annotate(project, root, audit)
    render(project, root, audit, matches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    main(parser.parse_args().project.resolve())
