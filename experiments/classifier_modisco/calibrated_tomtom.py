"""Annotate the selected 4/5 calibrated-IG motifs, preserving all 22 fits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path
import shutil

from .common import digest, event, require_allocation, write_json
from .report_breadth import COLORS, GLYPHS
from .tomtom_atlas import DATABASES, aligned_matrices, heights, query_id, run as tomtom_run

NAME = 'classifier_calibrated_tomtom_20260928'
PARENT = 'experiments/classifier_calibrated_filter_20260928/output'
ASSETS = 'experiments/classifier_modisco_support50ref_20260918'
CONTEXTS = ('ab', 'e13', 'e5', 'ead', 'hid', 'lb', 'o', 'wid')
RULE = ('4 of 5 columns strictly >0.5 bits; 0.2-bit flanks; width 5-30 bp; '
        'mean >=0.5 bits/base; total >=5 bits; concordant trimmed contribution sign')
LABELS = {
    'observed_active_mean': 'Mean over experimentally active contexts',
    'observed_active_sum': 'Sum over experimentally active contexts',
    'all_context_sum': 'Sum over all eight contexts',
    'balanced_four_families': 'Family-balanced sum: embryo, CNS, discs, ovary',
    'balanced_separate_brains': 'Family-balanced sum: adult and larval brain separated',
    'degree_1': 'Degree of pleiotropy 1', 'degree_2_5': 'Degree of pleiotropy 2–5',
    'degree_6_8': 'Degree of pleiotropy 6–8', 'context_specific': 'Context-specific (degree 1)',
    'family_restricted_multicontext': 'Multiple contexts within one family',
    'two_families': 'Exactly two active families',
    'three_or_more_families': 'Three or more active families',
    'all_enhancers': 'All QC-passing enhancers',
}


def label(value):
    if value.startswith('active_member_means_'):
        return 'Sum of within-family active-context means: '+value.removeprefix('active_member_means_').replace('_',' ')
    if value == 'family_degree_1': return 'Family degree of pleiotropy 1'
    if value.startswith('family_degree_ge_'):
        return 'Family degree of pleiotropy ≥'+value.removeprefix('family_degree_ge_')
    if value.startswith('active_family_means_'):
        return 'Sum of observed-active family means: '+value.removeprefix('active_family_means_').replace('_',' ')
    if value.startswith('degree_ge_'):
        return 'Degree of pleiotropy ≥'+value.removeprefix('degree_ge_')
    return LABELS.get(value, value.replace('_', ' '))


def select_four_of_five(comparison, specs):
    groups = []
    for group in comparison['groups']:
        task = group['task']
        groups.append(dict(task=task, elements=group['elements'],
            attribution=specs[task['target']], rows=group['variants']['5']['rows']))
    ids = [query_id(r) for g in groups for r in g['rows']]
    if len(ids) != len(set(ids)) or any(not r['quality']['passed'] for g in groups for r in g['rows']):
        raise ValueError('Duplicate or nonpassing query')
    return dict(groups=groups, rule=RULE, contexts=list(CONTEXTS))


def alignment_logos(row, match):
    target, qs, ts, span, overlap = aligned_matrices(
        row['trimmed_pwm'], match['reference']['pwm'], match['offset'], match['orientation'])
    if overlap != match['overlap']:
        raise ValueError('Displayed alignment disagrees with Tomtom')
    width,left,right=850,42,16
    step=(width-left-right)/span
    pieces=[f'<svg class="logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 284" '
        f'role="img" aria-label="Aligned discovery and reference motifs" data-overlap="{overlap}">',
        '<title>'+html.escape(row['consensus']+' / '+match['reference']['name'])+'</title>']
    for matrix,start,base,title in ((row['trimmed_pwm'],qs,116,'Discovered motif'),
                                    (target,ts,251,'JASPAR / database match')):
        pieces.append(f'<text x="{left}" y="{base-99}" font-size="13">{title}</text>')
        pieces.append(f'<rect x="{left+max(qs,ts)*step:.4f}" y="{base-92}" '
            f'width="{overlap*step:.4f}" height="92" fill="#eef5fc"/>')
        for k in range(span+1):
            x=left+k*step
            pieces.append(f'<path d="M{x:.4f} {base-92}V{base}" stroke="#dde5ed" stroke-width=".5"/>')
        for bits in (0,1,2):
            y=base-bits*45
            pieces.append(f'<path d="M{left} {y}H{width-right}" stroke="#cbd5df" stroke-width=".6"/>'
                f'<text x="32" y="{y+4}" text-anchor="end" font-size="12">{bits}</text>')
        for j,values in enumerate(heights(matrix)):
            y=float(base)
            for i in sorted(range(4),key=lambda i:values[i]):
                height=values[i]*45;y-=height
                if height>1e-5:
                    pieces.append(f'<path d="{GLYPHS[i]}" fill="{COLORS[i]}" fill-rule="evenodd" '
                        f'transform="translate({left+(start+j)*step:.4f} {y:.4f}) '
                        f'scale({(step-1)/100:.6f} {height/100:.6f})"/>')
    for k in range(span):
        if span<=30 or k%5==0:
            pieces.append(f'<text x="{left+(k+.5)*step:.4f}" y="270" text-anchor="middle" font-size="11">{k+1}</text>')
    pieces.append('</svg>')
    return ''.join(pieces)


def page_start(title, database_keys=None):
    keys=list(DATABASES) if database_keys is None else database_keys
    return ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>'+html.escape(title)+'</title><style>',
        'body{font:15px system-ui,sans-serif;color:#1d2936;max-width:1250px;margin:30px auto;padding:0 20px}',
        'a{color:#2463a0}table{border-collapse:collapse;width:100%;margin:18px 0}',
        'th,td{border-bottom:1px solid #dce1e6;padding:8px;text-align:left;vertical-align:top}',
        'nav,.table-wrap{overflow-x:auto}code{overflow-wrap:anywhere}section{margin:38px 0}',
        '.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}',
        'article{border:1px solid #dce1e6;border-radius:6px;padding:14px;min-width:0}',
        '.positive{border-top:4px solid #2466c4}.negative{border-top:4px solid #ce3646}',
        '.small,.muted{font-size:12px;color:#5d6873}.logo{display:block;width:100%;height:auto}',
        '.logo-label{font-size:12px;margin:12px 0 0}.hit{font-weight:600;color:#12694f}',
        '.weak{color:#785d3a}summary{cursor:pointer}h3{overflow-wrap:anywhere}',
        '@media(max-width:450px){.cards{grid-template-columns:1fr}}',
        '</style></head><body><h1>'+html.escape(title)+'</h1>',
        '<nav><a href="index.html">Comparison overview</a> · ',
        ' · '.join(f'<a href="{key}.html">{DATABASES[key]["name"]}</a>' for key in keys),
        '</nav>']


def methods(fit_count=22, discovery_note='No new attribution, discovery, or post-discovery clustering.'):
    return ('<details><summary>Methods and interpretation</summary>'
        '<p>IG64, 100 dinucleotide-shuffled references, eight calibrated probability outputs. '
        f'All {fit_count} TF-MoDISco fits use native enhancer intervals and the same eight-map numerical QC: '
        '40,309/40,338 enhancers, with all data splits included for exploratory discovery.</p>'
        '<p>'+RULE+'. '+html.escape(discovery_note)+'</p>'
        '<p>Observed-active sums use fixed experimental enhancer-table labels, not predicted labels; '
        'observed-active family sums instead mask whole families using their fixed experimental OR label. '
        'Within-family active-context means instead exclude inactive member contexts and divide by the number '
        'of experimentally active members in that family; inactive families contribute zero. '
        'Observed-active context MEANS divide each active-context coefficient by the enhancer’s number '
        'of experimentally active contexts; their coefficients total one. '
        'The other summaries do not mask contexts. Family-balanced sums add each family’s mean probability '
        '(not an OR). Families are embryo e5/e13, CNS ab/lb, imaginal discs ead/hid/wid and ovary o; '
        'the alternative scheme separates adult and larval brain. Families are biological groupings, '
        'not proven independent measurements. Contrasts subtract the mean of the other family means, '
        'or directly compare adult/larval brain or e13/e5. Exact coefficients are shown per section.</p>'
        '<p>Positive/negative means increasing/decreasing the selected scalar, not experimentally '
        'established activation/repression. For a contrast, negative motifs favor its opposite side. '
        'Rank is unique original seqlet-contributing enhancer support within each sign; percentage '
        'uses all discovery enhancers in that group. It is not sequence-scan prevalence or enrichment.</p>'
        '<p>Tomtom 5.5.9; Pearson similarity; complete scores; both strands; pseudocount 0.1; '
        'minimum overlap 5 bp (or the shorter motif if shorter than 5). Best match minimizes p-value, '
        'then q-value and target ID. All best hits are shown, including q &gt; 0.05. '
        'q-values are per query and database; no extra correction across queries, fits or databases. '
        'q is motif-similarity evidence, not binding probability or proof of TF identity. Database '
        'coverage, size and redundancy differ; do not directly rank databases by their q-values. '
        'JASPAR includes all CORE insects, not exclusively D. melanogaster.</p>'
        '<p>Logos show unmodified probabilities as information content against a uniform background, '
        'without small-sample correction. Discovery and reference logos share alignment columns; '
        'blank columns are display padding, not additions to the matched PWMs.</p>'
        '<p><a href="https://meme-suite.org/meme/doc/tomtom.html">Tomtom documentation</a> · '
        '<a href="audit.json">Motifs, coefficients and provenance</a> · '
        '<a href="queries_4of5.meme">Selected PWMs</a></p></details>')


def group_header(group):
    t = group['task']; spec = group['attribution']
    terms = ', '.join(f'{c}: {w:.4g}' for c,w in zip(CONTEXTS,spec['weights']))
    mask = 'Each coefficient is also multiplied by that enhancer’s observed binary label.' if spec['observed_mask'] else 'No observed or predicted activity mask.'
    if spec.get('observed_mean'):
        terms = ('For each enhancer: active contexts have weight 1 / number of experimentally active contexts; '
                 'inactive contexts have weight zero; total coefficient weight is one')
        mask = ('Experimental labels and the denominator stay fixed along the IG path. '
                'This is mean active-context probability, not predicted breadth.')
    if spec.get('observed_family_scheme'):
        mask = ('Each coefficient is multiplied by the observed OR label of its whole family. '
                'All members of an active family retain their original mean weight; no renormalization over active members. '
                'These fixed labels do not change along the IG path.')
    if spec.get('active_member_scheme'):
        terms = ('For each enhancer and family: active contexts have weight 1 / number of observed-active '
                 'members in that family; inactive contexts have weight zero')
        mask = ('A family with no active members contributes zero. Sum family averages without dividing by '
                'the number of active families. Experimental labels and these weights stay fixed along the IG path.')
    return (f'<h2>{html.escape(label(t["target"]))} — {html.escape(label(t["group"]))}</h2>'
        f'<p>{group["elements"]:,} discovery enhancers · {len(group["rows"])} retained patterns</p>'
        f'<details><summary>Exact attribution coefficients</summary><p>{terms}. {mask}</p></details>')


def match_text(match):
    ref = match['reference']; name = ref['name']
    return html.escape(name if name == ref['id'] else f'{name} ({ref["id"]})')


def render(audit, results, database=None, *, database_keys=None):
    keys=list(DATABASES) if database_keys is None else database_keys
    title = 'Calibrated IG motifs — '+(DATABASES[database]['name'] if database else 'database comparison')
    page = page_start(title,keys)
    page.append(f'<p>4-of-5 filter · {sum(len(g["rows"]) for g in audit["groups"])} patterns · '
        f'{len(audit["groups"])} fits · positive and negative contributions</p>')
    page.append(methods(len(audit['groups']),audit.get('discovery_note','No new attribution, discovery, or post-discovery clustering.')))
    page.append('<nav><ul>')
    for g in audit['groups']:
        t=g['task']
        page.append(f'<li><a href="#g{t["task"]}">{html.escape(label(t["target"]))} / {html.escape(label(t["group"]))}</a></li>')
    page.append('</ul></nav>')
    for g in audit['groups']:
        task=g['task']['task'];page.append(f'<section id="g{task}">'+group_header(g))
        if database:
            page.append(f'<p><a href="annotation/fit_{task:02d}/{database}/tomtom.tsv">All Tomtom matches for this fit</a></p>')
            for sign in ('positive','negative'):
                page.append(f'<h3>{sign.capitalize()} contributions</h3><div class="cards">')
                rows=[r for r in g['rows'] if r['sign']==sign]
                if not rows:page.append('<p>No retained patterns.</p>')
                for row in rows:
                    m=results[database]['best'][query_id(row)]
                    status='q ≤ 0.05' if m['q']<=.05 else 'not significant (q > 0.05)'
                    cls='hit' if m['q']<=.05 else 'weak'
                    page.append(f'<article class="{sign}" id="{query_id(row)}"><h3>#{row["rank"]} <code>{row["consensus"]}</code></h3>'
                        f'<p>{row["supporting_discovery_enhancers"]:,}/{g["elements"]:,} enhancers '
                        f'({100*row["support_fraction"]:.2f}%) · {row["seqlets"]:,} seqlets</p>'
                        f'<p>Best: <strong>{match_text(m)}</strong><br><span class="{cls}">q = {m["q"]:.3g} · {status}</span></p>')
                    page.append(alignment_logos(row,m))
                    page.append(f'<p class="small">p = {m["p"]:.3g}; E = {m["e"]:.3g}; overlap {m["overlap"]} bp; '
                        f'target strand {m["orientation"]}; offset {m["offset"]}.<br>'
                        f'{html.escape(row["pattern"])}; {row["quality"]["width"]} bp; '
                        f'{row["quality"]["mean_bits"]:.2f} bits/base.</p></article>')
                page.append('</div>')
        else:
            page.append('<div class="table-wrap"><table><tr><th>Sign / rank</th><th>Consensus / support</th>'
                + ''.join(f'<th>{DATABASES[key]["name"]}<br>Best hit and q</th>' for key in keys)+'</tr>')
            for row in g['rows']:
                page.append(f'<tr><td>{row["sign"]} #{row["rank"]}</td><td><code>{row["consensus"]}</code><br>'
                    f'{100*row["support_fraction"]:.2f}% ({row["supporting_discovery_enhancers"]:,})</td>')
                for key in keys:
                    m=results[key]['best'][query_id(row)];cls='hit' if m['q']<=.05 else 'weak'
                    flag='' if m['q']<=.05 else ' · not significant'
                    page.append(f'<td><a href="{key}.html#{query_id(row)}">{match_text(m)}</a><br>'
                        f'<span class="{cls}">q = {m["q"]:.3g}{flag}</span></td>')
                page.append('</tr>')
            page.append('</table></div>')
        page.append('</section>')
    page.append('</body></html>')
    return ''.join(page)


def run(project, root):
    require_allocation('cpu')
    config=json.loads((root/'config.json').read_text())
    for name, expected in config['source_hashes'].items():
        if digest(project/name)!=expected:raise ValueError('Changed input '+name)
    comparison=json.loads((project/PARENT/'comparison.json').read_text())
    audit=select_four_of_five(comparison, config['targets'])
    if len(audit['groups'])!=22 or sum(len(g['rows']) for g in audit['groups'])!=313:
        raise ValueError('Wrong 4/5 selection')
    audit.update(source_hashes=config['source_hashes'],databases=DATABASES)
    output=root/'output';output.mkdir(exist_ok=False)
    write_json(output/'audit.json',audit)
    shutil.copy2(project/PARENT/'queries_4of5.meme',output/'queries_4of5.meme')
    refs=output/'references';refs.mkdir()
    for meta in DATABASES.values():
        dest=refs/meta['file'];dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(project/ASSETS/'references'/meta['file'],dest)
    binary=project/ASSETS/'bin/tomtom'
    def fit(group):
        task=group['task']['task'];folder=output/'annotation'/f'fit_{task:02d}'
        folder.mkdir(parents=True,exist_ok=False)
        path=folder/'queries.json';write_json(path,dict(groups=[group]))
        tomtom_run(folder,path,binary,query_rule=RULE,references=refs)
        event('calibrated_tomtom_fit_done',task=task,queries=len(group['rows']))
        return {key:json.loads((folder/key/'matches.json').read_text()) for key in DATABASES}
    # Each fit invokes three database searches; 4 fits x 3 = at most 12 CPUs.
    with ThreadPoolExecutor(max_workers=4) as pool:
        fitted=list(pool.map(fit,audit['groups']))
    results={key:dict(database=DATABASES[key],best={},fit_provenance=[]) for key in DATABASES}
    for group, matches in zip(audit['groups'],fitted):
        for key,value in matches.items():
            if results[key]['best'].keys() & value['best'].keys():raise ValueError('Duplicate match')
            results[key]['best'].update(value['best'])
            results[key]['fit_provenance'].append(dict(task=group['task']['task'],
                **{k:v for k,v in value.items() if k!='best'}))
    summary={}
    for key,result in results.items():
        if len(result['best'])!=313:raise ValueError('Missing matches')
        write_json(output/f'{key}_matches.json',result)
        (output/f'{key}.html').write_text(render(audit,results,key))
        summary[key]=dict(queries=313,significant_q_le_005=sum(m['q']<=.05 for m in result['best'].values()))
    (output/'index.html').write_text(render(audit,results))
    write_json(output/'complete.json',dict(status='complete',summary=summary,
        manifest_sha256=digest(root/'MANIFEST.sha256'),
        files={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    event('calibrated_tomtom_complete',summary=summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();run(args.project,args.root)
