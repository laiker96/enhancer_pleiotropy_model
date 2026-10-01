"""Compare 4-of-5/6/7 PWM core rules on completed calibrated-IG motif fits."""
import argparse
from collections import Counter
from functools import partial
import html
import json
from pathlib import Path

import numpy as np

from .common import digest, event, require_allocation, write_json
from .dual_motif_pipeline import informative_core
from .simple_report import filter_native
from .report_breadth import sequence_logo
from .tomtom_atlas import meme_queries

NAME = 'classifier_calibrated_filter_20260928'
PARENT = 'experiments/classifier_calibrated_motifs_20260927'
PARENT_MANIFEST = 'c1899e36853f6097546e8187d98d979653804a9d7cf8b80bea0333ee99507b77'
WINDOWS = (5,6,7)


def comparison(variants):
    baseline={r['id']:r for r in variants['5']['rows']}
    result={}
    for size in WINDOWS:
        rows={r['id']:r for r in variants[str(size)]['rows']}
        common=rows.keys() & baseline.keys()
        result[str(size)]=dict(retained=len(rows),rescued=sorted(rows.keys()-baseline.keys()),
            lost=sorted(baseline.keys()-rows.keys()),
            changed_bounds=sorted(k for k in common if
                (rows[k]['quality']['start'],rows[k]['quality']['end']) !=
                (baseline[k]['quality']['start'],baseline[k]['quality']['end'])))
    return result


def render(data):
    esc=html.escape
    page=['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Calibrated IG motif filter comparison</title><style>',
        'body{font:15px system-ui,sans-serif;color:#222;max-width:1250px;margin:32px auto;padding:0 20px}',
        'table{border-collapse:collapse;width:100%;margin:18px 0}th,td{border-bottom:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}',
        'code{overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600;padding:12px 0}',
        '.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:18px}',
        'article{border:1px solid #ddd;border-radius:6px;padding:12px;overflow:hidden}.logo{width:100%;height:auto}',
        '.positive{border-top:4px solid #2466c4}.negative{border-top:4px solid #ce3646}',
        '.small{font-size:12px;color:#555}section{margin:32px 0}nav{overflow-x:auto}',
        '</style><h1>Calibrated IG: motif-filter sensitivity</h1>',
        '<p>Native enhancer intervals; all 22 TF-MoDISco fits. Compare sliding windows of 5, 6 or 7 bp, ',
        'each requiring at least 4 positions above 0.5 bits. Flank cutoff: 0.2 bits. ',
        'Retain widths 5-30 bp, mean information at least 0.5 bits/base, total at least 5 bits, and matching CWM sign.</p>',
        '<p>Raw patterns are unchanged. No extra clustering, motif scans or Tomtom matching. ',
        'Consensuses are not TF identities. Rank is unique discovery-enhancer support, not binding-site frequency. ',
        'A wider qualifying window can extend the retained span and cause a length/information/sign failure; ',
        'the filtered sets need not be nested. Full N-position windows are required.</p>',
        '<nav><table><tr><th>Fit</th><th>Group</th><th>4/5 (+ / -)</th><th>4/6 (+ / -)</th><th>4/7 (+ / -)</th></tr>']
    for group in data['groups']:
        task=group['task']; cells=[]
        for size in WINDOWS:
            rows=group['variants'][str(size)]['rows']
            cells.append(f"<td>{sum(r['sign']=='positive' for r in rows)} / {sum(r['sign']=='negative' for r in rows)}</td>")
        page.append(f"<tr><td><a href='#g{task['task']}'>{esc(task['target'])}</a></td><td>{esc(task['group'])}</td>{''.join(cells)}</tr>")
    page.append('</table></nav>')
    for group in data['groups']:
        task=group['task']
        page.append(f"<section id='g{task['task']}'><h2>{esc(task['target'])} / {esc(task['group'])}</h2><p>{group['elements']:,} discovery enhancers.</p>")
        for size in WINDOWS:
            variant=group['variants'][str(size)];change=group['comparison'][str(size)]
            page.append(f"<details {'open' if size==6 else ''}><summary>4 of {size}: {len(variant['rows'])} retained; "
                        f"{len(change['rescued'])} rescued, {len(change['lost'])} lost versus 4/5</summary>")
            for sign in ('positive','negative'):
                page.append(f'<h3>{sign.capitalize()}</h3><div class="cards">')
                rows=[r for r in variant['rows'] if r['sign']==sign]
                if not rows:page.append('<p>No retained patterns.</p>')
                for row in rows:
                    q=row['quality'];flag='Rescued versus 4/5' if row['id'] in change['rescued'] else 'Retained in 4/5'
                    page.append(f"<article class='{sign}'><strong>#{row['rank']} <code>{row['consensus']}</code></strong>"
                        f"<p class='small'>{esc(row['pattern'])}; {flag}</p>")
                    page.append(sequence_logo(row['trimmed_pwm'],row['consensus']))
                    page.append(f"<p>{row['supporting_discovery_enhancers']:,}/{group['elements']:,} enhancers "
                        f"({100*row['support_fraction']:.2f}%); {row['seqlets']:,} seqlets</p>"
                        f"<p class='small'>{q['width']} bp; {q['mean_bits']:.3f} bits/base; "
                        f"{q['total_bits']:.3f} bits total; original columns [{q['start']}, {q['end']})</p></article>")
                page.append('</div>')
            page.append('<details><summary>Exclusions and reasons</summary><ul>')
            for row in variant['exclusions']:
                page.append(f"<li><code>{esc(row['id'])}</code>: {esc(row['reason'])}</li>")
            page.append('</ul></details></details>')
        page.append('</section>')
    page.append('</html>')
    return ''.join(page)


def run(project,root):
    require_allocation('cpu')
    parent=project/PARENT
    if digest(parent/'MANIFEST.sha256')!=PARENT_MANIFEST:raise ValueError('Wrong discovery experiment')
    done=json.loads((parent/'complete.json').read_text())
    if done['status']!='complete' or len(done['fits'])!=22:raise ValueError('Incomplete raw motifs')
    output=root/'output';output.mkdir(exist_ok=False)
    result=dict(parent=PARENT,parent_complete_sha256=digest(parent/'complete.json'),
        rule=dict(windows=list(WINDOWS),min_informative=4,information_threshold=.5,
            flank_threshold=.2,min_width=5,max_width=30,min_mean_bits=.5,min_total_bits=5,
            full_windows_required=True,preserve_internal_gaps=True,require_matching_cwm_sign=True),
        filtered=True,reclustered=False,tomtom_run=False,groups=[],source_files={})
    for fit in done['fits']:
        folder=parent/fit['directory'];receipt=folder/'complete.json'
        if digest(receipt)!=fit['complete_sha256']:raise ValueError('Changed discovery receipt')
        record=json.loads(receipt.read_text())
        for name in ('motifs.h5','selection.json','raw_catalogue.json'):
            if digest(folder/name)!=record['files'][name]:raise ValueError('Changed raw fit '+name)
            result['source_files'][str((folder/name).relative_to(project))]=record['files'][name]
        selected=json.loads((folder/'selection.json').read_text())
        raw=json.loads((folder/'raw_catalogue.json').read_text())
        raw_by_id={r['pattern']:r for r in raw['patterns']}
        counts={s:record['audit'][s]['patterns'] for s in ('positive','negative')}
        group=dict(task=record['task'],elements=selected['elements'],raw_counts=counts,variants={})
        for size in WINDOWS:
            rows,excluded=filter_native(folder/'motifs.h5',folder.name,counts,
                core_filter=partial(informative_core,flank_threshold=.2,window_size=size,min_informative=4))
            if len(rows)+len(excluded)!=sum(counts.values()):raise ValueError('Lost pattern accounting')
            for row in rows:
                rawrow=raw_by_id[row['pattern']]
                if (row['seqlets']!=rawrow['seqlets'] or row['supporting_discovery_enhancers']!=rawrow['supporting_enhancers']):
                    raise ValueError('Filtering changed original discovery support')
                row['consensus']=''.join('ACGT'[i] for i in np.asarray(row['trimmed_pwm']).argmax(1))
                row['support_fraction']=row['supporting_discovery_enhancers']/selected['elements']
            ordered=[]
            for sign in ('positive','negative'):
                ranked=sorted((r for r in rows if r['sign']==sign),key=lambda r:(-r['supporting_discovery_enhancers'],r['id']))
                for rank,row in enumerate(ranked,1):row['rank']=rank
                ordered.extend(ranked)
            group['variants'][str(size)]=dict(rows=ordered,exclusions=excluded)
        group['comparison']=comparison(group['variants']);result['groups'].append(group)
        event('calibrated_filter_fit',task=fit['task'],retained={k:len(v['rows']) for k,v in group['variants'].items()})
    result['summary']={str(size):dict(
        positive=sum(r['sign']=='positive' for g in result['groups'] for r in g['variants'][str(size)]['rows']),
        negative=sum(r['sign']=='negative' for g in result['groups'] for r in g['variants'][str(size)]['rows']),
        rescued=sum(len(g['comparison'][str(size)]['rescued']) for g in result['groups']),
        lost=sum(len(g['comparison'][str(size)]['lost']) for g in result['groups']),
        changed_bounds=sum(len(g['comparison'][str(size)]['changed_bounds']) for g in result['groups']),
        exclusion_reasons=dict(Counter(r['reason'] for g in result['groups'] for r in g['variants'][str(size)]['exclusions'])))
        for size in WINDOWS}
    write_json(output/'comparison.json',result)
    (output/'report.html').write_text(render(result))
    for size in WINDOWS:
        data=dict(groups=[dict(rows=g['variants'][str(size)]['rows']) for g in result['groups']])
        (output/f'queries_4of{size}.meme').write_text(meme_queries(data))
    write_json(output/'complete.json',dict(status='complete',summary=result['summary'],
        source_complete_sha256=result['parent_complete_sha256'],
        files={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    event('calibrated_filter_complete',summary=result['summary'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();run(args.project,args.root)
