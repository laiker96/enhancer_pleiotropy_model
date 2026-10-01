"""Keep sequence, discovery-cluster, and Fi-NeMo support separate."""
import csv
import json

import numpy as np

from .common import digest, event, write_json
from .motif_occurrence import TARGETS, GROUPS, read_inputs, group_summary


def summarize(root, target):
    import polars as pl
    config, parent, metadata = read_inputs(root)
    completion = json.loads((root/'finemo_complete.json').read_text())
    if completion['status'] != 'complete' or completion['config_sha256'] != digest(root/'config.json'):
        raise ValueError('Fi-NeMo incomplete/config mismatch')
    result = root/TARGETS[target]
    sequence_receipt = json.loads((result/'sequence/complete.json').read_text())
    if sequence_receipt['status'] != 'complete': raise ValueError('Sequence scan incomplete')
    audit = json.loads((parent/TARGETS[target]/'report_audit.json').read_text())
    degree = metadata['labels'].sum(1)
    summaries, flattened, files = [], [], {}
    for group in audit['groups']:
        hits_frames, qc_frames = [], []
        for chunk in completion['chunks']:
            if chunk['target'] != TARGETS[target] or chunk['group'] != group['name']: continue
            directory = root/chunk['path']
            for name, expected in chunk['files'].items():
                if digest(directory/name) != expected: raise ValueError('Changed Fi-NeMo chunk')
            hits_frames.append(pl.read_parquet(directory/'hits.parquet'))
            qc_frames.append(pl.read_parquet(directory/'qc.parquet'))
        qc = pl.concat(qc_frames); hits = pl.concat(hits_frames)
        indices = qc['enhancer_index'].to_numpy()
        expected = np.flatnonzero(metadata['quality_pass'][:,target])
        np.testing.assert_array_equal(np.sort(indices), expected)
        converged = np.zeros(len(degree), bool)
        converged[qc.filter(pl.col('converged'))['enhancer_index'].to_numpy()] = True
        valid_hits = hits.filter(pl.col('converged'))
        for row in group['rows']:
            scan_path = result/'sequence'/(row['id'].replace('/','__')+'.npz')
            if digest(scan_path) != sequence_receipt['files'][scan_path.name]:
                raise ValueError('Changed sequence scan')
            with np.load(scan_path, allow_pickle=False) as saved: scores = dict(saved)
            np.testing.assert_array_equal(scores['ids'], metadata['ids'])
            carriers = np.zeros(len(degree), bool)
            motif_hits = valid_hits.filter(pl.col('id')==row['id'])
            carriers[motif_hits['enhancer_index'].unique().to_numpy()] = True
            comparisons, exact = [], []
            for split in ('train', 'validation', 'test'):
                specs = [(name,low,high,False) for name,low,high in GROUPS]
                specs += [(str(d),d,d,True) for d in range(1,9)]
                for name, low, high, is_exact in specs:
                    selected = (degree>=low) & (degree<=high) & (metadata['split']==split)
                    eligible = selected & converged
                    count = int((carriers & eligible).sum()); n = int(eligible.sum())
                    metric = group_summary(scores['best_fraction'], scores['null_best_fraction'],
                        scores['mean_importance'], metadata, target, low, high, split)
                    metric.update(group=name, split=split, finemo_carriers=count, finemo_n=n,
                        finemo_fraction=count/n if n else None,
                        finemo_nonconverged=int((selected & metadata['quality_pass'][:,target] & ~converged).sum()))
                    (exact if is_exact else comparisons).append(metric)
                    for threshold in metric['thresholds']:
                        flattened.append(dict(target=TARGETS[target], motif=row['id'], sign=row['sign'],
                            origin=group['name'], rank=row['rank'], split=split, group=name,
                            grouping='exact' if is_exact else 'nonoverlap', **threshold,
                            null_counts=json.dumps(threshold['null_counts']),
                            discovery_carriers=row['attribution_support']['hits'],
                            discovery_n=row['attribution_support']['n'],
                            finemo_carriers=count, finemo_n=n, finemo_fraction=metric['finemo_fraction'],
                            finemo_nonconverged=metric['finemo_nonconverged'],
                            attribution_carriers=metric['attribution_carriers'],
                            quality_excluded=metric['quality_excluded'],
                            mean_IG_at_80=metric['mean_importance']))
            summaries.append(dict(id=row['id'], origin=group['name'], sign=row['sign'],
                rank=row['rank'], discovery_support=row['attribution_support'],
                groups=comparisons, exact_degrees=exact, finemo_instances=len(motif_hits)))
            path = result/(row['id'].replace('/','__')+'_finemo_carriers.npz')
            np.savez_compressed(path, ids=metadata['ids'], carriers=carriers, converged=converged)
            files[path.name] = digest(path)
    summary = dict(target=TARGETS[target], rows=summaries,
        sequence_rule='Both strands; >= 0.8 * maximum log2 PWM odds, sensitivity 0.7/0.9; pseudocount mass .001.',
        null_rule='Ten paired dinucleotide-shuffled native sequences; descriptive baseline, no p/FDR claim.',
        finemo_rule='Projected CWM/IG; normalize retained CWM core, lambda .7 and cosine postfilter .7; native bounds; competitive within discovery group, never between groups.',
        importance_rule='Signed saved IG per base over union of >=80% scan hits, averaged over QC-passing carriers; noncarriers missing.',
        denominators='Sequence: all cohort enhancers. Fi-NeMo: saved-attribution QC pass AND converged per library. Cluster: original training discovery.',
        no_new_attributions=True, no_reclustering=True)
    write_json(result/'occurrence_summary.json', summary)
    with (result/'occurrence_metrics.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(flattened)
    for name in ('occurrence_summary.json', 'occurrence_metrics.tsv'): files[name] = digest(result/name)
    write_json(result/'summary_complete.json', dict(status='complete', files=files))
    event('occurrence_summary_complete', target=TARGETS[target], motifs=len(summaries))
    return audit, summary


def render(root, target, audit, summary):
    from .paper_figure3 import Figure, check_text_bounds, orient_pair
    config, parent, _ = read_inputs(root)
    matches = json.loads((parent/TARGETS[target]/'annotation/jaspar/matches.json').read_text())
    output = root/'output/pdf'; output.mkdir(parents=True, exist_ok=True)
    path = output/f'nonoverlap_{TARGETS[target]}_relative_pwm_finemo.pdf'
    if path.exists(): raise FileExistsError('Preserve existing PDF')
    figure = Figure(audit, path, root/'fonts')
    figure.c.setTitle('Native-enhancer motif occurrence | '+TARGETS[target])
    figure.c.setSubject('Native intervals; both contribution signs; non-overlapping breadth groups; frozen IG64 x100; sequence and predictive occurrence distinct.')
    metrics = {r['id']:r for r in summary['rows']}
    for group in audit['groups']:
        for sign in ('positive','negative'):
            label = str(group['minimum_breadth']) if group['minimum_breadth']==group['maximum_breadth'] else f"{group['minimum_breadth']}-{group['maximum_breadth']}"
            title = f"Degree of pleiotropy {label} | {sign}"
            figure.page(1050,680,title)
            figure.text(34,620,title,16,bold=True)
            figure.text(34,601,TARGETS[target]+' | native enhancer intervals | IG64 x 100 references',9)
            for x,text in [(38,'De novo PWM'),(200,'JASPAR match'),(364,'Match / q'),(490,'Discovery'),
                           (610,'Validation: 1'),(751,'Validation: 2-5'),(892,'Validation: 6-8')]:
                figure.text(x,573,text,9,bold=True)
            figure.text(610,557,'S: sequence >=80% score; N: shuffled null; F: Fi-NeMo. Bars: 0-100%.',7)
            rows = sorted((r for r in group['rows'] if r['sign']==sign), key=lambda r:r['rank'])[:5]
            color = '#2166AC' if sign=='positive' else '#B2182B'
            if not rows: figure.text(38,510,'No retained motif for this sign.',11)
            for number,row in enumerate(rows):
                y = 516-number*83
                match = matches['best'][row['id'].replace('/','__')]
                query,ref,qs,rs,span = orient_pair(dict(row,match=match)); step=min(11,140/span)
                figure.motif_logo(query,38+qs*step,y+4,step,height=24)
                figure.motif_logo(ref,200+rs*step,y+4,step,height=24)
                figure.text(38,y-11,f"#{row['rank']} {row['pattern'].split('/')[-1]}",8)
                figure.text(364,y+16,match['reference']['name'],10,bold=True)
                figure.text(364,y+1,f"q={match['q']:.3g}"+(' ns' if match['q']>=.05 else ''),8)
                support = row['attribution_support']
                figure.text(490,y+16,f"{100*support['fraction']:.1f}%",10)
                figure.text(490,y+1,f"{support['hits']}/{support['n']}",8)
                for x,(name,_,_) in zip((610,751,892),GROUPS):
                    value = next(m for m in metrics[row['id']]['groups'] if m['split']=='validation' and m['group']==name)
                    score = value['thresholds'][1]
                    for offset,fraction,tone in ((17,score['fraction'],color),(10,score['null_fraction'],'#999999'),
                                                 (3,value['finemo_fraction'],'#424242')):
                        figure.c.setFillColor(figure.color('#EEEEEE')); figure.c.rect(x,y+offset,118,4,fill=1,stroke=0)
                        if fraction is not None:
                            figure.c.setFillColor(figure.color(tone)); figure.c.rect(x,y+offset,118*fraction,4,fill=1,stroke=0)
                    f = value['finemo_fraction']
                    figure.text(x,y+27,f"S {100*score['fraction']:.1f}%  N {100*score['null_fraction']:.1f}%",7)
                    figure.text(x,y-10,('F NA' if f is None else f"F {100*f:.1f}%")+
                        f" ({value['finemo_carriers']}/{value['finemo_n']})",7)
                    importance = value['mean_importance']
                    figure.text(x,y-22,'IG/bp '+('NA' if importance is None else f'{importance:+.3g}'),7)
                figure.line(34,y-35,1016)
                figure.pages[-1]['patterns'].append(row['id'])
            figure.text(34,55,'Rank: original training cluster support. Logos: 0-2 bits. Sequence scans and Fi-NeMo support are not TF occupancy.',8)
            figure.text(34,42,'Fi-NeMo: lambda=0.7, signed CWMs, one competitive library per discovery group. Denominator excludes QC/optimizer failures.',8)
            figure.text(34,29,'Null: 10 dinucleotide shuffles per enhancer. Signed IG/bp: mean over sequence carriers. Full 70/80/90% results and counts: companion TSV.',8)
            figure.c.showPage()
    check_text_bounds(figure.text_bounds); figure.c.save()
    write_json(path.with_suffix('.source.json'), dict(target=TARGETS[target], pages=figure.pages,
        text_bounds=figure.text_bounds, summary_sha256=digest(root/TARGETS[target]/'occurrence_summary.json'),
        pdf_sha256=digest(path), visual_qa='pending manual local inspection'))
    write_json(root/TARGETS[target]/'figure_complete.json', dict(status='complete', file_base='experiment_root',
        files={str(p.relative_to(root)):digest(p) for p in (path,path.with_suffix('.source.json'))}))


def run(root):
    for target in range(2): render(root,target,*summarize(root,target))
