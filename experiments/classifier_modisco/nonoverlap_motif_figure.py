"""Top-five signed motifs and fixed-PWM cross-group validation comparisons."""
import argparse
import json
from pathlib import Path

from .common import digest, event, write_json
from .nonoverlap_motifs import GROUPS, LABELS, verify_receipt
from .dual_motif_pipeline import TARGETS
from .paper_figure3 import Figure, check_text_bounds, orient_pair


DATABASE_LABELS = {'jaspar':'JASPAR', 'flyfactorsurvey':'FlyFactorSurvey', 'flyreg':'FlyReg'}


def draw(audit, summary, matches, output, fonts, *, database='jaspar'):
    database_label=DATABASE_LABELS[database]
    figure=Figure(audit,output,fonts)
    figure.c.setTitle('Non-overlapping pleiotropy groups | '+summary['target'])
    if database!='jaspar':
        figure.c.setTitle('Non-overlapping pleiotropy groups | '+summary['target']+' | '+database_label)
    figure.c.setSubject('Native enhancers, IG64 x100; training-only discovery; fixed-PWM validation comparison')
    metrics={r['id']:r for r in summary['rows']}
    labels=dict(zip((g[0] for g in GROUPS),LABELS))
    for group in audit['groups']:
        for sign in ('positive','negative'):
            title=f"Degree of pleiotropy {labels[group['name']]} | {sign} contributions"
            figure.page(1000,650,title)
            figure.text(36,585,title,16,bold=True)
            figure.text(36,568,f"{summary['target']} | 100 references | training discovery n={group['discovery_elements']:,}",9)
            if audit.get('flank_note'):
                figure.text(580,568,audit['flank_note'],9)
            for x,text in [(42,'De novo PWM'),(214,database_label+' match'),(394,'Best match / q'),(524,'Cluster support'),
                           (642,'Validation: 1'),(750,'Validation: 2-5'),(858,'Validation: 6-8')]:
                figure.text(x,541,text,9,bold=True)
            figure.text(642,526,'Bars: sequence occurrence (0-100%); below: mean signed IG/bp',7)
            rows=sorted((r for r in group['rows'] if r['sign']==sign),key=lambda r:r['rank'])[:5]
            if not rows:
                figure.text(42,478,'No retained motif for this contribution sign.',11)
            for number,row in enumerate(rows):
                y=480-number*79
                match=matches['best'][row['id'].replace('/','__')]
                query,target,qs,ts,span=orient_pair(dict(row,match=match))
                step=min(12,145/span)
                figure.motif_logo(query,42+qs*step,y+7,step,height=24)
                figure.motif_logo(target,214+ts*step,y+7,step,height=24)
                figure.text(42,y-8,f"#{row['rank']}  {row['pattern'].split('/')[-1]}",8)
                name=match['reference']['name']
                size=10 if database=='jaspar' else min(10,116/figure.metrics.stringWidth(name,'AtlasBold',1))
                if size<7:raise ValueError('Reference label too long to display legibly: '+name)
                figure.text(394,y+21,name,size,bold=True)
                figure.text(394,y+5,f"q={match['q']:.3g}"+(' ns' if match['q']>=.05 else ''),8)
                support=row['attribution_support']
                figure.text(524,y+21,f"{100*support['fraction']:.1f}%",10)
                figure.text(524,y+5,f"{support['hits']}/{support['n']}",8)
                color='#2166AC' if sign=='positive' else '#B2182B'
                for x,(name,_,_) in zip((642,750,858),GROUPS):
                    metric=next(r for r in metrics[row['id']]['groups'] if r['split']=='validation' and r['group']==name)
                    fraction=metric['fraction']
                    if fraction is None:
                        figure.text(x,y+17,'Not quantifiable',7)
                    else:
                        figure.c.setFillColor(figure.color('#DDDDDD'));figure.c.rect(x,y+14,91,4,fill=1,stroke=0)
                        figure.c.setFillColor(figure.color(color));figure.c.rect(x,y+14,91*fraction,4,fill=1,stroke=0)
                        figure.text(x,y+24,f"{100*fraction:.1f}% ({metric['sequence_carriers']}/{metric['n']})",7)
                    value=metric['mean_importance']
                    figure.text(x,y-1,'IG: '+('NA' if value is None else f'{value:+.3g}'),8)
                    figure.text(x,y-13,f"carriers={metric['attribution_carriers']}",7,color=figure.MUTED)
                figure.line(36,y-28,964)
                figure.pages[-1]['patterns'].append(row['id'])
            note=('Motif logos: 0-2 bits. Support ranks are training-only; all panels use the same fixed PWMs and scan threshold.'
                if database=='jaspar' else
                'Motif logos: 0-2 bits. Ranks: training cluster support. Validation sequence scans: FIMO p<=1e-4; fixed de novo PWMs.')
            figure.text(36,55,note,8)
            figure.text(36,42,'Scan hits are not TF occupancy. IG is averaged over motif carriers; noncarriers are missing. Units: '+summary['units'],8)
            figure.text(36,28,'Group sizes are unequal; validation was previously examined. Negative patterns do not establish biological repression.',8)
            figure.c.showPage()
    check_text_bounds(figure.text_bounds)
    figure.c.save()
    return dict(pages=figure.pages,text_bounds=figure.text_bounds)


def render(project,root,config,target):
    result=root/TARGETS[target]
    verify_receipt(result,'report_complete.json');verify_receipt(result,'comparison_complete.json')
    audit=json.loads((result/'report_audit.json').read_text())
    summary=json.loads((result/'cross_group_summary.json').read_text())
    matches=json.loads((result/'annotation/jaspar/matches.json').read_text())
    output=root/'output/pdf';output.mkdir(parents=True,exist_ok=True)
    suffix=config.get('report_suffix','')
    path=output/f'nonoverlap_pleiotropy_{TARGETS[target]}_ig64_ref100{suffix}.pdf'
    if path.exists():raise FileExistsError('Preserve previous report')
    layout=draw(audit,summary,matches,path,project/config['assets']/'fonts')
    source=dict(target=TARGETS[target],groups=list(GROUPS),references=100,steps=64,**layout,
        filter_rule=audit['rules']['trim'],
        report_sha256=digest(result/'report_audit.json'),metrics_sha256=digest(result/'cross_group_summary.json'),
        matches_sha256=digest(result/'annotation/jaspar/matches.json'),pdf_sha256=digest(path),
        visual_qa='pending local page inspection')
    write_json(path.with_suffix('.source.json'),source)
    write_json(result/'figure_complete.json',dict(status='complete',files={
        str(p.relative_to(root)):digest(p) for p in (path,path.with_suffix('.source.json'))},
        file_base='experiment_root',visual_qa='pending'))
    event('nonoverlap_figure_complete',target=TARGETS[target],pdf=str(path),pages=len(layout['pages']))


def render_cached_databases(source_root, output, fonts):
    """Render four database variants from completed flank02 files only."""
    config=json.loads((source_root/'config.json').read_text())
    if config.get('flank_threshold')!=.2:raise ValueError('Expected the completed 0.2-bit sensitivity results')
    output.mkdir(parents=True,exist_ok=True)
    for target in TARGETS:
        result=source_root/target
        verify_receipt(result,'report_complete.json');verify_receipt(result,'comparison_complete.json')
        audit=json.loads((result/'report_audit.json').read_text())
        summary=json.loads((result/'cross_group_summary.json').read_text())
        if [g['name'] for g in audit['groups']] != [g[0] for g in GROUPS]:
            raise ValueError('Expected non-overlapping groups 1 / 2-5 / 6-8')
        expected={r['id'].replace('/','__') for g in audit['groups'] for r in g['rows']}
        for database in ('flyfactorsurvey','flyreg'):
            match_path=result/'annotation'/database/'matches.json'
            matches=json.loads(match_path.read_text())
            if matches['database']['key']!=database or set(matches['best'])!=expected:
                raise ValueError('Motif/database identifiers differ from frozen source')
            if matches['query_sha256']!=digest(result/'annotation/quality_passing_queries.meme'):
                raise ValueError('Annotation query changed')
            path=output/f'nonoverlap_pleiotropy_{target}_ig64_ref100_flanks02_{database}.pdf'
            if path.exists() or path.with_suffix('.source.json').exists():
                raise FileExistsError('Preserve existing output: '+str(path))
            layout=draw(audit,summary,matches,path,fonts,database=database)
            source=dict(target=target,database=matches['database'],groups=list(GROUPS),references=100,steps=64,
                **layout,filter_rule=audit['rules']['trim'],
                motif_ranking='Unchanged within-group original distinct discovery-enhancer support; top5 per sign.',
                occurrence_rule='Unchanged cached FIMO p<=1e-4 validation prevalence; not relative-PWM or Fi-NeMo support.',
                inputs={str(p.resolve()):digest(p) for p in (source_root/'config.json',result/'report_audit.json',
                    result/'cross_group_summary.json',match_path,result/'annotation/quality_passing_queries.meme')},
                renderer_sha256=digest(__file__),pdf_sha256=digest(path),visual_qa='pending manual page inspection',
                new_discovery=False,new_scanning=False,new_tomtom=False)
            write_json(path.with_suffix('.source.json'),source)
            event('nonoverlap_database_figure_complete',target=target,database=database,pdf=str(path),pages=len(layout['pages']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Render cached FlyFactorSurvey/FlyReg non-overlap PDFs; no new analysis.')
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fonts',type=Path,required=True)
    args=parser.parse_args()
    render_cached_databases(args.source_root,args.output,args.fonts)
