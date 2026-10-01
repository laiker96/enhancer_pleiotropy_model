"""Direct Trl/GAF versus Combgap sequence prevalence; no model attribution."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from .common import digest, event, require_allocation, write_json
from .original_sequence_report import parse_fimo_line
from .report_pdf import Atlas
from .tomtom_atlas import heights, read_meme

MOTIFS = ("MA0205.3", "MA2107.1")
NAMES = ("Trl / GAF", "Combgap / cg")
THRESHOLDS = (1e-5, 1e-4, 1e-3)
PRIMARY = 1e-4
COLORS = ("#CC6D25", "#176B79")


def wilson(k, n):
    if not n:
        return [None, None]
    if not 0 <= k <= n:
        raise ValueError("Invalid binomial count")
    z = 1.959963984540054
    p, denominator = k/n, 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0., center-half), min(1., center+half)]


def summarize(minimum_p, metadata):
    n = len(metadata["ids"])
    if (minimum_p.shape != (2, n) or not np.isfinite(minimum_p).all()
            or (minimum_p < 0).any() or (minimum_p > 1).any()):
        raise ValueError("Expected two aligned finite minimum-p tracks")
    rows = []
    for threshold in THRESHOLDS:
        hit = minimum_p <= threshold
        for split in ("all", "train", "validation", "test"):
            base = np.ones(n, bool) if split == "all" else metadata["split"] == split
            for degree in ("all", *range(1, 9)):
                mask = base if degree == "all" else base & (metadata["breadth"] == degree)
                size = int(mask.sum())
                a, b = hit[:, mask]
                counts = [int(a.sum()), int(b.sum())]
                rows.append(dict(split=split, degree=degree, threshold=threshold, n=size,
                    hits=counts, fraction=[k/size if size else None for k in counts],
                    ci95=[wilson(k, size) for k in counts],
                    both=int((a & b).sum()), gaf_only=int((a & ~b).sum()),
                    cg_only=int((b & ~a).sum()), neither=int((~a & ~b).sum()),
                    median_length=float(np.median(metadata["length"][mask])) if size else None,
                    higher="tie" if counts[0] == counts[1] else ("GAF" if counts[0] > counts[1] else "Combgap")))
    return rows


def render(root, summary, profiles, style):
    output = root/"output/pdf"; output.mkdir(parents=True)
    path = output/"gaf_combgap_prevalence_by_pleiotropy.pdf"
    report = Atlas(dict(groups=[], **style), path, root/"fonts")
    report.width, report.height = 1191, 842
    report.c.setPageSize((1191,842)); report.pages.append(dict(name="Fixed motif prevalence", patterns=[]))
    report.c.setTitle("Direct GAF and Combgap motif prevalence by exact activity breadth")
    report.c.setSubject("Fixed reference PWMs scanned in original enhancer intervals; no attribution or de novo discovery")
    report.text(36,795,"FIXED REFERENCE PWM SCAN  /  ORIGINAL ENHANCER INTERVALS",9,color=report.MUTED)
    report.text(36,757,"GAF or Combgap: which motif is more frequent?",27,bold=True)
    total = next(r for r in summary if r["threshold"]==PRIMARY and r["split"]=="all" and r["degree"]=="all")
    rows = [r for r in summary if r["threshold"]==PRIMARY and r["split"]=="all" and r["degree"]!="all"]
    report.text(36,732,f'{total["n"]:,} enhancers | Exact activity breadth 1-8 | FIMO site p <= 0.0001 | Both DNA strands',11,color=report.MUTED)
    for i,(ident,name) in enumerate(zip(MOTIFS,NAMES)):
        x=110+330*i
        report.text(x,698,name+"  ("+ident+")",12,bold=True,color=COLORS[i])
        report.logo(dict(heights=heights(profiles[ident]["pwm"])),x+10,632,17,48,coordinates=False)
    report.text(90,601,"Enhancers with at least one motif hit (%)",12,bold=True)
    x,y,w,h = 90,270,660,310
    for tick in range(0,101,20):
        at=y+h*tick/100
        report.line(x,at,x+w)
        report.text(x-12,at-3,str(tick),9,align="right",color=report.MUTED)
    for i,row in enumerate(rows):
        at=x+w*(i+.5)/8
        report.text(at,250,str(row["degree"]),11,bold=True,align="center")
        report.text(at,231,f'n={row["n"]:,}',8,align="center",color=report.MUTED)
    report.text(x+w/2,208,"Exact degree of pleiotropy (number of active contexts)",11,align="center")
    for j,color in enumerate(COLORS):
        points=[]
        report.c.setStrokeColor(report.color(color));report.c.setFillColor(report.color(color))
        for i,row in enumerate(rows):
            if not row["n"]: continue
            at=x+w*(i+.5)/8+(-2 if j==0 else 2)
            yy=y+h*row["fraction"][j]
            lo,hi=[y+h*v for v in row["ci95"][j]]
            report.c.setLineWidth(1)
            report.c.line(at,lo,at,hi);report.c.line(at-3,lo,at+3,lo);report.c.line(at-3,hi,at+3,hi)
            points.append((at,yy));report.c.circle(at,yy,3,stroke=0,fill=1)
        report.c.setLineWidth(2)
        for a,b in zip(points,points[1:]):report.c.line(*a,*b)
    report.text(800,669,"Both motifs can occur in one enhancer",12,bold=True)
    columns=(810,900,992,1070,1150)
    for xx,label in zip(columns,("Degree","N","GAF %","Cg %","Both %")):
        report.text(xx,625,label,10,bold=True,align="right")
    for i,row in enumerate(rows+[total]):
        yy=591-i*29
        values=(str(row["degree"]),f'{row["n"]:,}',
            *[f'{100*v:.1f}' if v is not None else 'NA' for v in row["fraction"]],
            f'{100*row["both"]/row["n"]:.1f}' if row["n"] else 'NA')
        for xx,value in zip(columns,values):report.text(xx,yy,value,10,align="right",bold=i==8)
    report.line(785,350,1155)
    report.text(800,307,"Overall: "+total["higher"]+" more frequent" if total['higher']!='tie' else "Overall: equal frequencies",13,bold=True)
    report.paragraph(800,282,
        f'GAF: {total["hits"][0]:,}/{total["n"]:,}; Combgap: {total["hits"][1]:,}/{total["n"]:,}. '
        f'Both: {total["both"]:,}; GAF only: {total["gaf_only"]:,}; Combgap only: {total["cg_only"]:,}.',350,10,15)
    winners=[next(r for r in summary if r['threshold']==t and r['split']=='all' and r['degree']=='all')['higher'] for t in THRESHOLDS]
    notes=[
        "Dots: distinct-enhancer fractions; error bars: descriptive 95% Wilson intervals. Multiple sites and reverse-strand hits count once per enhancer per motif. All enhancers are shown; separate training/validation/test counts are saved.",
        "Fixed JASPAR matrices, untrimmed: Trl/GAF 9 bp; Combgap 11 bp. Same training-derived RC-symmetric background and pseudocount0.1 for both. Site p is not enhancer-level FDR; equal p cutoffs do not guarantee equal detection power for different PWMs.",
        "Overall higher frequency at p=1e-5 / 1e-4 / 1e-3: "+" / ".join(winners)+". All threshold-by-degree counts are saved. Longer enhancers offer more chances for a hit; this is a descriptive prevalence plot, not a length-adjusted enrichment test.",
        "This comparison uses no model, attributions, TF-MoDISco motifs or Tomtom assignments. Sequence similarity alone does not establish TF binding, functional importance or greater contribution to pleiotropy."]
    yy=165
    for note in notes:yy=report.paragraph(36,yy,note,1119,9,12)-8
    report.text(36,26,"Matrices: jaspar.elixir.no/matrix/MA0205.3/ | jaspar.elixir.no/matrix/MA2107.1/",8,color=report.MUTED)
    report.c.showPage();report.c.save()
    write_json(path.with_suffix('.layout.json'),dict(pages=report.pages,text_bounds=report.text_bounds,visual_qa='pending'))


def main(project, root):
    require_allocation("cpu")
    config=json.loads((root/"config.json").read_text())
    for name,expected in config["inputs"].items():
        if digest(project/name)!=expected:raise ValueError("Changed frozen input: "+name)
    previous=project/config["sequence_report"]
    database=root/"jaspar2026_insects.meme"
    profiles=read_meme(database)
    if [profiles[k]['name'] for k in MOTIFS]!=['Trl','cg']:raise ValueError('Wrong fixed reference profiles')
    with np.load(previous/"cohort_metadata.npz",allow_pickle=False) as saved:metadata=dict(saved)
    n=len(metadata['ids'])
    if (n!=40338 or len(np.unique(metadata['ids']))!=n
            or not np.isin(metadata['breadth'],range(1,9)).all()
            or not np.isin(metadata['split'],['train','validation','test']).all()
            or (metadata['length']<=0).any()):raise ValueError('Invalid enhancer cohort')
    binary=root/'bin/fimo'
    if subprocess.check_output([str(binary),'--version'],text=True).strip()!='5.5.9':raise ValueError('Expected FIMO5.5.9')
    command=[str(binary),'--text','--thresh',str(max(THRESHOLDS)), '--motif-pseudo','0.1',
        '--bgfile',str(previous/'training_background.txt'),'--motif',MOTIFS[0],'--motif',MOTIFS[1],
        str(database),str(previous/'enhancers.fa')]
    best=np.ones((2,n));lookup={ident:i for i,ident in enumerate(MOTIFS)}
    widths=[len(profiles[k]['pwm']) for k in MOTIFS]
    event('fixed_pwm_scan_start',enhancers=n,motifs=list(MOTIFS),widths=widths)
    with (root/'stderr.log').open('x') as err,gzip.open(root/'hits.tsv.gz','wt') as archive:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err,text=True)
        try:
            header=next(process.stdout)
            if not header.startswith('motif_id\tmotif_alt_id\tsequence_name'):raise ValueError('Unexpected FIMO header')
            archive.write(header)
            for line in process.stdout:
                archive.write(line)
                if line.strip() and not line.startswith('#'):
                    parse_fimo_line(line,lookup,best,metadata['length'],widths)
            if process.wait():raise RuntimeError('FIMO failed')
        finally:
            if process.poll() is None:process.terminate();process.wait()
    summary=summarize(best,metadata)
    np.savez_compressed(root/'minimum_p.npz',minimum_p=best,motif_ids=np.array(MOTIFS),**metadata)
    report=dict(summary=summary,motifs=[profiles[k] for k in MOTIFS],command=command,inputs=config['inputs'],
        thresholds=THRESHOLDS,primary=PRIMARY,interval='original enhancer',model_used=False,
        definition='Distinct enhancers with >=1 fixed-PWM FIMO hit, both strands; one vote per enhancer per motif',
        limitations=['Motif widths/information and detection power differ','Longer enhancers offer more hit opportunities',
            'No TF binding or causal attribution implied','Site p is not enhancer-level FDR','All-cohort descriptive summary; split-specific counts also saved'])
    write_json(root/'summary.json',report)
    from .report_breadth import COLORS as BASE_COLORS,GLYPHS
    render(root,summary,profiles,dict(colors=BASE_COLORS,glyphs=GLYPHS))
    files=[root/'summary.json',root/'minimum_p.npz',root/'hits.tsv.gz']+list((root/'output/pdf').glob('*'))
    write_json(root/'complete.json',dict(status='complete',job=os.environ['SLURM_JOB_ID'],
        files={str(p.relative_to(root)):digest(p) for p in files},visual_qa='pending'))
    event('fixed_pwm_scan_complete',primary_overall=next(r for r in summary if r['threshold']==PRIMARY and r['split']=='all' and r['degree']=='all'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',required=True,type=Path);parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();main(args.project.resolve(),args.root.resolve())
