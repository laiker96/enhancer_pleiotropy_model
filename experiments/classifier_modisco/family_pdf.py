"""Vector PDF documenting signed motif audits, conservative families and matches."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .families import pair_key, choose_alignment
from .paper_motifs import PaperFigure, short_name
from .report_pdf import group_label, sha
from .tomtom_atlas import DATABASES, aligned_matrices, query_id, save_json


def chunks(items, size):
    return [items[i:i+size] for i in range(0, len(items), size)]


def compact_id(row):
    group = "1" if row["group"] == "exact_1" else ">="+row["group"].split("_")[-1]
    return group+" / "+("pos" if row["source_sign"] == "positive" else "neg")+row["pattern"].split("_")[-1]


class FamilyReport(PaperFigure):
    def __init__(self, data, results, pairs, output, font_dir):
        super().__init__(data, output, font_dir)
        self.results, self.pairs = results, pairs
        self.rows = {r["id"]:r for g in data["groups"] for r in g["rows"]}
        self.c.setTitle("Signed motif families across enhancer activity breadth")
        self.c.setSubject("Sign audit; conservative post-hoc families; enhancer unions; reference matches; full inventory")

    def begin(self, title, subtitle="", wide=False):
        self.start(1190 if wide else 595, 842, title+f" [{len(self.pages)+1}]")
        self.text(28, 811, "ENHANCER PLEIOTROPY  /  REPRODUCIBLE ANALYSIS", 8, color=self.MUTED)
        self.text(28, 777, title, 19 if wide else 17, bold=True)
        if subtitle:
            self.paragraph(28, 757, subtitle, self.width-56, 9, 13, self.MUTED)
        return 714

    def end(self):
        self.line(28, 42, self.width-28)
        self.text(28, 25, "Training-discovery support, not enrichment, occupancy or causal regulation.", 7, color=self.MUTED)
        self.text(self.width-28, 25, str(len(self.pages)), 8, align="right")
        self.c.showPage()

    def section(self, y, title, body):
        lines, line = 1, ""
        for word in body.split():
            candidate=(line+" "+word).strip()
            if self.metrics.stringWidth(candidate,"Atlas",10)>self.width-56 and line:
                lines+=1;line=word
            else:line=candidate
        if y-19-lines*14-22 < 60:
            self.end()
            y=self.begin("Methods and interpretation / continued")
        self.text(28, y, title, 12, bold=True, color=self.TEAL)
        return self.paragraph(28, y-19, body, self.width-56, 10, 14)-22

    def overview(self):
        y = self.begin("Signed motif families", "Sign audit, conservative grouping, reference matches and complete inventories | 17 September 2026")
        s = self.data["summary"]
        y = self.section(y, "The result", f"The 220 original group-specific patterns comprise 122 positive and 98 negative discovery clusters. The same sequence-information criterion retains 131 cores: 79 from positive clusters and 52 from negative clusters. After a separate sign audit, 79 positive and 47 negative patterns form {s['families']['positive']} positive and {s['families']['negative']} negative conservative families. Four discordant and one mixed-sign patterns remain visible but are withheld from the primary family panels.")
        y = self.section(y, "What was missing before", "Negative patterns were present in the saved TF-MoDISco results but excluded by the positive-only reporting layer. They were not deleted or absent from discovery. This report includes their quality filtering, signed contributions, reference matching and family assignments. The breadth-8 fit contains no retained negative discovery patterns; this is not proof that ubiquitous enhancers lack suppressive sequence features.")
        y = self.section(y, "What changed, and what did not", "This is a post-hoc grouping of existing motifs, not a new TF-MoDISco run. Original sequences, attributions, seqlets, cluster memberships and earlier reports are untouched. Family representation uses the union of supporting enhancers within each group. No new model inference, training, CUDA execution, cluster submission or external data transfer was required.")
        y = self.section(y, "How to read the figures", "Positive and negative family panels are separate. A family is represented by an existing medoid motif, not an averaged synthetic consensus. Sequence logos show information in bits; contribution logos show signed attribution in logit units with an explicitly labelled per-logo scale. Heatmaps display assigned-enhancer percentages, using one common 0-35% scale. Zeros mean no assigned family pattern in that fit, not verified motif absence.")
        y = self.section(y, "Navigation", "Next: the sign audit and flagged cases; reproducible methods and threshold sensitivity; positive and negative overview panels; a complete 58-family catalogue with three reference databases; membership/sign statistics for all 131 passing patterns; all 89 sequence-quality exclusions; provenance, checks and manuscript wording.")
        self.end()

    def sign_audit(self):
        y = self.begin("Sign audit: discovery label versus final core", "Core scores average sites within each supporting enhancer before averaging enhancers. A stable sign requires at least 75% agreement.")
        headers = ["Group", "Original + / -", "QC + / -", "Stable + / -", "Review"]
        xs = [28, 112, 225, 331, 453]
        for x, h in zip(xs, headers): self.text(x, y, h, 9, bold=True)
        y -= 24
        for g in self.data["groups"]:
            orig = Counter(r["source_sign"] for r in g["rows"])
            qc = Counter(r["source_sign"] for r in g["rows"] if r["quality"]["passed"])
            pol = Counter(r["polarity"] for r in g["rows"])
            values = [group_label(g), f"{orig['positive']} / {orig['negative']}",
                f"{qc['positive']} / {qc['negative']}", f"{pol['positive']} / {pol['negative']}",
                str(pol['mixed']+pol['discordant'])]
            for x, v in zip(xs, values): self.text(x,y,v,10)
            self.line(28,y-8,567); y -= 30
        y -= 12
        y = self.section(y,"Eight full-window sign mismatches; five passing patterns require review", "Eight of 98 negative-labelled patterns have a positive full-window mean contribution. Three fail the sequence-information filter. Of the five passing cases, four retain a positive core; one has a negative core mean but only 61.9% of supporting enhancers are negative. No positive-labelled passing pattern changes sign in this audit.")
        y = self.section(y,"Why the original label is not enough", "The label comes from the original positive/negative TF-MoDISco branch. In the pinned 2.5.2 implementation, early sign selection and checks precede subsequent motif aggregation, trimming and final pattern operations. Our previous summary retained that branch label while averaging the full saved window. The sign discrepancy is confirmed directly in stored seqlet contributions, not inferred from motif names. The exact historical operation responsible for each case is not recorded, so no specific pipeline step is claimed as the proven cause.")
        y = self.section(y,"What negative attribution means", "A negative core opposes the selected model score relative to the attribution baseline. Here that score is the mean forward/reverse-complement logit over each enhancer's observed-active contexts. It is not a direct activity-breadth score. Negative attribution does not by itself establish a silencer, a repressor TF or a causal effect on pleiotropy.")
        self.end()

    def signed_logo(self, scores, x, y, step, halfheight):
        plus = [sum(max(0,v) for v in col) for col in scores]
        minus = [sum(max(0,-v) for v in col) for col in scores]
        bound = max(plus+minus+[1e-12])
        scale = math.ceil(bound*10**(1-math.floor(math.log10(bound))))/10**(1-math.floor(math.log10(bound)))
        self.line(x, y, x+step*len(scores), self.MUTED)
        self.text(x-4,y+halfheight-2,f"+{scale:.2g}",6,align="right",color=self.MUTED)
        self.text(x-4,y-halfheight,f"-{scale:.2g}",6,align="right",color=self.MUTED)
        for pos, values in enumerate(scores):
            up = down = y
            for base in sorted(range(4),key=lambda i:(abs(values[i]),i)):
                value=values[base]; h=abs(value)*halfheight/scale
                if h <= 1e-8: continue
                if value>0: bottom=up; up+=h
                else: down-=h; bottom=down
                self.c.saveState(); self.c.translate(x+pos*step,bottom+h)
                self.c.scale((step-.35)/100,-h/100)
                self.c.setFillColor(self.color(self.data['colors'][base]))
                self.c.drawPath(self.glyphs[base],stroke=0,fill=1,fillMode=0);self.c.restoreState()
        return scale

    def flags(self):
        self.begin("Five patterns withheld from primary families", "Discordant means the core mean reverses the original branch label. Mixed means less than 75% within-enhancer sign agreement. No automatic relabelling.", wide=True)
        flags=[r for r in self.rows.values() if r['polarity'] in ('discordant','mixed')]
        for i,r in enumerate(flags):
            top=712-i*125;self.line(28,top+6,1162)
            self.text(28,top-12,r['id'],11,bold=True)
            s=r['signed'];lo,hi=s['ci95']
            self.text(28,top-31,f"{r['polarity'].upper()} | source: negative | n={r['supporting_enhancers']}",9)
            self.text(28,top-48,f"Full mean/bp {r['original_full_mean']:+.4f}; core {s['mean']:+.4f}",9)
            self.text(28,top-65,f"Core 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]",9)
            self.text(28,top-82,f"Enhancers: {100*s['positive_fraction']:.1f}% positive; {100*s['negative_fraction']:.1f}% negative",9)
            self.text(443,top-13,"Sequence core (0-2 bits)",9,bold=True)
            self.text(676,top-13,"Signed attribution (own scale)",9,bold=True)
            step=min(12,175/len(r['trimmed_pwm']))
            self.motif(r['trimmed_pwm'],443,top-85,step,36,axis=True)
            self.signed_logo(r['core_contributions'],695,top-64,step,27)
            m=self.results['jaspar']['best'][query_id(r)]
            self.text(919,top-13,'JASPAR best core match',9,bold=True)
            self.text(919,top-32,f"{short_name(m['reference'])}  q={m['q']:.2g}"+(' NS' if m['q']>.05 else ''),9)
            self.text(919,top-48,m['target_id'],8,color=self.MUTED)
            target,_,_,_,_=aligned_matrices(r['trimmed_pwm'],m['reference']['pwm'],m['offset'],m['orientation'])
            self.motif(target,919,top-96,min(9,220/len(target)),36)
            self.pages[-1]['patterns'].append(r['id'])
        self.end()

    def methods(self):
        y=self.begin("Methods: inputs, trimming and signed scores")
        sections=[
            ("Frozen model and discovery scope", "The legacy fine-tuned dilated CNN cnn_finetune_20260914, best checkpoint epoch 38, supplied the existing attributions. It was selected by validation performance at analysis launch, not newly selected here. Integrated Gradients used two seeded dinucleotide-shuffled full-2048-bp references, 32 integration steps and forward/RC averaging over observed-active context logits. TF-MoDISco used central 512-bp training windows. This report does not recompute any attribution."),
            ("Eight separate discovery groups", "Exactly 1 context is the reference group; >=2 through >=8 are cumulative, overlapping groups. All eligible training enhancers were used within each group, without balancing. The frozen discovery fits used TF-MoDISco 2.5.2, a full-fit cap of 20,000 seqlets per sign and target seqlet FDR 0.05. A pattern ID belongs to one fit; equal IDs in different groups do not imply equal motifs. Original full-fit files were verified against completion hashes."),
            ("Run-anchored cores", "Require a run of at least five consecutive columns with information strictly greater than 0.2 bits against uniform A/C/G/T background. Keep the contiguous interval from the first qualifying run's beginning through the last qualifying run's end. Internal positions remain intact. Sequence-quality filtering is independent of attribution sign. No renormalization beyond valid probability rows, small-sample correction, base deletion or reference-motif cropping is applied."),
            ("Enhancer-weighted attribution", "For each saved pattern, retain its aligned, already strand-oriented seqlets. Average their actual contribution matrices within each supporting enhancer; then average enhancer matrices, giving every enhancer one vote. The core mean/bp is the sum of the resulting matrix divided by core width. Record mean, median, positive/negative enhancer fractions, and a 95% percentile bootstrap interval from 2,000 enhancer resamples. The seed is fixed and pattern-specific. These descriptive intervals ignore discovery selection and are not multiplicity-adjusted significance tests."),
            ("Primary polarity rule", "A pattern enters the positive or negative family analysis only if its core mean agrees with its original discovery branch and >=75% of its supporting enhancers have that sign. A reversal is flagged discordant; lesser consistency is mixed. Zero contributions count toward neither sign. Review patterns retain their original labels, statistics and reference matches but do not merge into primary families. Bootstrap intervals are reported, not used to choose the sign threshold.")]
        for title,body in sections:y=self.section(y,title,body)
        self.end()
        y=self.begin("Methods: conservative family grouping")
        sections=[
            ("Sequence and contribution agreement", "Compare every pair within the same accepted polarity over both strands and all ungapped offsets. Require >=6 aligned bases, or the full shorter width when it is only 5 bp; >=70% of the shorter width and >=50% of the longer width; >=75% of each motif's total information in the overlap; and at least five aligned positions where both motifs exceed 0.2 bits."),
            ("Protecting distinguishing bases", "Reject an alignment if a position has different modal bases while both motifs give their modal base probability >=0.60 and information >=0.40 bits. Among the remaining alignments, require PWM similarity >=0.85 and signed contribution similarity >=0.70. PWM similarity is cosine after subtracting 0.25 from each probability, equivalent to Pearson for valid overlapping probability matrices; contribution similarity is cosine of the flattened, enhancer-weighted actual contribution matrices. Amplitude is not used as a similarity criterion."),
            ("No similarity-chain merging", "Use deterministic complete-link agglomeration: every cross-cluster pair must have an admissible alignment that passes both thresholds. Merge the candidate with greatest minimum pair similarity, where pair similarity is the mean of its PWM and contribution similarities; ties use sorted IDs. Alignment ties maximize the smaller similarity, then PWM, contribution, overlap and deterministic strand/offset preferences. A directly similar pair can remain in separate families because other members are incompatible. These are conservative subfamilies, not unique biological TF classes."),
            ("Representative and family support", "The medoid is the existing member with greatest mean pair similarity to the others; ties prefer greater support then ID. Keep all member PWMs and seqlet assignments. Within each discovery group, family support is the union of member enhancer indices. Never sum overlapping pattern counts, and never combine group-local indices across groups. Families are ordered by their maximum within-group representation, not by a pooled count across nested cohorts."),
            ("Sensitivity, not parameter optimization", "Repeat grouping with relaxed PWM/contribution thresholds 0.80/0.60 and strict 0.90/0.80, keeping overlap, information and conflict rules fixed. The primary 0.85/0.70 thresholds were chosen before inspecting resulting families. These are exploratory similarity cutoffs, not calibrated probabilities, biological boundaries or a replacement for held-out validation. Known TF names and Tomtom q-values do not influence any family assignment.")]
        for title,body in sections:y=self.section(y,title,body)
        self.end()

    def sensitivity(self):
        y=self.begin("Sensitivity and the ubiquitous GA-repeat pair")
        for x,h in zip([28,145,265,393],['PWM / contrib.','Positive families','Negative families','Total']):self.text(x,y,h,9,bold=True)
        y-=27
        for s in self.data['sensitivity']:
            p,n=s['signs']['positive']['families'],s['signs']['negative']['families']
            for x,v in zip([28,145,265,393],[f"{s['pwm']:.2f} / {s['contribution']:.2f}",p,n,p+n]):self.text(x,y,v,11)
            y-=29
        y-=14
        a,b='ge_8/pos_patterns/pattern_0','ge_8/pos_patterns/pattern_2'
        m=choose_alignment(self.pairs[pair_key(a,b)],.85,.70)
        y=self.section(y,"Strong direct match, separate complete-link subfamilies",f"Breadth-8 patterns 0 and 2 have an admissible {m['overlap']}-bp alignment with PWM similarity {m['pwm']:.3f}, contribution cosine {m['contribution']:.3f}, and information coverage {100*m['information_coverage'][0]:.1f}% / {100*m['information_coverage'][1]:.1f}%. They belong to {self.rows[a]['family']} and {self.rows[b]['family']}, respectively. Their direct similarity is real; the global complete-link rule keeps them separate because not every member of the two expanded families is compatible.")
        fa=next(f for f in self.data['families'] if a in f['members']);fb=next(f for f in self.data['families'] if b in f['members'])
        incompatible=sum(choose_alignment(self.pairs[pair_key(x,z)],.85,.70) is None for x in fa['members'] for z in fb['members'])
        y=self.section(y,"Do not equate these families with Trl versus CLAMP",f"There are {incompatible} incompatible cross-family member pairs out of {len(fa['members'])*len(fb['members'])}. Different best-reference names are not the reason for the split. The same GA-repeat class can span these conservative families. A 14-bp CLAMP reference and a 9-bp Trl reference can receive different Tomtom rankings as query cores and alignments change. Neither assignment establishes which TF binds the enhancer.")
        y=self.section(y,"Example of why enhancer unions matter", "Family P05 contains two breadth-8 patterns whose individual support counts sum to 54, but their union is 51 enhancers. Family N01 has a summed exactly-one-context support of 997 but a union of 961. The corrected counts are used throughout the heatmaps and catalogue. These examples concern cluster-assigned support, not sequence scanning or enrichment.")
        y=self.section(y,"Interpretation boundary", "Motif similarity and representation are descriptive. Group sizes, overlapping cohorts, finite seqlet caps and independent discovery fits affect what is recovered. A family missing from a group's clusters may still occur in its sequences. We have not tested differential enrichment between disjoint groups, inferred a monotonic pleiotropy effect, or validated a direct TF-specific causal role.")
        self.end()

    def overview_panel(self, sign):
        self.begin(sign.capitalize()+" contribution families", "Top 12 conservative families by maximum within-group representation. Logos are single existing medoids, not pooled family consensuses.", wide=True)
        families=[f for f in self.data['families'] if f['polarity']==sign][:12]
        for x,label in [(28,'Family / medoid'),(181,'Sequence: 0-2 bits'),(409,'Signed attribution: own logit scale'),(685,'Distinct-enhancer union (%)')]:self.text(x,718,label,10,bold=True)
        for i,g in enumerate(self.data['groups']):self.text(685+i*58+26,696,group_label(g),9,align='center')
        for i,f in enumerate(families):
            top=684-i*49;self.line(28,top,1162);r=self.rows[f['representative']]
            self.text(28,top-17,f['id']+f"  ({len(f['members'])} patterns)",10,bold=True)
            self.text(28,top-32,compact_id(r),8,color=self.MUTED)
            step=min(9,175/len(r['trimmed_pwm']))
            self.motif(r['trimmed_pwm'],187,top-41,step,31)
            self.signed_logo(r['core_contributions'],444,top-25,step,17)
            for j,g in enumerate(self.data['groups']):
                s=f['group_support'][g['name']];fraction=s['fraction'];x=685+j*58
                amount=min(1,fraction/.35)
                base=(.11,.43,.49) if sign=='positive' else (.62,.27,.14)
                self.c.setFillColorRGB(*[1-amount*(1-v) for v in base]);self.c.rect(x,top-43,54,35,stroke=0,fill=1)
                self.text(x+27,top-25,f"{100*fraction:.1f}" if s['count'] else '0',9,align='center',color='#FFFFFF' if amount>.65 else self.INK)
            self.pages[-1]['patterns'].append(f['id'])
        self.text(28,73,"Shared heatmap scale: 0-35%. Groups >=2..>=8 overlap. Zero means no assigned family member, not tested sequence absence.",9)
        self.text(28,57,"Sequence and attribution coordinates match within each row. Contribution amplitudes use labelled per-logo scales; compare numbers, not visual height.",9)
        self.end()

    def catalogue(self):
        for sign in ('positive','negative'):
            families=[f for f in self.data['families'] if f['polarity']==sign]
            for ci,batch in enumerate(chunks(families,5),1):
                self.begin(f"Complete {sign} family catalogue / {ci}","Each row shows one medoid. Full reference PWMs are aligned to that core; all match q-values belong to the medoid, not the entire family.",wide=True)
                for i,f in enumerate(batch):
                    top=717-i*129;self.line(28,top+4,1162);r=self.rows[f['representative']]
                    self.text(28,top-12,f"{f['id']} | {len(f['members'])} pattern(s) | medoid {r['id']}",11,bold=True)
                    self.text(28,top-34,f"Core width: {len(r['trimmed_pwm'])} bp",9)
                    self.text(28,top-49,f"Mean/bp: {r['signed']['mean']:+.4f}",9)
                    self.text(28,top-64,f"Sign agreement: {100*r['signed'][sign+'_fraction']:.1f}%",9)
                    pair_label=f"{f['minimum_pair_similarity']:.3f}" if len(f['members'])>1 else "n/a (singleton)"
                    self.text(28,top-79,"Min pair score: "+pair_label,9)
                    self.text(220,top-34,'Sequence core (0-2 bits)',8,bold=True)
                    self.text(414,top-34,'Signed attribution (own scale)',8,bold=True)
                    matches=[self.results[k]['best'][query_id(r)] for k in DATABASES]
                    aligned=[aligned_matrices(r['trimmed_pwm'],m['reference']['pwm'],m['offset'],m['orientation']) for m in matches]
                    anchor=max(a[1] for a in aligned);span=max(anchor-a[1]+a[3] for a in aligned);step=min(7,155/span)
                    self.motif(r['trimmed_pwm'],220+anchor*step,top-94,step,32)
                    self.signed_logo(r['core_contributions'],442+anchor*step,top-75,step,21)
                    for x,key,m,a in zip([628,809,993],DATABASES,matches,aligned):
                        self.text(x,top-34,dict(flyfactorsurvey='FlyFactorSurvey',flyreg='FlyReg',jaspar='JASPAR')[key],8,bold=True)
                        self.text(x,top-48,f"{short_name(m['reference'])} q={m['q']:.2g}"+(' NS' if m['q']>.05 else ''),8)
                        self.text(x,top-60,m['target_id'],7,color=self.MUTED)
                        t,qs,ts,_,overlap=a;assert overlap==m['overlap']
                        self.motif(t,x+(anchor-qs+ts)*step,top-96,step,32)
                    counts=' | '.join(("1" if g['name']=='exact_1' else ">="+str(g['minimum_breadth']))+f": {f['group_support'][g['name']]['count']}/{g['n']}" for g in self.data['groups'])
                    self.text(28,top-117,'Enhancer unions  '+counts,8)
                    self.pages[-1]['patterns'].append(f['id'])
                self.end()

    def inventory(self):
        rows=sorted([r for r in self.rows.values() if r['quality']['passed']],key=lambda r:(r.get('family','ZZ'),r['id']))
        for ci,batch in enumerate(chunks(rows,14),1):
            self.begin(f"All passing pattern memberships / {ci}","Every member variant: sequence core (0-2 bits), signed contribution logo (own logit scale), support, original sign and core statistics.",wide=True)
            for i,r in enumerate(batch):
                x=28+(i//7)*574;top=717-(i%7)*90
                s=r['signed'];m=self.results['jaspar']['best'][query_id(r)];lo,hi=s['ci95']
                self.line(x,top+5,x+548)
                self.text(x,top-8,r['id']+' | '+r.get('family','REVIEW'),9,bold=True)
                self.text(x,top-23,f"Source {r['source_sign']}; n={r['supporting_enhancers']}; core {len(r['trimmed_pwm'])} bp; sign agreement {100*s[r['source_sign']+'_fraction']:.1f}%",8)
                self.text(x+5,top-37,'Sequence',7,color=self.MUTED)
                self.text(x+196,top-37,'Signed attribution',7,color=self.MUTED)
                self.text(x+367,top-39,f"Mean/bp {s['mean']:+.4f}",7)
                self.text(x+367,top-51,f"95% CI {lo:+.3f}..{hi:+.3f}",7)
                self.text(x+367,top-63,f"JASPAR {short_name(m['reference'])} q={m['q']:.2g}"+(' NS' if m['q']>.05 else ''),7)
                self.text(x+367,top-75,m['target_id'],7,color=self.MUTED)
                step=min(7,140/len(r['trimmed_pwm']))
                self.motif(r['trimmed_pwm'],x+5,top-78,step,30)
                self.signed_logo(r['core_contributions'],x+206,top-60,step,16)
                self.pages[-1]['patterns'].append(r['id'])
            self.text(28,66,"Sign agreement is relative to the original source sign. REVIEW denotes a discordant or mixed-sign case, detailed earlier.",9)
            self.end()
        rows=sorted([r for r in self.rows.values() if not r['quality']['passed']],key=lambda r:r['id'])
        for ci,batch in enumerate(chunks(rows,32),1):
            self.begin(f"All sequence-quality exclusions / {ci}","These patterns remain in the original files. They are not assigned families or searched against reference databases in this analysis.",wide=True)
            xs=[28,390,497,610,755,948];headers=['Original group/pattern','Source sign','Support','Longest IC run','Full mean/bp','Required run']
            for x,h in zip(xs,headers):self.text(x,716,h,9,bold=True)
            for i,r in enumerate(batch):
                y=693-i*19
                for x,v in zip(xs,[r['id'],r['source_sign'],r['supporting_enhancers'],r['quality']['longest_run'],f"{r['original_full_mean']:+.5f}",'5 consecutive >0.2 bits']):self.text(x,y,v,9)
                self.line(28,y-6,1162);self.pages[-1]['patterns'].append(r['id'])
            self.end()

    def provenance(self, root):
        y=self.begin("Reference matching and interpretation")
        y=self.section(y,"Fresh searches on all 131 passing cores","Tomtom 5.5.9; Pearson distance; minimum overlap 5 bp; pseudocount 0.1; both target strands; complete scores including unaligned columns; reporting threshold q <=1. Queries retain original seqlet counts as nsites and uniform DNA background. Best hit means minimum p-value, then q-value, then ID. All 131 cores, including five review cases, were searched independently against all three reference databases. Only medoid matches represent families in the catalogue.")
        for key, result in self.results.items():
            count=sum(m['q']<=.05 for m in result['best'].values())
            y=self.section(y,result['database']['name'],f"{result['database']['count']} reference matrices; {count}/131 best matches at q <=0.05. Release: {result['database']['release']}. Reference logos retain their complete supplied probability matrices. NS marks a non-significant best hit, not evidence for a TF assignment.")
        y=self.section(y,"What q-values do not establish","Tomtom q-values quantify motif-similarity matching within a query/database search. They are not family-level significance, motif-discovery FDR, differential enrichment or binding probabilities. No additional adjustment across 131 queries or three databases is applied. FlyFactorSurvey includes FlyReg-derived profiles; the searches are not independent evidence. Similarity to Trl/GAF or CLAMP does not discriminate their binding without additional evidence.")
        y=self.section(y,"Rerun consistency","For all 79 previously reported positive cores, best targets, p-values and alignments are unchanged. Three FlyReg q-values vary slightly in this fresh search; the maximum absolute difference is 0.000747 and no q <=0.05 decision changes. The cause of that native q-value variation was not established here. This report always uses its own saved raw-search q-values, never copied statistics from another run.")
        y=self.section(y,"Primary references","TF-MoDISco method and data structure: github.com/jmschrei/tfmodisco-lite. Tomtom options and scoring: meme-suite.org/meme/doc/tomtom.html. Integrated Gradients and baseline interpretation: captum.ai/docs/extension/integrated_gradients. JASPAR CLAMP and Trl: jaspar.elixir.no/matrix/MA1700.1/ and /matrix/MA0205.3/. Database sources and content hashes are saved with the raw results.")
        self.end()
        y=self.begin("Reproduction, verification and limitations")
        y=self.section(y,"Implementation and artifacts","Analysis: experiments/classifier_modisco/families.py. PDF: experiments/classifier_modisco/family_pdf.py. Verification: scripts/verify_motif_families.py. Result root: results/classifier_modisco_families_20260917. analysis.json contains all 220 original records, core sign summaries, per-enhancer values, 58 families, union indices, thresholds and sensitivity results. pairs.json preserves all 4,162 within-polarity pair comparisons and admissible alignments. Per-database subdirectories retain raw Tomtom output, commands and input hashes.")
        y=self.section(y,"Execution","CPU-only local execution; no package installation or GPU access. Analysis and the three parallel Tomtom searches ran in tmux session motif_families_20260917, with log logs/motif_families_20260917.log. Working directory: enhancer_pleiotropy_model. Use python -m classifier_modisco.families analyze with --source, --source-audit, --previous and a fresh --root; then the search stage with --tomtom. Exact runnable commands and existing runtime paths are provided in the companion methods document.")
        y=self.section(y,"Checks required before hand-off","Input SHA256 checks, full PWM/seqlet consistency, reproduced signed means, complete-link membership, unique family coverage, union counts, core/reference identities, all displayed q-values, vector-only output, page coverage and text-overlap checks are recorded in verification.json. Focused synthetic tests cover reverse complements, distinguishing-base conflicts, short-overlap rejection, complete-link chaining, enhancer weighting and deterministic bootstrap. Every final page is rendered for visual inspection before delivery.")
        y=self.section(y,"Limits and next scientific checks","The groups are nested and not independent replicates. Pattern recovery depends on sample size and the seqlet cap. Counts measure assignment to the original clusters, not genome-wide motif prevalence. Bootstrap intervals condition on discovered supporters. Threshold sensitivity is descriptive, not independent replication. No held-out rescanning/enrichment, biological occupancy data, perturbation, cross-seed robustness or new classifier validation was performed. Families are candidate summaries of model-supported sequence patterns, not validated regulatory mechanisms.")
        self.end()
        y=self.begin("Draft manuscript legend and source manifest")
        y=self.section(y,"Figure legend","Conservative families of model-supported sequence motifs across enhancer activity breadth. Existing positive and negative TF-MoDISco patterns were trimmed to contiguous cores anchored by five consecutive positions above 0.2 bits. Passing cores underwent an enhancer-weighted contribution-sign audit, followed by within-polarity complete-link grouping using sequence and contribution similarity with informative-overlap and distinguishing-base constraints. Representative sequence and signed-contribution logos correspond to existing medoid patterns. Heatmaps show the fraction of discovery enhancers assigned at least one member seqlet, deduplicated within each group. Groups >=2 through >=8 overlap. Discordant and mixed-sign patterns are reported separately. Reference annotations are fresh core-based Tomtom matches and do not establish unique TF identity, enhancer enrichment or causality.")
        y=self.section(y,"Machine-readable provenance","Each original motifs.h5 and motif_importance.json was checked against its completion manifest. analysis.json records the exact input hashes. HDF5 sequence averages and contribution averages were checked against saved pattern matrices. All original source files and earlier PDFs remain unchanged. The adjacent PDF audit records every page's family/pattern coverage, recorded text bounds and the final PDF SHA256.")
        self.text(28,y,'Selected SHA256 identifiers',11,bold=True);y-=23
        for name,value in [('Analysis JSON',sha(root/'analysis.json')),('Pairwise comparisons',sha(root/'pairs.json')),('Exact MEME queries',sha(root/'quality_passing_queries.meme')),('Analysis source',sha(Path(__file__).with_name('families.py'))),('PDF source',sha(__file__))]:
            self.text(28,y,name,9,bold=True);y-=15
            self.text(28,y,value,6.6,font='AtlasMono');y-=24
        self.end()


def render(root, output, font_dir):
    data=json.loads((root/'analysis.json').read_text());pairs=json.loads((root/'pairs.json').read_text())
    assert sha(root/'pairs.json')==data['pairs_sha256']
    complete=json.loads((root/'complete.json').read_text());assert complete['audit_sha256']==sha(root/'analysis.json')
    results={k:json.loads((root/k/'matches.json').read_text()) for k in DATABASES}
    for key,r in results.items():
        assert r['source_audit_sha256']==sha(root/'analysis.json') and r['raw_output_sha256']==sha(root/key/'tomtom.tsv')
    if output.exists():raise FileExistsError(output)
    fig=FamilyReport(data,results,pairs,output,font_dir)
    fig.overview();fig.sign_audit();fig.flags();fig.methods();fig.sensitivity()
    fig.overview_panel('positive');fig.overview_panel('negative');fig.catalogue();fig.inventory();fig.provenance(root)
    fig.c.save()
    save_json(output.with_suffix('.audit.json'),dict(pdf=str(output),pdf_sha256=sha(output),
        analysis_sha256=sha(root/'analysis.json'),generator_sha256=sha(__file__),pages=fig.pages,text_bounds=fig.text_bounds))
    print(json.dumps(dict(event='family_pdf_complete',pages=len(fig.pages),path=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--font-dir',type=Path,default=Path('/usr/share/fonts/truetype/dejavu'))
    a=p.parse_args();render(a.root,a.output,a.font_dir)
