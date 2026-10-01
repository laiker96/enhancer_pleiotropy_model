"""Top-five signed native motif cores, without post-hoc motif clustering."""
import json
from pathlib import Path

from .common import write_json
from .report_pdf import Atlas
from .tomtom_atlas import DATABASES, aligned_matrices, heights, query_id


class SimpleAtlas(Atlas):
    def start(self,title,subtitle):
        self.page(1191,842,title)
        self.text(36,770,title,25,bold=True)
        self.text(36,746,subtitle,10,color=self.MUTED)

    def footer(self):
        self.line(36,37,1155)
        self.text(36,23,"Raw TF-MoDISco clusters; informative cores only | Frequency is sequence occurrence, not TF binding | Groups overlap",8,color=self.MUTED)
        self.text(1155,23,str(len(self.pages)),8,align="right",color=self.MUTED)
        self.c.showPage()

    def methods(self,matches):
        scope=self.data.get("report_scope", {})
        self.start(scope.get("title", "Enhancer-length TF-MoDISco motifs"),matches["database"]["name"]+" | Top 5 positive and negative motifs per pleiotropy group")
        y=692
        sections=[
            ("Direct native output; no extra clustering", "The already completed TF-MoDISco fits used attribution tensors restricted to each original enhancer interval. We read their raw motifs.h5 files directly, not the previous reclustered collection. The same frozen CNN epoch38, full2048 genomic input, mean observed-active-context logit, and 50 shuffled references are retained. No new attribution, clustering or merging was performed."),
            ("Short, informative motif cores", "Trim to the first through last run of at least five consecutive columns with information >0.2 bits. Keep only cores 5-30 bp long, with mean information >=0.5 bits/base and total information >=5 bits. The trimmed contribution sign must agree with the discovery sign. No internal bases are deleted or joined. Every exclusion and original-to-core coordinate is saved. The identical core is displayed, scanned and used for Tomtom."),
            ("Frequency ranking", "Scan both strands of original enhancers with FIMO5.5.9 at site p<=1e-4 and the fixed training-derived zero-order background. Frequency is the number of distinct enhancers with at least one hit divided by all enhancers in that breadth group. Rank by training frequency, separately for positive and negative patterns. Bars share a 0-100% axis. Validation percentages are shown as a separate check; test counts are saved separately. This site p is not enhancer-level FDR."),
            ("When fewer than five motifs appear", "Show fewer when the group has insufficient informative motifs. Motifs whose best possible sequence cannot reach the FIMO cutoff are not ranked as zero-frequency motifs: they remain in the complete core and Tomtom tables. No negative motif at >=7 or >=8 means too few negative seqlets were extracted, not absence of negative regulation. Positive/negative refers to attribution discovery, not proof of activation/repression at every sequence hit."),
            ("Reference matches and Combgap", "Tomtom5.5.9 uses Pearson similarity, minimum overlap5, pseudocount0.1, and both orientations. Show the best reference logo and its q, including nonsignificant matches. q is per-query/per-database, not globally adjusted across all searches. JASPAR contains the 11-bp Combgap/cg profile MA2107.1; its q and rank are recorded for every core even when it is not the best hit. This profile is absent from our pinned FlyReg and FlyFactorSurvey files. JASPAR here covers CORE insects, not only fly."),
        ]
        if scope:
            sections[0]=(sections[0][0], scope["intro"])
            sections[2]=(sections[2][0], sections[2][1].replace("original enhancers", scope["scan_region"]))
            sections[3]=(sections[3][0], sections[3][1].replace("No negative motif at >=7 or >=8 means too few negative seqlets were extracted, not absence of negative regulation.", scope["missing"]))
        if scope.get("rank_by") == "native_support":
            sections[1]=(sections[1][0], scope["trim_description"])
            sections[2]=("Attribution-supported ranking", scope["rank_description"])
            sections[3]=("Separate sequence prevalence and missing motifs", scope["missing_description"])
        for title,body in sections:
            self.text(36,y,title,13,bold=True,color=self.TEAL)
            y=self.paragraph(36,y-20,body,1119,10,15)-27
        self.text(36,y,"Combgap reference: jaspar.elixir.no/matrix/MA2107.1/ | Gene identity: flybase.org/reports/FBgn0000289",9,color=self.MUTED)
        self.footer()

    def card(self,row,match,x,y,jaspar):
        width,height=550,126
        color=self.TEAL if row["sign"]=="positive" else "#9D4D62"
        self.c.setStrokeColor(self.color(self.LINE));self.c.setLineWidth(.5)
        self.c.roundRect(x,y,width,height,4,stroke=1,fill=0)
        self.text(x+10,y+111,f'#{row["rank"]}  {row["pattern"]}  |  {len(row["trimmed_pwm"])} bp',9,bold=True,color=color)
        reference=match["reference"]
        name="Best: "+reference["name"]+(" ("+reference["id"]+")" if reference["id"]!=reference["name"] else "")
        size=8
        while self.metrics.stringWidth(name,"Atlas",size)>width-20:size-=.25
        if size<7:raise ValueError("Reference label needs a larger card")
        self.text(x+10,y+97,name,size)
        oriented,qstart,tstart,span,overlap=aligned_matrices(row["trimmed_pwm"],reference["pwm"],match["offset"],match["orientation"])
        if overlap!=match["overlap"]:raise ValueError("Tomtom alignment disagreement")
        step=min(13,310/span);origin=x+68
        self.text(x+10,y+66,"De novo",7,color=self.MUTED)
        self.text(x+10,y+32,"Reference",7,color=self.MUTED)
        self.logo(dict(heights=heights(row["trimmed_pwm"])),origin+qstart*step,y+60,step,25,coordinates=False)
        self.logo(dict(heights=heights(oriented)),origin+tstart*step,y+26,step,25,coordinates=False)
        stats=row["occurrence"]["train"];validation=row["occurrence"]["validation"]
        bx=x+402
        if self.data.get("report_scope", {}).get("rank_by") == "native_support":
            support=row["attribution_support"]
            self.text(bx,y+83,"Attribution-supported",8,color=color)
            self.text(bx,y+69,f'{100*support["fraction"]:.1f}%',10,bold=True,color=color)
            self.c.setFillColor(self.color("#E7EDF0"));self.c.rect(bx,y+58,128,6,stroke=0,fill=1)
            self.c.setFillColor(self.color(color));self.c.rect(bx,y+58,128*support["fraction"],6,stroke=0,fill=1)
            self.text(bx,y+44,f'{support["hits"]}/{support["n"]} discovery',8)
            if row["frequency_quantifiable"]:
                self.text(bx,y+30,f'Scan train {100*stats["fraction"]:.1f}%',8,color=self.MUTED)
                self.text(bx,y+18,f'Scan val {100*validation["fraction"]:.1f}%',8,color=self.MUTED)
            else:
                self.text(bx,y+30,"Scan N/A: cutoff",8,color=self.MUTED)
                self.text(bx,y+18,"unattainable",8,color=self.MUTED)
        else:
            self.text(bx,y+78,f'Train {100*stats["fraction"]:.1f}%',10,bold=True,color=color)
            self.c.setFillColor(self.color("#E7EDF0"));self.c.rect(bx,y+64,128,6,stroke=0,fill=1)
            self.c.setFillColor(self.color(color));self.c.rect(bx,y+64,128*stats["fraction"],6,stroke=0,fill=1)
            self.text(bx,y+51,f'{stats["hits"]}/{stats["n"]} enhancers',8)
            self.text(bx,y+36,f'Validation {100*validation["fraction"]:.1f}%',8,color=self.MUTED)
        self.text(x+10,y+10,f'Best q={match["q"]:.3g}'+(" (NS)" if match["q"]>.05 else ""),8,color=self.MUTED)
        if jaspar:
            cg=row["combgap"]
            self.text(x+220,y+10,f'Combgap q={cg["q"]:.3g}; rank {cg["rank"]}/296',8,color=self.MUTED)
        self.pages[-1]["patterns"].append(row["id"])

    def group(self,group,matches):
        name="Exactly 1" if group["name"]=="exact_1" else ">="+str(group["minimum_breadth"])
        support=self.data.get("report_scope", {}).get("rank_by") == "native_support"
        count=group['discovery_elements'] if support else group['n']
        label="discovery enhancers | Rank: native cluster support" if support else "training enhancers | Rank: sequence frequency"
        scope=self.data.get("report_scope", {})
        window_label=" | "+scope.get("window_label", "Central 512 bp; 2 references") if scope else ""
        self.start(name+" active context"+("" if group["name"]=="exact_1" else "s"),
            matches["database"]["name"]+f' | {count:,} {label} | Logos 0-2 bits; bars 0-100%'+window_label)
        for column,sign in enumerate(("positive","negative")):
            x=36+568*column
            self.text(x,714,sign.upper()+" CONTRIBUTION",13,bold=True,color=self.TEAL if column==0 else "#9D4D62")
            rows=sorted((r for r in group["rows"] if r["sign"]==sign and "rank" in r),key=lambda r:r["rank"])[:5]
            if not rows:
                message="No native motif passes the informative-core filters." if support else "No informative motif quantifiable at the fixed scan threshold."
                self.paragraph(x,681,message+" See the exclusion and discovery audit.",510,11,16)
            for i,row in enumerate(rows):
                self.card(row,matches["best"][query_id(row)],x,568-i*128,matches["database"]["key"]=="jaspar")
        self.footer()


def render(root,audit):
    output=root/"output/pdf";output.mkdir(parents=True)
    for key in DATABASES:
        matches=json.loads((root/"annotation"/key/"matches.json").read_text())
        prefix=audit.get("report_scope", {}).get("filename", "enhancer_native_top5")
        path=output/f"{prefix}_{key}.pdf"
        report=SimpleAtlas(audit,path,root/"fonts")
        report.c.setTitle(audit.get("report_scope", {}).get("title", "Native enhancer-length TF-MoDISco motifs")+" - "+matches["database"]["name"])
        rank="native attribution-supported enhancer fraction" if audit.get("report_scope", {}).get("rank_by") == "native_support" else "enhancer sequence frequency"
        report.c.setSubject("Top-five informative positive and negative cores by "+rank+"; no extra clustering")
        report.methods(matches)
        for group in audit["groups"]:report.group(group,matches)
        report.c.save()
        write_json(path.with_suffix(".layout.json"),dict(pages=report.pages,text_bounds=report.text_bounds,visual_qa="pending"))
