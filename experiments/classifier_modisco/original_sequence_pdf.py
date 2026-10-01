"""Vector-only paper overview and complete database-specific motif atlases."""
import json
import math
from pathlib import Path

from .common import digest, write_json
from .report_pdf import Atlas
from .tomtom_atlas import DATABASES, aligned_matrices, heights, query_id


def label(group):
    return "Exactly 1" if group["name"] == "exact_1" else ">="+str(group["minimum_breadth"])


def ordered(group, sign):
    return sorted((r for r in group["rows"] if r["sign"] == sign), key=lambda r: r["rank"])


class SequenceAtlas(Atlas):
    WIDTH, HEIGHT = 1191, 842
    NEGATIVE = "#9D4D62"

    def __init__(self, data, output, font_dir):
        super().__init__(data, output, font_dir)
        self.c.setTitle("Original-enhancer motifs: sequence occurrence and reference matches")
        self.c.setSubject("Corrected original intervals; positive and negative discovery patterns; not the old central-512 analysis")
        self.output = output

    def start(self, title, subtitle):
        self.page(self.WIDTH, self.HEIGHT, title)
        self.text(36, self.HEIGHT-72, title, 25, bold=True)
        self.text(36, self.HEIGHT-95, subtitle, 10, color=self.MUTED)

    def footer(self):
        self.line(36, 37, self.width-36)
        self.text(36, 23, "Original enhancer intervals | Sequence hits are not TF binding or predictive occurrences | Nested groups overlap", 8, color=self.MUTED)
        self.text(self.width-36, 23, str(len(self.pages)), 8, align="right", color=self.MUTED)
        self.c.showPage()

    def fit(self, value, width, size=9, bold=False):
        font = "AtlasBold" if bold else "Atlas"
        while self.metrics.stringWidth(value, font, size) > width and size >= 7:
            size -= .25
        if size < 7:
            raise ValueError("Label needs wrapping: "+value)
        return size

    def pwm(self, matrix, x, y, step, height):
        super().logo(dict(heights=heights(matrix)), x, y, step, height, coordinates=False)
        self.c.setDash(2, 2)
        self.line(x, y+height*.1, x+len(matrix)*step, "#BCC6CD")
        self.c.setDash()

    def bar(self, value, null, x, y, width, color):
        fraction, control = value["fraction"], null["fraction"]
        if fraction is None or control is None or not (0 <= fraction <= 1 and 0 <= control <= 1):
            raise ValueError("Invalid occurrence denominator")
        self.c.setFillColor(self.color("#E6ECEF"))
        self.c.rect(x, y, width, 5, fill=1, stroke=0)
        self.c.setFillColor(self.color(color))
        self.c.rect(x, y, width*fraction, 5, fill=1, stroke=0)
        self.c.setStrokeColor(self.color("#586671"))
        self.c.setLineWidth(1.2)
        self.c.line(x+width*control, y-2, x+width*control, y+7)

    def methods(self, database=None):
        self.start("Original-enhancer motif atlas", "Corrected discovery, frozen sequence scans and exploratory reference annotations | 18 September 2026")
        x, y, w = 36, 710, 1119
        sections = [
            ("01  What was discovered", "95 retained group-specific patterns (76 positive, 19 negative). The frozen 2048-bp legacy dilated-CNN classifier, epoch 38, sees native genomic context. Discovery uses only the original enhancer coordinates, not a fixed central-512 crop. The attribution target is the mean logit of observed-active contexts, averaged over forward/reverse-complement predictions; 50 dinucleotide-shuffled references."),
            ("02  Native refinement; no new logo-only surgery", "TF-MoDISco 2.5.2 native SimilarPatternsCollapser acts within each breadth group and sign. Final patterns are rebuilt from aligned seqlets, terminally trimmed at 30% of maximum absolute CWM magnitude, and retained with at least five consecutive sequence-information positions >0.2 bits and a concordant final sign. These saved PWMs are used unchanged for scanning, matching and logos. Internal weak columns are retained. Related patterns across groups are not unique TFs."),
            ("03  What the percentage bars mean", "FIMO 5.5.9 scans both strands of each original enhancer. A hit requires site p <= 1e-4 against a common reverse-complement-symmetric, zero-order background estimated from training enhancer bases. Each enhancer is counted once per motif. Denominators include all enhancers in that group/split; an interval shorter than a motif cannot contain a full hit. The companion table records this length eligibility. This site p is not an enhancer-level FDR."),
            ("04  Ranking, replication and controls", "Ranks use training sequence prevalence only, separately for each group and sign. All panels share a 0-100% bar scale. Gray ticks show one paired exact mono/dinucleotide-preserving shuffle per enhancer (descriptive, not a significance test). Sensitivity at p <= 1e-5 and 1e-3 and each motif's minimum attainable site p are saved; an unattainable cutoff is not evidence of biological absence. Validation and previously examined test intervals remain separate, not untouched confirmation. Cumulative breadth groups overlap."),
            ("05  Positive/negative patterns are not signed sequence matches", "A positive or negative label describes the original attribution-discovery pattern. PWM matching does not establish that a specific occurrence changes model activity, and a negative pattern is not proof of biological repression. No Fi-NeMo/predictive-occurrence calling was performed. Absence of a negative motif at >=7 or >=8 reflects insufficient extracted negative seqlets (50 and 18), not evidence that negative regulation is absent."),
            ("06  Database matching", "Tomtom 5.5.9; Pearson distance, minimum 5-bp overlap, 0.1 pseudocount, both orientations, complete alignment scores. Best means smallest p (then q and ID). Displayed q is per-query/per-database, not corrected again across 95 queries or three databases. Non-significant best matches are shown and labeled. Similarity does not identify a unique binding TF. FlyFactorSurvey includes FlyReg-derived profiles; JASPAR 2026 here is CORE insects, not fly-only."),
        ]
        for title, body in sections:
            self.text(x, y, title, 12, bold=True, color=self.TEAL)
            y = self.paragraph(x, y-18, body, w, size=10, leading=14)-21
        if database:
            self.text(x, y, database["name"]+" | "+database["release"], 11, bold=True)
            y -= 24
        self.text(x, y, "Methods: meme-suite.org/meme/doc/fimo.html | meme-suite.org/meme/doc/tomtom.html | github.com/jmschrei/tfmodisco-lite", 9, color=self.MUTED)
        self.text(x, y-19, "Companion audits preserve source hashes, original/refined mappings, commands, all hits, threshold sensitivity and exact denominators.", 9, color=self.MUTED)
        self.footer()

    def overview(self, sign, matches):
        color = self.TEAL if sign == "positive" else self.NEGATIVE
        self.start(sign.capitalize()+" patterns across pleiotropy levels", "Top three per group, ranked by training sequence prevalence | Logos: 0-2 bits | Bars: 0-100%; tick: shuffled control")
        self.text(36, 726, "Best FlyFactorSurvey match is a similarity annotation, not a TF assignment; NS means q > 0.05.", 9, color=self.MUTED)
        for i, group in enumerate(self.data["groups"]):
            top = 703-i*79
            self.text(40, top-13, label(group), 15, bold=True, color=color)
            self.text(40, top-31, f'n={group["n"]:,} train', 9, color=self.MUTED)
            rows = ordered(group, sign)
            if not rows:
                self.text(177, top-26, "No retained negative pattern; too few extracted negative seqlets.", 10, color=self.MUTED)
            for j, row in enumerate(rows[:3]):
                x = 180+j*325
                match = matches["best"][query_id(row)]
                title = f'#{row["rank"]} {row["pattern"].split("/")[-1]} | {match["reference"]["name"]}'
                self.text(x, top, title, self.fit(title, 300, 8), bold=False)
                matrix = row["trimmed_pwm"]
                step = min(10, 285/len(matrix))
                self.pwm(matrix, x+9, top-39, step, 29)
                metric = row["occurrence"][str(1e-4)]["train"]
                self.bar(metric["real"], metric["null"], x, top-57, 142, color)
                value = metric["real"]
                self.text(x+151, top-58, f'{100*value["fraction"]:.1f}% ({value["hits"]}/{value["n"]})', 8)
                self.text(x, top-70, f'q={match["q"]:.2g}'+("  NS" if match["q"] > .05 else ""), 7, color=self.MUTED)
                self.pages[-1]["patterns"].append(row["id"])
            self.line(36, top-74, self.WIDTH-36)
        self.footer()

    def card(self, group, row, match, x, y, width=550, height=199):
        color = self.TEAL if row["sign"] == "positive" else self.NEGATIVE
        self.c.setStrokeColor(self.color(self.LINE))
        self.c.setLineWidth(.5)
        self.c.roundRect(x, y, width, height, 4, stroke=1, fill=0)
        self.text(x+12, y+height-16, f'{row["sign"].upper()} #{row["rank"]} | {row["pattern"]}', 9, bold=True, color=color)
        self.text(x+width-12, y+height-16, f'{len(row["trimmed_pwm"])} bp | min site p={row["minimum_possible_p"]:.2g}', 8, align="right", color=self.MUTED)
        ref = match["reference"]
        text = "Best: "+ref["name"]+(" ("+ref["id"]+")" if ref["id"] != ref["name"] else "")
        self.text(x+12, y+height-32, text, self.fit(text, width-24, 9), bold=False)
        flag = "q <= 0.05" if match["q"] <= .05 else "NOT SIGNIFICANT"
        self.text(x+12, y+height-47, f'q={match["q"]:.3g} | {flag} | overlap={match["overlap"]} bp | strand {match["orientation"]}', 8, color=self.MUTED)
        matrix = row["trimmed_pwm"]
        oriented, qstart, tstart, span, overlap = aligned_matrices(matrix, ref["pwm"], match["offset"], match["orientation"])
        if overlap != match["overlap"]:
            raise ValueError("Displayed Tomtom alignment disagrees")
        step, origin = min(11, (width-80)/span), x+45
        self.text(x+12, y+134, "De novo", 7, color=self.MUTED)
        self.pwm(matrix, origin+qstart*step, y+95, step, 32)
        self.text(x+12, y+82, "Reference", 7, color=self.MUTED)
        self.pwm(oriented, origin+tstart*step, y+43, step, 32)
        for j, split in enumerate(("train", "validation", "test")):
            stats = row["occurrence"][str(1e-4)][split]
            v = stats["real"]
            bx = x+12+j*179
            title = (f'{split[:3].upper()} cutoff unattainable' if row["minimum_possible_p"] > 1e-4
                     else f'{split[:3].upper()} {100*v["fraction"]:.1f}% ({v["hits"]}/{v["n"]})')
            self.text(bx, y+25, title, 7)
            self.bar(v, stats["null"], bx, y+12, 153, color)
        self.pages[-1]["patterns"].append(row["id"])

    def catalogue(self, matches):
        for group in self.data["groups"]:
            rows = ordered(group, "positive")+ordered(group, "negative")
            for start in range(0, len(rows), 6):
                self.start(label(group)+f' active contexts | motifs {start+1}-{min(start+6,len(rows))}',
                    matches["database"]["name"]+" | Within-sign training prevalence rank | Both logos 0-2 bits; shared bp alignment | All bars 0-100%")
                self.text(36, 727, "Gray ticks: paired shuffled controls. TRAIN / VAL / TEST are separate cohorts. All retained motifs shown, including weak best matches.", 9, color=self.MUTED)
                for k, row in enumerate(rows[start:start+6]):
                    self.card(group, row, matches["best"][query_id(row)], 36+(k%2)*568, 502-(k//2)*215)
                self.footer()

    def repeats(self, report):
        self.start("GA and CA/GT repeats: frozen-model perturbations", "Validation only | 30 enhancers containing both nonoverlapping repeat families | One checkpoint, 10 perturbations per enhancer")
        summary = next(item for item in report["summary"] if item["group"] == "all")
        keys = ["GA_disrupted_drop", "CA_GT_disrupted_drop", "both_disrupted_drop", "nonrepeat_sham_drop", "interaction_WT_minus_GA_minus_CA_plus_double"]
        names = ["Disrupt GA", "Disrupt CA/GT", "Disrupt both", "Nonrepeat sham", "Interaction"]
        for column, (target, title) in enumerate((("mean_observed_active_logit", "Mean active-context logit"), ("sum_context_probabilities", "Sum of context probabilities"))):
            x = 36+column*571
            self.text(x, 686, title, 17, bold=True, color=self.TEAL)
            self.text(x, 664, "Mean drop and enhancer-bootstrap 95% interval", 10, color=self.MUTED)
            left, span, lo, hi = x+160, 350, -.2, .9
            scale = lambda v: left+(v-lo)/(hi-lo)*span
            self.c.setStrokeColor(self.color(self.LINE))
            self.c.line(scale(0), 280, scale(0), 620)
            for tick in (-.2, 0, .2, .4, .6, .8):
                self.text(scale(tick), 269, f"{tick:g}", 8, align="center", color=self.MUTED)
            for i, (key, name) in enumerate(zip(keys, names)):
                y = 600-i*65
                effect = summary["targets"][target][key]
                a, b = effect["ci95"]
                self.text(x, y-3, name, 10)
                self.c.setStrokeColor(self.color(self.NEGATIVE if i == 4 else self.TEAL))
                self.c.setFillColor(self.color(self.NEGATIVE if i == 4 else self.TEAL))
                self.c.setLineWidth(2)
                self.c.line(scale(a), y, scale(b), y)
                self.c.circle(scale(effect["mean"]), y, 4, fill=1, stroke=0)
                self.text(left, y-20, f'{effect["mean"]:.3f} [{a:.3f}, {b:.3f}]', 9, color=self.MUTED)
        self.paragraph(36, 214, "Interaction = WT - GA-disrupted - CA/GT-disrupted + double-disrupted. Its negative mean suggests modest non-additivity/compensation on these model-output scales; it is not evidence of biochemical synergy. The probability sum is a model breadth score, not calibrated biological breadth.", 1119, 11, 17)
        self.paragraph(36, 143, "Mononucleotide counts were preserved; dinucleotides and DNA shape can change. Shams match segment lengths, not composition or the number of changed bases. Four of 34 both-positive candidates were excluded because repeat masks overlap. Small, selected sample; no seed uncertainty. These results measure model sensitivity, not causality or TF binding.", 1119, 10, 15)
        self.footer()

    def save(self):
        self.c.save()
        write_json(self.output.with_suffix(".layout.json"), dict(pages=self.pages, text_bounds=self.text_bounds,
            source_audit="scan_audit.json", visual_qa="pending"))


def render(root, repeat_root):
    data = json.loads((root/"scan_audit.json").read_text())
    output = root/"output/pdf"
    output.mkdir(parents=True)
    fonts = root/"fonts"
    matches = {key: json.loads((root/"annotation"/key/"matches.json").read_text()) for key in DATABASES}
    paper = SequenceAtlas(data, output/"original_enhancer_motifs_paper.pdf", fonts)
    paper.methods()
    paper.overview("positive", matches["flyfactorsurvey"])
    paper.overview("negative", matches["flyfactorsurvey"])
    repeat_summary = repeat_root/"perturbations/summary.json"
    record = json.loads((repeat_root/"perturbations/complete.json").read_text())
    if record["files"]["summary.json"] != digest(repeat_summary):
        raise ValueError("Repeat perturbation summary checksum changed")
    paper.repeats(json.loads(repeat_summary.read_text()))
    paper.save()
    for key in DATABASES:
        atlas = SequenceAtlas(data, output/f"original_enhancer_motifs_{key}.pdf", fonts)
        atlas.methods(matches[key]["database"])
        atlas.catalogue(matches[key])
        atlas.save()
    write_json(root/"figure_provenance.json", dict(scan_audit_sha256=digest(root/"scan_audit.json"),
        repeat_summary_sha256=digest(repeat_summary), renderer_sha256=digest(__file__),
        matches={key:digest(root/"annotation"/key/"matches.json") for key in DATABASES},
        main_figure_top_n=3, database_atlases="all 95 patterns; within-sign ranks shown", visual_qa="pending"))
