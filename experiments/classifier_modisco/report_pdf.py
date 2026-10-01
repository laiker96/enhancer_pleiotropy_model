"""Vector motif atlas from the existing, checksum-verified PWM quality report.

Prepare uses the existing NumPy/h5py environment. Render uses ReportLab only.
Neither stage imports Torch, reruns discovery, or changes source PWMs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, attribution_config):
    from .report_breadth import (BASES, COLORS, GLYPHS, consensus, information_heights,
                                load_results, pwm_quality, rank_patterns)
    groups, sources = load_results(root)
    output = []
    for group in groups:
        rows = []
        for original_rank, row in enumerate(rank_patterns(group["rows"], "representation"), 1):
            qc = pwm_quality(row["sequence"])
            sensitive = pwm_quality(row["sequence"], threshold=.3)
            sequence = qc.pop("trimmed_sequence")
            rows.append(dict(pattern=row["pattern"], id=row["group_specific_pattern_id"],
                original_rank=original_rank, supporting_enhancers=row["supporting_enhancers"],
                representation=row["assigned_enhancer_fraction"], seqlets=row["seqlets"],
                attribution_per_original_bp=row["mean_contribution_per_base"],
                attribution_rank=row["within_group_importance_rank"], quality=qc,
                sensitivity_pass=sensitive["passed"], sensitivity_run=sensitive["longest_run"],
                trimmed_pwm=sequence.tolist(),
                heights=information_heights(sequence).tolist() if len(sequence) else [],
                majority_core=consensus(sequence)[1] if len(sequence) else "",
                original_majority_core=row["core"],
                full_pwm=row["sequence"].tolist()))
        output.append(dict(**group["definition"], n=group["complete"]["discovery_elements"],
                           negative_patterns=group["complete"]["patterns"]["negative"], rows=rows))
    sources["attribution_config.json"] = sha(attribution_config)
    sources["report_breadth.py"] = sha(Path(__file__).with_name("report_breadth.py"))
    sources["report_pdf.py"] = sha(__file__)
    return dict(groups=output, model=json.loads(attribution_config.read_text()),
                bases=BASES, colors=COLORS, glyphs=GLYPHS, sources=sources,
                rules=dict(primary_threshold_bits=.2, consecutive_positions=5,
                           sensitivity_threshold_bits=.3, strict_comparison=True,
                           background="uniform", small_sample_correction=False,
                           ranking="distinct supporting enhancers / all group enhancers"))


def vector_path(canvas, source):
    """Convert the report's four fixed SVG glyph paths without rasterization."""
    tokens = re.findall(r"[MLHVCSZ]|-?\d+(?:\.\d+)?", source)
    path = canvas.beginPath()
    x = y = 0
    start = (0, 0)
    control = None
    previous = None
    i = 0
    lengths = dict(M=2, L=2, H=1, V=1, C=6, S=4, Z=0)
    while i < len(tokens):
        command = tokens[i]
        i += 1
        if command not in lengths:
            raise ValueError("Unsupported logo glyph command: " + command)
        count = lengths[command]
        values = list(map(float, tokens[i:i+count]))
        if len(values) != count:
            raise ValueError("Incomplete glyph path")
        i += count
        if command == "M":
            x, y = values
            start = (x, y)
            path.moveTo(x, y)
        elif command == "L":
            x, y = values
            path.lineTo(x, y)
        elif command == "H":
            x = values[0]
            path.lineTo(x, y)
        elif command == "V":
            y = values[0]
            path.lineTo(x, y)
        elif command in {"C", "S"}:
            if command == "C":
                x1, y1, x2, y2, end_x, end_y = values
            else:
                x1, y1 = (2*x-control[0], 2*y-control[1]) if previous in {"C", "S"} else (x, y)
                x2, y2, end_x, end_y = values
            path.curveTo(x1, y1, x2, y2, end_x, end_y)
            control = (x2, y2)
            x, y = end_x, end_y
        else:
            path.close()
            x, y = start
        previous = command
    return path


def passing(group):
    return [row for row in group["rows"] if row["quality"]["passed"]]


def group_label(group):
    return "Exactly 1" if group["name"] == "exact_1" else f'≥{group["minimum_breadth"]}'


class Atlas:
    INK = "#182B3A"
    MUTED = "#566976"
    LINE = "#DCE5E9"
    TEAL = "#176B79"
    ACCENTS = ("#6B737E", "#497F8E", "#387D91", "#297990", "#246E8D", "#325E89", "#494C80", "#624174")

    def __init__(self, data, output, font_dir):
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.colors import HexColor
        self.color = HexColor
        self.metrics = pdfmetrics
        for name, filename in (("Atlas", "DejaVuSans.ttf"), ("AtlasBold", "DejaVuSans-Bold.ttf"),
                               ("AtlasMono", "DejaVuSansMono.ttf")):
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
        self.c = Canvas(str(output), pageCompression=1, invariant=1)
        self.c.setTitle("Motifs across pleiotropy levels - quality-filtered atlas")
        self.c.setAuthor("Enhancer pleiotropy model project")
        self.c.setSubject("De novo positive-pattern PWMs; representation-ranked; overlapping breadth groups")
        self.data = data
        self.total_pages = 2 + sum(math.ceil(len(passing(group))/10) for group in data["groups"])
        self.glyphs = [vector_path(self.c, source) for source in data["glyphs"]]
        self.pages = []
        self.text_bounds = []

    def text(self, x, y, value, size=9, bold=False, color=None, align="left", font=None):
        value = str(value)
        face = font or ("AtlasBold" if bold else "Atlas")
        width = self.metrics.stringWidth(value, face, size)
        left = x if align == "left" else x-width if align == "right" else x-width/2
        if left < 20 or left+width > self.width-20 or y < 16 or y+size > self.height-18:
            raise ValueError(f"Text outside page bounds: {value}")
        self.text_bounds.append(dict(page=len(self.pages), text=value, x=left, y=y, width=width, size=size))
        self.c.setFillColor(self.color(color or self.INK))
        self.c.setFont(face, size)
        self.c.drawString(left, y, value)

    def paragraph(self, x, y, value, width, size=9, leading=14, color=None):
        lines, line = [], ""
        for word in value.split():
            candidate = (line + " " + word).strip()
            if self.metrics.stringWidth(candidate, "Atlas", size) > width and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        for line in lines:
            self.text(x, y, line, size=size, color=color)
            y -= leading
        return y

    def line(self, x, y, end, color=None):
        self.c.setStrokeColor(self.color(color or self.LINE))
        self.c.setLineWidth(.5)
        self.c.line(x, y, end, y)

    def page(self, width, height, name):
        self.width, self.height = width, height
        self.c.setPageSize((width, height))
        self.pages.append(dict(name=name, width=width, height=height, patterns=[]))
        self.c.bookmarkPage(name)
        self.c.addOutlineEntry(name, name, 0)
        self.text(36, height-35, "ENHANCER PLEIOTROPY  /  DE NOVO MOTIF ATLAS", 8, color=self.MUTED)

    def footer(self):
        self.line(36, 37, self.width-36)
        self.text(36, 23, "Training-discovery patterns. Representation is clustering support, not enrichment or TF binding.", 7, color=self.MUTED)
        self.text(self.width-36, 23, f'{len(self.pages):02d} / {self.total_pages}', 7, align="right", color=self.MUTED)
        self.c.showPage()

    def logo(self, row, x, y, step, height, coordinates=True):
        heights = row["heights"]
        length = len(heights)
        width = step*length
        self.line(x, y, x+width)
        self.line(x, y+height, x+width)
        self.text(x-5, y-2, "0", 6, align="right", color=self.MUTED)
        self.text(x-5, y+height-2, "2", 6, align="right", color=self.MUTED)
        for pos, values in enumerate(heights):
            bottom = y
            for base in sorted(range(4), key=lambda i: (values[i], i)):
                h = values[base]*height/2
                if h > .000001:
                    self.c.saveState()
                    self.c.translate(x+pos*step, bottom+h)
                    self.c.scale((step-.45)/100, -h/100)
                    self.c.setFillColor(self.color(self.data["colors"][base]))
                    self.c.drawPath(self.glyphs[base], stroke=0, fill=1, fillMode=0)
                    self.c.restoreState()
                bottom += h
        if coordinates:
            start, end = row["quality"]["start0"]+1, row["quality"]["end0_exclusive"]
            self.text(x+step*.5, y-10, start, 6, align="center", color=self.MUTED)
            self.text(x+width-step*.5, y-10, end, 6, align="center", color=self.MUTED)

    def representation_bar(self, x, y, width, fraction, color):
        # Shared 0-35% axis; never normalize to each motif/group maximum.
        if not 0 <= fraction <= .35:
            raise ValueError("Representation exceeds the declared common 0-35% bar scale")
        self.c.setFillColor(self.color("#E7EDF0"))
        self.c.rect(x, y, width, 3, stroke=0, fill=1)
        self.c.setFillColor(self.color(color))
        self.c.rect(x, y, width*fraction/.35, 3, stroke=0, fill=1)

    def overview(self):
        from reportlab.lib.pagesizes import A3, landscape
        width, height = landscape(A3)
        self.page(width, height, "Overview")
        self.text(36, height-72, "Motifs across pleiotropy levels", 27, bold=True)
        self.text(36, height-96, "Top three quality-passing patterns per group, ranked by enhancer representation.", 11, color=self.MUTED)
        self.text(36, height-116, "Rows are overlapping, independently fitted groups. Columns are ranks, not matched motif families.", 9, color=self.MUTED)
        base_y, row_h = height-170, 72
        label_width = 142
        col_width = (width-72-label_width)/3
        top_rows = [row for group in self.data["groups"] for row in passing(group)[:3]]
        step = min(7.5, (col_width-42)/max(len(row["heights"]) for row in top_rows))
        for column in range(3):
            self.text(36+label_width+column*col_width+12, height-150, f"RANK {column+1}", 9, bold=True, color=self.MUTED)
        for i, group in enumerate(self.data["groups"]):
            y = base_y-(i+1)*row_h
            accent = self.ACCENTS[i]
            self.c.setFillColor(self.color("#F5F8FA" if i % 2 == 0 else "#FFFFFF"))
            self.c.rect(36, y, width-72, row_h, stroke=0, fill=1)
            self.c.setFillColor(self.color(accent))
            self.c.rect(36, y+8, 3, row_h-16, stroke=0, fill=1)
            label = group_label(group)+(" context" if group["name"] == "exact_1" else " contexts")
            self.text(48, y+46, label, 13, bold=True, color=accent)
            self.text(48, y+30, f'n = {group["n"]:,} enhancers', 8, color=self.MUTED)
            self.text(48, y+16, f'{len(passing(group))}/{len(group["rows"])} patterns pass', 8, color=self.MUTED)
            for j, row in enumerate(passing(group)[:3]):
                x = 36+label_width+j*col_width+12
                self.text(x, y+55, row["pattern"].split("/")[-1], 8, color=self.MUTED)
                self.text(x+col_width-24, y+55, f'{row["representation"]:.1%}', 11, bold=True, color=accent, align="right")
                self.logo(row, x+13, y+20, step, 27, coordinates=False)
                self.representation_bar(x, y+8, 82, row["representation"], accent)
                self.text(x+92, y+6, f'{row["supporting_enhancers"]:,} / {group["n"]:,}', 7, color=self.MUTED)
                self.pages[-1]["patterns"].append(row["id"])
        self.text(36, 69, "Logos: common 0-2 bit height and common bp width. Support bars: common 0-35% scale. Stored motif orientation is unchanged.", 8, color=self.MUTED)
        total = sum(len(passing(group)) for group in self.data["groups"])
        self.text(36, 54, f"{total} passing patterns in total; all appear on the following group pages. ≥2 through ≥8 are cumulative thresholds, not exact degrees.", 8, color=self.MUTED)
        self.footer()

    def group_page(self, group, index, rows, first_rank, page_index, page_count):
        from reportlab.lib.pagesizes import A3
        width, height = A3
        label = group_label(group)+(" active context" if group["name"] == "exact_1" else " active contexts")
        self.page(width, height, label+f" ({page_index}/{page_count})")
        accent = self.ACCENTS[index]
        all_rows = passing(group)
        self.text(36, height-73, label, 27, bold=True, color=accent)
        self.text(36, height-97, f'{group["n"]:,} training enhancers. {len(all_rows)} passing patterns; showing ranks {first_rank}-{first_rank+len(rows)-1}.', 10)
        self.text(36, height-116, f'{len(group["rows"])-len(all_rows)} positive patterns excluded. {sum(r["sensitivity_pass"] for r in all_rows)} also pass the >0.3-bit check. Page {page_index}/{page_count} for this group.', 9, color=self.MUTED)
        self.text(36, height-134, "Logos use 0-2 bits; horizontal coordinates refer to the original 50-bp alignment. Reverse complements are equivalent.", 8, color=self.MUTED)
        column_gap = 18
        card_width = (width-72-column_gap)/2
        nrows = math.ceil(len(rows)/2)
        card_height = min(220, (height-270-12*(nrows-1))/nrows)
        step = min(10, (card_width-43)/max(len(row["heights"]) for row in all_rows))
        for k, row in enumerate(rows):
            column, r = k % 2, k//2
            x = 36+column*(card_width+column_gap)
            y = height-160-(r+1)*card_height-r*12
            if y < 100:
                raise ValueError("Motif card overlaps the footer area")
            self.c.setStrokeColor(self.color(self.LINE))
            self.c.setFillColor(self.color("#FFFFFF"))
            self.c.roundRect(x, y, card_width, card_height, 5, stroke=1, fill=1)
            self.text(x+12, y+card_height-18, f'#{first_rank+k}  '+row["pattern"].split("/")[-1], 10, bold=True)
            self.text(x+card_width-12, y+card_height-18, f'{row["representation"]:.1%}', 13, bold=True, color=accent, align="right")
            self.text(x+12, y+card_height-33, f'{row["supporting_enhancers"]:,}/{group["n"]:,} enhancers; {row["seqlets"]:,} seqlets', 8, color=self.MUTED)
            self.representation_bar(x+card_width-84, y+card_height-29, 72, row["representation"], accent)
            logo_height = min(75, card_height-104)
            self.logo(row, x+25, y+60, step, logo_height)
            self.text(x+12, y+37, row["majority_core"], 8, font="AtlasMono")
            qc = row["quality"]
            status = "pass" if row["sensitivity_pass"] else "fail"
            self.text(x+12, y+24, f'IC {qc["trimmed_total_bits"]:.2f} bits; longest run {qc["longest_run"]} bp; >0.3 bits: {status}', 7.7, color=self.MUTED)
            self.text(x+12, y+11, f'Cols {qc["start0"]+1}-{qc["end0_exclusive"]}; attr/bp {row["attribution_per_original_bp"]:+.5f}; original rank #{row["original_rank"]}', 7.7, color=self.MUTED)
            self.pages[-1]["patterns"].append(row["id"])
        self.text(36, 67, "Consensus masks bases below 50% frequency with N. Information filtering uses the full PWM, not consensus length.", 8, color=self.MUTED)
        self.text(36, 53, "Attr/bp is the original per-enhancer mean attribution divided by the original 50-bp width; it is not recalculated after trimming.", 8, color=self.MUTED)
        self.footer()

    def methods(self):
        from reportlab.lib.pagesizes import A3
        width, height = A3
        self.page(width, height, "Methods and quality audit")
        self.text(36, height-73, "Reading the atlas", 27, bold=True)
        self.text(36, height-103, "A descriptive view of model-supported sequence patterns, not a motif-enrichment test.", 10, color=self.MUTED)
        y = height-145
        sections = [
            ("What the groups mean", "Exactly 1 is the context-specific reference. ≥2 through ≥8 are overlapping cumulative groups, not disjoint degrees or independent replicates. Each group was fitted separately. Pattern identifiers are local to a group; equal ranks or similar-looking logos do not establish matched motif families. Different cohort sizes and seqlet caps affect discovery power and support."),
            ("What representation measures", "Distinct discovery enhancers assigned at least one seqlet to a pattern, divided by all discovery enhancers in that group. Each enhancer counts once per pattern. The same enhancer may support several patterns. The numerator and denominator are unchanged by PWM trimming. This is not a motif scan, population prevalence, fold enrichment or significance estimate. Bars share a fixed 0-35% axis."),
            ("The PWM filter", "Information per position is 2 minus Shannon entropy of A/C/G/T frequencies, using log base 2, a uniform background and no small-sample correction. Terminal columns at or below 0.2 bits are trimmed. A pattern passes if at least five consecutive original positions each exceed 0.2 bits. Internal spacers are retained. The separate >0.3-bit five-position check is sensitivity analysis, not a second ranking. Short or degenerate biological motifs can fail these exploratory thresholds."),
            ("Discovery and interpretation", f'The frozen legacy fine-tuned dilated CNN ({self.data["model"]["model_id"]}, checkpoint epoch {self.data["model"]["checkpoint_epoch"]}) supplies Integrated Gradients for the mean forward/RC logit across observed-active contexts. Two dinucleotide-shuffled references use the full 2,048-bp input; discovery uses central 512 bp of quality-passing training enhancers only. TF-MoDISco 2.5.2 full fits use a 20,000-seqlet cap per sign and target seqlet FDR 0.05. Positive patterns can reflect one active context, not necessarily universal activity. No known-motif database, TF identification, cross-group matching or biological causality is claimed.'),
        ]
        for title, body in sections:
            self.text(36, y, title, 12, bold=True, color=self.TEAL)
            y = self.paragraph(36, y-19, body, width-72, 9, 14, self.MUTED)-18
        self.text(36, y, "Quality-filter accounting", 12, bold=True, color=self.TEAL)
        y -= 24
        columns = (36, 258, 377, 489, 601, width-36)
        for x, value in zip(columns, ("Group", "Enhancers", "Positive", ">0.2 pass", "Excluded", ">0.3 pass")):
            self.text(x, y, value, 9, bold=True, align="left" if x == 36 else "right")
        self.line(36, y-8, width-36)
        for group in self.data["groups"]:
            y -= 24
            values = (group_label(group), f'{group["n"]:,}', len(group["rows"]), len(passing(group)),
                      len(group["rows"])-len(passing(group)), sum(r["sensitivity_pass"] for r in group["rows"]))
            for x, value in zip(columns, values):
                self.text(x, y, value, 9, align="left" if x == 36 else "right")
        y -= 27
        rows = [row for group in self.data["groups"] for row in group["rows"]]
        n_pass = sum(row["quality"]["passed"] for row in rows)
        n_strict = sum(row["sensitivity_pass"] for row in rows)
        single = [row for row in rows if len(row["original_majority_core"]) == 1]
        single_excluded = sum(not row["quality"]["passed"] for row in single)
        negative = sum(group["negative_patterns"] for group in self.data["groups"])
        y = self.paragraph(36, y, f"Totals: {len(rows)} group-specific positive patterns, {n_pass} primary passes, {len(rows)-n_pass} exclusions, and {n_strict} stricter passes. {single_excluded}/{len(single)} single-base majority-core patterns are excluded. The {negative} negative patterns are not plotted or evaluated by this filter. These totals are not counts of unique motif families.", width-72, 9, 14, self.MUTED)-20
        self.text(36, y, "Sources and reproducibility", 12, bold=True, color=self.TEAL)
        y = self.paragraph(36, y-20, "Source: classifier_modisco_breadth_20260916 full-fit motifs.h5 and motif_importance.json files, verified against completion SHA256 hashes. The companion HTML preserves excluded patterns and original full logos. The PDF audit JSON records every plotted pattern, its source PWM, trim coordinates, original counts, source hashes and page assignment.", width-72, 9, 14, self.MUTED)-12
        self.text(36, y, "No new attribution, motif discovery, training or inference was performed to create this atlas.", 9, color=self.MUTED)
        self.footer()


def render(data, output, font_dir):
    audit_path = output.with_suffix(".audit.json")
    if output.exists() or audit_path.exists():
        raise FileExistsError("Refusing to overwrite the PDF or its audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    atlas = Atlas(data, output, font_dir)
    atlas.overview()
    for index, group in enumerate(data["groups"]):
        rows = passing(group)
        page_count = math.ceil(len(rows)/10)
        chunk_size = math.ceil(len(rows)/page_count)
        for page_index, start in enumerate(range(0, len(rows), chunk_size), 1):
            atlas.group_page(group, index, rows[start:start+chunk_size], start+1, page_index, page_count)
    atlas.methods()
    atlas.c.save()
    audit = dict(data, pages=atlas.pages, text_bounds=atlas.text_bounds, pdf_sha256=sha(output))
    with audit_path.open("x") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(pdf=str(output), pages=len(atlas.pages), sha256=sha(output))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    collect = sub.add_parser("prepare")
    collect.add_argument("--root", type=Path, required=True)
    collect.add_argument("--attribution-config", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    draw = sub.add_parser("render")
    draw.add_argument("--input", type=Path, required=True)
    draw.add_argument("--output", type=Path, required=True)
    draw.add_argument("--font-dir", type=Path, default=Path("/usr/share/fonts/truetype/dejavu"))
    args = parser.parse_args()
    if args.stage == "prepare":
        data = prepare(args.root, args.attribution_config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps(dict(output=str(args.output), groups=len(data["groups"]))))
    else:
        render(json.loads(args.input.read_text()), args.output, args.font_dir)


if __name__ == "__main__":
    main()
