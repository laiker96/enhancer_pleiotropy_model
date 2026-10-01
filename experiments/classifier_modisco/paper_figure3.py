"""Assemble Figure 3B-D from completed results; no scans or model inference.

Run from the project root with the existing ReportLab wheel on PYTHONPATH.
All inputs stay unchanged; the output includes the exact plotted values and
source hashes. Panel A is deliberately omitted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .report_pdf import Atlas
from .tomtom_atlas import aligned_matrices, heights, parse_matches, query_id, read_meme


SUPPORT = "results/classifier_modisco_support50ref_20260918/cecar_results"
PREVALENCE = "results/fixed_pwm_scan_20260918/cecar_results"
PERTURBATION = "results/repeat_relationship_20260918/cecar_results/perturbations"
PREFIX = "figure_3BCD_motifs_breadth_perturbation"
GA, CA, NEGATIVE, CONTROL = "#C46B22", "#167789", "#9C4F76", "#73808B"
WIDTH, HEIGHT = 540, 780
SUPPORT_MAX = .30


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def completed_json(root, name):
    completion = read_json(root / "complete.json")
    if completion.get("status") != "complete":
        raise ValueError(f"Incomplete input: {root}")
    if digest(root / name) != completion["files"][name]:
        raise ValueError(f"Input checksum mismatch: {root / name}")
    return read_json(root / name)


def select_patterns(audit, matches, positive_limit=2):
    """Same ranking as the source report, never chosen by TF name or q value."""
    selected = []
    for group in audit["groups"]:
        positive = sorted((r for r in group["rows"] if r["sign"] == "positive"),
                          key=lambda r: r["rank"])[:positive_limit]
        negative = sorted((r for r in group["rows"] if r["sign"] == "negative"),
                          key=lambda r: r["rank"])
        for row in positive + negative:
            support = row["attribution_support"]
            if (support["n"] != group["discovery_elements"]
                    or not 0 <= support["hits"] <= support["n"]
                    or not math.isclose(support["fraction"], support["hits"] / support["n"], abs_tol=1e-12)
                    or not 0 <= support["fraction"] <= SUPPORT_MAX):
                raise ValueError("Invalid support count or fraction outside shared bar axis")
            selected.append(dict(group=group["name"], discovery_n=group["discovery_elements"],
                **row, match=matches["best"][query_id(row)]))
    if len({r["id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate selected motif ID")
    return selected


def orient_pair(row):
    """Keep database PWM on its forward strand; reverse both displays if needed."""
    match = row["match"]
    query = row["trimmed_pwm"]
    target, qstart, tstart, span, overlap = aligned_matrices(
        query, match["reference"]["pwm"], match["offset"], match["orientation"])
    if overlap != match["overlap"]:
        raise ValueError("Saved Tomtom overlap disagrees with alignment")
    if match["orientation"] == "-":
        query = [list(reversed(v)) for v in reversed(query)]
        target = [list(reversed(v)) for v in reversed(target)]
        qstart, tstart = span - qstart - len(query), span - tstart - len(target)
    return query, target, qstart, tstart, span


def load_inputs(project, positive_limit=2):
    support, prevalence, perturbation = (project / p for p in (SUPPORT, PREVALENCE, PERTURBATION))
    audit = completed_json(support, "report_audit.json")
    fixed = completed_json(prevalence, "summary.json")
    edits = completed_json(perturbation, "summary.json")
    matches = read_json(support / "annotation/jaspar/matches.json")
    # Verify the raw match table and reference query file as well as the summary.
    if digest(support / "annotation/jaspar/tomtom.tsv") != matches["raw_output_sha256"]:
        raise ValueError("Tomtom raw-table checksum mismatch")
    if digest(support / "annotation/quality_passing_queries.meme") != matches["query_sha256"]:
        raise ValueError("Tomtom query checksum mismatch")
    reference_path = support / "references/jaspar2026_insects.meme"
    if digest(reference_path) != matches["database"]["sha256"]:
        raise ValueError("Reference database checksum mismatch")
    references = read_meme(reference_path)
    raw_best, _ = parse_matches((support / "annotation/jaspar/tomtom.tsv").read_text(),
                                set(matches["best"]), references)
    for ident, raw in raw_best.items():
        saved = matches["best"][ident]
        if any(saved[key] != value for key, value in raw.items()):
            raise ValueError("Saved match differs from original Tomtom table")
        if saved["reference"] != references[saved["target_id"]]:
            raise ValueError("Saved reference differs from original database PWM")
    if "50 shuffled references" not in audit["report_scope"]["native_discovery"]:
        raise ValueError("Wrong attribution reference protocol")
    rows = select_patterns(audit, matches, positive_limit=positive_limit)
    counts = [r for r in fixed["summary"] if r["split"] == "all"
              and r["threshold"] == 1e-4 and isinstance(r["degree"], int)]
    counts.sort(key=lambda r: r["degree"])
    if [r["degree"] for r in counts] != list(range(1, 9)) or sum(r["n"] for r in counts) != 40338:
        raise ValueError("Wrong exact-breadth prevalence cohort")
    for row in counts:
        for n, fraction, interval in zip(row["hits"], row["fraction"], row["ci95"]):
            if not math.isclose(n / row["n"], fraction, abs_tol=1e-12) or not interval[0] <= fraction <= interval[1]:
                raise ValueError("Invalid prevalence fraction or interval")
    if [m["id"] for m in fixed["motifs"]] != ["MA0205.3", "MA2107.1"]:
        raise ValueError("Wrong fixed reference motifs")
    if edits["split"] != "validation" or edits["included"] != 30 or edits["replicates"] != 10:
        raise ValueError("Wrong perturbation cohort")
    effects = next(r for r in edits["summary"] if r["group"] == "all")["targets"]["mean_observed_active_logit"]
    effects = {k: effects[k] for k in ("GA_disrupted_drop", "CA_GT_disrupted_drop",
                                     "both_disrupted_drop", "nonrepeat_sham_drop")}
    for effect in effects.values():
        if effect["n"] != 30 or not 0 <= effect["ci95"][0] <= effect["mean"] <= effect["ci95"][1] <= .8:
            raise ValueError("Perturbation value outside the declared plot range")
    paths = [support / "report_audit.json", support / "annotation/jaspar/matches.json",
             support / "annotation/jaspar/tomtom.tsv", support / "annotation/quality_passing_queries.meme",
             reference_path, prevalence / "summary.json", perturbation / "summary.json"]
    sources = {str(p.relative_to(project)): digest(p) for p in paths}
    return audit, rows, counts, effects, sources


class Figure(Atlas):
    def initialize(self):
        self.width, self.height = WIDTH, HEIGHT
        self.c.setPageSize((WIDTH, HEIGHT))
        self.pages.append(dict(name="Figure 3B-D", patterns=[]))
        self.c.setTitle("Figure 3B-D | Enhancer motifs, activity breadth and in silico perturbations")
        self.c.setSubject("Original enhancer intervals; native TF-MoDISco motifs; fixed PWM scans; frozen-model perturbations. Panel A omitted.")

    def motif_logo(self, pwm, x, y, step, height=12):
        # Vector letter heights are information in bits; no per-motif y rescaling.
        for pos, values in enumerate(heights(pwm)):
            bottom = y
            for base in sorted(range(4), key=lambda b: (values[b], b)):
                h = values[base] * height / 2
                if h > 1e-6:
                    self.c.saveState()
                    self.c.translate(x + pos * step, bottom + h)
                    self.c.scale((step - .3) / 100, -h / 100)
                    self.c.setFillColor(self.color(self.data["colors"][base]))
                    self.c.drawPath(self.glyphs[base], stroke=0, fill=1, fillMode=0)
                    self.c.restoreState()
                bottom += h

    def panel_b(self, rows):
        self.text(24, 750, "B", 12, bold=True)
        self.text(43, 750, "Motifs associated with enhancer activity breadth", 10, bold=True)
        self.text(43, 740, "Original enhancer intervals | 50 shuffled references | Training-only discovery", 7.2, color=self.MUTED)
        for x, title in ((46, "Breadth"), (143, "De novo PWM"), (278, "Best reference PWM"),
                         (356, "q"), (386, "Sign"), (451, "Support (%)")):
            self.text(x, 724, title, 7.2, bold=True, align="center")
        for x, title in ((46, "n train"), (143, "0-2 bits"), (278, "JASPAR insects"), (356, "Tomtom")):
            self.text(x, 713, title, 6.6, align="center", color=self.MUTED)
        for value in (0, 15, 30):
            self.text(402 + 70 * value / 30, 713, str(value), 6.5, align="center", color=self.MUTED)
        grouped = [(g, [r for r in rows if r["group"] == g]) for g in dict.fromkeys(r["group"] for r in rows)]
        top, row_height = 705, 22.5
        for group, members in grouped:
            bottom = top - row_height * len(members)
            self.line(24, top, 511)
            mid = (top + bottom) / 2
            label = "=1" if group == "exact_1" else ">=" + group.split("_")[1]
            self.text(46, mid + 3, label, 8.2, bold=True, align="center")
            self.text(46, mid - 9, f'{members[0]["discovery_n"]:,}', 6.8, align="center", color=self.MUTED)
            for row in members:
                y = top - row_height
                match = row["match"]
                query, target, qs, ts, span = orient_pair(row)
                step = min(8.5, 112 / span)
                origin = (112 - span * step) / 2
                self.motif_logo(query, 87 + origin + qs * step, y + 8, step)
                self.motif_logo(target, 222 + origin + ts * step, y + 8, step)
                pattern = ("p" if row["sign"] == "positive" else "n") + row["pattern"].rsplit("_", 1)[1]
                self.text(143, y + .8, pattern + " / rank " + str(row["rank"]), 6.5, align="center", color=self.MUTED)
                name = match["reference"]["name"]
                name = {"Trl": "Trl / GAF", "cg": "cg / Combgap"}.get(name, name)
                self.text(278, y + .8, name, 6.8, align="center",
                          color=self.MUTED if match["q"] > .05 else self.INK)
                self.text(369, y + 10, f'{match["q"]:.2g}', 6.6, align="right")
                if match["q"] > .05:
                    self.text(369, y + 1, "NS", 6.5, align="right", color=self.MUTED)
                color = NEGATIVE if row["sign"] == "negative" else (
                    GA if match["reference"]["id"] == "MA0205.3" else
                    CA if match["reference"]["id"] == "MA2107.1" else CONTROL)
                self.text(386, y + 7, "+" if row["sign"] == "positive" else "-", 9, bold=True,
                          align="center", color=NEGATIVE if row["sign"] == "negative" else self.INK)
                fraction = row["attribution_support"]["fraction"]
                self.c.setFillColor(self.color("#EBEEF0"))
                self.c.rect(402, y + 9, 70, 4, fill=1, stroke=0)
                self.c.setFillColor(self.color(color))
                self.c.rect(402, y + 9, 70 * fraction / SUPPORT_MAX, 4, fill=1, stroke=0)
                self.text(509, y + 7, f'{100 * fraction:.1f}', 7, align="right")
                self.pages[-1]["patterns"].append(row["id"])
                top = y
        self.line(24, top, 511)
        self.text(24, 243, "Top 2 positive motifs/group and all retained negative motifs; groups overlap. NS: q > 0.05.", 7, color=self.MUTED)
        self.text(24, 231, "Support = distinct original cluster-supporting enhancers / discovery enhancers; not sequence prevalence.", 6.7, color=self.MUTED)

    def panel_c(self, counts):
        self.text(24, 209, "C", 12, bold=True)
        self.text(43, 209, "Sequence prevalence", 9.5, bold=True)
        self.text(43, 196, "Enhancers with a reference-motif hit (%)", 7.1, color=self.MUTED)
        x, y, width, height, ymax = 48, 72, 198, 112, .70
        for tick in (0, 20, 40, 60):
            yy = y + height * tick / 100 / ymax
            self.line(x, yy, x + width)
            self.text(x - 8, yy - 2.5, str(tick), 7, align="right", color=self.MUTED)
        for i, row in enumerate(counts):
            self.text(x + i * width / 7, y - 13, row["degree"], 7, align="center")
        for j, color in enumerate((GA, CA)):
            points = []
            self.c.setStrokeColor(self.color(color))
            self.c.setFillColor(self.color(color))
            self.c.setLineWidth(.75)
            for i, row in enumerate(counts):
                xx = x + i * width / 7
                yy = y + height * row["fraction"][j] / ymax
                lo, hi = [y + height * v / ymax for v in row["ci95"][j]]
                self.c.line(xx, lo, xx, hi)
                self.c.line(xx - 2, lo, xx + 2, lo)
                self.c.line(xx - 2, hi, xx + 2, hi)
                points.append((xx, yy))
            self.c.setLineWidth(1.15)
            for a, b in zip(points, points[1:]):
                self.c.line(*a, *b)
            for xx, yy in points:
                self.c.circle(xx, yy, 2, fill=1, stroke=0)
        for j, (name, color) in enumerate((("Trl / GAF", GA), ("cg / Combgap", CA))):
            yy = 179 - j * 12
            self.c.setStrokeColor(self.color(color)); self.c.setLineWidth(1.5)
            self.c.line(58, yy + 2, 70, yy + 2)
            self.text(75, yy, name, 7, color=color)
        self.text(147, 43, "Exact number of active contexts", 7.5, align="center")
        self.text(147, 30, "n = 40,338; fixed PWMs; site p <= 0.0001", 6.6, align="center", color=self.MUTED)

    def panel_d(self, effects):
        self.text(278, 209, "D", 12, bold=True)
        self.text(297, 209, "In silico repeat disruption", 9.5, bold=True)
        self.text(297, 196, "30 validation enhancers; 10 shuffles each", 7.1, color=self.MUTED)
        x, width, y = 350, 157, 72
        for tick in (0., .2, .4, .6, .8):
            xx = x + width * tick / .8
            self.c.setStrokeColor(self.color(self.LINE)); self.c.setLineWidth(.5)
            self.c.line(xx, y, xx, 183)
            self.text(xx, y - 13, f'{tick:.1f}', 7, align="center", color=self.MUTED)
        for i, (key, label, color) in enumerate((
            ("GA_disrupted_drop", "GA", GA), ("CA_GT_disrupted_drop", "CA/GT", CA),
            ("both_disrupted_drop", "Both", "#424763"), ("nonrepeat_sham_drop", "Control", CONTROL))):
            effect = effects[key]
            yy = 169 - 25 * i
            self.text(340, yy - 2.5, label, 7.5, align="right", color=color)
            low, high = [x + width * v / .8 for v in effect["ci95"]]
            self.c.setStrokeColor(self.color(color)); self.c.setFillColor(self.color(color))
            self.c.setLineWidth(1.25)
            self.c.line(low, yy, high, yy)
            self.c.line(low, yy - 3, low, yy + 3); self.c.line(high, yy - 3, high, yy + 3)
            self.c.circle(x + width * effect["mean"] / .8, yy, 2.7, fill=1, stroke=0)
        self.text(407, 43, "Drop in mean active-context logit", 7.5, align="center")
        self.text(407, 30, "WT - perturbed; mean and 95% bootstrap CI", 6.6, align="center", color=self.MUTED)


def check_text_bounds(bounds):
    """Conservative text-only overlap check; manual inspection also required."""
    collisions = []
    for i, a in enumerate(bounds):
        for b in bounds[i + 1:]:
            if a["page"] != b["page"]:
                continue
            overlap_x = min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])
            overlap_y = min(a["y"] + a.get("height", a["size"]), b["y"] + b.get("height", b["size"])) - max(a["y"], b["y"])
            if overlap_x > .15 and overlap_y > .15:
                collisions.append((a["text"], b["text"]))
    if collisions:
        raise ValueError(f"Text overlaps: {collisions}")


def main(project):
    audit, rows, counts, effects, sources = load_inputs(project)
    out = project / "output/pdf"; out.mkdir(parents=True, exist_ok=True)
    path = out / (PREFIX + ".pdf")
    figure = Figure(audit, path, project / SUPPORT / "fonts")
    figure.initialize()
    figure.panel_b(rows); figure.panel_c(counts); figure.panel_d(effects)
    check_text_bounds(figure.text_bounds)
    figure.c.showPage(); figure.c.save()
    source_data = dict(
        inputs=sources, builder_sha256=digest(Path(__file__)),
        panels=["B", "C", "D"], panel_a="omitted at user request",
        selection="Top two positive motifs by original support rank per group, plus all four retained negatives; no TF-based selection or reclustering",
        b_support_axis=[0, 30], b_pwm_bits_axis=[0, 2],
        b_orientation="Reverse-complement both displayed matrices when needed to put database PWM on its forward strand; original query and q values unchanged",
        panel_b=rows, panel_c=counts, panel_d=effects,
        cohort_counts=dict(all=40338, exact_1=23100, ge_2=17238, ge_3=9591,
                           train_all=26930, train_ge_2=11626),
        software=dict(reportlab=__import__("reportlab").Version),
        text_overlap_check="passed", visual_qa="pending", pdf_sha256=digest(path),
        text_bounds=figure.text_bounds)
    with path.with_suffix(".source.json").open("w") as handle:
        json.dump(source_data, handle, indent=2, allow_nan=False); handle.write("\n")
    print(f'Created {path.relative_to(project)}: {len(rows)} motif rows; {len(counts)} exact-breadth groups; {len(effects)} perturbations', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    main(parser.parse_args().project.resolve())
