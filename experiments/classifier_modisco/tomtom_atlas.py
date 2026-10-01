"""Local Tomtom annotations of the existing quality-passing motif atlas.

No discovery, reclustering, PWM trimming changes, or model computation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import time

from .report_pdf import Atlas, group_label, passing, sha


DATABASES = {
    "flyfactorsurvey": dict(name="FlyFactorSurvey", file="motif_databases/FLY/fly_factor_survey.meme",
        count=656, source="https://meme-suite.org/meme-software/Databases/motifs/motif_databases.12.27.tgz",
        release="MEME motif archive 12.27, updated June 2025"),
    "flyreg": dict(name="FlyReg v2", file="motif_databases/FLY/flyreg.v2.meme",
        count=75, source="https://meme-suite.org/meme-software/Databases/motifs/motif_databases.12.27.tgz",
        release="Bergman & Pollard v2; MEME archive 12.27"),
    "jaspar": dict(name="JASPAR 2026 CORE insects", file="jaspar2026_insects.meme",
        count=296, source="https://jaspar.elixir.no/download/data/2026/CORE/JASPAR2026_CORE_insects_non-redundant_pfms_meme.txt",
        release="2026 CORE insects, non-redundant"),
}


def save_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_meme(path):
    """Read the explicit ACGT probability matrices in our three reference files."""
    text = path.read_text()
    if not re.search(r"ALPHABET\s*=\s*ACGT\s", text):
        raise ValueError("Expected explicit ACGT alphabet")
    motifs = {}
    for block in re.split(r"(?m)^MOTIF\s+", text)[1:]:
        lines = block.splitlines()
        names = lines[0].split(maxsplit=1)
        ident, name = names[0], names[-1]
        if ident in motifs:
            raise ValueError("Duplicate reference motif: " + ident)
        i = next(i for i, line in enumerate(lines) if line.startswith("letter-probability matrix:"))
        header = lines[i]
        if not re.search(r"alength=\s*4\b", header):
            raise ValueError("Non-DNA matrix")
        width = int(re.search(r"\bw=\s*(\d+)", header)[1])
        matrix = [list(map(float, line.split())) for line in lines[i+1:i+1+width]]
        if len(matrix) != width:
            raise ValueError("Truncated matrix")
        for row in matrix:
            if len(row) != 4 or any(not math.isfinite(v) or v < 0 for v in row) or abs(sum(row)-1) > 1e-4:
                raise ValueError("Invalid reference probabilities: " + ident)
        # Undo rounding error in the text export for display, without altering input files.
        matrix = [[v/sum(row) for v in row] for row in matrix]
        motifs[ident] = dict(id=ident, name=name, pwm=matrix,
            url=next((line[4:] for line in lines if line.startswith("URL ")), ""))
    if not motifs:
        raise ValueError("Empty reference database")
    return motifs


def query_id(row):
    return row["id"].replace("/", "__")


def meme_queries(data):
    lines = ["MEME version 5", "", "ALPHABET= ACGT", "", "strands: + -", "",
             "Background letter frequencies", "A 0.25 C 0.25 G 0.25 T 0.25", ""]
    for group in data["groups"]:
        for row in passing(group):
            pwm = row["trimmed_pwm"]
            lines.extend([f"MOTIF {query_id(row)}", f"letter-probability matrix: alength= 4 w= {len(pwm)} nsites= {row['seqlets']} E= 0"])
            for probabilities in pwm:
                total = sum(probabilities)
                if len(probabilities) != 4 or abs(total-1) > 1e-6:
                    raise ValueError("Invalid query probabilities")
                lines.append(" ".join(f"{v/total:.12g}" for v in probabilities))
            lines.append("")
    return "\n".join(lines)+"\n"


def parse_matches(text, expected, targets):
    rows = csv.DictReader(io.StringIO("\n".join(line for line in text.splitlines() if line and not line.startswith("#"))), delimiter="\t")
    best, counts = {}, dict.fromkeys(expected, 0)
    for row in rows:
        query, target = row["Query_ID"], row["Target_ID"]
        if query not in expected or target not in targets:
            raise ValueError("Unknown query or target in Tomtom output")
        match = dict(target_id=target, p=float(row["p-value"]), q=float(row["q-value"]),
            e=float(row["E-value"]), offset=int(row["Optimal_offset"]),
            overlap=int(row["Overlap"]), orientation=row["Orientation"])
        if (not all(math.isfinite(match[k]) for k in ("p", "q", "e"))
                or not 0 <= match["p"] <= 1 or not 0 <= match["q"] <= 1
                or match["orientation"] not in {"+", "-"}):
            raise ValueError("Invalid Tomtom statistics")
        counts[query] += 1
        key = (match["p"], match["q"], target)
        if query not in best or key < (best[query]["p"], best[query]["q"], best[query]["target_id"]):
            best[query] = match
    if set(best) != set(expected):
        raise ValueError("Not every query has a returned best match")
    return best, counts


def run(root, audit, tomtom, query_rule="unchanged 0.2-bit terminal trim; all 79 quality-passing positive patterns", *, references=None, database_keys=None):
    data = json.loads(audit.read_text())
    query = root/"quality_passing_queries.meme"
    with query.open("x") as handle:
        handle.write(meme_queries(data))
    expected = {query_id(row) for group in data["groups"] for row in passing(group)}
    parsed_queries = read_meme(query)
    assert set(parsed_queries) == expected
    version = subprocess.check_output([str(tomtom), "-version"], text=True).strip()
    if version != "5.5.9":
        raise ValueError("Expected the tested Tomtom 5.5.9")
    common = [str(tomtom), "-text", "-dist", "pearson", "-min-overlap", "5",
              "-motif-pseudo", "0.1", "-thresh", "1", "-verbosity", "2"]

    def search(item):
        key, metadata = item
        target = (root/"references" if references is None else references)/metadata["file"]
        motifs = read_meme(target)
        if len(motifs) != metadata["count"]:
            raise ValueError("Unexpected reference count")
        output = root/key
        output.mkdir()
        command = common+[str(query), str(target)]
        print(json.dumps(dict(event="tomtom_start", database=key, queries=len(expected), targets=len(motifs))), flush=True)
        started = time.monotonic()
        with (output/"tomtom.tsv").open("x") as stdout, (output/"stderr.log").open("x") as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
        best, counts = parse_matches((output/"tomtom.tsv").read_text(), expected, motifs)
        for qid, match in best.items():
            match["reference"] = motifs[match["target_id"]]
        result = dict(database=dict(metadata, key=key, sha256=sha(target)),
            best=best, returned_matches_per_query=counts, command=command, tomtom_version=version,
            tomtom_binary_sha256=sha(tomtom), query_sha256=sha(query), source_audit_sha256=sha(audit),
            raw_output_sha256=sha(output/"tomtom.tsv"), seconds=time.monotonic()-started,
            rules=dict(query=query_rule,
                distance="pearson", min_overlap=5, motif_pseudocount=.1, score_unaligned_columns=True,
                reverse_complements=True, best="minimum p-value; q-value and target ID break ties",
                report_q_threshold=1, significance_flag=.05,
                q_scope="Tomtom per-query, per-database q-values; no additional across-query/database adjustment"))
        save_json(output/"matches.json", result)
        print(json.dumps(dict(event="tomtom_done", database=key, queries=len(best),
            best_q_le_005=sum(m["q"] <= .05 for m in best.values()), seconds=result["seconds"])), flush=True)
        return result

    selected = list(DATABASES) if database_keys is None else list(database_keys)
    if not selected or len(selected)!=len(set(selected)) or any(k not in DATABASES for k in selected):
        raise ValueError("Expected a nonempty, unique subset of reference databases")
    with ThreadPoolExecutor(max_workers=min(3,len(selected))) as pool:
        results = list(pool.map(search, [(k,DATABASES[k]) for k in selected]))
    save_json(root/"complete.json", dict(status="complete", databases=[r["database"]["key"] for r in results],
        audit_sha256=sha(audit), query_sha256=sha(query), generator_sha256=sha(__file__)))


def heights(pwm):
    output = []
    for row in pwm:
        ic = max(0., 2+sum(p*math.log2(p) for p in row if p))
        output.append([p*ic for p in row])
    return output


def aligned_matrices(query, target, offset, orientation):
    """Tomtom offset is the query start relative to the oriented target start."""
    oriented = target if orientation == "+" else [list(reversed(row)) for row in reversed(target)]
    qstart, tstart = max(0, offset), max(0, -offset)
    span = max(qstart+len(query), tstart+len(oriented))
    overlap = min(qstart+len(query), tstart+len(oriented))-max(qstart, tstart)
    return oriented, qstart, tstart, span, overlap


class MatchAtlas(Atlas):
    def __init__(self, data, results, output, font_dir):
        super().__init__(data, output, font_dir)
        self.results = results
        self.total_pages = 1+len(data["groups"])
        self.c.setTitle("Pleiotropy motifs - " + results["database"]["name"] + " Tomtom matches")

    def summary(self):
        from reportlab.lib.pagesizes import A3, landscape
        width, height = landscape(A3)
        self.page(width, height, "Methods and match summary")
        self.text(36, height-75, self.results["database"]["name"], 29, bold=True)
        self.text(36, height-101, "De novo motifs and their best Tomtom reference matches", 16, color=self.TEAL)
        y = height-139
        sections = [
            ("What is shown", "All 79 quality-passing positive patterns from the previous atlas, in exactly the same within-group representation order. Every card shows the discovered PWM above the best reference PWM, aligned by Tomtom offset and strand. The shaded interval is their overlap. Letter heights encode 0-2 bits; a dashed line marks 0.2 bits. Logos use the original probabilities, without small-sample correction; Tomtom applies a 0.1 pseudocount for matching."),
            ("How matches were selected", f"Tomtom {self.results['tomtom_version']}; Pearson column similarity; minimum overlap 5 bp; both target strands; complete scores (unaligned columns included). Best means smallest p-value within this database, with q-value and target ID breaking ties. All best hits are shown, including non-significant matches. q <= 0.05 is highlighted; a larger q is explicitly marked not significant."),
            ("How to interpret q", "The displayed q is Tomtom's per-query, per-database multiple-testing estimate for motif similarity. It is not a motif-discovery FDR, enhancer enrichment q-value, binding probability or proof of TF identity. No additional correction across the 79 queries or the three database searches is applied. Different databases have different size, redundancy and coverage; q-values are not directly interchangeable evidence scales."),
            ("Why long flanks and related patterns remain", "The unchanged trim removes only outer columns until the first/last position above 0.2 bits. Isolated informative flank positions can therefore retain intervening weak columns. Internal columns are never deleted or concatenated. Discovery clusters have not been merged by PWM similarity: related motifs can share a best TF match without establishing identical specificity or binding."),
            ("Cohorts and reference provenance", f"Exactly 1 is the reference; >=2 through >=8 are overlapping cumulative training-discovery groups. Representation counts enhancers assigned a seqlet, not sequence-scan prevalence. Reference: {self.results['database']['release']}; {self.results['database']['count']} matrices. FlyFactorSurvey includes FlyReg-derived profiles, so those two databases are not independent. JASPAR uses all CORE insect profiles, not only D. melanogaster."),
        ]
        for title, body in sections:
            self.text(36, y, title, 12, bold=True, color=self.TEAL)
            y = self.paragraph(36, y-19, body, width-72, 10, 14, self.MUTED)-18
        self.text(36, y, "Best-match summary", 12, bold=True, color=self.TEAL)
        y -= 22
        cell = (width-72)/8
        for i, group in enumerate(self.data["groups"]):
            rows = passing(group)
            significant = sum(self.results["best"][query_id(row)]["q"] <= .05 for row in rows)
            x = 36+i*cell
            self.text(x, y, group_label(group), 11, bold=True)
            self.text(x, y-18, f"{significant}/{len(rows)} at q <= 0.05", 9, color=self.MUTED)
        y -= 51
        self.text(36, y, "Sources: MEME Tomtom documentation; MEME motif archive 12.27; official JASPAR 2026 downloads.", 9)
        self.text(36, y-16, "Reference/input hashes, commands, raw Tomtom tables and displayed alignments are retained in the companion audit.", 9, color=self.MUTED)
        self.footer()

    def draw_group(self, group, index):
        from reportlab.lib.pagesizes import A3, landscape
        width, height = landscape(A3)
        self.page(width, height, group_label(group))
        rows = passing(group)
        accent = self.ACCENTS[index]
        self.text(36, height-69, group_label(group)+" active context"+("" if group['name']=="exact_1" else "s"), 25, bold=True, color=accent)
        self.text(36, height-90, f"{self.results['database']['name']}  |  {group['n']:,} training enhancers  |  {len(rows)} discovered patterns", 10, color=self.MUTED)
        self.text(36, height-107, "Upper logo: discovery. Lower logo: best reference, oriented/aligned by Tomtom. Shading: aligned overlap. Dashed line: 0.2 bits.", 9, color=self.MUTED)
        nrows = math.ceil(len(rows)/2)
        gap, card_w = 12, (width-84)/2
        card_h = min(200, (height-174-(nrows-1)*gap)/nrows)
        for i, row in enumerate(rows):
            x = 36+(i%2)*(card_w+gap)
            top = height-127-(i//2)*(card_h+gap)
            self.card(row, i+1, group, x, top-card_h, card_w, card_h, accent)
            self.pages[-1]["patterns"].append(row["id"])
        self.footer()

    def card(self, row, rank, group, x, y, width, height, accent):
        match = self.results["best"][query_id(row)]
        ref = match["reference"]
        self.c.setStrokeColor(self.color(self.LINE))
        self.c.setLineWidth(.5)
        self.c.roundRect(x, y, width, height, 4, stroke=1, fill=0)
        self.text(x+10, y+height-14, f"#{rank}  {row['pattern'].split('/')[-1]}", 9, bold=True)
        self.text(x+width-10, y+height-14, f"{row['representation']:.1%}  ({row['supporting_enhancers']}/{group['n']:,} enhancers)", 8, align="right", color=accent)
        label = ref["name"] if ref["name"] == ref["id"] else f"{ref['name']} ({ref['id']})"
        # Reference names in these databases fit one line, with a deterministic font fit.
        label_size = 8
        while self.metrics.stringWidth("Best: "+label, "AtlasBold", label_size) > width-205:
            label_size -= .25
        if label_size < 6:
            raise ValueError("Reference label needs a larger card")
        self.text(x+10, y+height-29, "Best: "+label, label_size, bold=True)
        flag = "q <= 0.05" if match["q"] <= .05 else "not significant"
        self.text(x+width-10, y+height-29, f"q = {match['q']:.3g}  |  {flag}", 8, bold=True,
                  align="right", color=self.TEAL if match["q"] <= .05 else "#8B5E39")
        query = row["trimmed_pwm"]
        target, qs, ts, span, overlap = aligned_matrices(query, ref["pwm"], match["offset"], match["orientation"])
        if overlap != match["overlap"]:
            raise ValueError("Displayed alignment disagrees with Tomtom overlap")
        logo_h = min(55, (height-49)/2)
        step = min(12, (width-104)/span)
        start_x = x+76
        lower_y = y+5
        upper_y = lower_y+logo_h+6
        shared_start, shared_end = max(qs, ts), min(qs+len(query), ts+len(target))
        self.c.setFillColor(self.color("#EDF5F4"))
        self.c.rect(start_x+shared_start*step, lower_y, (shared_end-shared_start)*step,
                    2*logo_h+6, stroke=0, fill=1)
        self.text(x+9, upper_y+logo_h/2, "discovery", 7, color=self.MUTED)
        self.text(x+9, lower_y+logo_h/2, "reference "+match["orientation"], 7, color=self.MUTED)
        for matrix, start, baseline in ((query, qs, upper_y), (target, ts, lower_y)):
            self.logo(dict(heights=heights(matrix)), start_x+start*step, baseline, step, logo_h, coordinates=False)
            self.c.saveState()
            self.c.setDash(1, 2)
            self.line(start_x+start*step, baseline+logo_h*.1, start_x+(start+len(matrix))*step, "#AAB8BF")
            self.c.restoreState()


def render(root, audit, output, font_dir):
    data = json.loads(audit.read_text())
    complete = json.loads((root/"complete.json").read_text())
    if complete["audit_sha256"] != sha(audit):
        raise ValueError("Changed source atlas")
    output.mkdir(parents=True, exist_ok=True)
    for key in DATABASES:
        result = json.loads((root/key/"matches.json").read_text())
        if result["raw_output_sha256"] != sha(root/key/"tomtom.tsv"):
            raise ValueError("Changed Tomtom output")
        pdf = output/f"pleiotropy_motifs_tomtom_{key}_20260917.pdf"
        receipt = pdf.with_suffix(".audit.json")
        if pdf.exists() or receipt.exists():
            raise FileExistsError("Refusing to overwrite PDF or audit")
        atlas = MatchAtlas(data, result, pdf, font_dir)
        atlas.summary()
        for index, group in enumerate(data["groups"]):
            atlas.draw_group(group, index)
        atlas.c.save()
        save_json(receipt, dict(results=result, source_audit_sha256=sha(audit), pdf_sha256=sha(pdf),
            renderer_sha256=sha(__file__), pages=atlas.pages, text_bounds=atlas.text_bounds))
        print(json.dumps(dict(pdf=str(pdf), pages=len(atlas.pages), sha256=sha(pdf))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("run", "render"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--tomtom", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--font-dir", type=Path, default=Path("/usr/share/fonts/truetype/dejavu"))
    args = p.parse_args()
    if args.stage == "run":
        if not args.tomtom:
            p.error("run requires --tomtom")
        run(args.root, args.audit, args.tomtom)
    else:
        if not args.output:
            p.error("render requires --output")
        render(args.root, args.audit, args.output, args.font_dir)


if __name__ == "__main__":
    main()
