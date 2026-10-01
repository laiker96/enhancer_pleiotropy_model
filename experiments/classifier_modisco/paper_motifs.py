"""Run-anchored motif cores, fresh Tomtom searches, and publication figures.

CPU only. Original discovery PWMs, support counts, and reports are not changed.
"""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import statistics

from .report_pdf import Atlas, group_label, passing, sha
from .tomtom_atlas import (
    DATABASES, aligned_matrices, heights, query_id, read_meme, run, save_json,
)


RULE = ("Contiguous core from the first through last run of >=5 consecutive "
        "positions with information >0.2 bits; all 79 previously passing positive "
        "patterns retained; internal columns unchanged")


def run_core(pwm, threshold=.2, consecutive=5):
    if not pwm or consecutive < 1:
        raise ValueError("Empty PWM or invalid run length")
    for row in pwm:
        if (len(row) != 4 or any(not math.isfinite(p) or p < 0 for p in row)
                or abs(sum(row)-1) > 1e-6):
            raise ValueError("Invalid PWM probabilities")
    information = [sum(row) for row in heights(pwm)]
    starts = [i for i in range(len(pwm)-consecutive+1)
              if all(v > threshold for v in information[i:i+consecutive])]
    if not starts:
        raise ValueError("No qualifying informative run")
    start, end = starts[0], starts[-1]+consecutive
    return start, end, copy.deepcopy(pwm[start:end])


def prepare(audit, previous, root):
    data = json.loads(audit.read_text())
    data["core_provenance"] = dict(source_audit=str(audit), source_audit_sha256=sha(audit),
        previous_results=str(previous), previous_complete_sha256=sha(previous/"complete.json"),
        rule=RULE, generator_sha256=sha(__file__))
    widths = []
    for group in data["groups"]:
        for row in passing(group):
            start, end, pwm = run_core(row["full_pwm"])
            widths.append((len(row["trimmed_pwm"]), len(pwm)))
            row["previous_quality"] = copy.deepcopy(row["quality"])
            row["previous_trimmed_pwm"] = row["trimmed_pwm"]
            row["quality"].update(start0=start, end0_exclusive=end,
                trimmed_total_bits=sum(sum(c) for c in heights(pwm)))
            row["trimmed_pwm"], row["heights"] = pwm, heights(pwm)
            row["majority_core"] = "".join("ACGT"[c.index(max(c))] if max(c) >= .5 else "N" for c in pwm)
    if len(widths) != 79:
        raise ValueError("Expected the 79 previously passing motifs")
    data["core_summary"] = dict(patterns=len(widths), changed=sum(a != b for a, b in widths),
        old_median=statistics.median(a for a, b in widths), new_median=statistics.median(b for a, b in widths),
        old_max=max(a for a, b in widths), new_max=max(b for a, b in widths))
    root.mkdir()
    (root/"references").symlink_to(os.path.relpath(previous/"references", root), target_is_directory=True)
    save_json(root/"cores.audit.json", data)
    print(json.dumps(dict(event="cores_prepared", **data["core_summary"])), flush=True)


def raw_matches(path, qid):
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    rows = [r for r in csv.DictReader(lines, delimiter="\t") if r["Query_ID"] == qid]
    rows.sort(key=lambda r: (float(r["p-value"]), float(r["q-value"]), r["Target_ID"]))
    return {r["Target_ID"]: dict(rank=i, target_id=r["Target_ID"], p=float(r["p-value"]),
        q=float(r["q-value"]), offset=int(r["Optimal_offset"]), overlap=int(r["Overlap"]),
        orientation=r["Orientation"]) for i, r in enumerate(rows, 1)}


def contrast(data, root):
    previous = Path(data["core_provenance"]["previous_results"])
    refs = read_meme(root/"references"/DATABASES["jaspar"]["file"])
    contrasts = []
    for row in passing(data["groups"][-1])[:2]:
        qid = query_id(row)
        old = raw_matches(previous/"jaspar/tomtom.tsv", qid)
        new = raw_matches(root/"jaspar/tomtom.tsv", qid)
        result = dict(id=row["id"], query_id=qid, old_width=len(row["previous_trimmed_pwm"]),
            core_width=len(row["trimmed_pwm"]), references={})
        for target in ("MA1700.1", "MA0205.3"):
            result["references"][target] = dict(reference=refs[target], previous=old[target], core=new[target])
        contrasts.append(result)
    return contrasts


def short_name(reference):
    # Full unmodified matrix names/IDs are preserved in the supplement and audit.
    name = reference["name"].split("_")[0]
    return "CLAMP" if name.lower() == "clamp" else name


class PaperFigure(Atlas):
    def start(self, width, height, name):
        self.width, self.height = width, height
        self.c.setPageSize((width, height))
        self.pages.append(dict(name=name, width=width, height=height, patterns=[]))
        self.c.bookmarkPage(name)
        self.c.addOutlineEntry(name, name, 0)

    def motif(self, pwm, x, y, step, height, axis=False):
        """Vector logo; identical 0-2-bit scale; optional shared axis."""
        self.line(x, y, x+step*len(pwm))
        if axis:
            self.text(x-3, y-1, "0", 5.5, align="right", color=self.MUTED)
            self.text(x-3, y+height-2, "2", 5.5, align="right", color=self.MUTED)
        for pos, values in enumerate(heights(pwm)):
            bottom = y
            for base in sorted(range(4), key=lambda i: (values[i], i)):
                h = values[base]*height/2
                if h > 1e-6:
                    self.c.saveState()
                    self.c.translate(x+pos*step, bottom+h)
                    self.c.scale((step-.35)/100, -h/100)
                    self.c.setFillColor(self.color(self.data["colors"][base]))
                    self.c.drawPath(self.glyphs[base], stroke=0, fill=1, fillMode=0)
                    self.c.restoreState()
                bottom += h

    def main(self, result):
        # A single 190-mm-wide vector figure, not a reduced A3 report.
        width, height = 190*72/25.4, 278*72/25.4
        self.start(width, height, result["database"]["key"])
        self.c.setTitle("Representation-ranked motif cores and " + result["database"]["name"] + " matches")
        self.text(22, height-30, "Motif cores across enhancer activity breadth", 11, bold=True)
        self.text(22, height-46, result["database"]["name"]+" | top three patterns per group", 8, color=self.MUTED)
        self.text(22, height-61, "Upper: core. Lower: best reference. Aligned on the reference strand; shared 0-2 bits.", 7)
        self.text(22, height-75, "Support: distinct assigned enhancers / group size. q: fresh Tomtom search on cores.", 7)
        label_width, left = 48, 22
        col_width = (width-44-label_width)/3
        for col in range(3):
            self.text(left+label_width+col*col_width+8, height-95, f"Rank {col+1}", 8, bold=True)
        for gi, group in enumerate(self.data["groups"]):
            top = height-107-gi*77
            bottom = top-72
            self.line(left, top+2, width-left)
            self.text(left, top-20, group_label(group), 9, bold=True, color=self.ACCENTS[gi])
            self.text(left, top-34, "context" if gi == 0 else "contexts", 6.5, color=self.MUTED)
            self.text(left, top-47, f"n={group['n']:,}", 6.5, color=self.MUTED)
            for ri, row in enumerate(passing(group)[:3]):
                match = result["best"][query_id(row)]
                oriented, qs, ts, span, overlap = aligned_matrices(row["trimmed_pwm"], match["reference"]["pwm"], match["offset"], match["orientation"])
                assert overlap == match["overlap"]
                query_pwm = row["trimmed_pwm"]
                if match["orientation"] == "-":
                    query_pwm = [list(reversed(c)) for c in reversed(query_pwm)]
                    oriented = [list(reversed(c)) for c in reversed(oriented)]
                    qs, ts = span-qs-len(query_pwm), span-ts-len(oriented)
                x = left+label_width+ri*col_width+9
                step = min(7., (col_width-19)/span)
                pat = row["pattern"].split("_")[-1]
                self.text(x, top-9, f"P{pat}  {row['supporting_enhancers']}/{group['n']} ({100*row['representation']:.1f}%)", 7)
                self.motif(query_pwm, x+qs*step, top-33, step, 20)
                name = short_name(match["reference"])
                # GA-repeat motifs are not uniquely assigned to either TF.
                self.text(x, top-44, f"{name}  q={match['q']:.2g}"+(" *" if match['q']>.05 else ""), 7,
                          color=self.MUTED if match['q']>.05 else self.INK)
                self.motif(oriented, x+ts*step, bottom, step, 20)
                self.pages[-1]["patterns"].append(dict(id=row["id"], query_id=query_id(row),
                    reference=match["target_id"], q=match["q"], span=span, step=step,
                    query_start=qs, target_start=ts, orientation=match["orientation"],
                    display_query_reverse_complement=match["orientation"] == "-"))
        self.text(22, 48, "Core ends: first/last >=5-bp run with information >0.2 bits; internal positions retained.", 7)
        self.text(22, 35, "Groups >=2 to >=8 overlap. Motif similarity is not TF identity; * q >0.05.", 7)
        self.text(22, 22, "GA-repeat patterns: Trl/GAF- and CLAMP-like candidates; see the comparison supplement.", 7)
        self.c.showPage()

    def ambiguity(self, comparisons):
        width, height = 842, 595
        self.start(width, height, "CLAMP versus Trl")
        self.text(28, height-38, "GA-repeat motifs: CLAMP versus Trl/GAF", 18, bold=True)
        self.text(28, height-56, "Both candidates from the same JASPAR 2026 database; reference matrices are not cropped.", 10)
        for i, entry in enumerate(comparisons):
            row = next(r for r in passing(self.data["groups"][-1]) if r["id"] == entry["id"])
            x = 28+i*405
            self.text(x, height-91, f"Breadth 8, rank {i+1} / {row['pattern']}", 12, bold=True)
            self.text(x, height-111, f"Core {entry['old_width']} -> {entry['core_width']} bp; support {row['supporting_enhancers']}/355", 9)
            # Put the query in the GA-rich strand, then orient each target with it.
            q = [list(reversed(c)) for c in reversed(row["trimmed_pwm"])]
            matches = []
            for ident, refdata in entry["references"].items():
                m = refdata["core"]
                t, qs, ts, span, overlap = aligned_matrices(row["trimmed_pwm"], refdata["reference"]["pwm"], m["offset"], m["orientation"])
                matches.append((ident, refdata, [list(reversed(c)) for c in reversed(t)], span-qs-len(q), span-ts-len(t), span))
            # Shared query anchor across both pairwise alignments.
            anchor = max(m[3] for m in matches)
            span = max(anchor-m[3]+m[5] for m in matches)
            step = min(17., 305/span)
            lx, y = x+45, height-177
            self.text(x, y+38, "Discovered core (reverse complement)", 9, bold=True)
            self.motif(q, lx+anchor*step, y, step, 31, axis=True)
            for j, (ident, refdata, t, qs, ts, _) in enumerate(matches):
                baseline = y-105-j*105
                m, old = refdata["core"], refdata["previous"]
                self.text(x, baseline+75, f"{short_name(refdata['reference'])}  {ident}", 11, bold=True)
                self.text(x, baseline+59, f"Core: rank {m['rank']}, q={m['q']:.3g}, overlap {m['overlap']} bp", 9)
                self.text(x, baseline+44, f"Previous: rank {old['rank']}, q={old['q']:.3g}", 8, color=self.MUTED)
                start = lx+(anchor-qs+ts)*step
                if ident == "MA1700.1":
                    # Reference forward position 8 (C=0.754); map into current orientation.
                    forward = refdata["reference"]["pwm"]
                    col = 7 if t == forward else len(t)-1-7
                    self.c.setFillColor(self.color("#FFE6D5"))
                    self.c.rect(start+col*step, baseline, step, 32, stroke=0, fill=1)
                self.motif(t, start, baseline, step, 31, axis=True)
                self.pages[-1]["patterns"].append(dict(id=row["id"], reference=ident, **m))
        self.text(28, 143, "The highlighted CLAMP position is ~75% C; that preference is not consistent in the discovered repeats.", 10, bold=True)
        self.text(28, 123, "Tomtom ranks whole-motif similarity, not agreement at one position or evidence of factor-specific binding.", 9)
        self.text(28, 106, "Pearson comparisons and complete scores can favor a longer repeat match despite a diagnostic mismatch.", 9)
        self.text(28, 89, "The shorter Trl reference and 14-bp CLAMP reference are not equivalent hypotheses of TF occupancy.", 9)
        self.text(28, 66, "Interpretation: GA-repeat / Trl(GAF)-CLAMP-like, not a unique CLAMP assignment.", 12, bold=True)
        self.text(28, 42, "q-values are per query and per database; no additional correction across 79 queries or three databases.", 9, color=self.MUTED)
        self.text(28, 25, "Reference: JASPAR MA1700.1 (CLAMP), MA0205.3 (Trl). Original PWMs and annotations are retained.", 9, color=self.MUTED)
        self.c.showPage()

    def supplement_group(self, group, results):
        width, height = 1190, 842
        self.start(width, height, group["name"])
        self.text(28, height-38, group_label(group)+" active context(s): all retained motif cores", 18, bold=True)
        self.text(28, height-54, "Representation order unchanged. Each target uses its own optimal alignment to the same discovered core.", 10)
        self.text(28, height-70, "References are full length; letters use 0-2 bits. NS = q >0.05. Support is cluster assignment, not enrichment.", 9)
        starts = [212, 445, 678, 911]
        for x, label in zip(starts, ["Discovered core", "FlyFactorSurvey", "FlyReg v2", "JASPAR 2026 insects"]):
            self.text(x, height-95, label, 10, bold=True)
        rows = passing(group)
        row_height = min(70, (height-142)/len(rows))
        for ri, row in enumerate(rows):
            top = height-107-ri*row_height
            self.line(28, top+1, width-28)
            matches = [results[k]["best"][query_id(row)] for k in DATABASES]
            aligned = [aligned_matrices(row["trimmed_pwm"], m["reference"]["pwm"], m["offset"], m["orientation"]) for m in matches]
            anchor = max(a[1] for a in aligned)
            span = max(anchor-a[1]+a[3] for a in aligned)
            step = min(8, 215/span)
            baseline = top-row_height+8
            self.text(28, top-14, f"{ri+1:02d}  {row['pattern']}", 9, bold=True)
            self.text(28, top-28, f"{row['supporting_enhancers']}/{group['n']} ({100*row['representation']:.1f}%)", 9)
            self.text(28, top-41, f"Core {row['quality']['start0']+1}-{row['quality']['end0_exclusive']} / {len(row['full_pwm'])} bp", 8, color=self.MUTED)
            self.text(starts[0], top-12, f"{len(row['trimmed_pwm'])} bp; original pattern retained", 8)
            self.motif(row["trimmed_pwm"], starts[0]+anchor*step, baseline, step, 24)
            for x, m, a in zip(starts[1:], matches, aligned):
                t, qs, ts, _, overlap = a
                assert overlap == m["overlap"]
                label = f"{short_name(m['reference'])} [{m['target_id']}] q={m['q']:.2g}"+(" NS" if m['q']>.05 else "")
                self.text(x, top-12, label, 7.5)
                self.motif(t, x+(anchor-qs+ts)*step, baseline, step, 24)
                self.pages[-1]["patterns"].append(dict(id=row["id"], reference=m["target_id"], q=m["q"],
                    orientation=m["orientation"], query_start=anchor, target_start=anchor-qs+ts))
        self.text(28, 22, "Contiguous run-anchored cores. Fresh Tomtom 5.5.9 / Pearson / minimum overlap 5 bp / both strands / complete scores.", 9)
        self.c.showPage()


def render(root, output, font_dir):
    data = json.loads((root/"cores.audit.json").read_text())
    complete = json.loads((root/"complete.json").read_text())
    assert complete["audit_sha256"] == sha(root/"cores.audit.json")
    results = {k: json.loads((root/k/"matches.json").read_text()) for k in DATABASES}
    for k, result in results.items():
        assert result["raw_output_sha256"] == sha(root/k/"tomtom.tsv")
        assert result["source_audit_sha256"] == sha(root/"cores.audit.json")
    comparisons = contrast(data, root)
    comparison_path = root/"clamp_trl_comparison.json"
    if comparison_path.exists():
        assert json.loads(comparison_path.read_text()) == comparisons
    else:
        save_json(comparison_path, comparisons)
    manifest = dict(core_audit_sha256=sha(root/"cores.audit.json"), figures=[])
    for key, result in results.items():
        path = output/f"pleiotropy_paper_cores_{key}_20260917.pdf"
        if path.exists():
            raise FileExistsError(path)
        fig = PaperFigure(data, path, font_dir)
        fig.main(result)
        fig.c.save()
        manifest["figures"].append(dict(path=str(path), sha256=sha(path), pages=fig.pages, text_bounds=fig.text_bounds))
    path = output/"pleiotropy_paper_cores_supplement_20260917.pdf"
    if path.exists():
        raise FileExistsError(path)
    fig = PaperFigure(data, path, font_dir)
    fig.c.setTitle("Motif cores: CLAMP/Trl comparison and all 79 patterns")
    fig.ambiguity(comparisons)
    for group in data["groups"]:
        fig.supplement_group(group, results)
    fig.c.save()
    manifest["figures"].append(dict(path=str(path), sha256=sha(path), pages=fig.pages, text_bounds=fig.text_bounds))
    save_json(root/"figures.audit.json", manifest)
    print(json.dumps(dict(event="figures_complete", paths=[f["path"] for f in manifest["figures"]])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "search", "render"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--tomtom", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/pdf"))
    parser.add_argument("--font-dir", type=Path, default=Path("/usr/share/fonts/truetype/dejavu"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.audit, args.previous, args.root)
    elif args.stage == "search":
        run(args.root, args.root/"cores.audit.json", args.tomtom, query_rule=RULE)
    else:
        render(args.root, args.output, args.font_dir)


if __name__ == "__main__":
    main()
