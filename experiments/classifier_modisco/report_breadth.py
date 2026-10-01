"""Build an offline, checksum-verified report from the completed breadth fits."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import h5py
import numpy as np


BASES = "ACGT"
COLORS = ("#168344", "#2466c4", "#bd7900", "#ce3646")
# Vector glyphs have an exact 100 x 100 extent, so letter height encodes bits.
GLYPHS = (
    "M0 100L38 0H62L100 100H75L67 76H33L25 100ZM41 54H59L50 24Z",
    "M100 15L82 34C65 13 25 22 25 50S65 87 82 66L100 85C80 100 65 100 50 100C17 100 0 79 0 50S17 0 50 0C65 0 80 0 100 15Z",
    "M100 15L82 34C65 13 25 22 25 50S65 87 77 72V61H53V42H100V88C80 100 65 100 50 100C17 100 0 79 0 50S17 0 50 0C65 0 80 0 100 15Z",
    "M0 0H100V23H63V100H37V23H0Z",
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def information_heights(sequence):
    sequence = np.asarray(sequence, dtype=float)
    if (sequence.ndim != 2 or sequence.shape[1] != 4 or not len(sequence)
            or not np.isfinite(sequence).all() or (sequence < 0).any()
            or not np.allclose(sequence.sum(1), 1, atol=1e-7)):
        raise ValueError("Expected a finite normalized A/C/G/T probability matrix")
    entropy = -(sequence * np.log2(np.clip(sequence, 1e-300, 1))).sum(1)
    return sequence * np.maximum(0, 2 - entropy)[:, None]


def consensus(sequence):
    modal = "".join(BASES[i] for i in sequence.argmax(1))
    masked = "".join(base if p >= .5 else "N" for base, p in zip(modal, sequence.max(1)))
    return modal, masked.strip("N") or "No position ≥50%"


def pwm_quality(sequence, threshold=.2, min_consecutive=5):
    """Trim only terminal columns; test strict IC runs in original column order."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 2:
        raise ValueError("Information threshold must be between 0 and 2 bits")
    if not isinstance(min_consecutive, int) or min_consecutive < 1:
        raise ValueError("Minimum consecutive positions must be a positive integer")
    sequence = np.asarray(sequence, dtype=float)
    information = information_heights(sequence).sum(1)
    informative = information > threshold
    indices = np.flatnonzero(informative)
    start, end = (int(indices[0]), int(indices[-1]) + 1) if len(indices) else (0, 0)
    longest, run = 0, 0
    for passing in informative:
        run = run + 1 if passing else 0
        longest = max(longest, run)
    return dict(threshold=threshold, min_consecutive=min_consecutive,
                passed=longest >= min_consecutive, longest_run=longest,
                start0=start, end0_exclusive=end, original_width=len(sequence),
                trimmed_sequence=sequence[start:end].copy(),
                informative_positions=int(informative.sum()),
                trimmed_total_bits=float(information[start:end].sum()))


def group_label(group):
    low, high = group["minimum_breadth"], group["maximum_breadth"]
    return "Exactly 1 context" if low == high == 1 else f"≥{low} contexts"


def load_results(root):
    config_path = root / "package/config.json"
    config = json.loads(config_path.read_text())
    groups, audit = [], {"package/config.json": digest(config_path)}
    for definition in config["groups"]:
        name = definition["name"]
        directory = root / "full_consensus_inputs" / name
        complete = json.loads((directory / "complete.json").read_text())
        if (complete["status"] != "complete" or complete["group"] != definition
                or complete["analysis_config_sha256"] != audit["package/config.json"]):
            raise ValueError("Incomplete or mismatched discovery group: " + name)
        for filename in ("motifs.h5", "motif_importance.json", "complete.json"):
            path = directory / filename
            sha = digest(path)
            audit[str(path.relative_to(root))] = sha
            if filename != "complete.json" and sha != complete["outputs"][filename]:
                raise ValueError("Checksum mismatch: " + str(path))
        rows = json.loads((directory / "motif_importance.json").read_text())
        count = complete["discovery_elements"]
        with h5py.File(directory / "motifs.h5", "r") as handle:
            for sign, prefix in (("positive", "pos_patterns"), ("negative", "neg_patterns")):
                selected = [row for row in rows if row["ranking_direction"] == sign]
                expected = {prefix + "/" + key for key in handle.get(prefix, {})}
                if (len(selected) != complete["patterns"][sign]
                        or {r["pattern"] for r in selected} != expected):
                    raise ValueError("Pattern inventory differs: " + name)
                selected.sort(key=lambda row: (-abs(row["mean_contribution_per_base"]), row["pattern"]))
                for rank, row in enumerate(selected, 1):
                    support = row["supporting_enhancers"]
                    if (row["group"] != name or row["within_group_importance_rank"] != rank
                            or not 0 < support <= min(count, row["seqlets"])
                            or not np.isfinite(row["mean_contribution_per_base"])
                            or not np.isclose(row["assigned_enhancer_fraction"], support / count)):
                        raise ValueError("Invalid importance/support row: " + row["pattern"])
                    if sign == "positive":
                        sequence = handle[row["pattern"]]["sequence"][:]
                        information_heights(sequence)
                        if len(sequence) != row["aligned_seqlet_length"]:
                            raise ValueError("Pattern length differs")
                        row["sequence"] = sequence
                        row["consensus"], row["core"] = consensus(sequence)
        groups.append(dict(definition=definition, complete=complete,
                           rows=sorted((r for r in rows if r["ranking_direction"] == "positive"),
                                       key=lambda r: r["within_group_importance_rank"])))
    return groups, audit


def sequence_logo(sequence, label, position_offset=None):
    heights = information_heights(sequence)
    width, left, base, scale = 850, 42, 116, 50
    if position_offset is not None:
        width = max(280, 58 + 24 * len(sequence))
    step = (width - left - 16) / len(sequence)
    pieces = [f'<svg class="logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 158" '
              f'role="img" aria-label="{html.escape(label, quote=True)}">',
              f'<title>{html.escape(label)}</title>']
    for bits in (0, 1, 2):
        y = base - bits * scale
        pieces.append(f'<path d="M{left} {y}H{width - 12}" stroke="#dbe1e7" fill="none"/>')
        pieces.append(f'<text x="32" y="{y + 4}" text-anchor="end">{bits}</text>')
    pieces.append('<text x="15" y="68" text-anchor="middle" transform="rotate(-90 15 68)">bits</text>')
    for pos, values in enumerate(heights):
        y = float(base)
        for i in np.argsort(values, kind="stable"):
            height = float(values[i]) * scale
            y -= height
            if height > 0.00001:
                pieces.append(f'<path d="{GLYPHS[i]}" fill="{COLORS[i]}" fill-rule="evenodd" '
                              f'transform="translate({left + pos * step:.4f} {y:.4f}) '
                              f'scale({(step - 1) / 100:.6f} {height / 100:.6f})"/>')
    positions = [1] + list(range(10, len(sequence) + 1, 10))
    if position_offset is not None:
        positions = sorted({1, len(sequence)} | {
            p for p in range(3, len(sequence) - 1) if (p + position_offset) % 10 == 0})
    for pos in positions:
        label_position = pos + (position_offset or 0)
        pieces.append(f'<text x="{left + (pos - .5) * step:.2f}" y="134" text-anchor="middle">{label_position}</text>')
    pieces.append(f'<text x="{width / 2}" y="152" text-anchor="middle">Aligned motif position (bp)</text></svg>')
    return "".join(pieces)


def rank_patterns(rows, ranking):
    if ranking == "importance":
        return sorted(rows, key=lambda row: row["within_group_importance_rank"])
    if ranking == "representation":
        # All rows belong to one group: the common denominator makes ranking by
        # distinct enhancer count identical to ranking by enhancer proportion.
        return sorted(rows, key=lambda row: (-row["supporting_enhancers"], row["pattern"]))
    raise ValueError("Unknown ranking: " + ranking)


def motif_card(row, total, ranking="importance", display_rank=None):
    rank = row["within_group_importance_rank"] if display_rank is None else display_rank
    pattern = html.escape(row["pattern"].split("/")[-1])
    identifier = html.escape(row["group_specific_pattern_id"])
    title = f'{identifier}: {row["consensus"]}'
    score = f"{row['mean_contribution_per_base']:+.5f}"
    score_label, attribution_detail = "mean attribution / bp", ""
    if ranking == "representation":
        score = f"{row['supporting_enhancers'] / total:.1%}"
        score_label = "of group enhancers"
        attribution_detail = (f'<p class="support">Mean attribution / bp: '
                              f'{row["mean_contribution_per_base"]:+.5f} · '
                              f'Attribution rank: #{row["within_group_importance_rank"]}</p>')
    return f'''<article class="motif" data-rank="{rank}" data-pattern="{identifier}">
<header><h3>#{rank} <span>{pattern}</span></h3><p class="score">{score}
<span>{score_label}</span></p></header>
<p class="support"><strong>{row['supporting_enhancers']:,}</strong> / {total:,} supporting enhancers
({row['assigned_enhancer_fraction']:.1%}) · <strong>{row['seqlets']:,}</strong> seqlets</p>{attribution_detail}
<div class="logo-scroll">{sequence_logo(row['sequence'], title)}</div>
<div class="sequences"><div><span>Majority core</span><code>{row['core']}</code></div>
<div><span>Full modal consensus</span><code>{row['consensus']}</code></div></div>
</article>'''


def quality_motif_card(row, total, display_rank=None):
    quality, sensitivity = row["quality"], row["sensitivity"]
    passed = quality["passed"]
    identifier = html.escape(row["group_specific_pattern_id"])
    pattern = html.escape(row["pattern"].split("/")[-1])
    heading = f"#{display_rank}" if passed else "Excluded"
    state = "pass" if passed else "excluded"
    trimmed = quality["trimmed_sequence"]
    if len(trimmed):
        modal, core = consensus(trimmed)
        logo = sequence_logo(trimmed, identifier + ": trimmed " + modal, quality["start0"])
        position = (f'Original alignment columns {quality["start0"] + 1}–{quality["end0_exclusive"]} '
                    f'of {quality["original_width"]} (1-based); retained {len(trimmed)} bp. '
                    f'Zero-based slice [{quality["start0"]}, {quality["end0_exclusive"]}).')
    else:
        modal, core, logo = "No retained columns", "No retained columns", ""
        position = f'All {quality["original_width"]} columns are at or below 0.2 bits; trimmed PWM is empty.'
    reason = ("Passes: at least five consecutive positions above 0.2 bits." if passed else
              f'Excluded: longest run above 0.2 bits is {quality["longest_run"]} bp; requires at least 5 bp.')
    sensitivity_label = "passes" if sensitivity["passed"] else "does not pass"
    return f'''<article class="motif qc-motif" data-qc="{state}" data-pattern="{identifier}">
<header><h3>{heading} <span>{pattern}</span></h3><p class="score">{row['supporting_enhancers'] / total:.1%}
<span>of group enhancers</span></p></header>
<p class="support"><strong>{row['supporting_enhancers']:,}</strong> / {total:,} supporting enhancers
({row['assigned_enhancer_fraction']:.1%}) · <strong>{row['seqlets']:,}</strong> seqlets · Original representation rank: #{row['representation_rank']}</p>
<p class="qc-status">{reason}</p>
<p class="support">Longest run: {quality['longest_run']} bp · Informative positions: {quality['informative_positions']} ·
Trimmed total information: {quality['trimmed_total_bits']:.2f} bits.<br>
0.3-bit sensitivity: {sensitivity_label} (longest run {sensitivity['longest_run']} bp).</p>
<p class="support">{position}</p><div class="logo-scroll qc-logo">{logo}</div>
<div class="sequences"><div><span>Trimmed modal consensus</span><code>{modal}</code></div>
<div><span>Trimmed majority core</span><code>{core}</code></div></div>
<p class="support">Original mean attribution / bp: {row['mean_contribution_per_base']:+.5f} ·
Attribution rank: #{row['within_group_importance_rank']}. Attribution retains the original {quality['original_width']}-bp denominator.</p>
<details class="original-pwm"><summary>Original full PWM logo and consensus</summary>
<div class="logo-scroll">{sequence_logo(row['sequence'], identifier + ': original ' + row['consensus'])}</div>
<div class="sequences"><div><span>Original modal consensus</span><code>{row['consensus']}</code></div>
<div><span>Original majority core</span><code>{row['core']}</code></div></div></details></article>'''


def quality_summary(groups):
    rows = []
    passing_total, sensitivity_total, all_total = 0, 0, 0
    for group in groups:
        count = len(group["rows"])
        passing = sum(row["quality"]["passed"] for row in group["rows"])
        sensitive = sum(row["sensitivity"]["passed"] for row in group["rows"])
        passing_total += passing
        sensitivity_total += sensitive
        all_total += count
        rows.append(f'<tr><th scope="row">{group_label(group["definition"])}</th><td>{count}</td>'
                    f'<td>{passing}</td><td>{count - passing}</td><td>{sensitive}</td></tr>')
    return f'''<section class="methods qc-summary"><h2>PWM quality filter</h2>
<p><strong>{passing_total} / {all_total} positive patterns pass</strong> the primary filter.
Require at least five consecutive original positions, each strictly above 0.2 bits.
Trim only terminal columns at or below 0.2 bits; keep all internal columns and spacers unchanged.
No renormalization, clustering, rescanning or new model inference is performed.</p>
<p>Information per position = 2 − Shannon entropy of its A/C/G/T frequencies (log base 2).
Reference frequencies are uniform; no small-sample correction is applied.
These exploratory thresholds are not statistical significance tests. The stricter 0.3-bit check
retains {sensitivity_total} / {all_total} patterns, also requiring five consecutive positions.
Short or internally degenerate motifs can fail even when biologically relevant.</p>
<div class="qc-table-scroll"><table><thead><tr><th>Group</th><th>Original positive</th>
<th>Pass &gt;0.2 bits</th><th>Excluded</th><th>Pass &gt;0.3 bits</th></tr></thead>
<tbody>{''.join(rows)}</tbody><tfoot><tr><th>Total group-specific patterns</th><td>{all_total}</td>
<td>{passing_total}</td><td>{all_total - passing_total}</td><td>{sensitivity_total}</td></tr></tfoot></table></div>
<p>Rank passing motifs by unchanged representation, with pattern-ID tie-breaking.
All discovery enhancers remain in the denominator. Excluded patterns are preserved in expandable sections;
these are group-specific patterns, not deduplicated motif families.</p></section>'''


CSS = """
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#172637;font:16px/1.55 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:38px 28px 70px}h1{font-size:34px;line-height:1.2;margin:6px 0 14px}
h2{font-size:25px;margin:0}h3{font-size:21px;margin:0}p{margin:10px 0}.eyebrow{color:#526375;font-size:13px;letter-spacing:.08em;text-transform:uppercase}
.intro{max-width:900px}.notice{border-left:4px solid #22758b;background:#e6f0f4;padding:12px 18px;margin:22px 0}
nav{display:flex;flex-wrap:wrap;gap:9px;margin:24px 0}a{color:#125d7b}nav a{background:white;border:1px solid #cbd5df;border-radius:6px;padding:8px 13px;text-decoration:none;font-weight:600}
nav small{display:block;color:#526375;font-weight:400;font-size:12px}.group{margin:40px 0;scroll-margin-top:20px}
.group>header p{color:#526375;margin:5px 0 18px}.motif{background:white;border:1px solid #d8e0e8;border-radius:8px;padding:18px 22px;margin:14px 0;break-inside:avoid}
.motif header{display:flex;justify-content:space-between;gap:12px;align-items:start}.motif h3 span{font-size:15px;font-weight:400;color:#526375;margin-left:8px}
.score{font-variant-numeric:tabular-nums;font-size:21px;font-weight:650;line-height:1.15;margin:0;text-align:right;color:#125d7b}
.score span{display:block;font-size:12px;font-weight:400;margin-top:4px;color:#526375}.support{font-size:14px;color:#526375}
.logo-scroll{overflow-x:auto}.logo{display:block;min-width:760px;width:100%;height:auto}.logo text{font:12px system-ui,sans-serif;fill:#526375}
.sequences{font-size:13px;border-top:1px solid #e3e8ee;padding-top:10px}.sequences>div{display:grid;grid-template-columns:160px minmax(0,1fr);gap:10px;margin:4px 0}
.sequences span{color:#526375}code{font:14px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}.sequences code{letter-spacing:.06em}
details{margin:16px 0}summary{cursor:pointer;color:#125d7b;padding:10px 0;font-weight:600}details p{max-width:940px}
.methods{background:white;padding:22px;border:1px solid #d8e0e8;border-radius:8px}.methods li{margin:9px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
.legend{font-size:14px;color:#526375}.legend b{margin-right:14px}.top-link{font-size:13px;font-weight:400;margin-left:12px}
@media(max-width:600px){main{padding:24px 14px}h1{font-size:27px}.motif{padding:14px 12px}.sequences>div{grid-template-columns:1fr;gap:0}.notice{padding:10px 13px}.methods{padding:14px}.score{font-size:19px}}
@media print{body{background:white}main{max-width:none;padding:0}nav,.top-link{display:none}.motif{border-radius:0}.logo{min-width:0}.group{break-before:page}}
"""


QC_CSS = """
.qc-logo .logo{width:auto;min-width:0;height:158px;max-width:none}
.qc-status{font-size:14px;font-weight:600}.qc-motif[data-qc="excluded"]{border-left:4px solid #93651e}
.qc-table-scroll{overflow-x:auto}.qc-summary table{width:100%;min-width:620px;border-collapse:collapse;font-size:14px}
.qc-summary th,.qc-summary td{padding:10px 12px;text-align:right;border-bottom:1px solid #d8e0e8}
.qc-summary th:first-child{text-align:left}.qc-summary thead{background:#edf1f5}.qc-summary tfoot{font-weight:600}
.qc-motif header h3{min-width:0}.qc-motif header .score{flex-shrink:0}
"""


def build_report(root, attribution_config, top_n=5, ranking="importance", quality_filter=False):
    if top_n < 1:
        raise ValueError("top_n must be positive")
    rank_patterns([], ranking)
    if quality_filter and ranking != "representation":
        raise ValueError("PWM quality report requires representation ranking")
    groups, audit = load_results(root)
    if quality_filter:
        audit["experiments/classifier_modisco/report_breadth.py"] = digest(Path(__file__))
        for group in groups:
            for rank, row in enumerate(rank_patterns(group["rows"], "representation"), 1):
                row["representation_rank"] = rank
                row["quality"] = pwm_quality(row["sequence"])
                row["sensitivity"] = pwm_quality(row["sequence"], threshold=.3)
    model = json.loads(attribution_config.read_text())
    audit["attribution_config.json"] = digest(attribution_config)
    positive = sum(len(g["rows"]) for g in groups)
    negative = sum(g["complete"]["patterns"]["negative"] for g in groups)
    title, page_title = "Top motifs by pleiotropy degree", "Motifs by pleiotropy degree"
    ranking_intro = "mean attribution per base\nacross supporting enhancers."
    ranking_method = '''the saved within-group positive-pattern rank, descending absolute mean attribution / bp.
Actual contributions are summed per seqlet, averaged within each enhancer, then averaged across supporting enhancers
and divided by aligned length. Each supporting enhancer gets one vote. These are model-logit attribution units, not probabilities.'''
    representation_notice = ""
    if ranking == "representation":
        title = page_title = "Most represented motifs by pleiotropy degree"
        ranking_intro = "the proportion of discovery enhancers supporting each pattern."
        ranking_method = '''descending distinct supporting-enhancer count divided by all discovery enhancers in that group.
Each enhancer counts once per pattern, regardless of its number of assigned seqlets. Within a group the denominator is constant,
so this gives the same order as distinct enhancer count. Equal proportions are ordered by pattern ID;
the sequential rank does not imply a difference between tied proportions. Attribution score and seqlet count do not break ties.
The original attribution score/rank is retained as secondary information, not used for this ordering.'''
        representation_notice = '''<div class="notice"><strong>Representation = distinct supporting enhancers / all discovery enhancers in the group.</strong>
This is the proportion assigned at least one seqlet to the pattern, not a motif scan of every sequence.
It is not statistical enrichment or population motif prevalence. No fold enrichment, p-values or FDR-adjusted enrichment
tests are calculated. The seqlet cap and clustering can affect these proportions; they should not be compared as enrichment
between the overlapping, independently fitted groups.</div>'''
    introduction = (f'Top {top_n} positive-contribution patterns in each group, ranked by {ranking_intro} Expand a group to see its remaining positive patterns.\n'
                    f'All {positive} positive patterns are included; {negative} negative patterns are not displayed.')
    logo_scope = "full aligned motifs (50 bp)"
    if quality_filter:
        title = page_title = "Quality-filtered motifs by pleiotropy degree"
        introduction = (f'Top {top_n} passing positive patterns per group, ranked by unchanged enhancer representation. '
                        f'All {positive} positive patterns remain inspectable, including excluded patterns. '
                        f'{negative} negative patterns are not displayed or assessed by this filter.')
        logo_scope = "trimmed PWMs with original alignment coordinates; full logos expandable"
    pieces = [f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{page_title}</title>
<style>{CSS + (QC_CSS if quality_filter else '')}</style></head><body><main id="top"><div class="eyebrow">TF-MoDISco · full fits · training discovery</div>
<h1>{title}</h1>
<p class="intro">{introduction}</p>
<div class="notice"><strong>These groups overlap.</strong> Exactly 1 is the context-specific reference;
≥2 through ≥8 are cumulative thresholds, not disjoint degrees. ≥8 means all eight contexts.
Pattern IDs and ranks are local to each fit, not matched motif families across groups.</div>{representation_notice}<nav aria-label="Pleiotropy groups">''']
    for group in groups:
        definition, complete = group["definition"], group["complete"]
        pieces.append(f'<a href="#{definition["name"]}">{group_label(definition)}'
                      f'<small>{complete["discovery_elements"]:,} enhancers</small></a>')
    pieces.append('</nav><p class="legend">Sequence logos: '
                  + ' '.join(f'<b style="color:{color}">{base}</b>' for base, color in zip(BASES, COLORS))
                  + f' · common 0–2 bit scale · {logo_scope}. On narrow screens, scroll logos horizontally.</p>')
    if quality_filter:
        pieces.append(quality_summary(groups))
    pieces.append(f'''<details class="methods"><summary>How to read the scores, consensus and support</summary>
<ul><li><strong>Rank:</strong> {ranking_method}</li>
<li><strong>Supporting enhancers:</strong> distinct discovery enhancers assigned ≥1 seqlet to this pattern.
The percentage uses the entire discovery group as denominator. It is clustering support, not motif prevalence or enrichment;
one enhancer can support multiple patterns, and the seqlet cap can affect support.</li>
<li><strong>Logos:</strong> the saved sequence frequency matrix, with base height equal to frequency × (2 − Shannon entropy),
using a uniform reference and no small-sample correction. These are sequence logos, not contribution logos.</li>
<li><strong>Majority core:</strong> positions below 50% for every base are replaced by N, then only terminal Ns are removed.
Internal Ns are retained. This is a display threshold, not a significance test. Full modal consensus picks the most frequent base
at every position (A/C/G/T order breaks ties), including low-information flanks.</li>
<li>Stored motif orientation is arbitrary; reverse complements can describe the same motif. Similar-looking patterns have not
been formally merged, matched between groups, or assigned to transcription factors.</li></ul></details>''')
    for group in groups:
        definition, complete = group["definition"], group["complete"]
        rows = rank_patterns(group["rows"], ranking)
        total = complete["discovery_elements"]
        if quality_filter:
            passing = [row for row in rows if row["quality"]["passed"]]
            excluded = [row for row in rows if not row["quality"]["passed"]]
            pieces.append(f'<section class="group" id="{definition["name"]}"><header><h2>{group_label(definition)}'
                          f'<a class="top-link" href="#top">Back to top</a></h2><p>{total:,} training enhancers · '
                          f'{len(passing)} passing / {len(rows)} positive patterns · {len(excluded)} excluded · '
                          f'{complete["patterns"]["negative"]} negative patterns (not shown)</p></header>')
            if not passing:
                pieces.append('<p>No positive patterns pass the primary filter in this group.</p>')
            pieces.extend(quality_motif_card(row, total, rank) for rank, row in enumerate(passing[:top_n], 1))
            if len(passing) > top_n:
                pieces.append(f'<details class="remaining"><summary>Show remaining {len(passing) - top_n} passing patterns '
                              f'(ranks {top_n + 1}–{len(passing)})</summary>')
                pieces.extend(quality_motif_card(row, total, rank) for rank, row in enumerate(passing[top_n:], top_n + 1))
                pieces.append('</details>')
            if excluded:
                pieces.append(f'<details class="excluded-patterns"><summary>Show {len(excluded)} excluded low-information patterns and reasons</summary>')
                pieces.extend(quality_motif_card(row, total) for row in excluded)
                pieces.append('</details>')
            pieces.append('</section>')
            continue
        pieces.append(f'<section class="group" id="{definition["name"]}"><header><h2>{group_label(definition)}'
                      f'<a class="top-link" href="#top">Back to top</a></h2><p>{total:,} training enhancers · '
                      f'{len(rows)} positive patterns · {complete["patterns"]["negative"]} negative patterns (not shown)</p></header>')
        pieces.extend(motif_card(row, total, ranking, rank) for rank, row in enumerate(rows[:top_n], 1))
        if len(rows) > top_n:
            pieces.append(f'<details class="remaining"><summary>Show remaining {len(rows) - top_n} positive patterns '
                          f'(ranks {top_n + 1}–{len(rows)})</summary>')
            pieces.extend(motif_card(row, total, ranking, rank) for rank, row in enumerate(rows[top_n:], top_n + 1))
            pieces.append('</details>')
        pieces.append('</section>')
    parameters = groups[0]["complete"]["parameters"]
    provenance_note = ("The report-generator path is relative to the project directory. "
                       "Its hash records the implementation of both fixed QC thresholds. " if quality_filter else "")
    pieces.append(f'''<section class="methods"><h2>Methods &amp; interpretation</h2>
<p><strong>Frozen model:</strong> <code>{html.escape(model['model_id'])}</code>, epoch {model['checkpoint_epoch']},
validation macro AP {model['validation_macro_ap']:.4f}. This is the legacy fine-tuned dilated CNN selected at attribution launch,
not the subsequently corrected classifier.</p>
<p><strong>Target:</strong> mean forward/reverse-complement logit over each enhancer’s observed-active contexts.
Integrated Gradients uses {model['references']} dinucleotide-preserving shuffled references of the full 2,048-bp input,
starting at {model['steps']} integration steps (adaptive 64/128 on failed convergence).
Motif discovery uses only the central 512 bp and quality-passing training enhancers, with no enhancer balancing or downsampling.</p>
<p><strong>Discovery:</strong> TF-MoDISco {parameters['modisco_version']}, eight separate full fits capped at
{parameters['max_seqlets_per_metacluster']:,} seqlets per sign; target seqlet FDR {parameters['target_seqlet_fdr']:.2f}.
The report reads <code>motifs.h5</code>, not the pilot results. No known-motif database enters discovery.</p>
<p><strong>Limits:</strong> positive patterns support the model’s mean active-context score, which can be driven by one context;
they do not establish activity in every context, TF binding or biological causality. Different cohort sizes and context composition
affect detection power. Absence from a group is not evidence of irrelevance. No held-out enrichment, formal cross-group matching,
cross-group statistical test or experimental perturbation is claimed.</p>
<details><summary>Reproducibility: source paths &amp; verified SHA256 hashes</summary>
<p>{provenance_note}Inputs are relative to the report’s results directory. The attribution configuration is supplied separately.
Every HDF5 and importance-summary hash was verified against its completion record before rendering.
The HTML embeds all logos and requires no network connection, JavaScript or separate image files.</p>
<p>Checkpoint: <code>{html.escape(model['checkpoint'])}</code></p>
<pre>{html.escape(json.dumps(audit, indent=2))}</pre></details></section></main></body></html>''')
    return "\n".join(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attribution-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--rank-by", choices=("importance", "representation"), default="importance")
    parser.add_argument("--quality-filter", action="store_true",
                        help="Trim ends at 0.2 bits and require 5 consecutive positions >0.2 bits; include a 0.3-bit sensitivity check. Requires --rank-by representation.")
    args = parser.parse_args()
    document = build_report(args.root, args.attribution_config, args.top, args.rank_by, args.quality_filter)
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing report: " + str(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), bytes=args.output.stat().st_size,
                          sha256=digest(args.output), top_per_group=args.top, ranking=args.rank_by)))


if __name__ == "__main__":
    main()
