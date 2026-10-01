"""Compact Figure 3b-d: reference-motif importance and per-context perturbation boxes.

Uses completed CPU scan results and existing frozen-classifier predictions.
The original Figure 3B-D files and builder are preserved.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .paper_figure3 import (Figure as OriginalFigure, SUPPORT, PERTURBATION,
    GA, CA, digest, load_inputs, orient_pair, check_text_bounds, completed_json)

PREFIX = "figure_3bcd_motif_importance_activity"
IMPORTANCE = "results/jaspar_importance_20260918/cecar_results"
CG_IMPORTANCE = "results/jaspar_importance_20260918/exact_degrees"
LABELS = ["1"] + [f"[{k}-8]" for k in range(2, 8)] + ["8"]
VARIANTS = ["WT", "GA_disrupted", "CA_GT_disrupted", "both_disrupted", "nonrepeat_sham"]
VARIANT_LABELS = ["WT", "GA disrupted", "CA/GT disrupted", "Both disrupted", "Control"]
VARIANT_COLORS = ["#B5B5B5", GA, CA, "#77728E", "#FFFFFF"]
CONTEXTS = ["ab", "e13", "e5", "ead", "hid", "lb", "o", "wid"]
WIDTH, HEIGHT = 510, 740
SIGN_COLORS = {'positive': '#2166AC', 'negative': '#B2182B'}
IMPORTANCE_STYLES = {
    'jaspar': dict(count=296, name='JASPAR insects', highlights={'MA0205.3': GA, 'MA2107.1': CA},
                  legend=((280, 'Other JASPAR motifs', '#AFAFAF'), (384, 'Trl / GAF', GA), (456, 'cg', CA))),
    'flyfactorsurvey': dict(count=656, name='FlyFactorSurvey', highlights={'FBgn0013263': GA},
                           legend=((280, 'Other FlyFactorSurvey motifs', '#AFAFAF'), (437, 'GAF', GA))),
}


def load_cg_overlay(project, importance):
    """Use the existing JASPAR cg curve; the pinned FlyFactorSurvey has no cg."""
    if importance.get('database_key') != 'flyfactorsurvey' or importance.get('grouping') != 'exact':
        raise ValueError('cg overlay requires exact-degree FlyFactorSurvey curves')
    jaspar = completed_json(project / CG_IMPORTANCE, 'summary.json')
    for key in ('groups', 'group_labels', 'grouping', 'split', 'metric', 'overlap_policy',
                'missing_policy', 'site_p', 'references', 'interval'):
        if jaspar[key] != importance[key]:
            raise ValueError(f'Incompatible cg overlay: {key}')
    # The reference library differs; all other scan/attribution inputs must match.
    shared = {k: v for k, v in importance['input_hashes'].items() if not k.endswith('.meme')}
    if shared != {k: v for k, v in jaspar['input_hashes'].items() if not k.endswith('.meme')}:
        raise ValueError('Incompatible cg overlay input hashes')
    matches = [r for r in jaspar['trajectories'] if r['id'] == 'MA2107.1' and r['name'] == 'cg']
    if len(matches) != 1 or not np.isfinite(matches[0]['means']).all():
        raise ValueError('Missing or ambiguous JASPAR cg curve')
    row = matches[0]
    if any(r['group_enhancers'] != row['group_enhancers'] for r in importance['trajectories']):
        raise ValueError('Incompatible cg overlay cohort sizes')
    return dict(database_key='jaspar', summary_path=CG_IMPORTANCE + '/summary.json',
                summary_sha256=digest(project / CG_IMPORTANCE / 'summary.json'), profile=row)


def box_statistics(values, probability=True):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite enhancer-level values required")
    if probability and ((values < 0).any() or (values > 1).any()):
        raise ValueError("Expected probabilities in [0,1]")
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    iqr = q3 - q1
    inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
    return dict(n=len(values), q1=float(q1), median=float(median), q3=float(q3),
                low=float(inside.min()), high=float(inside.max()),
                fliers=values[(values < inside.min()) | (values > inside.max())].tolist())


def load_activity(project):
    root = project / PERTURBATION
    receipt = json.loads((root / "complete.json").read_text())
    path = root / "predictions.npz"
    if digest(path) != receipt["files"]["predictions.npz"]:
        raise ValueError("Changed raw perturbation predictions")
    with np.load(path, allow_pickle=False) as saved:
        values = saved["probabilities"]
        if list(saved["variants"]) != VARIANTS or values.shape != (30, 10, 5, 8):
            raise ValueError("Unexpected variants or prediction dimensions")
        if not np.isfinite(values).all():
            raise ValueError("Missing values require an explicit boxplot exclusion policy")
        np.testing.assert_allclose(values[:, :, 0], np.repeat(values[:, :1, 0], 10, axis=1))
        # Each enhancer is one observation, never ten pseudoreplicated shuffles.
        averaged = values.mean(axis=1, dtype=np.float64)
        ids = saved["ids"].tolist()
    boxes = [[box_statistics(averaged[:, j, k]) for j in range(5)] for k in range(8)]
    return dict(contexts=CONTEXTS, variants=VARIANTS, ids=ids,
                mean_probabilities=averaged.tolist(), boxes=boxes,
                predictions_sha256=digest(path), metric="Forward/RC mean predicted probability; shuffle replicates averaged within enhancer")


def paired_log2_odds_ratios(logits, include_sham=False):
    """One log2 geometric-mean mutant/WT odds ratio per enhancer and context."""
    logits = np.asarray(logits, dtype=np.float64)
    if (logits.ndim != 4 or logits.shape[2:] != (5, 8)
            or not logits.shape[0] or not logits.shape[1] or not np.isfinite(logits).all()):
        raise ValueError("Finite enhancer x replicate x five variants x eight contexts logits required")
    np.testing.assert_allclose(logits[:, :, 0], np.repeat(logits[:, :1, 0], logits.shape[1], axis=1),
                               rtol=1e-5, atol=1e-5)
    # Pair each mutant with its own enhancer's WT, not a pooled WT average.
    end = 5 if include_sham else 4
    return (logits[:, :, 1:end] - logits[:, :, :1]).mean(axis=1) / np.log(2.)


def load_odds_ratios(project, include_sham=False):
    root = project / PERTURBATION
    receipt = json.loads((root / "complete.json").read_text())
    path = root / "predictions.npz"
    if receipt['status'] != 'complete' or digest(path) != receipt['files']['predictions.npz']:
        raise ValueError("Incomplete or changed perturbation predictions")
    with np.load(path, allow_pickle=False) as saved:
        logits = saved['logits']
        if list(saved['variants']) != VARIANTS or logits.shape != (30, 10, 5, 8):
            raise ValueError("Unexpected variants or prediction dimensions")
        log2_or = paired_log2_odds_ratios(logits, include_sham=include_sham)
        ids = saved['ids'].tolist()
    if len(ids) != 30 or len(set(ids)) != 30:
        raise ValueError("Thirty distinct paired enhancers required")
    variants = VARIANTS[1:] if include_sham else VARIANTS[1:4]
    boxes = [[box_statistics(log2_or[:, j, k], probability=False) for j in range(len(variants))] for k in range(8)]
    return dict(contexts=CONTEXTS, variants=variants, reference='WT', ids=ids,
                log2_odds_ratios=log2_or.tolist(), odds_ratios=np.exp2(log2_or).tolist(), boxes_log2=boxes,
                predictions_sha256=digest(path), scale='odds-ratio-wt',
                replicates_per_enhancer=10, observations_per_box=30,
                metric='exp(mean_replicates(mutant mean-forward/RC logit minus matched WT mean-forward/RC logit))',
                boxplot_space='log2 odds ratio; quantiles and 1.5-IQR whiskers computed in log2 space',
                excluded_display_conditions=['WT'] if include_sham else ['WT', 'nonrepeat_sham'],
                caveats=['Model-predicted odds, not experimentally measured activity ratios',
                         'Odds derived from mean logits, not from mean forward/RC probabilities',
                         'Geometric mean across shuffles; enhancer is the observation, not each shuffle']
                         + (['Nonrepeat-shuffle control matches segment lengths, not exact mutation count or local composition across segments']
                            if include_sham else []))


def limits_and_ticks(trajectories):
    values = [v for r in trajectories for v in r["means"] if v is not None]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Missing/nonfinite importance trajectories")
    low, high = min(0., min(values)), max(0., max(values))
    if low == high:
        low, high = -.01, .01
    span = high - low
    unit = 10 ** math.floor(math.log10(span / 4))
    step = next(v * unit for v in (1, 2, 2.5, 5, 10) if v * unit >= span / 4)
    low, high = math.floor(low / step) * step, math.ceil(high / step) * step
    ticks = np.arange(low, high + step / 2, step)
    return low, high, ticks


class NatureFigure(OriginalFigure):
    INK = "#161616"
    MUTED = "#4A4A4A"
    LINE = "#D7D7D7"

    def initialize(self, revised_layout=False, motif_count=20):
        self.revised_layout = revised_layout
        extra_height = max(0, motif_count - 20) * 14.4 if revised_layout else 0
        self.width, self.height = WIDTH, HEIGHT + extra_height
        self.c.setPageSize((self.width, self.height))
        self.pages.append(dict(name="Figure 3b-d", patterns=[]))
        self.c.setTitle("Figure 3b-d | Motif importance across degree of pleiotropy")
        self.c.setSubject("Original-enhancer motifs; all 296 JASPAR insect profiles; predicted activity of WT, disrupted and control sequences")

    def segment(self, x1, y1, x2, y2, color=None, width=.5):
        self.c.setStrokeColor(self.color(color or self.INK))
        self.c.setLineWidth(width); self.c.line(x1, y1, x2, y2)

    def y_axis_title(self, x, y, label, size=7):
        length = self.metrics.stringWidth(label, 'Atlas', size)
        ascent, descent = self.metrics.getAscentDescent('Atlas', size)
        left, bottom, width = x - ascent, y - length / 2, ascent - descent
        if left < 20 or left + width > self.width - 20 or bottom < 16 or bottom + length > self.height - 18:
            raise ValueError(f'Y-axis title outside page bounds: {label}')
        self.text_bounds.append(dict(page=len(self.pages), text=label, x=left, y=bottom,
                                     width=width, height=length, size=size, rotation=90, role='y_axis_title'))
        self.c.saveState(); self.c.translate(x, y); self.c.rotate(90)
        self.c.setFillColor(self.color(self.INK)); self.c.setFont('Atlas', size)
        self.c.drawCentredString(0, 0, label); self.c.restoreState()

    def panel_b(self, rows):
        offset = self.height - HEIGHT
        self.text(20, 714 + offset, "b", 8, bold=True)
        for x, label in ((51, "Degree of"), (144, "De novo motif"), (258, "JASPAR match"),
                         (350, "q"), (378, "+/-"), (443, "Support (%)")):
            self.text(x, 714 + offset, label, 6.2, align="center")
        self.text(51, 706 + offset, "pleiotropy", 6.2, align="center")
        for value in (0, 50, 100):
            self.text(398 + 68 * value / 100, 703 + offset, str(value), 5.5, align="center")
        top, height = 697 + offset, 14.4
        grouped = [(g, [r for r in rows if r["group"] == g]) for g in dict.fromkeys(r["group"] for r in rows)]
        for number, (group, members) in enumerate(grouped):
            bottom = top - height * len(members)
            mid = (top + bottom) / 2
            self.text(51, mid + 2, LABELS[number], 6.5, align="center")
            self.text(51, mid - 6, f'n={members[0]["discovery_n"]:,}', 5.2, align="center")
            for row in members:
                y = top - height
                match = row["match"]
                query, target, qs, ts, span = orient_pair(row)
                step = min(7, 89 / span)
                offset = (89 - span * step) / 2
                self.motif_logo(query, 99 + offset + qs * step, y + 2, step, height=10.5)
                self.motif_logo(target, 205 + offset + ts * step, y + 2, step, height=10.5)
                self.text(300, y + 4, match["reference"]["name"], 5.7)
                q = f'{match["q"]:.2g}' + (" ns" if match["q"] > .05 else "")
                self.text(369, y + 4, q, 5.5, align="right")
                if self.revised_layout:
                    self.c.setFillColor(self.color(SIGN_COLORS[row['sign']]))
                    self.c.rect(374, y + 2.5, 8, 8, stroke=0, fill=1)
                self.text(378, y + 3.5, "+" if row["sign"] == "positive" else "-", 6.5,
                          bold=self.revised_layout, color='#FFFFFF' if self.revised_layout else None, align="center")
                fraction = row["attribution_support"]["fraction"]
                self.c.setFillColor(self.color("#DEDEDE")); self.c.rect(398, y + 5, 68, 3, stroke=0, fill=1)
                self.c.setFillColor(self.color("#737373")); self.c.rect(398, y + 5, 68 * fraction, 3, stroke=0, fill=1)
                self.text(489, y + 3, f'{100 * fraction:.1f}', 5.8, align="right")
                self.pages[-1]["patterns"].append(row["id"])
                top = y
            if number != len(grouped) - 1:
                self.segment(30, top, 490, top, self.LINE, .35)

    def panel_c(self, importance, cg_overlay=None, panel_label='c'):
        trajectories = importance["trajectories"]
        style = IMPORTANCE_STYLES[importance.get("database_key", "jaspar")]
        highlights = dict(style['highlights'])
        legend = style['legend']
        if cg_overlay is not None:
            trajectories = trajectories + [cg_overlay['profile']]
            highlights[cg_overlay['profile']['id']] = CA
            legend = ((255, 'Other FlyFactorSurvey', '#AFAFAF'),
                      (384, 'GAF', GA), (426, 'cg (JASPAR)', CA))
        labels = ["8" if label == "[8-8]" else label
                  for label in importance.get("group_labels", LABELS)]
        if len(labels) != 8 or any(len(row["means"]) != 8 for row in trajectories):
            raise ValueError("Eight labelled degree groups required")
        self.text(20, 387, panel_label, 8, bold=True)
        if not self.revised_layout:
            self.text(43, 387, "Mean motif importance (logit / bp)", 7)
        for x, label, color in legend:
            self.segment(x, 389, x + 10, 389, color, 1.3)
            self.text(x + 14, 386.7, label, 5.8)
        x, y, width, height = 53, 258, 428, 112
        if self.revised_layout:
            x, width = 63, 418
            self.y_axis_title(26, y + height / 2, 'Mean motif importance (logit / bp)')
        low, high, ticks = limits_and_ticks(trajectories)
        self.segment(x, y, x, y + height); self.segment(x, y, x + width, y)
        for tick in ticks:
            yy = y + height * (tick - low) / (high - low)
            self.segment(x - 2.5, yy, x, yy)
            self.text(x - 6, yy - 2, f'{tick:.3g}', 6, align="right")
        for i, label in enumerate(labels):
            xx = x + width * i / 7
            self.segment(xx, y - 2.5, xx, y)
            self.text(xx, y - 12, label, 6, align="center")
        def draw(row, color, highlight=False):
            points = [(x + width * i / 7, y + height * (value - low) / (high - low))
                      if value is not None else None for i, value in enumerate(row["means"])]
            for a, b in zip(points, points[1:]):
                if a is not None and b is not None:
                    self.segment(*a, *b, color, 1.2 if highlight else .35)
            self.c.setFillColor(self.color(color))
            for point in points:
                if point is not None:
                    self.c.circle(*point, 1.8 if highlight else .65, fill=1, stroke=0)
        for row in trajectories:
            if row["id"] not in highlights:
                draw(row, "#C2C2C2")
        for row in trajectories:
            if row["id"] in highlights:
                draw(row, highlights[row["id"]], True)
        self.text(x + width / 2, 233, "Degree of pleiotropy", 7, align="center")

    def panel_d(self, activity):
        self.text(20, 208, "d", 8, bold=True)
        if not self.revised_layout:
            self.text(43, 208, "Predicted activity (probability)", 7)
        at = 66
        for label, color in zip(VARIANT_LABELS, VARIANT_COLORS):
            self.c.setFillColor(self.color(color)); self.c.setStrokeColor(self.color(self.INK)); self.c.setLineWidth(.4)
            self.c.rect(at, 192, 5, 5, stroke=1, fill=1)
            self.text(at + 9, 192, label, 6)
            at += 9 + self.metrics.stringWidth(label, 'Atlas', 6) + 15
        x, y, width, height = 43, 65, 443, 115
        if self.revised_layout:
            x, width = 53, 433
            self.y_axis_title(26, y + height / 2, 'Predicted activity (probability)')
        self.segment(x, y, x, y + height); self.segment(x, y, x + width, y)
        for tick in (0, .5, 1):
            yy = y + height * tick
            self.segment(x - 2.5, yy, x, yy)
            self.text(x - 6, yy - 2, f'{tick:g}', 6, align="right")
        for i, context in enumerate(activity["contexts"]):
            center = x + width * (i + .5) / 8
            self.segment(center, y - 2.5, center, y)
            self.text(center, y - 13, context, 6.2, align="center")
            for j, box in enumerate(activity["boxes"][i]):
                xx, bw = center + (j - 2) * 7.4, 5.7
                low, q1, median, q3, high = [y + height * box[key] for key in ("low", "q1", "median", "q3", "high")]
                self.segment(xx, low, xx, high, width=.45)
                self.segment(xx - bw / 3, low, xx + bw / 3, low, width=.45)
                self.segment(xx - bw / 3, high, xx + bw / 3, high, width=.45)
                self.c.setFillColor(self.color(VARIANT_COLORS[j])); self.c.setStrokeColor(self.color(self.INK)); self.c.setLineWidth(.45)
                self.c.rect(xx - bw / 2, q1, bw, q3 - q1, stroke=1, fill=1)
                self.segment(xx - bw / 2, median, xx + bw / 2, median, width=.65)
                self.c.setFillColor(self.color(self.INK))
                for value in box["fliers"]:
                    self.c.circle(xx, y + height * value, .75, stroke=0, fill=1)
        self.text(x + width / 2, 36, "Context", 7, align="center")


    def panel_d_odds_ratios(self, activity, axis='log', panel_label='d'):
        if axis not in ('log', 'linear'):
            raise ValueError('Unknown odds-ratio axis')
        self.text(20, 208, panel_label, 8, bold=True)
        axis_label = 'Odds ratio vs WT' + (' (log scale)' if axis == 'log' else '')
        if not self.revised_layout:
            self.text(43, 208, axis_label, 7)
        labels = ['Matched control' if v == 'nonrepeat_sham' else VARIANT_LABELS[VARIANTS.index(v)]
                  for v in activity['variants']]
        colors = ['#D8D8D8' if v == 'nonrepeat_sham' else VARIANT_COLORS[VARIANTS.index(v)]
                  for v in activity['variants']]
        at = 110
        for label, color in zip(labels, colors):
            self.c.setFillColor(self.color(color)); self.c.setStrokeColor(self.color(self.INK)); self.c.setLineWidth(.4)
            self.c.rect(at, 192, 5, 5, stroke=1, fill=1)
            self.text(at + 9, 192, label, 6)
            at += 9 + self.metrics.stringWidth(label, 'Atlas', 6) + 15
        values = np.asarray(activity['log2_odds_ratios'])
        if axis == 'linear':
            low, high = 0., max(1.5, math.ceil(float(np.exp2(values).max()) * 2) / 2)
            ticks = np.arange(0., high + .25, .5)
        else:
            low = 2 * math.floor(min(0., float(values.min())) / 2)
            high = 2 * math.ceil(max(0., float(values.max())) / 2)
            if low == high:
                low, high = -2, 2
            ticks = (0.02, 0.1, 0.5, 1., 2., 4.)
        x, y, width, height = 43, 65, 443, 115
        if self.revised_layout:
            x, width = 53, 433
            self.y_axis_title(26, y + height / 2, axis_label)
        def axis_y(value):
            return y + height * (value - low) / (high - low)
        def ordinate(value):
            # Preserve the existing log-space box statistics; change only display.
            return axis_y(2. ** value if axis == 'linear' else value)
        self.segment(x, y, x, y + height); self.segment(x, y, x + width, y)
        # Decimal labels show the actual OR; avoid opaque fractions such as 1/64.
        for ratio in ticks:
            tick = ratio if axis == 'linear' else math.log2(ratio)
            if not low <= tick <= high:
                continue
            yy = axis_y(tick)
            self.segment(x - 2.5, yy, x, yy)
            self.text(x - 6, yy - 2, f'{ratio:g}', 6, bold=ratio == 1., align='right')
        self.c.saveState(); self.c.setDash(2, 2)
        self.segment(x, ordinate(0), x + width, ordinate(0), self.MUTED, .55)
        self.c.restoreState()
        for i, context in enumerate(activity['contexts']):
            center = x + width * (i + .5) / 8
            self.segment(center, y - 2.5, center, y)
            self.text(center, y - 13, context, 6.2, align='center')
            for j, box in enumerate(activity['boxes_log2'][i]):
                xx, bw = center + (j - (len(activity['variants']) - 1) / 2) * 10, 7.4
                lower, q1, median, q3, upper = [ordinate(box[key]) for key in ('low', 'q1', 'median', 'q3', 'high')]
                self.segment(xx, lower, xx, upper, width=.45)
                self.segment(xx - bw / 3, lower, xx + bw / 3, lower, width=.45)
                self.segment(xx - bw / 3, upper, xx + bw / 3, upper, width=.45)
                self.c.setFillColor(self.color(colors[j])); self.c.setStrokeColor(self.color(self.INK)); self.c.setLineWidth(.45)
                self.c.rect(xx - bw / 2, q1, bw, q3 - q1, stroke=1, fill=1)
                self.segment(xx - bw / 2, median, xx + bw / 2, median, width=.65)
                self.c.setFillColor(self.color(self.INK))
                for value in box['fliers']:
                    self.c.circle(xx, ordinate(value), .75, stroke=0, fill=1)
        self.text(x + width / 2, 36, 'Context', 7, align='center')


def main(project, importance_root=IMPORTANCE, prefix=PREFIX, activity_scale='probability', odds_axis='log', include_sham=False, highlight_cg=False, revised_layout=False):
    audit, rows, _, _, sources = load_inputs(project, positive_limit=4 if revised_layout else 2)
    importance = completed_json(project / importance_root, "summary.json")
    style = IMPORTANCE_STYLES[importance.get("database_key", "jaspar")]
    if (importance["profiles"] != style['count'] or len(importance["trajectories"]) != style['count']
            or importance["split"] != "train" or importance["references"] != 50):
        raise ValueError("Expected complete fixed-reference 50-reference training importance")
    if not set(style['highlights']) <= {r['id'] for r in importance['trajectories']}:
        raise ValueError("Missing required highlight profiles")
    cg_overlay = load_cg_overlay(project, importance) if highlight_cg else None
    if activity_scale not in ('probability', 'odds-ratio-wt'):
        raise ValueError('Unknown activity scale')
    odds_ratio = activity_scale == 'odds-ratio-wt'
    activity = load_odds_ratios(project, include_sham=include_sham) if odds_ratio else load_activity(project)
    if odds_ratio and prefix == PREFIX:
        prefix += '_odds_ratio_wt'
        if odds_axis == 'linear':
            prefix += '_linear'
        if include_sham:
            prefix += '_control'
    if highlight_cg:
        prefix += '_cg'
    if revised_layout:
        prefix += '_expanded'
    path = project / 'output/pdf' / (prefix + '.pdf')
    figure = NatureFigure(audit, path, project / SUPPORT / 'fonts')
    figure.initialize(revised_layout=revised_layout, motif_count=len(rows))
    figure.panel_b(rows); figure.panel_c(importance, cg_overlay=cg_overlay)
    if odds_ratio:
        figure.panel_d_odds_ratios(activity, axis=odds_axis)
    else:
        figure.panel_d(activity)
    overlay_note = ' plus JASPAR cg (MA2107.1)' if highlight_cg else ''
    figure.c.setSubject(f"Panel c: all {style['count']} {style['name']} profiles{overlay_note}; panel b: unchanged JASPAR motif annotations; panel d: frozen perturbation predictions")
    check_text_bounds(figure.text_bounds)
    figure.c.showPage(); figure.c.save()
    sources[str(Path(importance_root) / 'summary.json')] = digest(project / importance_root / 'summary.json')
    sources[PERTURBATION + '/predictions.npz'] = activity['predictions_sha256']
    result = dict(panel_a='omitted', panel_b=rows, panel_c=importance, panel_d=activity,
                  inputs=sources, builder_sha256=digest(Path(__file__)), pdf_sha256=digest(path),
                  b_support_axis_percent=[0, 100], groups=LABELS,
                  panel_c_groups=importance["groups"], panel_c_group_labels=importance["group_labels"],
                  panel_c_highlights=style['highlights'], panel_d_axis=odds_axis if odds_ratio else 'linear',
                  text_bounds=figure.text_bounds, text_overlap_check='passed', visual_qa='pending')
    if cg_overlay is not None:
        sources[cg_overlay['summary_path']] = cg_overlay['summary_sha256']
        result['panel_c_overlay'] = cg_overlay
        result['panel_c_highlights'] = {**style['highlights'], 'MA2107.1': CA}
    if revised_layout:
        result['layout'] = dict(page_size_pt=[figure.width, figure.height], panel_cd_titles=False,
                              panel_cd_y_axis_titles=True, sign_colors=SIGN_COLORS,
                              motif_count=len(rows), selection='Up to four positive motifs by original support rank per group, plus all retained negatives; unchanged information filters')
    path.with_suffix('.source.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    boxes = 8 * len(activity['variants'])
    print(f"Created {path.relative_to(project)}: {style['count']} motif profiles{overlay_note}; {boxes} context/condition boxplots", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--importance', default=IMPORTANCE)
    parser.add_argument('--prefix', default=PREFIX)
    parser.add_argument('--activity-scale', choices=('probability', 'odds-ratio-wt'), default='probability')
    parser.add_argument('--odds-axis', choices=('log', 'linear'), default='log')
    parser.add_argument('--include-sham', action='store_true', help='Include matched nonrepeat-shuffle/WT odds-ratio boxes')
    parser.add_argument('--highlight-cg', action='store_true', help='Overlay the matched exact-degree JASPAR cg curve on FlyFactorSurvey curves')
    parser.add_argument('--revised-layout', action='store_true', help='Four positive motifs per group, coloured sign boxes, and y-axis titles instead of panel headings')
    args = parser.parse_args()
    main(args.project.resolve(), args.importance, args.prefix, args.activity_scale, args.odds_axis, args.include_sham, args.highlight_cg, args.revised_layout)
