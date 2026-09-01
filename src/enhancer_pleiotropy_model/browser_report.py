"""Quantitatively validate aligned observed/predicted browser tracks."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pyBigWig

from .constants import ASSAYS, CONTEXTS
from .io import atomic_write_json, read_bed_intervals, sha256_file
from .metrics import (
    correlation,
    correlation_structure,
    regression_metrics,
    target_pca_projection_metrics,
    tissue_pattern_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-metadata", required=True, type=Path)
    parser.add_argument("--master-dhs-bed", required=True, type=Path)
    parser.add_argument("--h3k27ac-peaks-bed", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--active-quantile", default=0.90, type=float)
    parser.add_argument("--variable-quantile", default=0.75, type=float)
    parser.add_argument("--representatives-per-class", default=5, type=int)
    parser.add_argument("--representative-span-bp", default=10_000, type=int)
    parser.add_argument("--minimum-separation-bp", default=100_000, type=int)
    parser.add_argument("--igv-genome", default="dm6")
    parser.add_argument("--seed", default=20260829, type=int)
    return parser.parse_args()


def resolve_recorded_path(recorded: str, metadata_path: Path) -> Path:
    path = Path(recorded)
    if path.is_absolute() and path.exists():
        return path
    candidates = [Path.cwd() / path]
    candidates.extend(parent / path for parent in metadata_path.resolve().parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve path from browser metadata: {recorded}")


def load_track_matrix(
    paths: list[Path], chromosome: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    starts: np.ndarray | None = None
    ends: np.ndarray | None = None
    columns = []
    chromosome_header: dict[str, int] | None = None
    for path in paths:
        with pyBigWig.open(str(path)) as bigwig:
            header = dict(bigwig.chroms())
            intervals = bigwig.intervals(chromosome)
        if intervals is None:
            raise ValueError(f"{path}: no intervals for {chromosome}")
        current_starts = np.fromiter((row[0] for row in intervals), dtype=np.int64)
        current_ends = np.fromiter((row[1] for row in intervals), dtype=np.int64)
        values = np.fromiter((row[2] for row in intervals), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: signal contains non-finite values")
        if starts is None:
            starts, ends, chromosome_header = current_starts, current_ends, header
        elif not (
            np.array_equal(starts, current_starts)
            and np.array_equal(ends, current_ends)
        ):
            raise ValueError(f"{path}: intervals do not match the first track")
        elif header != chromosome_header:
            raise ValueError(f"{path}: chromosome header does not match")
        columns.append(values)
    assert starts is not None and ends is not None and chromosome_header is not None
    return starts, ends, np.column_stack(columns), chromosome_header


def interval_overlap_mask(
    bin_starts: np.ndarray,
    bin_ends: np.ndarray,
    intervals: Iterable[tuple[int, int]],
) -> np.ndarray:
    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    if not ordered:
        return np.zeros(len(bin_starts), dtype=bool)
    starts = np.asarray([item[0] for item in ordered], dtype=np.int64)
    maximum_ends = np.maximum.accumulate(
        np.asarray([item[1] for item in ordered], dtype=np.int64)
    )
    previous = np.searchsorted(starts, bin_ends, side="left") - 1
    result = np.zeros(len(bin_starts), dtype=bool)
    valid = previous >= 0
    result[valid] = maximum_ends[previous[valid]] > bin_starts[valid]
    return result


def informative_mask(
    labels: np.ndarray, active_quantile: float, variable_quantile: float
) -> tuple[np.ndarray, dict[str, float]]:
    activity = labels.mean(axis=1)
    variability = labels.std(axis=1)
    activity_threshold = float(np.quantile(activity, active_quantile))
    active = activity >= activity_threshold
    variability_threshold = float(np.quantile(variability[active], variable_quantile))
    return active & (variability >= variability_threshold), {
        "activity_threshold_log1p_mean": activity_threshold,
        "variability_threshold_log1p_sd": variability_threshold,
    }


def extended_regression_metrics(
    labels: np.ndarray, predictions: np.ndarray, contexts: tuple[str, ...]
) -> dict[str, object]:
    result = regression_metrics(labels, predictions, contexts)
    for index, context in enumerate(contexts):
        residual = predictions[:, index] - labels[:, index]
        result["by_context"][context]["rmse"] = float(
            np.sqrt(np.square(residual).mean())
        )
        result["by_context"][context]["bias"] = float(residual.mean())
        target_sd = float(labels[:, index].std())
        result["by_context"][context]["prediction_to_target_sd"] = (
            float(predictions[:, index].std() / target_sd)
            if target_sd > 0
            else float("nan")
        )
    for metric in ("rmse", "bias", "prediction_to_target_sd"):
        result["macro"][metric] = float(
            np.nanmean(
                [values[metric] for values in result["by_context"].values()]
            )
        )
    return result


def top_context_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    if not len(labels):
        return float("nan")
    return float((labels.argmax(axis=1) == predictions.argmax(axis=1)).mean())


def assay_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    masks: dict[str, np.ndarray],
    active_quantile: float,
    variable_quantile: float,
    contexts: tuple[str, ...],
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    labels = np.log1p(np.maximum(observed, 0))
    predictions = np.log1p(np.maximum(predicted, 0))
    informative, thresholds = informative_mask(
        labels, active_quantile, variable_quantile
    )
    all_eligible = np.ones(len(labels), dtype=bool)
    result: dict[str, object] = {
        "bins": len(labels),
        "transform": "log1p(max(signal, 0))",
        "per_context": extended_regression_metrics(labels, predictions, contexts),
        "tissue_pattern_all_variable": tissue_pattern_metrics(
            labels, predictions, all_eligible
        ),
        "tissue_pattern_informative": tissue_pattern_metrics(
            labels, predictions, informative
        ),
        "informative_thresholds": thresholds,
        "informative_bins": int(informative.sum()),
        "top_context_accuracy_informative": top_context_accuracy(
            labels[informative], predictions[informative]
        ),
        "correlation_structure": correlation_structure(
            labels, predictions, contexts
        ),
        "target_pca": target_pca_projection_metrics(
            labels, predictions, contexts
        ),
    }
    strata_rows: list[dict[str, object]] = []
    for name, mask in masks.items():
        if mask.sum() < 2:
            continue
        regression = extended_regression_metrics(labels[mask], predictions[mask], contexts)
        patterns = tissue_pattern_metrics(
            labels[mask], predictions[mask], np.ones(int(mask.sum()), dtype=bool)
        )
        strata_rows.append(
            {
                "stratum": name,
                "bins": int(mask.sum()),
                "macro_pearson": regression["macro"]["pearson"],
                "macro_spearman": regression["macro"]["spearman"],
                "macro_rmse": regression["macro"]["rmse"],
                "tissue_pattern_mean_pearson": patterns["mean_pearson"],
                "tissue_pattern_finite_bins": patterns["finite_windows"],
                "top_context_accuracy": top_context_accuracy(
                    labels[mask], predictions[mask]
                ),
            }
        )
    return result, strata_rows, informative


def row_correlations(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    left = labels - labels.mean(axis=1, keepdims=True)
    right = predictions - predictions.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.full(len(left), np.nan),
        where=denominator > 0,
    )


def block_representatives(
    chromosome: str,
    starts: np.ndarray,
    ends: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    assay: str,
    span_bp: int,
    count: int,
    minimum_separation_bp: int,
) -> list[dict[str, object]]:
    block_ids = starts // span_bp
    rows = []
    for block_id in np.unique(block_ids):
        mask = block_ids == block_id
        label = np.log1p(np.maximum(observed[mask], 0)).mean(axis=0)
        prediction = np.log1p(np.maximum(predicted[mask], 0)).mean(axis=0)
        rows.append(
            {
                "chromosome": chromosome,
                "start": int(starts[mask].min()),
                "end": int(ends[mask].max()),
                "assay": assay,
                "activity": float(label.mean()),
                "variability": float(label.std()),
                "pattern_pearson": correlation(label, prediction),
                "mae": float(np.abs(label - prediction).mean()),
                "observed_log1p": label.tolist(),
                "predicted_log1p": prediction.tolist(),
                "observed_top_context": CONTEXTS[int(label.argmax())],
                "predicted_top_context": CONTEXTS[int(prediction.argmax())],
            }
        )
    activities = np.asarray([row["activity"] for row in rows])
    variable = np.asarray([row["variability"] for row in rows])
    eligible = [
        row
        for row in rows
        if row["activity"] >= np.quantile(activities, 0.75)
        and row["variability"] >= np.quantile(variable, 0.50)
        and math.isfinite(row["pattern_pearson"])
    ]

    def select(category: str, reverse: bool) -> list[dict[str, object]]:
        ordered = sorted(
            eligible,
            key=lambda row: (row["pattern_pearson"], -row["mae"]),
            reverse=reverse,
        )
        selected = []
        for row in ordered:
            midpoint = (row["start"] + row["end"]) // 2
            if all(
                abs(midpoint - (other["start"] + other["end"]) // 2)
                >= minimum_separation_bp
                for other in selected
            ):
                selected.append(dict(row, category=category, rank=len(selected) + 1))
            if len(selected) == count:
                break
        return selected

    return select("success", True) + select("failure", False)


def atomic_write_bigwig(
    path: Path,
    chromosome_header: dict[str, int],
    chromosome: str,
    starts: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".bw", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with pyBigWig.open(str(temporary), "w") as bigwig:
            bigwig.addHeader(list(chromosome_header.items()))
            for offset in range(0, len(values), 100_000):
                slc = slice(offset, offset + 100_000)
                bigwig.addEntries(
                    [chromosome] * len(starts[slc]),
                    starts[slc].tolist(),
                    ends=ends[slc].tolist(),
                    values=values[slc].astype(np.float64).tolist(),
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, list):
                    value = ",".join(f"{number:.8g}" for number in value)
                values.append(str(value))
            handle.write("\t".join(values) + "\n")
    temporary.replace(path)


def write_bed(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            name = (
                f"{row['assay']}_{row['category']}_{row['rank']}"
                f"_rho={row['pattern_pearson']:.3f}"
            )
            handle.write(
                f"{row['chromosome']}\t{row['start']}\t{row['end']}\t{name}\t0\n"
            )
    temporary.replace(path)


def write_igv_session(
    path: Path,
    genome: str,
    chromosome: str,
    region_start: int,
    region_end: int,
    source_paths: dict[str, dict[str, dict[str, Path]]],
    residual_paths: dict[str, dict[str, Path]],
    bookmarks: Path,
) -> None:
    root = ET.Element(
        "Session",
        genome=genome,
        locus=f"{chromosome}:{region_start + 1}-{region_end}",
        version="3",
    )
    resources = ET.SubElement(root, "Resources")
    panel = ET.SubElement(root, "Panel", height="1600", name="DataPanel", width="1400")
    colors = {
        "observed": "44,123,182",
        "predicted": "215,25,28",
        "residual": "117,107,177",
    }
    for assay in ASSAYS:
        for context in CONTEXTS:
            tracks = [
                (source, source_paths[source][assay][context])
                for source in ("observed", "predicted")
            ] + [("residual", residual_paths[assay][context])]
            for source, track_path in tracks:
                relative = os.path.relpath(track_path, path.parent)
                name = f"{assay.upper()} {context} {source}"
                ET.SubElement(resources, "Resource", name=name, path=relative)
                ET.SubElement(
                    panel,
                    "Track",
                    color=colors[source],
                    displayMode="COLLAPSED",
                    height="32",
                    id=relative,
                    name=name,
                    renderer="BAR_CHART",
                    visible="true",
                    windowFunction="mean",
                )
    relative_bed = os.path.relpath(bookmarks, path.parent)
    ET.SubElement(resources, "Resource", name="Representative loci", path=relative_bed)
    ET.SubElement(
        panel,
        "Track",
        color="0,0,0",
        displayMode="EXPANDED",
        height="60",
        id=relative_bed,
        name="Representative loci",
        visible="true",
    )
    ET.indent(root, space="  ")
    temporary = path.with_suffix(path.suffix + ".tmp")
    ET.ElementTree(root).write(temporary, encoding="UTF-8", xml_declaration=True)
    temporary.replace(path)


def finite_or_none(value: object) -> object:
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def format_value(value: object) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def html_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    heading = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(format_value(row.get(column)))}</td>" for column in columns
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table>"


def matrix_table(matrix: list[list[float]], contexts: tuple[str, ...]) -> str:
    values = np.asarray(matrix, dtype=np.float64)
    rows = []
    for index, context in enumerate(contexts):
        row = {"context": context}
        row.update({name: values[index, j] for j, name in enumerate(contexts)})
        rows.append(row)
    return html_table(rows, ["context", *contexts])


def histogram_svg(values: np.ndarray, title: str) -> str:
    finite = values[np.isfinite(values)]
    counts, edges = np.histogram(finite, bins=20, range=(-1, 1))
    width, height, margin = 640, 220, 32
    maximum = max(int(counts.max()), 1)
    bars = []
    usable_width = width - 2 * margin
    for index, count in enumerate(counts):
        x = margin + index * usable_width / len(counts)
        bar_width = usable_width / len(counts) - 1
        bar_height = (height - 2 * margin) * int(count) / maximum
        bars.append(
            f'<rect x="{x:.1f}" y="{height-margin-bar_height:.1f}" '
            f'width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2c7bb6"/>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">'
        f'<text x="{margin}" y="18">{html.escape(title)}</text>'
        + "".join(bars)
        + f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" '
        f'y2="{height-margin}" stroke="black"/>'
        f'<text x="{margin}" y="{height-6}">-1</text>'
        f'<text x="{width-margin-8}" y="{height-6}">1</text></svg>'
    )


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def render_report(
    path: Path,
    metrics: dict[str, dict[str, object]],
    per_context_rows: list[dict[str, object]],
    strata_rows: list[dict[str, object]],
    representative_rows: list[dict[str, object]],
    pattern_values: dict[str, np.ndarray],
    provenance: dict[str, object],
) -> None:
    sections = []
    for assay in ASSAYS:
        assay_metrics_value = metrics[assay]
        sections.append(f"<h2>{assay.upper()}</h2>")
        sections.append(
            html_table(
                [row for row in per_context_rows if row["assay"] == assay],
                [
                    "context",
                    "pearson",
                    "spearman",
                    "rmse",
                    "mae",
                    "r2",
                    "bias",
                    "prediction_to_target_sd",
                ],
            )
        )
        pattern = assay_metrics_value["tissue_pattern_informative"]
        sections.append(
            "<p><strong>Informative-bin tissue-pattern Pearson:</strong> "
            f"mean {format_value(pattern['mean_pearson'])}, median "
            f"{format_value(pattern['median_pearson'])}; top-context accuracy "
            f"{format_value(assay_metrics_value['top_context_accuracy_informative'])}. "
            "Informative thresholds were fixed by the configured validation-bin "
            "activity and variability quantiles.</p>"
        )
        sections.append(histogram_svg(pattern_values[assay], f"{assay} tissue-pattern Pearson"))
        structure = assay_metrics_value["correlation_structure"]
        sections.append(
            "<p><strong>Over-correlation (predicted minus observed, off-diagonal "
            f"mean):</strong> {format_value(structure['mean_prediction_minus_true'])}; "
            f"matrix MAE {format_value(structure['mean_absolute_error'])}.</p>"
        )
        sections.append("<h3>Observed context correlation</h3>")
        sections.append(matrix_table(structure["true_matrix"], CONTEXTS))
        sections.append("<h3>Predicted context correlation</h3>")
        sections.append(matrix_table(structure["predicted_matrix"], CONTEXTS))
        pca_rows = [
            {
                "PC": item["component"],
                "target_variance": item["target_explained_variance_ratio"],
                "prediction_pearson": item["prediction_pearson"],
                "prediction_r2": item["prediction_r2"],
            }
            for item in assay_metrics_value["target_pca"]["components"]
        ]
        sections.append("<h3>Target PCA modes</h3>")
        sections.append(
            html_table(
                pca_rows,
                ["PC", "target_variance", "prediction_pearson", "prediction_r2"],
            )
        )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Browser-track validation</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse;margin:1rem 0;font-size:.88rem}}th,td{{border:1px solid #ccc;padding:.35rem .5rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eee;position:sticky;top:0}}code{{background:#eee;padding:.1rem .25rem}}.warning{{border-left:5px solid #d95f02;padding:.7rem;background:#fff3e6}}svg{{max-width:640px;border:1px solid #ddd;margin:.5rem 0}}
</style></head><body>
<h1>Observed versus predicted validation tracks</h1>
<p class="warning">This checkpoint was selected on this chr2L validation interval. Results are model-development QC, not an untouched test estimate.</p>
<p>Metrics use aligned native model bins and <code>log1p(max(signal, 0))</code>. ATAC bins are 16 bp; H3K27ac bins are 64 bp. No thresholds were tuned on the test chromosome.</p>
<h2>Run provenance</h2>
{html_table([provenance], ['analysis_date_utc','git_commit','chromosome','region','checkpoint_epoch','checkpoint_sha256','seed'])}
{''.join(sections)}
<h2>Regulatory strata</h2>
<p>Each native bin is classified by interval overlap with the master DHS and consensus H3K27ac peak BEDs.</p>
{html_table(strata_rows, ['assay','stratum','bins','macro_pearson','macro_spearman','macro_rmse','tissue_pattern_mean_pearson','top_context_accuracy'])}
<h2>Representative loci</h2>
<p>Successes and failures are deterministic, active/variable 10-kb blocks ranked by across-context Pearson and separated spatially. They are bookmarks, not independent statistical units.</p>
{html_table(representative_rows, ['assay','category','rank','chromosome','start','end','activity','variability','pattern_pearson','mae','observed_top_context','predicted_top_context'])}
<h2>Files</h2>
<ul><li><a href="metrics.json">Complete metrics JSON</a></li>
<li><a href="per_context_metrics.tsv">Per-context metrics TSV</a></li>
<li><a href="stratified_metrics.tsv">Stratified metrics TSV</a></li>
<li><a href="representative_loci.tsv">Representative loci TSV</a></li>
<li><a href="representative_loci.bed">IGV bookmark BED</a></li>
<li><a href="igv_session_with_residuals.xml">IGV session including residual tracks</a></li>
<li><a href="analysis_config.json">Configuration and input hashes</a></li></ul>
<h2>Interpretation limits</h2><p>Correlations describe agreement, not calibration or causality. Native bins overlap through the sliding-window averaging process and are not independent observations. Peak-overlap strata depend on the supplied BED definitions.</p>
</body></html>"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not 0 < args.active_quantile < 1 or not 0 < args.variable_quantile < 1:
        raise ValueError("Activity and variability quantiles must lie between 0 and 1")
    if (
        args.representatives_per_class < 1
        or args.representative_span_bp < 1
        or args.minimum_separation_bp < 0
    ):
        raise ValueError("Representative-locus parameters are invalid")
    metadata = json.loads(args.browser_metadata.read_text(encoding="utf-8"))
    contexts = tuple(metadata["contexts"])
    if contexts != CONTEXTS:
        raise ValueError(f"Expected context order {CONTEXTS}, found {contexts}")
    chromosome = metadata["chromosome"]
    region_start = int(metadata["region"]["start"])
    region_end = int(metadata["region"]["end"])
    dhs_intervals = read_bed_intervals(args.master_dhs_bed).get(chromosome, ())
    h3_intervals = read_bed_intervals(args.h3k27ac_peaks_bed).get(chromosome, ())

    source_paths = {
        source: {
            assay: {
                context: resolve_recorded_path(
                    metadata["output_bigwigs"][source][assay][context],
                    args.browser_metadata,
                )
                for context in contexts
            }
            for assay in ASSAYS
        }
        for source in ("observed", "predicted")
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    residual_directory = args.output_directory / "residuals"
    residual_paths = {assay: {} for assay in ASSAYS}
    metrics: dict[str, dict[str, object]] = {}
    all_strata_rows = []
    all_representative_rows = []
    all_per_context_rows = []
    pattern_values: dict[str, np.ndarray] = {}
    chromosome_header: dict[str, int] | None = None

    for assay in ASSAYS:
        starts, ends, observed, observed_header = load_track_matrix(
            [source_paths["observed"][assay][context] for context in contexts],
            chromosome,
        )
        pred_starts, pred_ends, predicted, predicted_header = load_track_matrix(
            [source_paths["predicted"][assay][context] for context in contexts],
            chromosome,
        )
        if not (
            np.array_equal(starts, pred_starts)
            and np.array_equal(ends, pred_ends)
            and observed_header == predicted_header
        ):
            raise ValueError(f"{assay}: observed/predicted track geometry differs")
        chromosome_header = observed_header
        if np.any(observed < 0) or np.any(predicted < 0):
            raise ValueError(f"{assay}: source tracks must be nonnegative")
        dhs_mask = interval_overlap_mask(starts, ends, dhs_intervals)
        h3_mask = interval_overlap_mask(starts, ends, h3_intervals)
        masks = {
            "DHS_and_H3K27ac_peak": dhs_mask & h3_mask,
            "DHS_only": dhs_mask & ~h3_mask,
            "H3K27ac_peak_only": ~dhs_mask & h3_mask,
            "background": ~dhs_mask & ~h3_mask,
        }
        assay_result, strata_rows, informative = assay_metrics(
            observed,
            predicted,
            masks,
            args.active_quantile,
            args.variable_quantile,
            contexts,
        )
        metrics[assay] = assay_result
        for context in contexts:
            all_per_context_rows.append(
                dict(
                    assay=assay,
                    context=context,
                    **assay_result["per_context"]["by_context"][context],
                )
            )
        all_strata_rows.extend(dict(assay=assay, **row) for row in strata_rows)
        all_representative_rows.extend(
            block_representatives(
                chromosome,
                starts,
                ends,
                observed,
                predicted,
                assay,
                args.representative_span_bp,
                args.representatives_per_class,
                args.minimum_separation_bp,
            )
        )
        logged_observed = np.log1p(observed)
        logged_predicted = np.log1p(predicted)
        pattern_values[assay] = row_correlations(
            logged_observed[informative], logged_predicted[informative]
        )
        for index, context in enumerate(contexts):
            path = residual_directory / f"residual.{context}.{assay}.bw"
            atomic_write_bigwig(
                path,
                observed_header,
                chromosome,
                starts,
                ends,
                logged_predicted[:, index] - logged_observed[:, index],
            )
            residual_paths[assay][context] = path

    if chromosome_header is None:
        raise RuntimeError("No assays were analyzed")
    per_context_columns = [
        "assay",
        "context",
        "n",
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "r2",
        "bias",
        "prediction_to_target_sd",
    ]
    strata_columns = [
        "assay",
        "stratum",
        "bins",
        "macro_pearson",
        "macro_spearman",
        "macro_rmse",
        "tissue_pattern_mean_pearson",
        "tissue_pattern_finite_bins",
        "top_context_accuracy",
    ]
    representative_columns = [
        "assay",
        "category",
        "rank",
        "chromosome",
        "start",
        "end",
        "activity",
        "variability",
        "pattern_pearson",
        "mae",
        "observed_top_context",
        "predicted_top_context",
        "observed_log1p",
        "predicted_log1p",
    ]
    write_tsv(
        args.output_directory / "per_context_metrics.tsv",
        all_per_context_rows,
        per_context_columns,
    )
    write_tsv(
        args.output_directory / "stratified_metrics.tsv",
        all_strata_rows,
        strata_columns,
    )
    write_tsv(
        args.output_directory / "representative_loci.tsv",
        all_representative_rows,
        representative_columns,
    )
    bookmarks = args.output_directory / "representative_loci.bed"
    write_bed(bookmarks, all_representative_rows)
    session_path = args.output_directory / "igv_session_with_residuals.xml"
    write_igv_session(
        session_path,
        args.igv_genome,
        chromosome,
        region_start,
        region_end,
        source_paths,
        residual_paths,
        bookmarks,
    )
    provenance = {
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "chromosome": chromosome,
        "region": f"{region_start}-{region_end}",
        "checkpoint_epoch": metadata["checkpoint"]["epoch"],
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "seed": args.seed,
    }
    input_tracks = {
        source: {
            assay: {
                context: {
                    "path": str(source_paths[source][assay][context]),
                    "sha256": sha256_file(source_paths[source][assay][context]),
                }
                for context in contexts
            }
            for assay in ASSAYS
        }
        for source in ("observed", "predicted")
    }
    analysis_config = {
        **provenance,
        "browser_metadata": {
            "path": str(args.browser_metadata.resolve()),
            "sha256": sha256_file(args.browser_metadata),
        },
        "master_dhs_bed": {
            "path": str(args.master_dhs_bed.resolve()),
            "sha256": sha256_file(args.master_dhs_bed),
        },
        "h3k27ac_peaks_bed": {
            "path": str(args.h3k27ac_peaks_bed.resolve()),
            "sha256": sha256_file(args.h3k27ac_peaks_bed),
        },
        "input_tracks": input_tracks,
        "parameters": {
            "active_quantile": args.active_quantile,
            "variable_quantile": args.variable_quantile,
            "representatives_per_class": args.representatives_per_class,
            "representative_span_bp": args.representative_span_bp,
            "minimum_separation_bp": args.minimum_separation_bp,
            "igv_genome": args.igv_genome,
            "seed": args.seed,
        },
        "software": {
            package: importlib.metadata.version(package)
            for package in ("enhancer-pleiotropy-model", "numpy", "scipy", "pyBigWig")
        },
        "residual_definition": "log1p(predicted) - log1p(observed)",
    }
    atomic_write_json(
        args.output_directory / "analysis_config.json", finite_or_none(analysis_config)
    )
    complete_metrics = {
        "provenance": provenance,
        "definitions": {
            "signal_transform": "log1p(max(signal, 0))",
            "tissue_pattern": (
                "Pearson across the eight contexts within each native genomic bin"
            ),
            "overcorrelation": (
                "mean(predicted context-pair correlation - observed context-pair "
                "correlation) across 28 off-diagonal pairs"
            ),
            "residual": "log1p(predicted) - log1p(observed)",
        },
        "assays": metrics,
    }
    atomic_write_json(
        args.output_directory / "metrics.json", finite_or_none(complete_metrics)
    )
    render_report(
        args.output_directory / "index.html",
        metrics,
        all_per_context_rows,
        all_strata_rows,
        all_representative_rows,
        pattern_values,
        provenance,
    )
    print(
        json.dumps(
            {
                "event": "browser_report_complete",
                "output_directory": str(args.output_directory),
                "report": str(args.output_directory / "index.html"),
                "residual_tracks": len(ASSAYS) * len(CONTEXTS),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
