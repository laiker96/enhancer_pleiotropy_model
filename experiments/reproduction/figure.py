"""Deterministic, portable replay of the last paper figure, including examples."""
import copy
import json
from pathlib import Path

from classifier_modisco.paper_figure3 import check_text_bounds
from classifier_modisco.paper_figure3_examples import ExampleFigure, EXTRA_HEIGHT
from .common import digest, write_json


def replay(cfg, output=None):
    source_path = Path(cfg["paths"]["figure_source"])
    style_path = Path(cfg["paths"]["figure_style"])
    manifest = json.loads(source_path.with_name("manifest.json").read_text())
    for path in (source_path, style_path):
        if digest(path) != manifest["files"][path.name]:
            raise ValueError("Historical figure input changed: "+str(path))
    source = json.loads(source_path.read_text())
    style = json.loads(style_path.read_text())
    path = Path(output) if output else Path(cfg["paths"]["work"]) / "figure3_replay.pdf"
    if path.exists() or path.with_suffix(".receipt.json").exists():
        raise FileExistsError("Use a new figure output filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    font_dir = Path(cfg["paths"]["font_directory"])
    figure = ExampleFigure(style, path, font_dir)
    figure.initialize(revised_layout=True, motif_count=len(source["panel_b"]))
    figure.height += EXTRA_HEIGHT
    figure.c.setPageSize((figure.width, figure.height))
    figure.c.setTitle("Figure 3b-e | Native enhancer motifs with uniform information filter")
    figure.c.setSubject("Original 50-reference attribution; uniformly re-filtered raw motifs and new JASPAR matches")
    figure.panel_b(source["panel_b"])
    figure.examples_panel(copy.deepcopy(source["panel_c"]))
    figure.panel_c(source["panel_d"], panel_label="d")
    figure.panel_d_odds_ratios(source["panel_e"], axis="linear", panel_label="e")
    check_text_bounds(figure.text_bounds)
    figure.c.showPage()
    figure.c.save()
    write_json(path.with_suffix(".receipt.json"), dict(status="complete", pdf_sha256=digest(path),
        historical_pdf_sha256=manifest["historical_pdf_sha256"],
        byte_identical=digest(path)==manifest["historical_pdf_sha256"],
        inputs=manifest["files"], text_bounds=figure.text_bounds,
        fonts={p.name: digest(p) for p in (font_dir / n for n in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf"))},
        mode="Historical source-data replay, not a new model analysis"))
    print(path)
