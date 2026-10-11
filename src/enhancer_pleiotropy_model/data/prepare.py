"""Build the regression and classification datasets described by ``configs/data.yaml``."""

from __future__ import annotations

import time
from pathlib import Path

from ..constants import ASSAYS, CONTEXTS, SOURCE_BIN_BP
from ..io import write_json
from .background import prepare_backgrounds
from .enhancers import prepare_enhancers
from .peaks import build_h3k27ac_consensus
from .profiles import bigwig_path, build_profiles
from .windows import build_windows


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def input_paths(config: dict) -> dict:
    """Resolve and check every input file before any work starts."""
    inputs = config["inputs"]
    paths = {key: Path(inputs[key]) for key in
             ("reference_fasta", "blacklist", "master_dhs", "master_dhs_summits", "catalog")}
    paths["bigwigs"] = [bigwig_path(inputs["bigwigs"], c, a) for a in ASSAYS for c in CONTEXTS]
    pattern = inputs.get("h3k27ac_peak_file", "{library}/{library}_peaks.broadPeak")
    paths["h3k27ac_peaks"] = {
        context: [Path(inputs["h3k27ac_peaks"]) / pattern.format(library=library)
                  for library in inputs["h3k27ac_libraries"][context]]
        for context in CONTEXTS}
    files = [*(paths[k] for k in ("reference_fasta", "blacklist", "master_dhs", "master_dhs_summits", "catalog")),
             *paths["bigwigs"], *(p for ps in paths["h3k27ac_peaks"].values() for p in ps)]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files (check the paths in your data config):\n  "
                                + "\n  ".join(missing))
    return paths


def prepare(config: dict, only: str | None = None) -> None:
    paths = input_paths(config)
    root = Path(config["output"])
    regression, classification = root / "regression", root / "classification"
    r = config["regression"]
    if only in (None, "regression"):
        log("building the H3K27ac consensus peak set")
        consensus = regression / "h3k27ac_consensus.bed"
        build_h3k27ac_consensus(paths["h3k27ac_peaks"], consensus, regression / "h3k27ac_consensus.json")
        log("sampling regression windows (about 10 minutes)")
        windows = build_windows(
            reference_fasta=paths["reference_fasta"], blacklist=paths["blacklist"],
            master_dhs=paths["master_dhs"], master_dhs_summits=paths["master_dhs_summits"],
            h3k27ac_peaks=consensus, bigwig_paths=paths["bigwigs"], splits=config["splits"],
            output=regression / "windows.tsv.gz", metadata_path=regression / "windows.json",
            target_bp=r["target_bp"], flank_bp=r["flank_bp"], train_stride_bp=r["train_stride_bp"],
            validation_stride_bp=r["validation_stride_bp"],
            background_to_peak_ratio=r["background_to_peak_ratio"], block_size_bp=r["block_size_bp"],
            seed=r["seed"])
        log(f"windows per split: {windows['window_counts']} "
            "(manuscript: train 757,401, validation 53,313, test 140,318)")
        for assay in ASSAYS:
            log(f"extracting {assay} profiles")
            build_profiles(windows=regression / "windows.tsv.gz", bigwig_dir=Path(config["inputs"]["bigwigs"]),
                           output_dir=regression / "profiles" / assay, assay=assay, contexts=CONTEXTS,
                           target_bp=r[f"{assay}_profile_bp"], bin_bp=SOURCE_BIN_BP)
        write_json(regression / "prepared.json", {"config": {k: v for k, v in config.items() if k[0] != "_"},
                                                  "window_counts": windows["window_counts"]})
    if only in (None, "classification"):
        if not (regression / "windows.tsv.gz").is_file():
            raise FileNotFoundError("Build the regression dataset first: background comes from its windows")
        e = config["enhancers"]
        log("building the enhancer classification splits")
        audit = prepare_enhancers(catalog=paths["catalog"], reference_fasta=paths["reference_fasta"],
                                  blacklist=paths["blacklist"], splits=config["splits"], output=classification,
                                  classes=e["classes"], percentile=float(e["h3k27ac_background_percentile"]))
        log(f"{audit['catalog_enhancers']:,} catalog enhancers, {audit['retained']:,} retained "
            f"({ {s: v['n'] for s, v in audit['splits'].items()} }), excluded: {audit['exclusions']} "
            "(manuscript: 40,455 enhancers, 40,338 retained: 26,930 / 4,062 / 9,346)")
        b = config["background"]
        log("matching genomic background to enhancers by GC content")
        summary = prepare_backgrounds(enhancer_dir=classification, windows=regression / "windows.tsv.gz",
                                      seed=b["seed"], test_seed=b["test_seed"], caliper=b["gc_caliper"])
        log(f"backgrounds per split: { {s: v['backgrounds'] for s, v in summary.items()} }")
    log(f"done -> {root}")
