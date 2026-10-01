"""Run the unchanged production preprocessing modules on staged v4 inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np
import pyBigWig
import torch
import yaml

from stage_v4_inputs import CONTEXTS, require_compute_node, sha256


def run_module(module, arguments):
    command = [sys.executable, "-u", "-m", "enhancer_pleiotropy_model.preprocessing." + module,
               *map(str, arguments)]
    print("RUN " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def validate_tracks(config):
    fasta_index = Path(config["inputs"]["reference_fasta"] + ".fai")
    chroms = {line.split("\t")[0]: int(line.split("\t")[1]) for line in fasta_index.read_text().splitlines()}
    directory = Path(config["inputs"]["bigwig_directory"])
    stage = directory.parents[1]
    selection = json.loads((stage / "provenance/atlas/selection.json").read_text())
    staging = json.loads((stage / "staging.json").read_text())
    for assay in ("atac", "h3k27ac"):
        for context in CONTEXTS:
            path = directory / f"{context}.{assay}.mean.background_tmm.bw"
            metadata = json.loads(path.with_suffix(".json").read_text())
            expected = {x for x in selection[f"{assay}_libraries"] if x.startswith(context + "_")}
            if (metadata["context"] != context or metadata["assay"] != assay
                    or set(metadata["libraries"]) != expected
                    or metadata["normalization_method"] != "tmm_background_10kb_v1"
                    or metadata["status"] != "ok"
                    or metadata["library_n"] != len(expected)
                    or metadata["output"]["sha256"] != staging["inputs"][str(path.relative_to(stage))]):
                raise ValueError(f"Unexpected track provenance: {path}")
            with pyBigWig.open(str(path)) as track:
                if track.chroms() != chroms:
                    raise ValueError(f"BigWig/reference chromosome mismatch: {path}")
            print(f"Track header/provenance verified: {context} {assay}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    require_compute_node()
    started = time.monotonic()
    config = yaml.safe_load(args.config.read_text())
    if tuple(config["contexts"]) != CONTEXTS or config["specificity_finetuning"]["enabled"]:
        raise ValueError("Expected eight-context fresh base configuration")
    if not torch.__version__.startswith("2.5."):
        raise ValueError(f"Expected existing Torch 2.5 environment, got {torch.__version__}")
    validate_tracks(config)
    root = Path(config["output_directory"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "run_config.yaml").write_text(args.config.read_text())
    data = root / "data"
    inputs, sampling, geometry = config["inputs"], config["sampling"], config["profiles"]
    consensus = data / "h3k27ac_consensus_union.bed"
    run_module("h3_peaks", ["--peak-directory", inputs["h3k27ac_peak_directory"],
               "--output", consensus, "--metadata", consensus.with_suffix(".metadata.json"),
               "--contexts", *CONTEXTS])
    windows = data / "windows.tsv.gz"
    arguments = ["--reference-fasta", inputs["reference_fasta"], "--blacklist-bed", inputs["blacklist_bed"],
                 "--master-dhs-bed", inputs["master_dhs_bed"], "--master-dhs-summits-bed", inputs["master_dhs_summits_bed"],
                 "--h3k27ac-peaks-bed", consensus, "--bigwig-directory", inputs["bigwig_directory"],
                 "--signal-assays", "atac", "h3k27ac", "--omit-signal-summaries",
                 "--window-size", sampling["central_target_bp"], "--context-flank-size", sampling["context_flank_bp"],
                 "--stride", sampling["training_stride_bp"], "--validation-stride", sampling["validation_stride_bp"],
                 "--split-strategy", config["split_strategy"], "--block-size", sampling["block_size_bp"],
                 "--background-to-peak-ratio", sampling["background_to_peak_ratio"], "--seed", config["seed"],
                 "--output", windows, "--metadata", data / "windows.metadata.json"]
    for split in ("train", "validation", "test"):
        arguments += [f"--{split}-chromosomes", *config["chromosome_splits"][split]]
        arguments += [f"--{split}-regions", *config["region_splits"][split]]
    run_module("windows", arguments)
    def profiles(assay):
        run_module("profiles", ["--dataset", windows, "--bigwig-directory", inputs["bigwig_directory"],
                   "--output-directory", data / "profiles" / assay, "--assay", assay,
                   "--target-size", geometry[f"{assay}_target_bp"], "--bin-size", geometry["source_bin_bp"],
                   "--contexts", *CONTEXTS])
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(profiles, ("atac", "h3k27ac")))
    metadata = json.loads((data / "windows.metadata.json").read_text())
    outputs = {str(windows.relative_to(root)): sha256(windows)}
    for assay in ("atac", "h3k27ac"):
        pmeta = json.loads((data / "profiles" / assay / "profiles.metadata.json").read_text())
        if tuple(pmeta["contexts"]) != CONTEXTS:
            raise ValueError("Unexpected prepared profile context order")
        for split, count in metadata["window_counts"].items():
            path = data / "profiles" / assay / f"{split}_profiles.npy"
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected = (count, geometry[f"{assay}_target_bp"] // geometry["source_bin_bp"], 8)
            if array.shape != expected or array.dtype != np.float32:
                raise ValueError(f"Unexpected profile shape/type: {path}")
            outputs[str(path.relative_to(root))] = sha256(path)
    smoke = json.loads(json.dumps(config))
    smoke["output_directory"] = str(root / "smoke")
    (root / "smoke_config.yaml").write_text(json.dumps(smoke, indent=2) + "\n")
    (root / "smoke").mkdir()
    (root / "smoke/data").symlink_to("../data", target_is_directory=True)
    record = dict(status="complete", device="cpu", seconds=time.monotonic() - started,
                  host=platform.node(), python=sys.version, numpy=np.__version__, torch=torch.__version__,
                  pyBigWig=pyBigWig.__version__, config_sha256=sha256(args.config),
                  window_counts=metadata["window_counts"], output_hashes=outputs,
                  initialization="fresh base model; no v3 checkpoint", context_order=CONTEXTS,
                  selection="same CREsted scientific validation composite; chr3R not used for selection")
    (root / "prepared.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
