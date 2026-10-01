#!/usr/bin/env python3
"""Allowlisted source-only distribution. Dry-run unless --write is specified."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("src/enhancer_pleiotropy_model", "experiments/reproduction",
               "experiments/classifier_transfer", "experiments/classifier_motifs", "experiments/classifier_modisco")
FILES = ("README.md", "pyproject.toml", ".gitignore", "LICENSES/AlphaGenome-Apache-2.0.txt",
         "docs/figures/best_model_architecture.png", "docs/figures/best_model_architecture.svg",
         "docs/figures/best_model_architecture.pdf", "docs/figures/best_model_architecture.provenance.json",
         "docs/reproduction.md", "docs/reproduction_science.md",
         "docs/reproduction_verification.md",
         "docs/paper_benchmarks.md",
         "docs/legacy_regressor_readme.md", "docs/classifier_calibration_20260921.md",
         "scripts/reproduce.py", "scripts/stage_v4_inputs.py", "scripts/prepare_v4_dataset.py",
         "scripts/prepare_v4_motifs.py", "scripts/export_figure3_inputs.py", "scripts/export_reproduction_release.py",
         "scripts/summarize_paper_benchmarks.py",
         "tests/test_reproduction.py", "tests/test_calibrated_attribution.py", "tests/test_paper_benchmarks.py",
         "tests/test_browser_report.py", "tests/test_browser_tracks.py", "tests/test_data.py",
         "tests/test_metrics.py", "tests/test_model.py", "tests/test_preprocessing.py",
         "tests/test_sequence.py", "tests/test_training.py")
REPRODUCTION = ("config.yaml", "regression.yaml", "requirements-train.txt", "requirements-modisco.txt",
    "slurm/site.example.yaml", "slurm/environment.example.sh", "slurm/task.sbatch", "slurm/submit.py",
    "figure3/README.md", "figure3/source.json", "figure3/style.json", "figure3/manifest.json",
    "benchmarks/models.json")


def files():
    paths = {ROOT / name for name in FILES} | {ROOT / "reproduce" / name for name in REPRODUCTION}
    for folder in DIRECTORIES: paths.update((ROOT / folder).rglob("*.py"))
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Missing or unsafe release member: "+str(path))
        if path.stat().st_size > 5_000_000:
            raise ValueError("Unexpected large source file: "+str(path))
    return sorted(paths)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    selected = files()
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in selected}
    print(json.dumps(dict(files=len(selected), uncompressed_bytes=sum(p.stat().st_size for p in selected),
        output=str(args.output), write=args.write, excluded="data, checkpoints, logs, runtime environments, site.yaml, dated launchers"), indent=2))
    if not args.write: return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Stable metadata and gzip timestamp; refuses to overwrite an existing archive.
    with args.output.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            contents = [(str(p.relative_to(ROOT)), p.read_bytes()) for p in selected]
            contents.append(("RELEASE_MANIFEST.json", (json.dumps(manifest, indent=2)+"\n").encode()))
            for name, content in contents:
                info = tarfile.TarInfo("enhancer_pleiotropy_model/"+name)
                info.size = len(content); info.mode = 0o644
                tar.addfile(info, io.BytesIO(content))


if __name__ == "__main__": main()
