"""A tiny synthetic drosophila-cCREs output tree for end-to-end CPU tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from enhancer_pleiotropy_model.constants import ASSAYS, CONTEXTS

CHROMOSOMES = {"chrA": 150_000, "chrB": 120_000, "chrC": 120_000}
LIBRARIES = {c: [f"{c}_h3k27ac_rep1", f"{c}_h3k27ac_rep2"] for c in CONTEXTS}


def write_inputs(root: Path, rng: np.random.Generator) -> None:
    import pyBigWig

    genome = {chrom: "".join(rng.choice(list("ACGT"), size=n)) for chrom, n in CHROMOSOMES.items()}
    (root / "dm6.fa").write_text("".join(f">{c}\n{s}\n" for c, s in genome.items()))
    (root / "blacklist.bed").write_text("chrA\t100000\t101000\n")
    rows, dhs, summits, peaks = [], [], [], {lib: [] for libs in LIBRARIES.values() for lib in libs}
    for chrom, length in CHROMOSOMES.items():
        for k, summit in enumerate(range(5_000, length - 5_000, 2_500)):
            start, end = summit - 150, summit + 150
            dhs.append(f"{chrom}\t{start}\t{end}\n")
            summits.append(f"{chrom}\t{summit}\t{summit + 1}\n")
            membership = rng.random(8) < .5
            membership[k % 8] = True
            percentile = np.where(rng.random(8) < .7, rng.uniform(.61, 1, 8), rng.uniform(0, .59, 8))
            row = {"master_dhs_id": f"DHS{len(rows):05d}", "chrom": chrom, "start": start, "end": end,
                   "summit": summit, "regulatory_class": "distal_enhancer_like" if k % 7 else "promoter_associated",
                   "blacklist_overlap": int(chrom == "chrA" and 99_000 < summit < 102_000)}
            for i, context in enumerate(CONTEXTS):
                row[f"{context}__context_membership"] = int(membership[i])
                row[f"{context}__h3k27ac_max_500_background_percentile"] = float(percentile[i])
                row[f"{context}__h3k27ac_active"] = int(membership[i] and percentile[i] > .6)
                if row[f"{context}__h3k27ac_active"]:
                    for library in LIBRARIES[context]:
                        peaks[library].append(f"{chrom}\t{start - 200}\t{end + 200}\n")
            rows.append(row)
    (root / "master_dhs.bed").write_text("".join(dhs))
    (root / "master_dhs_summits.bed").write_text("".join(summits))
    pd.DataFrame(rows).to_csv(root / "master_elements_wide.tsv.gz", sep="\t", index=False)
    for library, lines in peaks.items():
        (root / "peaks" / library).mkdir(parents=True, exist_ok=True)
        (root / "peaks" / library / f"{library}_peaks.broadPeak").write_text("".join(lines) or "chrA\t1\t2\n")
    (root / "tracks").mkdir()
    for assay in ASSAYS:
        for context in CONTEXTS:
            with pyBigWig.open(str(root / "tracks" / f"{context}.{assay}.mean.background_tmm.bw"), "w") as bw:
                bw.addHeader(list(CHROMOSOMES.items()))
                for chrom, length in CHROMOSOMES.items():
                    starts = np.arange(0, length, 64, dtype=np.int64)
                    values = rng.gamma(.5, 2.0, len(starts)).astype(np.float64)
                    bw.addEntries(chrom, int(starts[0]), values=values.tolist(), span=64, step=64)


@pytest.fixture(scope="session")
def workspace(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("workspace")
    inputs = root / "inputs"
    inputs.mkdir()
    write_inputs(inputs, np.random.default_rng(0))
    config = {
        "output": str(root / "data"),
        "inputs": {"reference_fasta": str(inputs / "dm6.fa"), "blacklist": str(inputs / "blacklist.bed"),
                   "master_dhs": str(inputs / "master_dhs.bed"),
                   "master_dhs_summits": str(inputs / "master_dhs_summits.bed"),
                   "catalog": str(inputs / "master_elements_wide.tsv.gz"), "bigwigs": str(inputs / "tracks"),
                   "h3k27ac_peaks": str(inputs / "peaks"), "h3k27ac_libraries": LIBRARIES},
        "splits": {"train": {"chromosomes": ["chrA"], "regions": []},
                   "validation": {"chromosomes": ["chrB"], "regions": []},
                   "test": {"chromosomes": ["chrC"], "regions": []}},
        "regression": {"seed": 1, "target_bp": 512, "flank_bp": 768, "train_stride_bp": 256,
                       "validation_stride_bp": 512, "background_to_peak_ratio": 1.0, "block_size_bp": 1_000_000,
                       "atac_profile_bp": 512, "h3k27ac_profile_bp": 1536},
        "enhancers": {"classes": ["distal_enhancer_like", "proximal_enhancer_like"],
                      "h3k27ac_background_percentile": 0.6},
        "background": {"seed": 2, "test_seed": 3, "gc_caliper": 0.05},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(config))
    from enhancer_pleiotropy_model.config import load_config
    from enhancer_pleiotropy_model.data.prepare import prepare
    prepare(load_config(root / "data.yaml"))
    return root


def repository_config(name: str, workspace: Path, **values) -> list[str]:
    """Overrides that point a repository config at the synthetic workspace and keep runs tiny."""
    overrides = {"device": "cpu", "data": str(workspace / "data" / name),
                 "output": str(workspace / "runs" / ("regressors" if name == "regression" else "classifiers"))}
    overrides.update(values)
    return [f"{k}={v}" for k, v in overrides.items()]
