from argparse import Namespace

import numpy as np
import pyBigWig

from enhancer_pleiotropy_model.constants import CONTEXTS
from enhancer_pleiotropy_model.preprocessing.h3_peaks import (
    build_consensus,
    merge_intervals,
    replicate_supported_intervals,
)
from enhancer_pleiotropy_model.preprocessing.profiles import binned_means
from enhancer_pleiotropy_model.preprocessing.windows import (
    build_split_regions,
    containing_split,
    parse_region,
)


def test_interval_merge_and_replicate_support():
    replicates = [
        {"chr2L": [(10, 20), (40, 50)]},
        {"chr2L": [(15, 25), (70, 80)]},
    ]
    assert replicate_supported_intervals(replicates) == {"chr2L": [(10, 25)]}
    assert merge_intervals({"chr2L": [(1, 3), (3, 5), (9, 10)]}) == {
        "chr2L": [(1, 5), (9, 10)]
    }


def test_consensus_ignores_an_excluded_context(tmp_path):
    peak_directory = tmp_path / "peaks"
    peak_directory.mkdir()
    for context in (*CONTEXTS, "e11"):
        (peak_directory / f"{context}_h3k27ac_rep1_peaks.broadPeak").write_text(
            "chr2L\t10\t20\n"
        )
    output = tmp_path / "consensus.bed"
    metadata = tmp_path / "consensus.metadata.json"
    result = build_consensus(
        Namespace(
            peak_directory=peak_directory,
            output=output,
            metadata=metadata,
            contexts=CONTEXTS,
        )
    )
    assert result["union_peak_count"] == 1
    assert "e11" not in result["context_summary"]


def test_bigwig_profile_bins_are_exact_means(tmp_path):
    path = tmp_path / "signal.bw"
    writer = pyBigWig.open(str(path), "w")
    writer.addHeader([("chr2L", 16)])
    starts = list(range(16))
    writer.addEntries(
        ["chr2L"] * 16,
        starts,
        ends=[start + 1 for start in starts],
        values=[float(value) for value in range(16)],
    )
    writer.close()
    with pyBigWig.open(str(path)) as reader:
        observed = binned_means(
            reader,
            "chr2L",
            np.asarray([0, 4]),
            np.asarray([8, 12]),
            bin_size=4,
        )
    expected = np.asarray([[1.5, 5.5], [5.5, 9.5]], dtype=np.float32)
    np.testing.assert_allclose(observed, expected)


def test_region_split_requires_complete_input_containment():
    genome = {"chr2L": "A" * 100, "chr3R": "A" * 80}
    regions = build_split_regions(
        genome,
        {"train": set(), "validation": set(), "test": {"chr3R"}},
        {
            "train": ["chr2L:50-100"],
            "validation": ["chr2L:0-50"],
            "test": [],
        },
    )
    assert parse_region("chr2L:50-100") == ("chr2L", 50, 100)
    assert containing_split(regions, "chr2L", 10, 40) == "validation"
    assert containing_split(regions, "chr2L", 60, 90) == "train"
    assert containing_split(regions, "chr2L", 40, 60) is None
