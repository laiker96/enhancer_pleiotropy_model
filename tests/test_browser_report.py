from pathlib import Path

import numpy as np

from enhancer_pleiotropy_model.browser_report import (
    assay_metrics,
    interval_overlap_mask,
    row_correlations,
    write_bed,
)
from enhancer_pleiotropy_model.constants import CONTEXTS


def test_interval_overlap_mask_obeys_half_open_bed_coordinates():
    starts = np.asarray([0, 10, 20, 30])
    ends = np.asarray([10, 20, 30, 40])
    observed = interval_overlap_mask(starts, ends, [(5, 7), (20, 25), (40, 50)])
    np.testing.assert_array_equal(observed, [True, False, True, False])


def test_perfect_predictions_have_perfect_metrics():
    observed = np.arange(1, 81, dtype=np.float64).reshape(10, 8)
    masks = {"all": np.ones(10, dtype=bool)}
    metrics, strata, informative = assay_metrics(
        observed,
        observed.copy(),
        masks,
        active_quantile=0.5,
        variable_quantile=0.5,
        contexts=CONTEXTS,
    )
    assert metrics["per_context"]["macro"]["pearson"] == 1.0
    assert metrics["per_context"]["macro"]["rmse"] == 0.0
    assert metrics["tissue_pattern_informative"]["mean_pearson"] == 1.0
    assert metrics["top_context_accuracy_informative"] == 1.0
    assert strata[0]["macro_pearson"] == 1.0
    assert informative.any()


def test_row_correlations_handle_constant_predictions():
    labels = np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.float64)
    predictions = np.asarray([[2, 4, 6], [1, 1, 1]], dtype=np.float64)
    values = row_correlations(labels, predictions)
    assert np.isclose(values[0], 1.0)
    assert np.isnan(values[1])


def test_bookmark_bed_has_no_header(tmp_path: Path):
    output = tmp_path / "bookmarks.bed"
    write_bed(
        output,
        [
            {
                "assay": "atac",
                "category": "success",
                "rank": 1,
                "pattern_pearson": 0.75,
                "chromosome": "chr2L",
                "start": 100,
                "end": 200,
            }
        ],
    )
    assert output.read_text().startswith("chr2L\t100\t200\tatac_success_1_rho=0.750\t0")
