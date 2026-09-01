import numpy as np

from enhancer_pleiotropy_model.metrics import (
    correlation_structure,
    scientific_composite,
    tissue_pattern_metrics,
)


def test_tissue_pattern_pearson_is_one_for_affine_predictions():
    labels = np.asarray([[0, 1, 2], [3, 1, 0]], dtype=np.float64)
    predictions = 4 * labels + 7
    result = tissue_pattern_metrics(labels, predictions, np.ones(2, dtype=bool))
    assert np.isclose(result["mean_pearson"], 1.0)


def test_correlation_structure_reports_overcorrelation():
    rng = np.random.default_rng(7)
    shared = rng.normal(size=(200, 1))
    labels = shared + rng.normal(scale=1.0, size=(200, 3))
    predictions = shared + rng.normal(scale=0.05, size=(200, 3))
    result = correlation_structure(labels, predictions, ("a", "b", "c"))
    assert result["mean_prediction_minus_true"] > 0
    assert result["fraction_more_correlated_than_true"] == 1.0


def test_scientific_composite_uses_both_assays_and_metric_families():
    metrics = {
        "atac": {
            "window_mean": {"macro": {"pearson": 0.8}},
            "tissue_pattern": {"mean_pearson": 0.4},
        },
        "h3k27ac": {
            "window_mean": {"macro": {"pearson": 0.6}},
            "tissue_pattern": {"mean_pearson": 0.2},
        },
    }
    assert np.isclose(scientific_composite(metrics), 0.5)
