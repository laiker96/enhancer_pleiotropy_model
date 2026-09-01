import numpy as np

from enhancer_pleiotropy_model.browser_tracks import (
    accumulate_profile,
    build_grid_windows,
)


def test_sliding_grid_is_globally_aligned_and_complete():
    windows, skipped = build_grid_windows("A" * 4096, 0, 4096, stride=256)
    assert windows[0].input_start == 0
    assert windows[-1].input_end == 4096
    assert all(window.target_start % 256 == 0 for window in windows)
    assert skipped == {"ambiguous_sequence": 0, "blacklist_overlap": 0}


def test_overlap_aggregation_has_expected_per_bin_mean():
    totals = np.zeros((6, 1), dtype=np.float32)
    support = np.zeros(6, dtype=np.uint16)
    accumulate_profile(
        totals,
        support,
        np.asarray([[1], [2], [3], [4]], dtype=np.float32),
        global_start_bin=0,
        output_start_bp=0,
        bin_size=1,
    )
    accumulate_profile(
        totals,
        support,
        np.asarray([[5], [6], [7], [8]], dtype=np.float32),
        global_start_bin=0,
        output_start_bp=2,
        bin_size=1,
    )
    observed = totals[:, 0] / support
    np.testing.assert_allclose(observed, [1, 2, 4, 5, 7, 8])
