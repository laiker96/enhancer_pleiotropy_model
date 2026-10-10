import numpy as np
from scipy.special import expit, logit

from enhancer_pleiotropy_model.calibration import apply, fit


def test_recovers_known_platt_parameters():
    rng = np.random.default_rng(0)
    raw = rng.uniform(.01, .99, (20000, 8))
    truth = expit(.7 * logit(raw) - .4)
    labels = (rng.random(raw.shape) < truth).astype(float)
    calibration = fit(raw, labels)
    assert np.allclose(calibration["a"], .7, atol=.05)
    assert np.allclose(calibration["b"], -.4, atol=.05)
    assert np.allclose(apply(raw, calibration), truth, atol=.02)


def test_calibration_preserves_ranking():
    raw = np.linspace(.01, .99, 50)[:, None].repeat(8, 1)
    calibrated = apply(raw, {"a": [.5] * 8, "b": [1.] * 8, "probability_clip": 1e-7})
    assert np.all(np.diff(calibrated, axis=0) > 0)
