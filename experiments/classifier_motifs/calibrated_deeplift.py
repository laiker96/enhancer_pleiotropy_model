"""Eight calibrated DeepLIFT readouts; isolated from production IG.

Use the existing pinned tensor-op adapter. Calibration is a scalar rescale
rule applied separately to EACH enhancer/reference pair before averaging.
"""
import torch
from torch import nn

from .captum_deeplift import (DEEPLIFT_EPS, hypothetical_projection,
                            require_pinned_captum)


class ProbabilityReadout(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Captum batches [inputs, references]. Concatenating orientations here
        # would break that pairing for the tensor-op rescale rules.
        return (self.model(x).sigmoid()+self.model(x.flip((1, 2))).sigmoid())*.5


def calibrate(probability, a, b, epsilon=1e-7):
    return torch.sigmoid(a*torch.logit(probability.clamp(epsilon, 1-epsilon))+b)


def calibration_multiplier(p, reference_p, a, b, epsilon=1e-7):
    """Exact unary secant, with analytic derivative near equal endpoints.

    This avoids an unsupported torch.logit DeepLIFT rule. A scalar chain of
    rescale rules telescopes to this same secant (away from fallback cases).
    It is not rescaling an attribution already averaged across references.
    """
    q, reference_q = calibrate(p, a, b, epsilon), calibrate(reference_p, a, b, epsilon)
    dp = p-reference_p
    clipped = p.clamp(epsilon, 1-epsilon)
    derivative = a*q*(1-q)/(clipped*(1-clipped))
    derivative = torch.where((p >= epsilon) & (p <= 1-epsilon), derivative, 0.)
    use_secant = dp.abs() > DEEPLIFT_EPS
    safe_dp = torch.where(use_secant, dp, torch.ones_like(dp))
    return torch.where(use_secant, (q-reference_q)/safe_dp, derivative), q, reference_q


def paired_calibrated_deeplift(model, x, baseline, a, b, should_stop=lambda: False):
    from captum.attr import DeepLift
    from captum.attr._core.deep_lift import SUPPORTED_NON_LINEAR
    require_pinned_captum()
    if (model.training or x.ndim != 3 or x.shape != baseline.shape or x.shape[1] != 4
            or not len(x) or a.shape != (8,) or b.shape != (8,)
            or any(type(m) in SUPPORTED_NON_LINEAR for m in model.modules())):
        raise ValueError('Adapted frozen model, eight coefficients and paired ACGT inputs required')
    if not all(bool(torch.isfinite(v).all() & ((v == 0) | (v == 1)).all()
                    & (v.sum(1) == 1).all()) for v in (x, baseline)):
        raise ValueError('Finite one-hot endpoints required')
    wrapper = ProbabilityReadout(model).eval()
    with torch.no_grad():
        p, reference_p = wrapper(x), wrapper(baseline)
        multiplier, q, reference_q = calibration_multiplier(p, reference_p, a, b)
    explainer = DeepLift(wrapper, eps=DEEPLIFT_EPS)
    values = []
    for context in range(8):
        if should_stop():
            raise TimeoutError('DeepLIFT stopped at context boundary')
        value = explainer.attribute(x.clone().requires_grad_(True), baselines=baseline,
            target=context, custom_attribution_func=hypothetical_projection).detach()
        values.append(value*multiplier[:, context, None, None])
    hypothetical = torch.stack(values, 1)
    actual = (hypothetical*x[:, None]).sum(2)
    difference = q-reference_q
    result = dict(hypothetical=hypothetical, actual=actual,
                  difference=difference, delta=actual.sum(2)-difference,
                  probabilities=q, reference_probabilities=reference_q)
    if not all(bool(torch.isfinite(v).all()) for v in result.values()):
        raise ValueError('Nonfinite calibrated DeepLIFT output')
    return result
