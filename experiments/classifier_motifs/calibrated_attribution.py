"""Full-input, calibrated eight-context IG plus the observed-active mean logit.

Only computational batches vary. Every enhancer/reference pair uses the same
Gauss-Legendre nodes and its own dinucleotide reference. No model training.
"""
import numpy as np
import torch
from torch import nn

from .attribution import active_weights
from .context_attribution import CONTEXTS, _quadrature
from classifier_transfer.data import digest
from classifier_transfer.models import (DilatedBlock, RetainedHeadsClassifier,
                                        make_regressor, RETAINED_HEADS_READOUT)
from classifier_transfer.fast_convolution import interleaved_block_forward

CHECKPOINT_SHA = '7dc8eb721fbd4f292eba95ff2ea44399cfd1aae5a596e8d56e6d7ff25a95628a'
TARGETS = tuple('calibrated_probability_'+c for c in CONTEXTS) + ('mean_active_logit',)


def load_classifier(path, device):
    if digest(path) != CHECKPOINT_SHA:
        raise ValueError('Wrong background-trained checkpoint')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    settings = saved['settings']
    if (saved['epoch'] != 38 or settings['architecture'] != 'cnn'
            or settings['training']['readout'] != RETAINED_HEADS_READOUT
            or settings['training']['background'] !=
               dict(enhancers_per_background=3, selection='enhancer_only')):
        raise ValueError('Wrong architecture, training population or checkpoint epoch')
    model = RetainedHeadsClassifier(make_regressor('cnn'))
    model.load_state_dict(saved['state_dict'], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite weights')
    DilatedBlock.forward = interleaved_block_forward
    return model.eval().requires_grad_(False).to(device)


class CalibratedTargets(nn.Module):
    def __init__(self, model, calibration, expected_checkpoint_sha256=CHECKPOINT_SHA):
        super().__init__()
        if (calibration['checkpoint_sha256'] != expected_checkpoint_sha256
                or tuple(calibration['contexts']) != CONTEXTS
                or calibration['fitting_population'] != 'enhancers_only'
                or calibration['method'] != 'sigmoid'
                or calibration['probability_clip'] != 1e-7):
            raise ValueError('Incompatible calibration contract')
        self.model = model
        for key in ('a', 'b'):
            value = torch.tensor(calibration[key], dtype=torch.float32)
            if value.shape != (8,) or not torch.isfinite(value).all():
                raise ValueError('Invalid calibration coefficients')
            self.register_buffer(key, value)
        if not (self.a > 0).all():
            raise ValueError('Calibration must preserve each context ranking')
        self.epsilon = calibration['probability_clip']
        self.eval()

    def endpoints(self, x):
        # Batch orientations together; gradients of the flip map back to x.
        forward, reverse = self.model(torch.cat((x, x.flip((1, 2))), 0)).chunk(2)
        probability = (forward.sigmoid()+reverse.sigmoid())*.5
        q = torch.sigmoid(self.a*torch.logit(probability.clamp(
            self.epsilon, 1-self.epsilon))+self.b)
        return dict(orientation_logits=torch.stack((forward, reverse), 1),
                    logits=(forward+reverse)*.5, probabilities=probability,
                    calibrated_probabilities=q)

    def forward(self, x, weights):
        values = self.endpoints(x)
        return torch.cat((values['calibrated_probabilities'],
            (values['logits']*weights).sum(1, keepdim=True)), 1)


def integrate(target_model, x, baseline, labels, steps=64, internal_batch=512,
              target_batch=1, should_stop=lambda: False, target_count=9):
    """Independent pairs [N,4,L], jointly batched across pairs/nodes/targets.

    The returned hypothetical scores subtract EACH reference's observed-base
    projection BEFORE reference averaging. Set target_count=8 to omit only the
    historical mean-logit output; the default nine-target workflow is unchanged.
    Do not project an averaged gradient
    against an averaged reference. No parameter or higher-order gradients.
    """
    if (target_model.training or x.ndim != 3 or not len(x) or x.shape[1] != 4
            or x.shape != baseline.shape or len(labels) != len(x)
            or internal_batch < len(x) or steps < 1 or target_count not in (8, 9)
            or not 1 <= target_batch <= target_count):
        raise ValueError('Invalid attribution shapes, mode or batching')
    weights = active_weights(labels).to(dtype=x.dtype)
    if not torch.stack([torch.isfinite(v).all() & ((v == 0) | (v == 1)).all()
                        & (v.sum(1) == 1).all() for v in (x, baseline)]).all():
        raise ValueError('Expected finite normalized ACGT one-hot endpoints')
    nodes, quadrature = _quadrature(steps)
    nodes = torch.as_tensor(nodes, device=x.device, dtype=x.dtype)
    quadrature = torch.as_tensor(quadrature, device=x.device, dtype=x.dtype)
    integrated = torch.zeros((len(x), target_count, 4, x.shape[-1]), device=x.device, dtype=x.dtype)
    directions = torch.eye(target_count, device=x.device, dtype=x.dtype)
    finite = torch.ones((), device=x.device, dtype=torch.bool)
    group = internal_batch//len(x)
    for begin in range(0, steps, group):
        if should_stop():
            raise TimeoutError('Stopped at integration batch boundary')
        alpha = nodes[begin:begin+group]
        points = (baseline[None]+alpha[:, None, None, None]*(x-baseline)[None]).flatten(0, 1).requires_grad_(True)
        repeated_weights = weights.repeat(len(alpha), 1)
        def outputs(values):
            scores = target_model(values, repeated_weights)[:, :target_count]
            return scores.sum(0), torch.isfinite(scores).all()
        if target_batch == 1:
            scores, valid = outputs(points)
        else:
            scores, pullback, valid = torch.func.vjp(outputs, points, has_aux=True)
        finite &= valid
        for first in range(0, target_count, target_batch):
            end = min(first+target_batch, target_count)
            if target_batch == 1:
                grad = torch.autograd.grad(scores[first], points, retain_graph=end < target_count)[0][None]
            else:
                grad = torch.vmap(lambda direction: pullback(direction,
                    retain_graph=end < target_count, create_graph=False)[0])(directions[first:end])
            grad = grad.reshape(end-first, len(alpha), *x.shape)
            integrated[:, first:end] += (grad*quadrature[
                None, begin:begin+len(alpha), None, None, None]).sum(1).transpose(0, 1)
        if target_batch != 1:
            del pullback
    hypothetical = integrated-(integrated*baseline[:, None]).sum(2, keepdim=True)
    actual = (hypothetical*x[:, None]).sum(2)
    with torch.no_grad():
        difference = (target_model(x, weights)-target_model(baseline, weights))[:, :target_count]
    result = dict(hypothetical=hypothetical, actual=actual,
                  delta=actual.sum(2)-difference, target_difference=difference)
    if not torch.stack([finite, *(torch.isfinite(v).all() for v in result.values())]).all():
        raise ValueError('Nonfinite calibrated IG')
    return {key: value.detach() for key, value in result.items()}


def quality(delta, difference):
    # Probability targets are on a 0..1 scale, unlike the historical logit.
    if delta.shape != difference.shape or delta.shape[-1] not in (8, 9):
        raise ValueError('Expected eight probabilities or nine historical targets')
    absolute = np.asarray([.002]*8+[.02])[:delta.shape[-1]]
    return np.abs(delta) <= absolute+.05*np.abs(difference)
