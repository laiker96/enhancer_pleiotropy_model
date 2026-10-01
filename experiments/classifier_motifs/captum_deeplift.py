"""Isolated experimental Captum adapter; never used by production IG jobs."""
import copy
from functools import lru_cache
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from enhancer_pleiotropy_model.model import SoftmaxPooling1d

from .attribution import active_weights
from .dual_attribution import targets


CAPTUM_COMMIT = '0c903909b1e920d6911f88754894a5f230ed4c82'
CAPTUM_DEEPLIFT_SHA256 = '0bfb84f066d8f7126276a63c83c377afc1672916bd4492a6304abab2068b1012'
# Captum's derivative fallback threshold, NOT an acceptance tolerance. At1e-10,
# near-equal FP32 activations on CECAR yield batch-dependent secant multipliers.
# The unchanged batch/RC/completeness tests pass at1e-7 on the failing CPU host.
DEEPLIFT_EPS = 1e-7


@lru_cache(maxsize=1)
def require_pinned_captum():
    from captum.attr._core import deep_lift
    if hashlib.sha256(Path(deep_lift.__file__).read_bytes()).hexdigest() != CAPTUM_DEEPLIFT_SHA256:
        raise ValueError('Use the isolated, pinned Captum development implementation')


class FunctionalGELU(nn.Module):
    """Same forward operation, avoiding reused nonlinear module hooks for RC."""
    def __init__(self, approximate):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        return F.gelu(x, approximate=self.approximate)


class DecomposedLayerNorm(nn.Module):
    """Expose the identical normalization algebra to Captum tensor-op rules.

    This changes the attribution rule, not the intended model function. FP32
    forward/input-gradient equivalence must be checked against fused LayerNorm.
    """
    def __init__(self, original):
        super().__init__()
        self.normalized_shape = original.normalized_shape
        self.eps = original.eps
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, x):
        dims = tuple(range(-len(self.normalized_shape), 0))
        centered = x-x.mean(dim=dims, keepdim=True)
        value = centered*torch.rsqrt(centered.square().mean(dim=dims, keepdim=True)+self.eps)
        if self.weight is not None:
            value = value*self.weight
        if self.bias is not None:
            value = value+self.bias
        return value


def decomposed_softmax(x, dim=-1):
    # No data-dependent maximum subtraction: the pinned Captum max/softmax
    # paths fail conservation in composed learned pooling. The unshifted
    # expression is equivalent on this checkpoint's validated finite range.
    # Fail, rather than silently clamp/renormalize, if that range is exceeded.
    exp = x.exp()
    total = exp.sum(dim=dim, keepdim=True)
    if not (torch.isfinite(exp).all() & torch.isfinite(total).all()
            & (total >= torch.finfo(x.dtype).tiny).all()):
        raise ValueError('Softmax exponent range unsafe for the experimental adapter')
    return exp/total


class DecomposedSoftmaxPooling(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.pool_size = original.pool_size
        self.logit_scale = original.logit_scale
        self.logit_bias = original.logit_bias

    def forward(self, hidden, attention_mask):
        if hidden.ndim != 3 or attention_mask.shape != (hidden.shape[0],hidden.shape[2]):
            raise ValueError('Softmax pooling received incompatible shapes')
        remainder = hidden.shape[-1] % self.pool_size
        if remainder:
            hidden = F.pad(hidden,(0,self.pool_size-remainder))
            attention_mask = F.pad(attention_mask.bool(),(0,self.pool_size-remainder),value=False)
        batch,channels,length = hidden.shape
        values = hidden.reshape(batch,channels,length//self.pool_size,self.pool_size)
        mask = attention_mask.reshape(batch,length//self.pool_size,self.pool_size)
        logits = values*self.logit_scale.view(1,channels,1,1)+self.logit_bias.view(1,channels,1,1)
        logits = logits.masked_fill(~mask.unsqueeze(1),-1e4)
        weights = decomposed_softmax(logits.float(),-1).to(hidden.dtype)
        pooled_mask = mask.any(-1)
        return (weights*values).sum(-1)*pooled_mask.unsqueeze(1),pooled_mask


def adapt_model(model, decompose_layernorm=True):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen evaluation model required')
    adapted = copy.deepcopy(model)
    changes = []
    def replace(parent, prefix=''):
        for name, child in list(parent.named_children()):
            path = prefix+name
            if type(child) is nn.LayerNorm and decompose_layernorm:
                setattr(parent, name, DecomposedLayerNorm(child))
                changes.append(dict(module=path, original='LayerNorm', replacement='DecomposedLayerNorm'))
            elif type(child) is nn.GELU:
                setattr(parent, name, FunctionalGELU(child.approximate))
                changes.append(dict(module=path, original='GELU', replacement='FunctionalGELU'))
            elif type(child) is SoftmaxPooling1d:
                setattr(parent,name,DecomposedSoftmaxPooling(child))
                changes.append(dict(module=path,original='SoftmaxPooling1d',replacement='DecomposedSoftmaxPooling'))
            else:
                replace(child, path+'.')
    replace(adapted)
    return adapted.eval().requires_grad_(False), changes


class DualTarget(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, weights):
        return targets(self.model, x, weights)


def hypothetical_projection(multipliers, inputs, baselines):
    # Projection is per example/reference BEFORE averaging, like production IG.
    return tuple(m-(m*b).sum(1, keepdim=True) for m,b in zip(multipliers,baselines))


def paired_deeplift(model, x, baseline, labels, eps=DEEPLIFT_EPS):
    """Matched pairs, never Captum DeepLiftShap's cross-example reference pool.

    Caller supplies independently selected reference pairs and averages them.
    Captum's default channel attribution differs from hypothetical nucleotide
    attribution; completeness below uses observed-base projections, not the
    sum of all hypothetical ACGT possibilities.
    """
    from captum.attr import DeepLift
    from captum.attr._core.deep_lift import SUPPORTED_NON_LINEAR
    require_pinned_captum()
    if (model.training or x.ndim != 3 or not len(x) or x.shape[1] != 4
            or baseline.shape != x.shape or len(labels) != len(x)):
        raise ValueError('Aligned ACGT inputs and an eval model required')
    if any(type(m) in SUPPORTED_NON_LINEAR for m in model.modules()):
        raise ValueError('Functional nonlinearities required before reusing model for RC')
    if not torch.stack([torch.isfinite(v).all() & ((v == 0) | (v == 1)).all()
                        & (v.sum(1) == 1).all() for v in (x,baseline)]).all():
        raise ValueError('Finite one-hot endpoints required')
    wrapper = DualTarget(model).eval()
    weights = active_weights(labels)
    explainer = DeepLift(wrapper, eps=eps)
    values = []
    for target in range(2):
        values.append(explainer.attribute(x.clone().requires_grad_(True), baselines=baseline,
            additional_forward_args=(weights,), target=target,
            custom_attribution_func=hypothetical_projection).detach())
    hyp = torch.stack(values, 1)
    actual = (hyp*x[:,None]).sum(2)
    with torch.no_grad():
        difference = wrapper(x,weights)-wrapper(baseline,weights)
    result = dict(hypothetical=hyp, actual=actual, delta=actual.sum(2)-difference,
                  target_difference=difference)
    if not torch.stack([torch.isfinite(v).all() for v in result.values()]).all():
        raise ValueError('Nonfinite DeepLIFT result')
    return result
