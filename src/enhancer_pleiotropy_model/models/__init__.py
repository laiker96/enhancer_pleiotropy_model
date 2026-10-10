"""Model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch

from ..constants import CONTEXTS
from .classifiers import MANUSCRIPT_READOUT, READOUTS, classifier_class
from .enhancernet import EnhancerNetAttentionRegressor, EnhancerNetCNNRegressor
from .regressors import AttentionRegressor, CNNRegressor, DenseRegressor

REGRESSORS = {"cnn": CNNRegressor, "attention": AttentionRegressor, "dense": DenseRegressor,
              "enhancernet_cnn": EnhancerNetCNNRegressor,
              "enhancernet_attention": EnhancerNetAttentionRegressor}
ARCHITECTURES = tuple(REGRESSORS)

# Architecture names stored in checkpoints written by the original research code.
_LEGACY_REGRESSOR_NAMES = {
    "v4_dilated_cnn_profile_regressor_v1": "cnn",
    "enformer_like_dense_atac_h3k27ac_profile_regressor_v1": "attention",
    "v4_flatten_dense_profile_regressor_v1": "dense",
    "v4_enhancernet_cnn_profile_regressor_v1": "enhancernet_cnn",
    "v4_enhancernet_attention_profile_regressor_v1": "enhancernet_attention",
}
_LEGACY_READOUTS = {"retained_assay_hidden_v1": "assay_heads", "legacy": "new_head"}

__all__ = ["ARCHITECTURES", "READOUTS", "MANUSCRIPT_READOUT", "make_regressor", "load_regressor",
           "make_classifier", "load_classifier", "use_interleaved_dilation"]


def use_interleaved_dilation(model, enabled: bool = True):
    """Compute dilated convolutions in interleaved form (inference and attribution only)."""
    from .layers import DilatedBlock
    for module in model.modules():
        if isinstance(module, DilatedBlock):
            module.interleaved = enabled
    return model


def make_regressor(architecture: str, output_scaling: dict | None = None, dropout: float = .1,
                   head_dropout: float = .1):
    if architecture not in REGRESSORS:
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {ARCHITECTURES}")
    return REGRESSORS[architecture](context_count=len(CONTEXTS), dropout=dropout, head_dropout=head_dropout,
                                    output_scaling=output_scaling)


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_regressor(path: Path):
    """Return ``(model, checkpoint)`` for a regressor ``best_model.pt``."""
    checkpoint = _load(path)
    meta = checkpoint["architecture"]
    if isinstance(meta, dict):  # original research checkpoints
        architecture = _LEGACY_REGRESSOR_NAMES[meta["name"]]
        scaling = meta.get("output_scaling")
    else:
        architecture, scaling = meta, checkpoint.get("output_scaling")
    if tuple(checkpoint["contexts"]) != CONTEXTS:
        raise ValueError(f"{path}: unexpected context order")
    model = make_regressor(architecture, scaling)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    checkpoint["architecture_name"] = architecture
    return model, checkpoint


def make_classifier(architecture: str, readout: str, seed: int, prevalence, parent: Path | None = None):
    """Build a classifier; ``parent`` is a regressor checkpoint (None trains from scratch).

    The encoder is built (or loaded) after ``torch.manual_seed(seed)`` and the new
    head after ``torch.manual_seed(seed + 100000)``, so paired scratch and
    pretrained runs start from identical head weights. The output bias is set to
    the logit of the training prevalence of each context.
    """
    torch.manual_seed(seed)
    regressor = make_regressor(architecture) if parent is None else load_regressor(parent)[0]
    if parent is not None and not isinstance(regressor, REGRESSORS[architecture]):
        raise ValueError(f"{parent} is not a {architecture} regressor")
    torch.manual_seed(seed + 100000)
    model = classifier_class(architecture, readout)(regressor)
    prevalence = torch.as_tensor(prevalence, dtype=torch.float32)
    if not torch.all((prevalence > 0) & (prevalence < 1)):
        raise ValueError("Every context needs positive and negative training examples")
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.logit(prevalence))
    return model


def load_classifier(path: Path):
    """Return ``(model, checkpoint)`` for a classifier ``best_model.pt``."""
    checkpoint = _load(path)
    settings = checkpoint["settings"]
    architecture = settings["architecture"]
    readout = settings.get("readout") or _LEGACY_READOUTS[settings["training"].get("readout", "legacy")]
    model = classifier_class(architecture, readout)(make_regressor(architecture))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    checkpoint["architecture_name"], checkpoint["readout"] = architecture, readout
    return model.eval(), checkpoint
