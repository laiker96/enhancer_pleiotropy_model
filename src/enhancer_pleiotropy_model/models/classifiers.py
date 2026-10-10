"""Eight-context enhancer classifiers built on a regressor encoder.

A classifier reuses a regressor's encoder, either freshly initialised
("scratch") or with pretrained weights ("pretrained"), and adds a new readout.
Outputs are eight logits in ``CONTEXTS`` order.

Readouts:

* ``assay_heads`` (``cnn``, ``attention``; manuscript choice) - keep the regressor's
  ATAC and H3K27ac hidden layers (LN-Linear-GELU-Dropout), drop only their final
  signal projections, average them into one ATAC and three 512-bp H3K27ac
  summaries (4 x 384 = 1,536 features) and apply one linear layer.
* ``new_head`` - discard the regressor heads. ``cnn``/``attention``: average three
  central 512-bp segments of the latent sequence and apply a small MLP;
  ``dense``: linear layer on the shared dense representation; ``enhancernet_*``:
  linear layer on the 256-d encoder output. This is the only readout for
  ``dense`` and ``enhancernet_*``.
"""

from __future__ import annotations

import torch
from torch import nn

from ..constants import INPUT_BP
from .regressors import ATAC_SLICE, H3K27AC_SLICE

READOUTS = ("assay_heads", "new_head")
MANUSCRIPT_READOUT = {"cnn": "assay_heads", "attention": "assay_heads", "dense": "new_head",
                      "enhancernet_cnn": "new_head", "enhancernet_attention": "new_head"}


def _full_mask(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 3 or sequence.shape[1:] != (4, INPUT_BP):
        raise ValueError(f"Expected one-hot inputs of shape [batch, 4, {INPUT_BP}]")
    return torch.ones((len(sequence), INPUT_BP), dtype=torch.bool, device=sequence.device)


class AssayHeadsClassifier(nn.Module):
    def __init__(self, regressor) -> None:
        super().__init__()
        self.convolutional_body = regressor.convolutional_body
        self.transformer = regressor.transformer
        self.atac_features = regressor.atac_head[:-1]
        self.h3k27ac_features = regressor.h3k27ac_head[:-1]
        self.head = nn.Sequential(nn.Linear(4 * 384, 8))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        mask = _full_mask(sequence)
        hidden, mask, _ = self.convolutional_body(sequence, mask, mask)
        hidden = hidden.transpose(1, 2)
        for block in self.transformer:
            hidden = block(hidden, mask)
        atac = self.atac_features(hidden[:, ATAC_SLICE]).mean(dim=1)                    # 32 x 16 bp -> 384
        h3 = hidden[:, H3K27AC_SLICE].reshape(len(sequence), 24, 4, 192).mean(dim=2)    # 24 x 64 bp
        h3 = self.h3k27ac_features(h3).reshape(len(sequence), 3, 8, 384).mean(dim=2)    # 3 x 512 bp
        return self.head(torch.cat((atac, h3.flatten(1)), dim=1))


class SegmentClassifier(nn.Module):
    """``new_head`` readout for the dilated CNN and attention encoders."""

    def __init__(self, regressor) -> None:
        super().__init__()
        self.convolutional_body = regressor.convolutional_body
        self.transformer = regressor.transformer
        self.head = nn.Sequential(nn.LayerNorm(576), nn.Linear(576, 192), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(192, 8))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        mask = _full_mask(sequence)
        hidden, mask, _ = self.convolutional_body(sequence, mask, mask)
        hidden = hidden.transpose(1, 2)
        for block in self.transformer:
            hidden = block(hidden, mask)
        pooled = hidden[:, H3K27AC_SLICE].reshape(len(sequence), 3, 32, 192).mean(dim=2)
        return self.head(pooled.flatten(1))


class DenseClassifier(nn.Module):
    def __init__(self, regressor) -> None:
        super().__init__()
        self.convolutional_body = regressor.convolutional_body
        self.dense = regressor.dense
        self.head = nn.Sequential(nn.Linear(256, 8))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        mask = _full_mask(sequence)
        hidden, mask, _ = self.convolutional_body(sequence, mask, mask)
        return self.head(self.dense((hidden.transpose(1, 2) * mask.unsqueeze(-1)).flatten(1)))


class EnhancerNetClassifier(nn.Module):
    def __init__(self, regressor) -> None:
        super().__init__()
        self.encoder = regressor.encoder
        self.head = nn.Sequential(nn.Linear(256, 8))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence))


def classifier_class(architecture: str, readout: str):
    if readout == "assay_heads":
        if architecture not in ("cnn", "attention"):
            raise ValueError("The assay_heads readout exists only for cnn and attention")
        return AssayHeadsClassifier
    if readout != "new_head":
        raise ValueError(f"Unknown readout {readout!r}; choose from {READOUTS}")
    if architecture.startswith("enhancernet"):
        return EnhancerNetClassifier
    return DenseClassifier if architecture == "dense" else SegmentClassifier
