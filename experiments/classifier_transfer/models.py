"""Matched classifiers and a no-attention replacement for the frozen regressor."""

import torch
from torch import nn
from torch.nn import functional as F

from enhancer_pleiotropy_model.model import EnformerLikeJointProfileRegressor
from .enhancernet import MODELS as ENHANCERNET_MODELS, NAMES as ENHANCERNET_NAMES, EnhancerNetClassifier


DILATIONS = (1, 2, 4, 8, 16, 32)
CNN_NAME = "v4_dilated_cnn_profile_regressor_v1"
DENSE_NAME = "v4_flatten_dense_profile_regressor_v1"
RETAINED_HEADS_READOUT = "retained_assay_hidden_v1"


class DilatedBlock(nn.Module):
    def __init__(self, dilation, dimension=192, dropout=.1):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.conv = nn.Conv1d(dimension, dimension, 3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, mask):
        values = self.norm(hidden) * mask.unsqueeze(-1)
        values = self.conv(values.transpose(1, 2)).transpose(1, 2)
        return (hidden + self.dropout(F.gelu(values))) * mask.unsqueeze(-1)


class CNNRegressor(EnformerLikeJointProfileRegressor):
    """Same stem/heads/output geometry; six dilated convolutions replace attention."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.convolution_filters != (96, 128, 160, 192):
            raise ValueError("This experiment requires the frozen 4x stem")
        self.transformer = nn.ModuleList(DilatedBlock(d, dropout=kwargs.get("dropout", .1)) for d in DILATIONS)
        self.transformer_layers = self.transformer_heads = self.transformer_feedforward_dimension = 0


class DenseRegressor(EnformerLikeJointProfileRegressor):
    """Unchanged stem, global position-specific flatten/dense representation, profile heads."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.convolution_filters != (96, 128, 160, 192) or self.h3k27ac_output_pool_size != 4:
            raise ValueError("Dense experiment requires the frozen 4x stem and output geometry")
        self.transformer = nn.ModuleList()
        self.transformer_layers = self.transformer_heads = self.transformer_feedforward_dimension = 0
        self.dense = nn.Sequential(
            nn.Linear(128 * 192, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(kwargs.get("dropout", .1)),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(kwargs.get("dropout", .1)))
        self.atac_head = nn.Sequential(nn.Linear(256, 32 * self.context_count))
        self.h3k27ac_head = nn.Sequential(nn.Linear(256, 24 * self.context_count))

    def initialize_output_means(self, atac_means, h3k27ac_means):
        from enhancer_pleiotropy_model.alphagenome_loss import soft_clip
        for assay, means, bins in (("atac", atac_means, 32), ("h3k27ac", h3k27ac_means, 24)):
            values = torch.as_tensor(means)
            if values.shape != (self.context_count,) or not torch.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid output means")
            if self.output_scaling is not None:
                values = soft_clip(values / torch.tensor(self.output_scaling[assay]))
            self._initialize_head_means(getattr(self, f"{assay}_head"), values.repeat(bins))

    def forward(self, one_hot, attention_mask, atac_target_mask, h3k27ac_target_mask):
        if one_hot.ndim != 3 or one_hot.shape[1:] != (4, 2048):
            raise ValueError("Dense model requires [batch, 4, 2048] inputs")
        # Flatten loses variable-length/crop equivariance: reject unsupported geometry explicitly.
        for mask, lo, hi in ((atac_target_mask, 768, 1280), (h3k27ac_target_mask, 256, 1792)):
            expected = torch.zeros_like(attention_mask)
            expected[:, lo:hi] = True
            if mask.shape != expected.shape or not torch.equal(mask, expected):
                raise ValueError("Dense model requires centered 512/1536-bp target masks")
        hidden, mask, _ = self.convolutional_body(one_hot, attention_mask, atac_target_mask)
        hidden = hidden.transpose(1, 2) * mask.unsqueeze(-1)
        representation = self.dense(hidden.flatten(1))
        outputs = tuple(F.softplus(getattr(self, f"{assay}_head")(representation)).reshape(
            len(one_hot), bins, self.context_count) for assay, bins in (("atac", 32), ("h3k27ac", 24)))
        if self.output_scaling is not None:
            from enhancer_pleiotropy_model.alphagenome_loss import inverse_soft_clip
            outputs = tuple(inverse_soft_clip(v) * getattr(self, f"{assay}_output_means")
                            for assay, v in zip(("atac", "h3k27ac"), outputs, strict=True))
        return outputs


def make_regressor(architecture, output_scaling=None):
    if architecture not in {"attention", "cnn", "dense", *ENHANCERNET_MODELS}:
        raise ValueError("Unknown encoder architecture")
    cls = {"cnn": CNNRegressor, "dense": DenseRegressor, "attention": EnformerLikeJointProfileRegressor,
           **ENHANCERNET_MODELS}[architecture]
    return cls(model_size="4x", dropout=.1, head_dropout=.1,
               h3k27ac_output_pool_size=4, output_scaling=output_scaling)


def load_regressor(path, architecture):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    meta = saved["architecture"]
    expected = {"cnn": CNN_NAME, "dense": DENSE_NAME, **ENHANCERNET_NAMES,
                "attention": "enformer_like_dense_atac_h3k27ac_profile_regressor_v1"}[architecture]
    if meta["name"] != expected or tuple(saved["contexts"]) != ("ab", "e13", "e5", "ead", "hid", "lb", "o", "wid"):
        raise ValueError("Checkpoint architecture/context order mismatch")
    model = make_regressor(architecture, meta.get("output_scaling"))
    model.load_state_dict(saved["state_dict"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite pretrained weights")
    return model, saved


class Classifier(nn.Module):
    def __init__(self, regressor):
        super().__init__()
        self.convolutional_body = regressor.convolutional_body
        self.transformer = regressor.transformer
        # Three central 512-bp segment means retain position without attention pooling.
        self.head = nn.Sequential(nn.LayerNorm(576), nn.Linear(576, 192), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(192, 8))

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[1:] != (4, 2048):
            raise ValueError("Expected [batch, 4, 2048] ACGT inputs")
        mask = torch.ones((len(sequence), 2048), dtype=torch.bool, device=sequence.device)
        hidden, mask, _ = self.convolutional_body(sequence, mask, mask)
        hidden = hidden.transpose(1, 2)
        for block in self.transformer:
            hidden = block(hidden, mask)
        pooled = hidden[:, 16:112].reshape(len(sequence), 3, 32, 192).mean(dim=2)
        return self.head(pooled.flatten(1))


class DenseClassifier(nn.Module):
    def __init__(self, regressor):
        super().__init__()
        self.convolutional_body = regressor.convolutional_body
        self.dense = regressor.dense
        self.head = nn.Sequential(nn.Linear(256, 8))

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[1:] != (4, 2048):
            raise ValueError("Expected [batch, 4, 2048] ACGT inputs")
        mask = torch.ones((len(sequence), 2048), dtype=torch.bool, device=sequence.device)
        hidden, mask, _ = self.convolutional_body(sequence, mask, mask)
        hidden = hidden.transpose(1, 2) * mask.unsqueeze(-1)
        return self.head(self.dense(hidden.flatten(1)))


class RetainedHeadsClassifier(nn.Module):
    """Transfer both assay MLPs except their final signal-output projections.

    Preserve regressor target geometry before the nonlinear heads: ATAC at
    16 bp in the center 512 bp; H3K27ac at 64 bp across the center 1536 bp.
    Pool hidden activations into one ATAC and three H3K27ac 512-bp summaries.
    """
    def __init__(self, regressor):
        super().__init__()
        if regressor.h3k27ac_output_pool_size != 4:
            raise ValueError("Retained heads require the 4x regressor output geometry")
        self.convolutional_body = regressor.convolutional_body
        self.transformer = regressor.transformer
        self.atac_features = regressor.atac_head[:-1]
        self.h3k27ac_features = regressor.h3k27ac_head[:-1]
        self.head = nn.Sequential(nn.Linear(4 * 384, 8))

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[1:] != (4, 2048):
            raise ValueError("Expected [batch, 4, 2048] ACGT inputs")
        mask = torch.ones((len(sequence), 2048), dtype=torch.bool, device=sequence.device)
        hidden, mask, _ = self.convolutional_body(sequence, mask, mask)
        hidden = hidden.transpose(1, 2)
        for block in self.transformer:
            hidden = block(hidden, mask)
        atac = self.atac_features(hidden[:, 48:80]).mean(dim=1)
        h3 = hidden[:, 16:112].reshape(len(sequence), 24, 4, 192).mean(dim=2)
        h3 = self.h3k27ac_features(h3).reshape(len(sequence), 3, 8, 384).mean(dim=2)
        return self.head(torch.cat((atac, h3.flatten(1)), dim=1))


def initialize_classifier(architecture, seed, prevalence, checkpoint=None, readout="legacy"):
    if readout not in {"legacy", RETAINED_HEADS_READOUT}:
        raise ValueError("Unknown classifier readout")
    if readout == RETAINED_HEADS_READOUT and architecture not in {"attention", "cnn"}:
        raise ValueError("Retained assay heads apply only to attention and dilated CNN")
    torch.manual_seed(seed)
    regressor = make_regressor(architecture) if checkpoint is None else load_regressor(checkpoint, architecture)[0]
    # Paired scratch/transfer classifiers get exactly the same new head weights.
    torch.manual_seed(seed + 100000)
    cls = (RetainedHeadsClassifier if readout == RETAINED_HEADS_READOUT else
           EnhancerNetClassifier if architecture in ENHANCERNET_MODELS else
           DenseClassifier if architecture == "dense" else Classifier)
    model = cls(regressor)
    prevalence = torch.as_tensor(prevalence, dtype=torch.float32)
    if prevalence.shape != (8,) or not torch.all((prevalence > 0) & (prevalence < 1)):
        raise ValueError("Training requires both label classes in every context")
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.logit(prevalence))
    return model
