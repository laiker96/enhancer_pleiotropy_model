"""enhancerNet pooled encoder, adapted to the frozen v4 profile interface.

Architecture source: https://github.com/laiker96/enhancerNet
Commit f64e3381ee26109f00d184214444a4f9bf14d926, transformer_model.py,
NNBlocks.py and pretrain_model_with_ATAC.py. No released weights are used.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from enhancer_pleiotropy_model.model import EnformerLikeJointProfileRegressor
from enhancer_pleiotropy_model.alphagenome_loss import inverse_soft_clip, soft_clip


UPSTREAM_COMMIT = "f64e3381ee26109f00d184214444a4f9bf14d926"
NAMES = {"enhancernet_cnn": "v4_enhancernet_cnn_profile_regressor_v1",
         "enhancernet_attention": "v4_enhancernet_attention_profile_regressor_v1"}


class EnhancerNetEncoder(nn.Module):
    def __init__(self, with_attention):
        super().__init__()
        blocks = []
        channels = 4
        for filters, kernel in zip((256, 60, 60, 120), (7, 3, 5, 3)):
            blocks.append(nn.Sequential(nn.Conv1d(channels, filters, kernel),
                nn.BatchNorm1d(filters), nn.ReLU(), nn.Dropout1d(.1), nn.MaxPool1d(2, 2)))
            channels = filters
        self.convs = nn.Sequential(*blocks)
        if with_attention:
            position = torch.arange(5000).unsqueeze(1)
            frequency = torch.exp(torch.arange(0, 120, 2) * (-math.log(10000.) / 120))
            pe = torch.zeros(1, 5000, 120)
            pe[0, :, 0::2] = torch.sin(position * frequency)
            pe[0, :, 1::2] = torch.cos(position * frequency)
            self.register_buffer("positional_encoding", pe)
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=120, nhead=8, dim_feedforward=256,
                dropout=.2, activation="gelu", batch_first=True, norm_first=False)
            for _ in range(2 if with_attention else 0)])
        self.dense = nn.Sequential(nn.Linear(240, 256), nn.Dropout(.4), nn.ReLU(),
                                   nn.Linear(256, 256), nn.Dropout(.4), nn.ReLU())

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[1:] != (4, 2048):
            raise ValueError("enhancerNet requires [batch, 4, 2048] ACGT inputs")
        hidden = self.convs(sequence).transpose(1, 2)
        if self.transformer:
            hidden = hidden + self.positional_encoding[:, :hidden.shape[1]]
            for layer in self.transformer:
                hidden = layer(hidden)
        # Max before mean matches the upstream concatenation order.
        return self.dense(torch.cat((hidden.max(dim=1).values, hidden.mean(dim=1)), dim=1))


class EnhancerNetRegressor(EnformerLikeJointProfileRegressor):
    def __init__(self, *, with_attention, **kwargs):
        # Reuse the validated scaling interface, but replace all trainable modules.
        super().__init__(**kwargs)
        if (kwargs.get("model_size", "4x") != "4x" or self.context_count != 8
                or self.h3k27ac_output_pool_size != 4):
            raise ValueError("enhancerNet adaptation requires eight-context 4x output geometry")
        del self.convolutional_body, self.transformer
        self.encoder = EnhancerNetEncoder(with_attention)
        self.convolution_filters = (256, 60, 60, 120)
        self.convolution_kernels = (7, 3, 5, 3)
        self.transformer_layers = 2 if with_attention else 0
        self.transformer_heads = 8 if with_attention else 0
        self.transformer_feedforward_dimension = 256 if with_attention else 0
        self.atac_head = nn.Sequential(nn.Linear(256, 32 * self.context_count))
        self.h3k27ac_head = nn.Sequential(nn.Linear(256, 24 * self.context_count))

    def initialize_output_means(self, atac_means, h3k27ac_means):
        for assay, means, bins in (("atac", atac_means, 32), ("h3k27ac", h3k27ac_means, 24)):
            values = torch.as_tensor(means)
            if values.shape != (8,) or not torch.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid output means")
            if self.output_scaling is not None:
                values = soft_clip(values / torch.tensor(self.output_scaling[assay]))
            self._initialize_head_means(getattr(self, f"{assay}_head"), values.repeat(bins))

    def forward(self, one_hot, attention_mask, atac_target_mask, h3k27ac_target_mask):
        if attention_mask.shape != (len(one_hot), 2048) or not attention_mask.all():
            raise ValueError("enhancerNet requires unpadded 2048-bp inputs")
        for mask, lo, hi in ((atac_target_mask, 768, 1280), (h3k27ac_target_mask, 256, 1792)):
            expected = torch.zeros_like(attention_mask)
            expected[:, lo:hi] = True
            if mask.shape != expected.shape or not torch.equal(mask, expected):
                raise ValueError("enhancerNet requires centered 512/1536-bp target masks")
        representation = self.encoder(one_hot)
        outputs = tuple(F.softplus(getattr(self, f"{assay}_head")(representation)).reshape(
            len(one_hot), bins, 8) for assay, bins in (("atac", 32), ("h3k27ac", 24)))
        if self.output_scaling is not None:
            outputs = tuple(inverse_soft_clip(v) * getattr(self, f"{assay}_output_means")
                            for assay, v in zip(("atac", "h3k27ac"), outputs, strict=True))
        return outputs


class EnhancerNetCNNRegressor(EnhancerNetRegressor):
    def __init__(self, **kwargs):
        super().__init__(with_attention=False, **kwargs)


class EnhancerNetAttentionRegressor(EnhancerNetRegressor):
    def __init__(self, **kwargs):
        super().__init__(with_attention=True, **kwargs)


class EnhancerNetClassifier(nn.Module):
    def __init__(self, regressor):
        super().__init__()
        self.encoder = regressor.encoder
        self.head = nn.Sequential(nn.Linear(256, 8))

    def forward(self, sequence):
        return self.head(self.encoder(sequence))


MODELS = {"enhancernet_cnn": EnhancerNetCNNRegressor,
          "enhancernet_attention": EnhancerNetAttentionRegressor}
