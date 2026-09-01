# Model and checkpoint contract

The default 4x architecture has four convolution/pooling blocks followed by
four relative-position Transformer blocks. The total downsampling factor is
16. Separate ATAC and H3K27ac heads map the shared representation to eight
nonnegative context profiles.

Checkpoint output order is fixed to:

```text
ab, e13, e5, ead, hid, lb, o, wid
```

A release checkpoint contains the architecture, context order, profile
metadata, dataset hash, target standardization, target means, epoch, and
checkpoint score in addition to the state dictionary.

`load_model()` rejects unsupported decoder/cross-attention checkpoints and
loads state dictionaries strictly. This prevents a partially loaded model
from being mistaken for the production architecture.

## Loss contract

`training.loss.name` selects the objective without changing the model output
contract. Supported values are:

- `poisson_atac_standardized_log1p_huber_h3k27ac`;
- `crested_cosine_mse_log_both`.

The CREsted option is a PyTorch port of
[`CosineMSELogLoss`](https://github.com/aertslab/CREsted/blob/main/src/crested/tl/losses/_cosinemse_log.py).
It is calculated independently for ATAC and H3K27ac tensors shaped
`[batch, bins, contexts]`. Log-MSE is reduced across the complete assay tensor;
cosine similarity is calculated across the last, eight-context dimension at
each bin. The two assay losses are summed with equal weight.

Multipliers, the maximum dynamic cosine weight, and the optional minimum
target-vector norm are explicit configuration values. `minimum_target_norm: 0`
reproduces CREsted's inclusion of zero vectors; a positive value masks
low-norm target bins from only the cosine term. Log-MSE always includes every
bin.

Every best-model and restart checkpoint records the full resolved
configuration. A restart is rejected if the loss configuration differs from
the checkpoint, preventing accidental continuation under a different
objective.

## Specificity fine-tuning contract

When `specificity_finetuning.enabled` is true, training has two sequential,
independently restartable stages:

1. the base stage trains on the complete prepared training table and writes
   `model/`;
2. the specificity stage initializes from `model/best_model.pt`, resets the
   optimizer at the configured lower learning rate, and writes
   `model_specific_finetune/`.

The specificity score is the Gini index across contexts after averaging each
assay's target profile over genomic bins. ATAC thresholds use ATAC and joint
peak windows; H3K27ac thresholds use H3K27ac and joint peak windows. Each
threshold is fitted on the training split as `mean + k * standard deviation`.
The stage-two set is the union of windows passing either threshold; random
background windows cannot enter it. Training-derived thresholds are applied
unchanged to validation.

Stage-two checkpoint selection uses the scientific composite calculated only
on the specific validation subset. The log also retains complete-validation
losses and metrics so loss of broad performance can be detected. Release
checkpoints record the threshold values, subset counts, initialization
checkpoint hash, and training stage.
