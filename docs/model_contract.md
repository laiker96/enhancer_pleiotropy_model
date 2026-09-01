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
