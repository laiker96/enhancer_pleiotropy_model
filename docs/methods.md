# Methods

What the data preparation, models, calibration and attribution do, and how the code was verified against the manuscript.

## Data and models

**Regression data.**

- Inputs are 2,048-bp genomic windows. Every window whose central 512 bp overlaps a DHS or an
  H3K27ac peak is used, plus an equal number of random background windows.
- Targets are background-normalized signal: ATAC in 32 × 16-bp bins over the central 512 bp,
  and H3K27ac in 24 × 64-bp bins over the central 1,536 bp, for all eight contexts.
- The loss is adapted from AlphaGenome: per-context Poisson + multinomial terms over bins, plus
  an auxiliary term across contexts.

**Regressor architectures** (`--architecture`):

- `cnn` — the residual convolutional stem from the figure, followed by six dilated convolutions.
- `attention` — the same stem with four relative-position transformer blocks.
- `dense` — the same stem, flattened into two dense layers.
- `enhancernet_cnn`, `enhancernet_attention` — the earlier DeepSTARR-style enhancerNet
  ([laiker96/enhancerNet](https://github.com/laiker96/enhancerNet)), without or with a transformer.

**Classification data.** An enhancer is an enhancer-like master DHS that is active in at least
one context. Active means it is a member of that context's DHS set *and* its H3K27ac signal is
above the 60th percentile of that context's genomic background. The model input is the 2,048 bp
centred on the DHS summit. This gives 40,338 enhancers:

| Split | Enhancers |
|---|---|
| Train | 26,930 |
| Validation | 4,062 |
| Test | 9,346 |

**Classifier training.**

- **Initialization:** the classifier reuses the regressor's encoder, either pretrained or random.
- **Readout:** `cnn` and `attention` keep the regressor's assay hidden layers ([figure](figures/best_model_architecture.svg)). The other
  architectures use a new linear head.
- **Learning rates:** with pretraining, the encoder learns at 0.1× the rate of the head.
- **Background:** each batch adds one GC-matched genomic background (all-negative) per three enhancers.
- **Augmentation:** reverse complement.
- **Checkpoint selection:** best validation macro average precision over 40 epochs.
- **Predictions** average the forward and reverse-complement probabilities.

**Calibration** (`epm calibrate`).

- **Methods compared:** no change, temperature scaling, and Platt (sigmoid) scaling, fitted per
  context with a positive slope so that rankings are preserved.
- **Evaluation:** five spatially blocked validation folds. Any training example whose DNA overlaps
  a held-out example is removed.
- **Populations:** enhancers only, or enhancers plus their matched backgrounds.
- **Final fit:** the method with the lowest out-of-fold log loss is refitted on all validation
  enhancers. In the manuscript, Platt scaling on enhancers only was selected.

**Attribution maps** (`epm attribute`) are integrated gradients of the eight calibrated
probabilities over the full 2,048 bp:

- **References:** 100 dinucleotide-shuffled copies of each enhancer, seeded by enhancer ID.
- **Integration:** 64 Gauss–Legendre steps.
- **Maps:** actual and hypothetical contributions, averaged over references, with their standard
  errors.
- **Quality check:** for every reference, the contributions must sum to the change in
  probability within 0.002 + 5%.

**The manuscript model** is `cnn`, pretrained, seed 20260914, selected at epoch 38 (test, chr3R):

| Metric | Value |
|---|---|
| Macro average precision | 0.637 |
| Macro AUROC | 0.854 |
| Enhancer vs. matched background, average precision | 0.941 |

## Reproducibility

This code was verified against the original research implementation:

- **Data:** rebuilt from the same drosophila-cCREs outputs, the datasets are identical to the
  ones used in the manuscript — every window, profile, enhancer and background.
- **Training:** on the same GPU and software versions, every architecture trains to bit-identical
  weights. Both trainers seed every random generator and use deterministic CUDA kernels.
- **Calibration:** given the same validation predictions, the method comparison and both final
  calibrators are identical to the manuscript's.
- **Attribution:** recomputing a manuscript chunk on a different GPU (GTX 1060 instead of RTX 2080)
  reproduced every reference sequence exactly. The maps agree to within 1e-7 (values up to 0.12;
  correlation 1.0) and pass the same quality checks.
- **Other hardware:** a different GPU model or library version changes the numbers slightly, not
  the conclusions.
