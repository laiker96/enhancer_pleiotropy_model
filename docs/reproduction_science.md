# Scientific contracts and reference results

## Data and targets

The regressor uses master DHS/H3K27ac-consensus windows plus genomic background,
not just the active enhancer catalog. Inputs are 2,048 genomic bp. ATAC targets
cover central 512 bp (32 bins of 16 bp); H3K27ac targets cover central 1,536 bp
(24 bins of 64 bp, pooled from 16-bp source bins). Track ordering is always
`ab,e13,e5,ead,hid,lb,o,wid`.

The 40,455-row v4 active enhancer catalog supplies binary classifier labels:
context DHS membership AND H3K27ac atlas percentile >0.60. After full-input
containment, blacklist, valid DNA and exact/RC cross-split duplicate checks,
40,338 enhancers remain: 26,930 train, 4,062 validation, 9,346 test.
Validation is chr2L 0–11,751,856; training uses chr2L 11,761,856–23,513,712 and
the configured other chromosomes; test is chr3R. Complete inputs must remain
inside a split. chr3R has already been inspected historically: it is not a
pristine independent holdout for a new paper claim.

The classification label describes the **catalog enhancer centered in the
input**, not a per-base enhancer segmentation. Context zeros and sampled
background are operational negatives, not proof of inactivity. Labels derive
from the same assays used for pretraining; this is transfer across objectives,
not independent functional validation.

Background training adds one genomic background per three enhancers (25% of
examples), retaining all enhancers and the same 421 updates per epoch. Matching
uses same chromosome and GC fraction within 0.02 for both full 2,048 bp and
central 512 bp, unique matches, and exact/RC duplicate exclusions across splits.
The background **center** avoids master DHS/H3K27ac consensus; its flanks need
not be devoid of regulatory sequence. Test matching uses seed 20260915;
train/validation matching uses 20260916 and excludes locked test backgrounds.

## Architecture and initialization

All share a convolutional stem with channels 96,128,160,192, giving 128 latent
positions at 16-bp stride. Dilated CNN uses six residual convolutions with
dilations 1,2,4,8,16,32 in place of the attention mixer. Flatten/dense removes
the mixer, flattens 128×192 positions and uses two shared 256-unit dense blocks.
Changing mixer/readout is an architecture comparison, not a single-variable
attention ablation. All requested models are trained from scratch as regressors.

CNN/attention classifier readouts:

- `legacy`: discard both assay hidden MLPs; pool three spatial encoder segments,
  then use a new 576→192→8 readout. Transfer the stem/mixer only for fine-tuning.
- `retained_assay_hidden_v1`: retain the two regression LayerNorm/192→384/GELU/
  dropout hidden blocks. Pool one ATAC segment and three H3K27ac segments;
  a new 1,536→8 projection predicts context logits. Remove the final signal
  projectors, Softplus and output scaling. Fine-tune all transferred layers.

Dense retains its shared flatten/dense representation and replaces the final
profile projectors with 256→8. Its historical readout key is `legacy`; this
does **not** imply its shared dense hidden layers are discarded.

Scratch and pretrained fits of a given readout share the new classifier-head
initialization seed and training-label prevalence bias. Scratch fits train the
same architecture with a randomly initialized encoder. Seeds are 20260914/15/16.
One parent regressor per architecture supplies all its classifier seed fits.

## Losses, scaling, schedules, selection

Regressor: AlphaGenome-style profile count + positional losses, with RNA-style
cross-context count/distribution auxiliary loss; positional and cross-context
weights 5, auxiliary weight 0.1, equal sum of assays. There are **no RNA labels
or RNA output head**. Each track is divided by its nonzero training-bin mean
and soft-clipped above 10; H3 pooling precedes fitting. No validation/test data
fit the transform. Saved model predictions are in original background-TMM units.

Regressor schedule: 40 epochs, batch 64, one-epoch linear warmup to 1e-4, four-
epoch cosine transition to 5e-5, then validation-plateau halving (patience 3,
minimum 1e-6). AdamW, weight decay 0.01, clip norm 1, FP16 training, random RC
probability 0.5, forward/RC validation. Select `best_model.pt` by the original
scientific composite, not training loss. The separate low-overcorrelation
checkpoint is diagnostic and is not silently substituted as a transfer parent.

Classifier: unweighted multilabel BCE on eight logits, AdamW, weight decay
0.01, clip norm 1, batch 64 enhancers (+21/22 backgrounds when enabled), FP16,
random RC 0.5. One-epoch warmup then cosine decay to 10% of peak over the
remaining 39 epochs. Scratch encoder/head peak LR = 1e-4/1e-4; fine-tuned
transferred layers/new head = 1e-5/1e-4. Retained assay hidden layers use the
lower transferred-layer LR. Evaluation averages forward/RC **probabilities**.
Select by enhancer-only validation macro average precision, breaking ties by
BCE, even when training with background. Test and background AP do not select
epochs in this protocol.

## Historical benchmark (not a new rerun)

Best combined-validation background classifier among the previously completed
suite: corrected dilated CNN fine-tune seed 20260914, selected epoch 38 of 40.
SHA256 `7dc8eb721fbd4f292eba95ff2ea44399cfd1aae5a596e8d56e6d7ff25a95628a`.
This is the new analysis configuration's architecture/initialization target;
fresh stochastic training need not reproduce this checkpoint byte-for-byte.

| Population | Validation macro AP | Historical test macro AP |
|---|---:|---:|
| Active vs other enhancers | 0.648744 | 0.637113 |
| Active vs paired genomic background | 0.933890 | 0.940572 |
| Active vs other enhancers + background pool | 0.586317 | 0.582833 |

The active/background comparison is balanced per context. The combined
comparison has a different prevalence. These APs must not be compared as if
they measured the same task. Three-seed combined-validation means were
0.585163 (dilated CNN) and 0.584155 (dense); no significant superiority is
claimed. Dense seed 20260916 was nearly tied at 0.585455. Source: historical
`docs/classifier_calibration_20260921.md` and its locked model/prediction audit.

## Attribution contract and reuse

Model is frozen/eval, FP32, deterministic algorithms, TF32/autocast disabled.
For each reference, 64 Gauss–Legendre points approximate input-space IG, with
100 full-input dinucleotide-preserving shuffled references and ID-keyed seed
20260916. Reference construction preserves counts via randomized Euler trails;
it is not asserted to sample every valid shuffle uniformly. Production does
not adaptively increase integration points or references. Failed numerical
quality is retained/flagged, not silently repaired with a new protocol.

Targets are eight calibrated probabilities plus historical mean observed-
active-context logit. Calibrate after averaging forward/RC probabilities:
positive-slope sigmoid applied to clipped probability logit. The calibration
is inside the graph; all eight probability outputs contribute regardless of
the observed label. The ninth target deliberately keeps an observed-active
label mask. Sum of the eight probability targets is expected active-context
count for the known-enhancer population, not an estimate of genome-wide prevalence.

The pilot checks FP32 endpoints against cached FP16 validation inference,
batching equivalence, IG64 vs IG128 native-map agreement, and per-reference
completeness. Each probability tolerance is 0.002 + 0.05×absolute endpoint
change; legacy mean-logit and summed-breadth tolerances follow the original
kernel contract. The latter are recorded with saved maps; numerical
completeness is not proof of biological attribution accuracy.

Actual maps have shape [enhancer,9,2048]; hypothetical maps
[enhancer,9,4,2048], in ACGT order. Reference-specific baseline projections
are subtracted before averaging references. Maps support later linear sums,
fixed-weight contrasts, tissue-group means, motif interval sums and native/
flanking analyses without new gradients. They do **not** recover nonlinear
family OR, a newly changed calibration, hard threshold breadth, arbitrary
mutant effects, or per-context logit IG from probability IG. Per-reference
full tensors/path Jacobians are not retained; arbitrary new covariance/SE
calculations cannot be reconstructed. Standard-error summaries are saved.

Native TF-MoDISco discovery uses only passing training enhancers grouped as
1 / 2–5 / 6–8 by observed labels (or the explicit cumulative option). Padding
is excluded from the null and every seqlet; full genomic flanks influenced the
model but are excluded from native motif discovery. TF-MoDISco 2.5.2 clustering
is reused, with the audited length-boundary adapter, 20,000 seqlets/sign cap,
target seqlet FDR 0.05 and deterministic group order permutation.

Filter native cluster representatives: first-to-last 5-position windows with
at least four columns >0.5 bits; trim/extend terminal flanks at the configured
threshold (default 0.2); retain widths 5–30 bp, mean information ≥0.5 bits,
total ≥5 bits and matching contribution sign. Preserve internal low-info
positions. No extra merging/clustering. Support = distinct training enhancers
with cluster-assigned seqlets / passing discovery group size, **not** sequence-
scan motif prevalence or Fi-NeMo predictive occurrence. Trimming a PWM does
not recalculate original cluster membership; this support limitation is explicit.
Tomtom annotations are similarity matches, not TF occupancy or causal proof.
