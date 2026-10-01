# Historical regressor README (preserved before the reproducibility cleanup)

Archived on 2026-09-21 without deleting the previous instructions below.
These describe earlier regressors and site-specific workflows, not the current
transfer-learning default. Some links are relative to the repository root.
Use [the current README](../README.md) for the portable v4 workflow.

This repository contains the minimal reproducible pipeline used to train and
load the Drosophila eight-context joint ATAC/H3K27ac sequence model. It is
deliberately narrower than the experimental workspace from which it was
extracted.

The default model is the best-performing **4x Enformer-like joint profile
regressor**. It accepts a 2,048-bp one-hot DNA sequence and predicts:

- ATAC over the central 512 bp as 32 x 16-bp bins;
- H3K27ac over the central 1,536 bp as 24 x 64-bp bins;
- both assays in `ab`, `e13`, `e5`, `ead`, `hid`, `lb`, `o`, and `wid`.

`e11` is intentionally excluded.

## Repository layout

```text
config/default.yaml                 Default data/model/training configuration
src/enhancer_pleiotropy_model/      Importable model and pipeline code
src/.../preprocessing/              Window, peak, and BigWig processing
scripts/create_environment.sh       Project-local mamba environment
Snakefile                           Reproducible preprocessing and training DAG
cluster/                            Slurm launcher; never computes on login node
tests/                              Focused unit and checkpoint-compatibility tests
docs/                               Data and model contracts
```

Raw data, prepared arrays, checkpoints, logs, and reports are ignored by Git.

## Required inputs

The default workflow expects:

1. dm6 FASTA;
2. dm6 blacklist BED;
3. master DHS BED and summit BED;
4. H3K27ac replicate broadPeak files, named
   `<context>_h3k27ac_rep<number>_peaks.broadPeak`;
5. normalized mean BigWigs named
   `<context>.<assay>.mean.background_tmm.bw`.

Peak files determine sampling strata. Regression labels are always extracted
from the BigWigs. BAM files are not required when normalized BigWigs already
exist.

Edit paths in `config/default.yaml`; do not commit raw data.

## Environment

Install all packages with mamba into the repository-local `.venv` prefix:

```bash
bash scripts/create_environment.sh
mamba activate "$PWD/.venv"
```

The setup writes exact installed versions to `environment.lock.txt`.

## Prepare data and train

Inspect the workflow first:

```bash
.venv/bin/snakemake --configfile config/default.yaml --dry-run
```

Run preprocessing on CPU:

```bash
.venv/bin/snakemake --configfile config/default.yaml --cores 4 prepared_data
```

The final 40-epoch run uses `config/final_4x.yaml`. It trains on chrX, chr2R,
chr3L, chr4, chrY, the two configured unplaced scaffolds, and the right half of
chr2L. The left half of chr2L is validation and chr3R is test. A 10-kb gap is
excluded around the chr2L midpoint, and every complete 2,048-bp input must fit
inside one split.

```bash
.venv/bin/snakemake --configfile config/final_4x.yaml --cores 4 prepared_data
```

Run the complete workflow on a CUDA host:

```bash
mkdir -p logs
tmux new -d -s enhancer_pleiotropy_train \
  '.venv/bin/snakemake --configfile config/final_4x.yaml --cores 4 --resources gpu=1 --rerun-incomplete 2>&1 | tee logs/train.log'
tail -f logs/train.log
```

The Slurm launchers in `cluster/` reject the login node. For the final run,
first create the repository-local environment on a CPU compute node, then
submit training with an `afterok` dependency:

```bash
environment_job=$(sbatch --parsable cluster/setup_environment.sbatch)
sbatch --dependency="afterok:${environment_job}" cluster/train_final_4x.sbatch
```

The final launcher defaults to the broad base stage, which trains for up to 40
epochs. After inspecting that checkpoint, submit the independently restartable
specificity stage with a dependency on the completed base job:

```bash
base_job=$(sbatch --parsable cluster/train_final_4x.sbatch)
sbatch --dependency="afterok:${base_job}" \
  --export=ALL,TRAINING_STAGE=specificity \
  cluster/train_final_4x.sbatch
```

The specificity stage loads the best base checkpoint and fine-tunes the full
network for up to 10 epochs at an initial learning rate of `1e-5`. Raw BigWigs
are not needed on the GPU node once prepared arrays have been copied.
High-Gini peaks are assigned to their dominant context and resampled to equal
context counts. Rare training contexts are repeated by at most two-fold;
specificity validation is downsampled to equal natural context counts.

## Load a checkpoint

```python
from enhancer_pleiotropy_model import load_model

model, metadata = load_model("results/default_4x/model/best_model.pt")
model.eval()
print(metadata.contexts)
```

For tabular sequence inference:

```bash
.venv/bin/enhancer-predict \
  --checkpoint results/default_4x/model/best_model.pt \
  --sequences sequences.tsv \
  --output predictions.npz \
  --reverse-complement-ensemble
```

`sequences.tsv` must contain `id` and `sequence`; production checkpoints
expect 2,048 unambiguous A/C/G/T bases.

## Observed/predicted browser tracks

Generate paired observed and predicted BigWigs across the final chr2L
validation interval, followed by a portable IGV session:

```bash
.venv/bin/enhancer-browser-tracks \
  --checkpoint results/default_4x/model/best_model.pt \
  --reference-fasta data/raw/reference/dm6.fa \
  --blacklist-bed data/raw/reference/dm6.blacklist.bed \
  --observed-bigwig-directory data/raw/bigwig \
  --output-directory results/default_4x/browser/chr2L_validation \
  --chromosome chr2L \
  --region-start 0 \
  --region-end 11751856 \
  --stride 256 \
  --batch-size 64 \
  --device cuda \
  --mixed-precision fp16
```

The command creates observed/predicted pairs for both assays and all eight
contexts plus `igv_session.xml`. It uses a complete genome-anchored sliding
grid, not the balanced training-table subset. Every overlapping prediction is
averaged at each native model bin (16 bp for ATAC and 64 bp for H3K27ac), and
the observed tracks use the identical bins and support mask. Expensive
inference is checkpointed in `.partial_predictions.npz`.

The current local checkpoint was selected on chr2L validation. These browser
tracks are therefore appropriate for qualitative model QC, not as an
untouched test-set performance estimate.

Evaluate the frozen checkpoint on the v3 distal/proximal enhancer-like set
active in at least one retained context (about 40,000 elements):

```bash
.venv/bin/enhancer-evaluate-catalog \
  --catalog ../drosophila_ccre_contact_analysis_bundle_v3/catalog/master_elements_wide.tsv.gz \
  --catalog-long ../drosophila_ccre_contact_analysis_bundle_v3/catalog/master_elements_long.tsv.gz \
  --split-dataset ../enhancer_transformer/outputs/prepared/dataset.tsv \
  --reference-fasta data/raw/reference/dm6.fa \
  --checkpoint results/default_4x/model/best_model.pt \
  --output-directory results/default_4x/enhancer_catalog_evaluation
```

Inference uses a forward/reverse-complement average by default and saves
restartable chunks. Activity thresholds are selected on validation only;
test metrics therefore remain held out. In each context, the negative class
contains enhancers active in another retained context, making this a direct
test of context specificity rather than an easy enhancer/background test.

Calibrate enhancer probabilities from the cached regressor summaries without
retraining the sequence model:

```bash
.venv/bin/enhancer-calibrate \
  --predictions results/default_4x/enhancer_catalog_evaluation/predicted_features.npz \
  --catalog ../drosophila_ccre_contact_analysis_bundle_v3/catalog/master_elements_wide.tsv.gz \
  --output-directory results/default_4x/enhancer_calibration
```

The command compares the existing geometric-mean score, a fuzzy percentile
AND, smooth products of separately calibrated ATAC-membership and
H3K27ac-activity probabilities with either local or cross-context assay
features, one logistic calibrator per context using its matching ATAC and
H3K27ac summaries, ATAC-only and H3K27ac-only eight-feature ablations, and a
linear 16-to-8 calibrator using all context/assay summaries.
It fits model weights on the left half of the existing validation chromosome,
selects classification thresholds on the right half, excludes a 10-kb buffer,
and evaluates once on the existing test chromosome. The regressor's training
examples are not used to fit the calibrators. Outputs include `metrics.json`,
reloadable parameters and predictions, reliability bins, and a concise
`summary.md`. The saved component probabilities and their product are intended
as continuous analysis scores; the product is not assumed to be an exact joint
probability because the assays are correlated.

For continuous ATAC and H3K27ac breadth with additive context/group contributions
on the final checkpoints, see [Continuous assay breadth](docs/continuous_breadth.md).
This directly scores the existing signal outputs and includes saturation and
log-signal diagnostics; no classifier or additional inference is needed.

For the opt-in cluster fine-tuning pilot that jointly trains signals, breadth
and related-context discrimination, see [Joint breadth training](docs/joint_breadth_training.md).

For the separate catalog-defined expected-count target, see
[Expected activity breadth](docs/expected_breadth.md).
The local CUDA workflow fits probability heads from cached predictions and
reports expected-count errors, calibration, group allocation, assay ablations,
and paired genomic-block uncertainty. These evaluations are exploratory.

Audit whether the joint calibrator reduces cross-group confusion and excessive
tissue breadth:

```bash
.venv/bin/enhancer-audit-calibrator \
  --predictions results/default_4x/enhancer_calibration/calibrated_predictions.npz \
  --parameters results/default_4x/enhancer_calibration/calibrator_parameters.npz \
  --output-directory results/default_4x/enhancer_calibration/audit
```

The audit compares the original geometric score and joint linear calibrator on
the saved test predictions without refitting. It reports standardized linear
coefficients, related-context false positives, tissue-breadth error, and true
versus predicted context-correlation matrices. The predefined related groups
are adult/larval brain, embryonic stages, and the three imaginal discs.

Evaluate the model-derived continuous enhancer score without fitting a binary
classifier:

```bash
.venv/bin/enhancer-evaluate-continuous-activity \
  --predictions results/default_4x/enhancer_catalog_evaluation/predicted_features.npz \
  --catalog ../drosophila_ccre_contact_analysis_bundle_v3/catalog/master_elements_wide.tsv.gz \
  --bigwig-directory data/raw/bigwig \
  --output-directory results/default_4x/continuous_activity \
  --split test
```

The score is the geometric mean of ATAC and H3K27ac signal. Observed ATAC is
re-extracted from the normalized BigWigs over the exact summit-centered 512 bp
used by the ATAC output; observed H3K27ac is the maximum of the catalog's three
500-bp windows. Predictions use the analogous 512-bp summaries. The command
reports per-context signal correlations, within-enhancer tissue-pattern
correlations, continuous participation-ratio pleiotropy, related-context
contrasts, hard-breadth descriptive subsets, and assay-state strata. It does
not fit or tune a classifier, and hard catalog calls are used only to describe
subsets.

Build the preferred master-element state table without combining raw assay
scales:

```bash
.venv/bin/enhancer-build-master-states \
  --catalog ../drosophila_ccre_contact_analysis_bundle_v3/catalog/master_elements_wide.tsv.gz \
  --split-dataset ../enhancer_transformer/outputs/prepared/dataset.tsv \
  --predictions results/default_4x/enhancer_catalog_evaluation/predicted_features.npz \
  --windows results/default_4x/data/windows.tsv.gz \
  --atac-train-profiles results/default_4x/data/profiles/atac/train_profiles.npy \
  --h3k27ac-train-profiles results/default_4x/data/profiles/h3k27ac/train_profiles.npy \
  --bigwig-directory data/raw/bigwig \
  --output-directory results/default_4x/master_element_states
```

This extracts observed signal directly from the normalized BigWigs using the
model's exact target geometry: the central 512 bp for ATAC and the maximum of
three adjacent 512-bp means for H3K27ac. Each assay/context is converted to a
positive robust-z excess in `log1p` space using only genomic-background
training windows on chromosomes shared by the local and final training
schemes. chr2L, chr3R, and chrX are excluded from fitting the transform.
ATAC and H3K27ac pleiotropy remain separate. The optional joint score is the
minimum of their standardized excess values, so one high assay cannot
compensate for the other. The output contains all non-blacklisted distal and
proximal enhancer-like master elements as observed states; cached frozen-model
predictions are attached to the active subset on which inference was already
run. Hard catalog activity calls are annotations and do not set the continuous
score.

Calibrate the two assay summaries to a common empirical training-background
percentile scale and reevaluate the frozen model with:

```bash
.venv/bin/python -m enhancer_pleiotropy_model.master_element_calibration \
  --master-states results/default_4x/master_element_states/master_element_states.npz \
  --windows results/default_4x/data/windows.tsv.gz \
  --atac-train-profiles results/default_4x/data/profiles/atac/train_profiles.npy \
  --h3k27ac-train-profiles results/default_4x/data/profiles/h3k27ac/train_profiles.npy \
  --output-directory results/default_4x/master_element_percentile_calibration
```

The empirical reference contains only `genomic_background` rows from the
training chromosomes. Each assay/context summary is mapped to its frozen
mid-rank background percentile and then to positive percentile excess with
`max(0, 2 * percentile - 1)`. The command records the fitted reference,
distribution diagnostics, calibrated assay and joint-min states, and
observed-versus-predicted performance. This is an evaluation transform; it
does not retrain or modify the regressor.

Generate the complete quantitative validation report reproducibly with:

```bash
XDG_CACHE_HOME="$PWD/.cache" .venv/bin/snakemake \
  --configfile config/default.yaml \
  --cores 4 \
  --rerun-incomplete \
  browser_validation_report \
  --resources gpu=1
```

This target creates per-context accuracy metrics, tissue-pattern metrics,
observed/predicted context-correlation matrices, target-PCA diagnostics,
DHS/H3K27ac/background stratification, deterministic success/failure
bookmarks, 16 signed residual BigWigs, a combined IGV session, and a
self-contained `index.html`. Every quantitative input is SHA-256 hashed in
`analysis_config.json`. Definitions and standalone commands are documented in
[`docs/browser_validation.md`](docs/browser_validation.md).

## Training defaults

- stochastic reverse complement with probability 0.5;
- forward/RC ensemble for validation;
- the reusable default configuration keeps ATAC raw Poisson NLL and H3K27ac
  train-standardized log1p SmoothL1;
- the final 40-epoch experiment in `config/final_4x.yaml` instead applies
  CREsted `CosineMSELogLoss` independently to both assays, with cosine
  similarity calculated across the eight contexts at each output bin;
- AdamW with weight decay 0.01;
- linear warmup to `1e-4`; the final configuration uses a four-epoch cosine
  transition to `5e-5` before plateau scheduling;
- validation-plateau reduction by 0.5 after three epochs;
- `best_model.pt`, selected by the scientific composite combining window and
  tissue-pattern Pearson;
- `best_low_overcorrelation_model.pt`, independently selected by the lowest
  mean positive excess of predicted versus observed pairwise context
  correlations on regulatory validation windows;
- rolling `last_checkpoint.pt` restart state at batch and epoch boundaries.

For CREsted runs, training and validation logs separately report each assay's
log-MSE, mean context-vector cosine similarity, dynamic cosine weight, and
combined loss. The checkpoint records the exact loss configuration and
multipliers. The loss change does not alter the prepared data or chromosome
splits.

The CREsted-style specificity stage fits assay-specific Gini thresholds on
training peaks only. Window activity is the mean target signal in each of the
eight contexts. A peak is retained when its ATAC Gini exceeds the ATAC
training-peak mean plus one standard deviation, or its H3K27ac Gini exceeds
the corresponding H3K27ac threshold. Background windows are excluded from
this stage. The fine-tuned checkpoint is selected on the matching
context-balanced specificity validation subset, while full-validation metrics are also
logged every epoch. Outputs are written to `model_specific_finetune/`; the
broad base outputs remain in `model/`.

The test chromosome is excluded from training and validation and is evaluated
only after model and analysis choices are frozen.

### Combine broad and specificity checkpoints

The validation-only residual ensemble retains the base checkpoint's mean
`log1p` activity across contexts at every output bin while blending its
context-centered residuals with those from the specificity checkpoint. It
selects separate ATAC and H3K27ac blend coefficients on balanced specificity
validation, subject to at most a 1% decrease in the complete-validation
scientific composite. The test split is not loaded or evaluated for selection.

```bash
sbatch cluster/evaluate_residual_ensemble.sbatch
```

The default comparison uses base epoch 31 and specificity epoch 1. Results are
written to `residual_ensemble_base31_specific1/metrics.json`, with all tested
coefficient pairs in `alpha_grid.tsv`. This combines predictions at inference
time and does not alter either checkpoint.

## Tests

```bash
.venv/bin/pytest
```

The included enhancer classifier is intentionally limited to lightweight
calibration of the frozen regressor. Because the catalog labels are derived
from ATAC and H3K27ac, it should not be interpreted as independent functional
validation or as a genome-wide caller trained against background sequence.
