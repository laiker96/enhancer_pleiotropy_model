# Enhancer pleiotropy model

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
  '.venv/bin/snakemake --configfile config/default.yaml --cores 4 --resources gpu=1 --rerun-incomplete 2>&1 | tee logs/train.log'
tail -f logs/train.log
```

The Slurm launchers in `cluster/` reject the login node. For the final run,
first create the repository-local environment on a CPU compute node, then
submit training with an `afterok` dependency:

```bash
environment_job=$(sbatch --parsable cluster/setup_environment.sbatch)
sbatch --dependency="afterok:${environment_job}" cluster/train_final_4x.sbatch
```

The final training launcher calls the restartable trainer directly, so raw
BigWigs are not needed on the GPU node once prepared arrays have been copied.

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

## Training defaults

- stochastic reverse complement with probability 0.5;
- forward/RC ensemble for validation;
- ATAC raw Poisson NLL;
- H3K27ac train-standardized log1p SmoothL1;
- AdamW with weight decay 0.01;
- linear warmup to `1e-4`; the final configuration uses a four-epoch cosine
  transition to `5e-5` before plateau scheduling;
- validation-plateau reduction by 0.5 after three epochs;
- scientific checkpoint score combining window and tissue-pattern Pearson;
- restartable batch and epoch checkpoints.

The test chromosome is excluded from training and validation and is evaluated
only after model and analysis choices are frozen.

## Tests

```bash
.venv/bin/pytest
```

The repository does not yet include the enhancer classifier. Because the
catalog labels are derived from ATAC and H3K27ac, the first enhancer caller
should calibrate a score from the regressor outputs. A classifier can be added
later if it uses independent functional labels or demonstrably improves
held-out performance.
