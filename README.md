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

Run the complete workflow on a CUDA host:

```bash
mkdir -p logs
tmux new -d -s enhancer_pleiotropy_train \
  '.venv/bin/snakemake --configfile config/default.yaml --cores 4 --resources gpu=1 --rerun-incomplete 2>&1 | tee logs/train.log'
tail -f logs/train.log
```

The Slurm launcher in `cluster/` rejects the login node and submits the
Snakemake workflow to a compute node:

```bash
sbatch cluster/train_default.sbatch
```

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

## Training defaults

- stochastic reverse complement with probability 0.5;
- forward/RC ensemble for validation;
- ATAC raw Poisson NLL;
- H3K27ac train-standardized log1p SmoothL1;
- AdamW with weight decay 0.01;
- linear warmup to `1e-4`, then `5e-5`;
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
