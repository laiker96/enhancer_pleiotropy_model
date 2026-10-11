# Usage details

Reference for the `epm` commands; start with the [README](../README.md).

## Reproducing all models in the manuscript

The architecture comparison has 35 models:

- **Regressors:** 5 architectures.
- **Classifiers:** 30 = 5 architectures × {pretrained, scratch} × seeds 20260914, 20260915 and 20260916.

```bash
epm manuscript            # print the 35 commands
epm manuscript --run      # run them one after another on this machine
```

On a Slurm cluster, submit each command as its own job. Pretrained classifiers must wait for
their regressor:

```bash
for arch in cnn attention dense enhancernet_cnn enhancernet_attention; do
  job=$(sbatch --parsable scripts/slurm_job.sh epm train-regressor --architecture $arch --resume)
  for seed in 20260914 20260915 20260916; do
    sbatch --dependency=afterok:$job scripts/slurm_job.sh epm train-classifier --architecture $arch --init pretrained --seed $seed --resume
    sbatch scripts/slurm_job.sh epm train-classifier --architecture $arch --init scratch --seed $seed --resume
  done
done
```

`--resume` continues from the last checkpoint, so a job that hit its time limit can simply be
resubmitted.

Approximate cost on one GTX 1060 (6 GB):

| Run | Time |
|---|---|
| One regressor, 40 epochs | ~110 h |
| One classifier, 40 epochs | ~6 h |
| Calibration | ~2 min |
| Attribution maps, all 40,338 enhancers | ~35 min per 16 enhancers (~1,500 GPU-hours); RTX 2080: 13 min (~540 h) |

Recent GPUs are several times faster. Attribution is split into independent 16-enhancer chunks;
an interrupted job resumes from its last completed block of references.

## Configuration

| File | Used by | Contents |
|---|---|---|
| [`configs/data.yaml`](../configs/data.yaml) | `prepare-data` | Input paths, chromosome splits, window sampling, enhancer labels, background matching |
| [`configs/regressor.yaml`](../configs/regressor.yaml) | `train-regressor` | Architecture, optimiser, learning-rate schedule, loss weights |
| [`configs/classifier.yaml`](../configs/classifier.yaml) | `train-classifier` | Architecture, `pretrained`/`scratch`, seed, readout, optimiser |
| [`configs/attribution.yaml`](../configs/attribution.yaml) | `attribute` | Classifier, references, integration steps, batching |
| [`configs/manuscript.yaml`](../configs/manuscript.yaml) | `manuscript` | The 35-run model matrix |

Override any value without editing the file:

```bash
--set key.subkey=value                                   # repeatable
epm train-regressor --set training.batch_size=32 --set output=runs/test
```

For a quick end-to-end check, add `--set training.max_batches_per_epoch=20 --set training.epochs=1`.

Splits are by chromosome: validation is chr2L 0–11.75 Mb, test is chr3R, and everything else is
training. Training never reads the test split.

## Inputs and outputs

`epm prepare-data` reads only drosophila-cCREs outputs, from one of two places:

- **The archived atlas (default, `configs/data.yaml`).** This is the exact input of the manuscript, the
  drosophila-cCREs v4 export on Zenodo ([10.5281/zenodo.23296290](https://doi.org/10.5281/zenodo.23296290)).
  `epm prepare-data` on it reproduces the manuscript's training datasets exactly.
- **Your own pipeline run (`configs/data_pipeline.yaml`).** Point `ccre:` at a checkout of drosophila-cCREs
  (written for commit
  [`8508143`](https://github.com/laiker96/drosophila-cCREs/commit/850814395a5482a31bab8628526d8eaabb4d2987),
  the eight-context atlas) and set its run IDs. `prepare-data` prints the manuscript's window and enhancer
  counts next to yours, so you can see whether your run reproduces the same dataset.

The files used, and where a pipeline run writes them:

| Input | In the archive (`drosophila_ccre_regulatory_analysis_bundle_v4/`) | In a pipeline run |
|---|---|---|
| dm6 genome and ENCODE blacklist v2 | `reference/dm6.fa`, `reference/dm6.blacklist.bed` | `references/dm6/dm6.fa`, `dm6.blacklist.bed` |
| Master DHS set and summits | `regulatory/master_dhs/master_dhs{,_summits}.bed` | `results/<project>/<master run>/atac/master/master_dhs{,_summits}.bed` |
| Normalized ATAC and H3K27ac bigWigs (16) | `regulatory/normalized_mean_bigwig/` | `results/<project>/<catalog run>/activity/catalog/tracks/` |
| H3K27ac replicate peaks | `regulatory/h3k27ac/peaks/<library>_peaks.broadPeak` | `results/<project>.chip_histone.dm6/<qc run>/peaks/<library>/<library>_peaks.broadPeak` |
| Regulatory-element catalog | `catalog/master_elements_wide.tsv.gz` | `results/<project>/<catalog run>/activity/catalog/master_elements_wide.tsv.gz` |

It writes:

```text
data/regression/        windows.tsv.gz (951,032 windows: sequence, split, source)
                        profiles/{atac,h3k27ac}/{train,validation,test}_profiles.npy
data/classification/    {train,validation,test}.npz (40,338 enhancers: one-hot codes, 8 labels)
                        enhancers.tsv.gz (all 40,455 catalog enhancers, split or exclusion reason)
                        background/{train,validation,test}.npz (GC-matched genomic negatives)
runs/regressors/<architecture>/                  best_model.pt, history.json, metrics.json
runs/classifiers/<architecture>_<init>_<seed>/   best_model.pt, history.json, calibration.json,
                                                 calibration_report.md
runs/attribution/<run>/chunks/chunk_<first>.npz  attribution maps, 16 enhancers per file
```

Each attribution chunk holds, for its enhancers (order: train, validation, test):

| Array | Shape | Meaning |
|---|---|---|
| `actual` | `[n, 8, 2048]` | contribution of the observed base at each position, per context |
| `hypothetical` | `[n, 8, 4, 2048]` | contribution each base (A, C, G, T) would make |
| `actual_reference_se`, `hypothetical_reference_se` | as above | standard error across references |
| `actual_reference_halves` | `[n, 2, 8, 2048]` | maps from references 1–50 and 51–100 |
| `sequence`, `labels`, `ids`, `split`, `chrom`, `summit` | | the enhancer (sequence as 0–3 = A/C/G/T) |
| `native_offset`, `native_length` | `[n]` | the DHS interval inside the input: `offset : offset + length` |
| `calibrated_probabilities`, `probabilities`, `logits` | `[n, 8]` | model outputs |
| `delta`, `target_difference`, `quality_pass` | | completeness check per reference and context |

Because integrated gradients are additive, maps for any fixed weighting of the eight contexts
(their sum, the mean over active contexts, a contrast) are the same weighting of these maps.
