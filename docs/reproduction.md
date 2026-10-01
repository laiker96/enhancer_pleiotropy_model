# Portable reproduction guide

## Input data

The canonical input is `drosophila_ccre_regulatory_analysis_bundle_v4.tar`:

```text
SHA256 0202c8a4fdb1105da658e7895dc2916670f634ec1b4aecb4cd942914e9e19052
root   drosophila_ccre_regulatory_analysis_bundle_v4
```

The `stage-inputs` stage checks the entire archive, its member manifest,
context/sample registry and every extracted member. It stages only required
files, including `catalog/noncontributing_dhs_p60/active_enhancers_wide.tsv.gz`.
It refuses existing destinations. Inputs are never overwritten.

Required scientific inputs: dm6 FASTA and index, dm6 blacklist, master DHS
intervals and summits, selected H3K27ac replicate peaks, 16 context/assay mean
BigWigs, normalization/selection provenance, and the active enhancer catalog.
The 81,035-element master DHS includes the corrected v4 coordinates. Ovary
H3K27ac is CUT&RUN; wing-disc replacements are ChIP-seq. Alias staging retains
those distinctions and the original checksummed track-selection provenance.

BigWigs must already use the v4 background-TMM normalization; preprocessing
verifies metadata and chromosome lengths. Rebuilding alignments, peaks or TMM
from FASTQ is outside this repository's entry boundary. Providing a different
archive requires a separately reviewed input adapter; do not bypass hashes or
silently reuse the old split counts. A public data deposit is still needed for
an outside user to reproduce training without obtaining the export from us.

## Configuration and small runs

Start with `--recipe best`: one dilated-CNN regressor and six classifiers
(scratch/fine-tuning × three seeds), retaining assay hidden heads and training
with 25% background. `--recipe architectures` compares all three primary
architectures (3 regressors + 18 classifiers). `--recipe readouts` tests
replacement/retention in CNN and attention with enhancer-only training
(2 + 24). `--recipe full` selects all supported combinations (3 + 60).

Use the **same recipe at every stage**, including inventory, attribution and
resumption. Recipe outputs live under `paths.work/RECIPE`. Omitting the option
retains the previous behavior and original work directory. Recipe selection
and configuration hashes are recorded in submission/training provenance.
Named recipes cannot override scientific training/attribution parameters.

Paths are relative to `project_root`, which is relative to the configuration
file. Copy `reproduce/config.yaml` **within reproduce/** and choose a new
`paths.work` for every scientifically distinct run. CLI examples below use
the default file; pass `--config reproduce/my_run.yaml` when using a copy.
Do not combine a named recipe with manually changed matrix lists unless that
override is intentional; named recipe selections take precedence.

For the smallest matched best-architecture comparison, change these fields:

```yaml
architectures: [cnn]
readouts: [retained_assay_hidden_v1]
populations: [background]
seeds: [20260914]
```

This schedules one regressor and two classifiers (scratch and fine-tuned).
Keep the existing `analysis_classifier` selection. Add `legacy` to `readouts`
to compare dropping versus retaining assay hidden heads. The full configuration
contains three seeds but **one pretrained regressor per architecture**, as in
the original comparison; this is not three independent pretraining seeds.

Training uses 40 epochs. Shortening `epochs` changes the classifier cosine
schedule and is not a faithful early pause. Preserve 40 for reproductions;
use stop markers/time-limited resumptions for an interrupted full schedule.
GPU memory needs differ by architecture. Do not assume batch 64 with added
background fits an 8-GB GPU; test in an allocation before a large submission.

## Slurm execution

Copy `reproduce/slurm/site.example.yaml` to `site.yaml` and edit both Python
paths (for the README environments, `.venv-reproduce/bin/python` and
`.venv-modisco/bin/python`), CPU/GPU partitions, account, GPU request, array
concurrency, memory/time limits and optional environment-module shell script.
The shell script is a trusted user-controlled file, not downloaded code.
No hostnames or allocation names are built into the portable execution path.

Run commands from the repository root. These are dry runs:

```bash
.venv-reproduce/bin/python reproduce/slurm/submit.py preprocessing --site reproduce/slurm/site.yaml
.venv-reproduce/bin/python reproduce/slurm/submit.py training --site reproduce/slurm/site.yaml
.venv-reproduce/bin/python reproduce/slurm/submit.py analysis --site reproduce/slurm/site.yaml
```

Add `--recipe best` to each command for the recommended smaller comparison.
The staged input directory is shared and immutable: `stage-inputs` runs once.
For another recipe using already staged inputs, submit `prepare-regression`,
then `prepare-classifiers --after JOBID`, rather than repeating `stage-inputs`.
Do not delete or overwrite the shared inputs to change recipes.

Add `--execute` to each only when ready. Do not submit independent pipelines
simultaneously without a dependency: finish preprocessing, then training,
then analysis. To queue ahead, pass `--after JOBID` using the final job ID of
the preceding pipeline. Submission receipts under `work/submissions/` record
commands, config hashes and any partially submitted DAG. Check receipts if
submission fails; do not blindly submit the whole pipeline twice.

```text
preprocessing: stage-inputs -> prepare-regression -> prepare-classifiers
training:      regressor array -> classifier array -> evaluate -> calibrate
analysis:      numerical pilot -> attribution array -> MoDISco group array
```

Each arrow is `afterok`. Default classifier training waits for all three
parents, including the scratch fits; this is deliberately simple. Array
limits apply per array, not globally across unrelated submissions. Request
one GPU per task. Restrict the attribution pipeline to one GPU type, or run
`attribute-pilot` independently on each type before its full shards. The
pilot's hardware/software/input gate prevents unvalidated GPU types from
running full attribution. A pilot timeout/failure blocks downstream jobs.

For more control submit a single stage and selected task indices:

```bash
.venv-reproduce/bin/python scripts/reproduce.py plan
.venv-reproduce/bin/python reproduce/slurm/submit.py train-classifier \
  --tasks 0,1,2 --site reproduce/slurm/site.yaml
```

`plan` shows the ordered matrix; task indices start at zero. `attribute` tasks
are shards 0–79 by default. `modisco` tasks are the configured breadth groups.
No GPU work runs in the submission process: the task verifies its hostname is
inside the Slurm allocation, checks exactly one visible CUDA GPU and imports
the checkout's own source, not an old experiment snapshot.

## Resume, monitor, and stop

Logs: `logs/reproduce/epm_STAGE-JOB_ARRAYTASK.log`. Use `squeue -u "$USER"`
and `tail -f` on the relevant log. Training logs include current learning
rates; attribution is frozen inference and has no learning rate.

Classifier checkpoints: `work/classifiers/RUN_ID/{best_model,last_checkpoint}.pt`.
Regressors: `work/regressors/ARCH/model/`. Slurm sends USR1 three minutes before
the time limit. Training creates a `STOP` marker, saves at a safe boundary,
and returns nonzero (75) when paused, so dependent jobs are not released.
Hard kill/OOM can still lose work since the most recent checkpoint.

To resume, inspect the log and checkpoint, **manually clear only that run's
STOP marker**, and submit its single stage/task with `--resume --execute`.
After a failed array/dependency, submit fresh downstream jobs with dependencies
on the successful replacement; a Slurm `DependencyNeverSatisfied` job will
not repair itself. Do not rerun already complete array members unnecessarily.
Preprocessing/evaluation/MoDISco use fresh outputs and do not support partial
resume; preserve incomplete outputs for inspection and use a new work root or
explicitly recover the failed stage before retrying.

Attribution saves reference accumulator state every completed reference block
at checkpoints and resumes automatically after input/signature checks. A
`work/attribution/STOP` marker stops it at a safe boundary. Each shard has its
own progress/completion receipt and exclusive task lock. Moving inputs or
changing code/settings invalidates the pilot/signature; use a fresh analysis
directory rather than mixing scientific runs.

## Evaluation and inventory

`evaluate` locks every configured model's validation-selected checkpoint before
opening the test data. It writes per-model prediction caches and metrics for
enhancer-only, active-versus-matched-background and combined populations.
AP is average precision; trapezoidal PR area is reported separately. Compare
models on the same population/prevalence, not AP values across populations.
Full reports retain per-context results, breadth, calibration and group metrics.

`calibrate` fits the preselected positive-slope sigmoid method to known-enhancer
validation predictions. It is a **deployment fit**, not a new cross-validation
calibration comparison. Its training-set fit quality must not be called held-out
performance. The historical population comparison is documented separately.

```bash
CUDA_VISIBLE_DEVICES='' .venv-reproduce/bin/python scripts/reproduce.py inventory --local-cpu
```

`work/model_inventory.tsv` includes planned/partial/complete entries, architecture,
readout, training mode/population/seed, checkpoint paths/hashes, selected epochs,
current LRs, validation metrics and completed test APs. This refresh does not
open test DNA or select checkpoints. The original experiment inventory is not
modified by this workflow.
The refreshed table also includes selected-epoch LRs, classifier peak LRs,
initialization checkpoint paths/hashes, separate validation/test background
populations, their per-context AP/AUROC, and regressor profile/central-signal
correlations and R². Missing metrics remain blank. Metrics are taken from the
selected epoch, not whichever epoch completed most recently.

The [paper benchmark tables](paper_benchmarks.md) are a separate, frozen record
of 94 completed historical models. They include all relevant strategies and
additional historical architectures, without implying that every historical
architecture is supported by the new launcher. Regenerate/check the tables
using `python scripts/summarize_paper_benchmarks.py --check` (no Torch/GPU).
Their source JSON uses artifact-root-relative checkpoint paths, not login
names or home directories. These paths do not constitute a public checkpoint
download. Do not merge new-run measurements into the reference table until
their inputs, selections and evaluation populations have been reviewed.

## Attribution and motif output

Attributions are under `work/attribution/`: full-input observed-base scores,
hypothetical A/C/G/T scores, reference SEs/half-reference comparisons, endpoint
predictions, per-reference completeness deltas, labels, native intervals,
reference hashes and input/code provenance. There are eight calibrated-context
targets and one mean observed-active-context logit target. The first eight have
no label mask. The sum is expected context count, not hard predicted breadth.
See [the scientific contract](reproduction_science.md) for reuse limitations.

`motif_databases` maps a name to a local MEME file. For example:

```yaml
motif_databases:
  jaspar: data/raw/motifs/jaspar2026_insects.meme
  flyfactorsurvey: data/raw/motifs/fly_factor_survey.meme
  flyreg: data/raw/motifs/flyreg.v2.meme
```

Use the same database files/releases as the historical analysis for comparable
q-values: JASPAR 2026 CORE insects (296), MEME motif archive 12.27 FlyFactorSurvey
(656), FlyReg v2 (75). The files are not redistributed here. The report records
each supplied database hash and the Tomtom command/version. Empty configuration
performs de novo discovery without annotation; it does not silently fetch a DB.

`work/motifs/TARGET/GROUP/` contains raw signed `motifs.h5`, the length-aware
discovery audit, filtered PWMs/report JSON, `queries.meme`, Tomtom tables and
per-database matches, plus a self-contained `report.html`. All passing motifs
are shown, ranked within sign. The support denominator is explicit. Change
`discovery.target` to `mean_active_logit` or `calibrated_probability_CONTEXT`
to reuse saved maps; select a distinct target/group output or new work root
for a changed filter. No post-hoc motif reclustering is performed.

Training also writes a source/config fingerprint under `work/provenance/`;
restarts reject a changed checkout/configuration. Use an unchanged source
release while a run is in progress. The export's `RELEASE_MANIFEST.json`
records every shipped file checksum. Only load trusted `.pt` files: these
historical checkpoint formats use PyTorch pickle loading.

## Scope of verification

CPU unit tests exercise the wrapper contracts, model transfer, optimizer
resumption, no-test training, numerical attribution identities, background
matching and source-data checks. The figure is rendered and visually checked.
These are not a full v4 preprocessing/training/attribution/MoDISco rerun. GPU
throughput, memory and final metrics require execution on the target cluster.
Fixed seeds and pinned primary versions improve reproducibility but do not
guarantee bitwise training equality across GPU types and numerical libraries.
