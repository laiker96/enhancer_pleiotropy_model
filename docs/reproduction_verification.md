# Reproduction verification

## Source publication and README figure, 2026-10-01

The publication includes the approved compact architecture figure for
`background__cnn_finetune_20260914` in PNG, SVG and PDF formats, with its
original provenance record. Its checkpoint hash and 1,244,168 parameters
match the benchmark snapshot; its copied PDF hash matches the original.
The README embeds the PNG and links the vector versions. Visual inspection
of a fresh PDF render found no clipped/overlapping labels. Dashed signal
branches are explicitly pretraining-only, not classifier signal outputs.

The source allowlist also includes the AlphaGenome third-party license and
the existing core model/preprocessing tests. It excludes full research
datasets, checkpoints, logs, runtime environments, personal site settings
and unrelated dated launch scripts.

**76 CPU tests passed**, locally and from a fresh source-only export, with
CUDA disabled. This comprises the three reproduction/calibration/benchmark
suites below plus `test_browser_report.py`, `test_browser_tracks.py`,
`test_data.py`, `test_metrics.py`, `test_model.py`, `test_preprocessing.py`,
`test_sequence.py` and `test_training.py`. All 216 exported source members
were verified against their SHA256 manifest. The runtime was the existing
local environment, not a clean dependency installation. A targeted scan
found no private keys or obvious embedded credential values in the allowlist;
this is not a comprehensive security audit.

Local test session: `paper_release_checks_20261001`, repository root, with
output at `logs/paper_release_checks_20261001.log`. It completed. During a
future run, monitor that log with `tail -f`; stop only the corresponding tmux
session if needed. No training, CUDA inference or remote scientific jobs
were started or changed for publication. The public data deposit and fresh
end-to-end GPU validation remain outstanding.

## Model recipes and paper benchmarks, 2026-10-01

Prepared locally, with no new training, CUDA inference, remote job changes,
data upload, Git commit or push.

- Added `best`, `architectures`, `readouts` and `full` matrix selections to
  the existing configuration/CLI/Slurm launcher. Separate work directories,
  unchanged scientific settings and selection recorded in job provenance.
- Added a machine-independent metrics/checkpoint snapshot for all 94 completed
  transfer-comparison models (7 regressors, 87 classifiers). Source inventory
  SHA256: `a40197312967b1122bcedaa2971508b5ddf7feb87da64de5acd38046fad67ef5`.
  The historical inventory is preserved; private absolute paths are replaced
  by artifact-root-relative identifiers in the public snapshot.
- Generated [paper benchmark tables](paper_benchmarks.md): 29 three-seed
  classifier conditions, matched-population comparisons, breadth/context
  diagnostics, per-context AP, regressor metrics, learning rates and all
  selected checkpoint paths/hashes. Representative seeds are validation-only
  selections; seed mean ± sample SD is distinct from single-checkpoint scores.
- Expanded future-run inventory refresh with selected-epoch background and
  per-context metrics, regressor correlations/R², selected/current LR and
  parent checkpoint provenance. No test inference occurs during a refresh.

Checks actually run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src:experiments:scripts:tmp/finemo_cpu_deps \
  .venv/bin/python -m pytest tests/test_reproduction.py \
  tests/test_calibrated_attribution.py tests/test_paper_benchmarks.py \
  -o addopts='' -p no:cacheprovider -q
```

**42 passed** locally, and **42 passed** from a newly extracted source-only
export without the research data/checkpoints/results directory. All 203
exported source members passed their SHA256 manifest checks. The clean-export
tests used the existing local Python runtime (including the existing h5py
dependency directory), not a newly installed environment. Source code came
from the export. The benchmark report regenerated exactly from its shipped
JSON. Every published source metric was checked against the original TSV
where available locally. Slurm dry-run dependencies/recipe propagation,
configuration counts, shell syntax and `git diff --check` passed.

This verifies CPU contracts and packaging, **not a new end-to-end training
reproduction**. Before claiming an externally self-contained paper release:

1. Preserve the published source commit used by each experiment. The remote `main` inspected
   on 2026-10-01 was still `42c17b32fb1d8a4ee7a233fb22af7e8444f1dd06`
   (2026-09-01), before the subsequently authorized source publication.
2. Deposit the exact v4 processed-data export with a stable URL/checksum;
   deposit selected pretrained checkpoints if offering inference without
   retraining. The current input path is a user-supplied archive, not a
   downloader. This workflow does not rebuild FASTQ alignments/normalization.
3. Validate a clean dependency installation and a full target-cluster
   preprocessing/training/calibration/IG pilot. Full data/GPU stages were
   not executed during this update; exact stochastic checkpoint equality
   across GPUs is not promised.
4. If reproducing the newest manuscript Figure 3, connect the September
   29–30 masked-context/family analysis. The portable figure replay currently
   ships the older September 20 source-data fixture only.

The portable training scope is the 2,048-bp dilated CNN, attention and
flatten/dense CNN. The table includes historical EnhancerNet and shorter-input
results, but the generic launcher does not yet implement those extra suites.

## Original cleanup, 2026-09-21

This was a local code/documentation cleanup, not a new scientific training run.
Existing data, checkpoints, historical launchers and remote jobs were preserved.
No Git commit/push, external transfer or Slurm submission was performed.

## What changed

- Portable stage CLI/configuration and generic dry-run-first Slurm orchestration.
- v4 staging, regression preprocessing, classifier/background rebuilding,
  three regressor architectures and the scratch/transfer/readout matrix.
- Validation checkpoint locks, test-population evaluation, deployment calibration,
  and a separate compact model-inventory refresh.
- Gated/resumable full-context IG, native-length signed TF-MoDISco filtering,
  Tomtom annotations and HTML reporting, reusing existing numerical code.
- Checksummed source-data replay of the final Figure 3 B–E; old README archived.
- An allowlisted source exporter, with no raw data, checkpoints or site settings.

One backwards-compatible change was made to the existing attribution module:
`CalibratedTargets` can accept an explicit expected checkpoint SHA for a new
fit. Its historical default remains pinned and the numerical implementation
is unchanged. Historical running jobs use their frozen packages and are not
affected by this checkout change.

## Checks actually run

1. **95 passed, 7 skipped** in the broader CPU regression suite (75.92 s).
   Six skips require the pinned TF-MoDISco native runtime; one needs ReportLab
   in the training test interpreter. Independently, the figure renderer was
   exercised using the existing reporting interpreter and ReportLab 4.4.9.
   Covered model transfer for legacy/retained/dense readouts, CPU training and
   exact restart on tiny synthetic data, background sampling, no-test training,
   IG linearity/reference resumption, native boundaries, motif filters and the
   portable discovery wrapper with synthetic HDF5 and a mocked clustering fit.

   Command, from the project root (the final path supplies already existing
   local h5py/Numba dependencies, not alternative scientific source):

   ```bash
   CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:experiments:scripts:tmp/finemo_cpu_deps \
     .venv/bin/python -m pytest tests/test_reproduction.py \
     tests/test_calibrated_attribution.py tests/test_retained_heads.py \
     tests/test_classifier_transfer.py tests/test_dense_transfer.py \
     tests/test_background_training.py tests/test_modisco_original.py \
     tests/test_dual_motif_pipeline.py tests/test_modisco_simple_report.py \
     -o addopts='' -ra
   ```

   This ran in tmux `reproduce_cleanup_checks`, project root, log
   `logs/reproduction_cleanup_checks.log`. It completed; no test process is
   intentionally left running. For an active test session, monitor with
   `tail -f logs/reproduction_cleanup_checks.log`; stop only that session with
   `tmux kill-session -t reproduce_cleanup_checks` if it is still present.

2. **Clean export: 26 passed** in the two standalone shipped suites, extracted
   into a new `/tmp` directory without the workspace's data/results/cluster
   scripts. Verified every source checksum in `RELEASE_MANIFEST.json`.

3. **Historical best classifier compatibility:** checked SHA
   `7dc8eb721fbd4f292eba95ff2ea44399cfd1aae5a596e8d56e6d7ff25a95628a`, strict
   state-dict load in the portable loader, 1,244,168 parameters, finite [1,8]
   output on a synthetic 2,048-bp CPU input. CUDA was not initialized.

4. **Figure replay:** text-bound validation and visual PNG inspection; local
   ReportLab 4.4.4 produced pixel-identical 605×1700 renders. Using the existing
   ReportLab 4.4.9 wheel, both the workspace and clean-export replays reproduce
   the exact historical PDF SHA:
   `5c4b6c3a800bf12eabbaec76c4d875a5b86886a3c4e07386c4162d9f2924c9a3`.
   The PDF skill's render-and-inspect workflow was used for this check.

5. Python compilation, Bash syntax checks, Slurm dry-run command/dependency
   inspection, configuration parity against the archived parent regressor,
   and `git diff --check` passed. Source export is about 0.7 MB compressed.

## Not verified by this cleanup

The entire large v4 dataset was not rebuilt, no full model was retrained, no
GPU pilot/full attribution was launched, and native TF-MoDISco clustering or
real Tomtom database searches were not rerun. The new GPU/motif adapters need
their real target-cluster execution checks before calling the workflow fully
end-to-end reproduced. A clean dependency installation was not performed;
primary-version requirements and installation instructions are supplied.
Historical metric values are reference results, not new measurements.

Before external publication, make the checksummed v4 export (and optionally
selected checkpoints/motif DB provenance) available via a stable data deposit.
The input archive currently has no automatic public download location here.
The figure replay is independently usable from its shipped source values.
