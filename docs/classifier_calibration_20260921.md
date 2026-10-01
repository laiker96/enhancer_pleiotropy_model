# Calibrated all-context breadth: two population comparison

2026-09-21. User asked to calibrate probabilities and identify the best
enhancer/background classifier, then explicitly requested comparing both
known-enhancer and enhancer-plus-background calibration populations first.

## Classifier choice

Ranked the 45 completed background-trained classifier fits in the 94-model
20260919 inventory snapshot by combined **validation** macro average precision
at their already selected checkpoints. Best: `background__cnn_finetune_20260914`,
2048-bp dilated CNN, corrected retained regression-hidden-layer readout,
regressor-pretrained, 25% background training exposure, selected epoch38 of40.
Each original checkpoint was selected by enhancer-only validation AP, not by
the combined metric. No checkpoint was selected from test performance.

SHA256: `7dc8eb721fbd4f292eba95ff2ea44399cfd1aae5a596e8d56e6d7ff25a95628a`.
Checkpoint relative to the historical artifact root:
`experiments/classifier_background_20260916/runs/cnn_finetune_20260914/best_model.pt`.
This is NOT the enhancer-only legacy CNN used for existing motif attributions.

| Evaluation population | Validation AP | Previously logged test AP |
|---|---:|---:|
| Active vs other enhancers | 0.648744 | 0.637113 |
| Active vs paired background | 0.933890 | 0.940572 |
| Active vs other enhancers + full background pool | 0.586317 | 0.582833 |

AP means average precision, not trapezoidal PR area. The active/background
comparison is balanced within each context; the combined comparison includes
the full background pool and has lower prevalence. Values across populations
are not interchangeable. Test chromosome chr3R was inspected historically.

Dense-CNN finetune seed20260916 is nearly tied on combined validation AP:
0.585455 versus0.586317. Dense has higher active/background AP0.943575 but lower
enhancer-only AP0.631894. Three-seed combined validation means: dilated CNN
0.585163, dense0.584155; no meaningful superiority significance is claimed.

## Calibration contract

Local CPU only; no GPU, inference, neural-network training, remote mutation,
motif discovery, new IG, threshold tuning, or model-inventory edits.
Reuse `results/cnn_length_pr_20260918/evaluation/predictions.npz` and its locked
checkpoint/collection provenance. The code explicitly loads validation arrays
only. Full-file checksums include the archive containing test predictions,
but test entries are not used for calibration fitting or method selection.

Validation cohort: 4,062 known enhancers and 4,062 fixed paired backgrounds.
Compare calibrators fitted on:

1. Known enhancers only.
2. Exactly 1:1 known enhancers and matched genomic backgrounds.

For both, evaluate on enhancers, backgrounds, and the combined population.
Both scalar functions include ALL eight outputs, with NO observed-active mask.
Population choice changes calibration parameters, not which output contexts
are included for an individual sequence.

Methods, independently per context:

- Raw: identity.
- Temperature: sigmoid(a * score), positive a.
- Sigmoid/Platt-type: sigmoid(a * score + b), positive a.

`score = logit(clip(mean(forward_probability, RC_probability), 1e-7, 1-1e-7))`.
Calibration is AFTER orientation probability averaging, not applied to mean
orientation logits or separately per orientation. This matches saved inference.
New scalar: sum of the eight calibrated probabilities. No hard or soft
threshold-based predicted-label count was substituted.

Fit maximum likelihood with SciPy L-BFGS-B, analytic gradients, a>=1e-6,
unbounded intercept, max1000iterations, ftol1e-12, gtol1e-8. All optimizers
converged; final slopes did not hit their numerical lower bound. No class
reweighting, invented background prevalence, or tuning to motif results.
References: [probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
and [temperature scaling](https://proceedings.mlr.press/v70/guo17a.html).

## Cross-fitting and leakage controls

Five approximately equal contiguous enhancer-coordinate folds on chr2L
validation; each background stays with its paired enhancer. For each fold,
exclude any calibration-training pair whose enhancer OR background 2048-bp
window overlaps ANY held-out enhancer/background window. Half-open genomic
coordinates and chromosome identities are respected.

Held-out pairs:813/813/812/812/812. Training pairs AFTER purging:
1847/1760/1763/1832/1752. Each pair is evaluated exactly once. The two population
conditions use identical held-out folds and purged pairs. The original model
is never refitted. Calibration method selected by lowest aggregate-context
OOF log loss within the corresponding fitting population, then that method
refitted on all validation members of that population for a future pilot.
Final-fit coefficients are not evaluated on their own fitting data here.

These are validation-development comparisons, not an untouched final test:
the validation cohort previously selected the checkpoint/model, and the
calibration method is now also chosen using it. Cross-fitting does not erase
that selection history. No statistical significance or final-test gain claimed.

## Results

Sigmoid calibration selected for BOTH fitting populations. ECE is macro
absolute calibration error across ten equal-width probability bins; lower is
better, but bin-sensitive. Log loss and Brier complement reliability diagrams
and are not pure calibration-only quantities.

| Calibration/evaluation population | Raw log loss | Calibrated | Raw ECE | Calibrated | Raw Brier | Calibrated |
|---|---:|---:|---:|---:|---:|---:|
| Known enhancers | .37905 | .37301 | .03653 | .01290 | .12188 | .12053 |
| 1:1 enhancer/background | .22244 | .22102 | .01746 | .00628 | .06858 | .06789 |

Known-enhancer scalar mean moves1.72281->1.91233 versus observed1.90030;
bias-.17749->+.01203, RMSE1.14449->1.12322, R2 .38763->.41019.
MAE does NOT improve: .79790->.80405. This is not a claim that calibration
improves every prediction metric or improves motif accuracy.

Mixed-population scalar mean1.07542->.95359 versus observed.95015;
bias+.12527->+.00345, MAE.61297->.59026. However the mixed calibrator gives
mean1.53463 among known enhancers (observed1.90030) and.37255 among zero-labelled
backgrounds. Near-zero overall bias can coexist with opposing subgroup biases.

Applying the enhancer-only calibrator to the mixture worsens mixture log loss
.22244->.23210, and background predicted breadth .42804->.59810. Therefore
neither population's calibration should be presented as universally valid.
Use enhancer-only calibration for the known-enhancer motif question; retain
the mixed calibration for that explicit enhancer/background evaluation design.
Neither yields genome-wide activity probabilities, and background labels are
catalogue negatives, not experimentally proven inactivity in every context.

## Files, reproduction and verification

Results: `results/classifier_calibration_20260921/`:
`report.md`, `metrics.json` (all populations/methods/context reliability bins,
per-observed-breadth summaries and fold coefficients), `calibrators.json`
(two frozen eight-context calibration sets), `validation_oof.npz`,
`complete.json` (source/output hashes), `verification.json`.

The final calibration coefficients require the EXACT checkpoint hash and
context order ab/e13/e5/ead/hid/lb/o/wid. Existing IG maps belong to another
model and cannot be transformed into these new maps. FP32 real-model CUDA
endpoint replay, the differentiable IG wrapper, new attribution convergence,
and independent test calibration performance remain untested. The calibration
cache used the original float16-autocast inference; check endpoint agreement
in a bounded FP32 pilot before any full new attribution run.

From the project root (choose a fresh output to repeat):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=experiments .venv/bin/python -m unittest discover -s tests -p test_calibrate_breadth.py -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=experiments .venv/bin/python -m classifier_transfer.calibrate_breadth \
  --bundle-root results/cnn_length_pr_20260918/evaluation \
  --data-root results/classifier_background_training_20260916/data \
  --inventory results/model_inventory_20260919/source_snapshot.json \
  --output FRESH_OUTPUT_DIRECTORY
CUDA_VISIBLE_DEVICES='' PYTHONPATH=experiments .venv/bin/python scripts/check_breadth_calibration.py \
  --root results/classifier_calibration_20260921 \
  --data-root results/classifier_background_training_20260916/data
```

Checker refuses an existing verification receipt. Seven focused CPU tests
cover calibration recovery, positive-slope monotonicity, invalid inputs,
boundary/tied scores, label-free application, pairing and overlap purging.
Independent real-data verification replays all30 fold calibrators, recomputes
18 metric sets, checks hashes and brute-force verifies no training/held-out
window overlap. NumPy2.2.6/SciPy1.15.2; no new dependencies installed.
All seven tests, all30 fold replays, all18 metric-set recomputations, source/
output hashes and brute-force overlap checks passed. Python compilation and
whitespace checks also passed. Calibration of real-model CUDA attributions and
final-test performance were not tested.

Completed tmux session `calibration_breadth_20260921`; command above ran through
`tee logs/calibration_breadth_20260921.log`, CUDA hidden, project root working
directory. Monitor with `tail -f logs/calibration_breadth_20260921.log`.
While running, stop this local task only with
`tmux send-keys -t calibration_breadth_20260921 C-c`. No Slurm job was submitted.

## Distribution figure: exact experimental degrees

User requested this plot on2026-09-21. Generated
`output/pdf/degree_pleiotropy_calibrated_breadth_validation.pdf` plus PNG preview
and `.source.json` with per-degree counts/quantiles/means, input/output hashes,
script hash and rendering versions. Source is the saved
`validation_oof.npz` field `enhancers_only__sigmoid`, selecting only rows with
population `enhancer`. All4,062 validation enhancers; no background, test or
in-sample final-fit probabilities included. Experimental degree is the sum of
eight observed binary labels; predicted breadth sums all eight calibrated
probabilities, without an active-label mask. Groups are EXACT1..8, not cumulative.

Uniform-width Scott-bandwidth violins show the full observed range; width is
within-group density, not group size. White boxes show median/IQR, whiskers
extend to1.5IQR; all extremes remain in the violins. Orange diamonds/line show
means; grey dashed line marks predicted=experimental. Sample counts are shown
under each degree. Correlations reproduced: Pearson.645422, Spearman.543855.
All counts and means match the locked metrics to1e-12; all input hashes and
array validity checks passed. PDF is one page, rendered with Poppler and
visually verified (no clipping/overlap); text extraction and compilation passed.

Reproduce from the project root, choosing a fresh prefix to avoid overwrites:

```bash
CUDA_VISIBLE_DEVICES='' ../.venv/bin/python scripts/plot_calibrated_breadth.py \
  --source results/classifier_calibration_20260921 \
  --output output/pdf/NEW_PREFIX
```

Uses the existing parent plotting environment (Matplotlib3.11.1); no packages
installed, model inference, GPU work, refitting or remote job changes. This
figure remains validation-development evidence, not an independent test result.

## Binary breadth with context-specific F1 thresholds

Requested on 2026-09-21 as the binary counterpart to the continuous figure.
New outputs are in `results/classifier_binary_breadth_20260921/`; the original
calibration outputs are unchanged. The figure is
`output/pdf/degree_pleiotropy_binary_calibrated_f1_validation.pdf`, with PNG
and `.source.json` provenance. Implementation:
`experiments/classifier_transfer/threshold_breadth.py` and
`scripts/plot_binary_breadth.py`.

Use the same 4,062 known validation enhancers, context order
`ab,e13,e5,ead,hid,lb,o,wid`, and enhancer-only sigmoid calibration. A context
is active when its calibrated probability is **greater than or equal to**
its own threshold. Predicted degree counts those eight calls, without an
observed-label mask or a minimum-one-context correction. Zero is allowed.

Threshold criterion: maximize per-context F1 over every distinct fitting
score; equal-score examples cannot be split; tied F1 optima choose the higher
threshold. This tunes classification decisions, not probability calibration,
ranking, breadth error or background specificity. See the
[decision-threshold documentation](https://scikit-learn.org/stable/modules/classification_threshold.html).

Evaluation reuses the five spatial folds and original paired-window overlap
purge. For each fold, use its saved calibrator on that fold's fitting rows to
select thresholds, then apply them to its held-out calibrated predictions.
Both calibration fitting and threshold tuning exclude the evaluated enhancer
and overlapping fitting DNA. Crucially, we do NOT tune on pooled OOF labels
and evaluate those thresholds on the same labels. Threshold fitting and
calibration can share the fitting set; the outer held-out fold is untouched.
Model/checkpoint/calibration-method selection previously used validation, so
these remain development results, not independent final-test estimates.

| Prediction | Macro context F1 | Pearson | Spearman | Breadth MAE | Breadth RMSE | Breadth R2 | Mean predicted degree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuous calibrated sum | Not applicable | 0.645422 | 0.543855 | 0.804051 | 1.123216 | 0.410186 | 1.912325 |
| Count at calibrated probability >= 0.5 | 0.571551 | 0.569628 | 0.434736 | 0.987445 | 1.453108 | 0.012847 | 1.536189 |
| Count at cross-fitted maximum-F1 thresholds | 0.625652 | 0.575972 | 0.485357 | 1.146480 | 1.668709 | -0.301818 | 2.514032 |

Observed mean degree is 1.900295. F1 thresholding improves context-level F1
but overcalls breadth overall. Exact breadth agreement is 32.79%, within-one
agreement 72.38%, and 307 enhancers have zero called contexts. The mean for
observed degree eight rises from 5.3812 (continuous) to 7.2292 (binary), but
degree-one mean rises from 1.4983 to 1.8590. Better high-degree means do not
imply better overall breadth prediction. Continuous sums remain more accurate
on this cohort; do not replace the IG scalar with a nondifferentiable count.

For subsequent independent evaluation, `thresholds.json` freezes a separate
full-validation-fit threshold vector to use with the exact saved final
enhancer-only calibrator and checkpoint. Those full-fit thresholds are NOT
the OOF thresholds used in this plot:

| Context | Full-validation-fit threshold |
| --- | ---: |
| ab | 0.2732641633 |
| e13 | 0.2884207605 |
| e5 | 0.2709019656 |
| ead | 0.3094380245 |
| hid | 0.3464340580 |
| lb | 0.2818491594 |
| o | 0.3286168541 |
| wid | 0.3723942029 |

Blue mirrored histograms show integer counts without KDE smoothing: widths
are proportional to frequency within each observed degree and normalized to
the same maximum width per group. White boxes show median/IQR, whiskers
1.5 IQR; orange diamonds mark means, dashed grey line marks identity. All
observations including outliers remain represented in the histograms.

Reproduce from the project root; choose fresh output directories/prefixes:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=experiments ../.venv/bin/python -m unittest discover -s tests -p test_threshold_breadth.py -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=experiments ../.venv/bin/python -m classifier_transfer.threshold_breadth \
  --source results/classifier_calibration_20260921 \
  --data-root results/classifier_background_training_20260916/data \
  --output FRESH_ANALYSIS_DIRECTORY
CUDA_VISIBLE_DEVICES='' ../.venv/bin/python scripts/plot_binary_breadth.py \
  --source FRESH_ANALYSIS_DIRECTORY \
  --output output/pdf/FRESH_FIGURE_PREFIX
```

Verification: six threshold unit tests and seven existing calibration tests
passed. All 48 fitted thresholds were independently checked by exhaustive
candidate evaluation; held-out calibration replay, disjoint DNA windows,
exact-once coverage, baseline metric parity, hashes and figure distributions
passed. One-page PDF rendered with Poppler and visually checked; compilation
passed. Existing NumPy/SciPy/Matplotlib environment; no dependency changes,
CUDA, new inference, training, test evaluation or remote-job changes.
