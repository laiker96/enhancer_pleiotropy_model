# Browser-track validation

This analysis compares the observed and model-predicted BigWigs produced by
`enhancer-browser-tracks`. It operates on the complete genome-anchored sliding
grid and does not use the balanced training-table subset.

## Reproducible target

From the repository root, run:

```bash
XDG_CACHE_HOME="$PWD/.cache" .venv/bin/snakemake \
  --configfile config/default.yaml \
  --cores 4 \
  --rerun-incomplete \
  browser_validation_report \
  --resources gpu=1
```

The `browser_tracks` rule requires one CUDA GPU. The downstream report is CPU
only. If the 32 source tracks already exist and match the configured target,
Snakemake runs only the report. Parameters are under `browser_tracks` and
`browser_report` in `config/default.yaml`.

For the final split/checkpoint, substitute `config/final_4x.yaml`. Never use the
chr3R test split to choose thresholds, checkpoints, or report parameters.

## Definitions

All quantitative comparisons use aligned native output bins and

\[
z = \log(1 + \max(x, 0)).
\]

ATAC has 16-bp bins and H3K27ac has 64-bp bins. Reported per-context metrics
are Pearson, Spearman, RMSE, MAE, R2, signed bias, and the ratio of predicted to
observed standard deviations.

The tissue-pattern Pearson is the Pearson correlation across the eight
contexts *within one genomic bin*. The informative-bin summary uses bins above
the configured activity quantile and, among those bins, above the configured
context-variability quantile. These thresholds are derived only from observed
validation signal and are recorded in `metrics.json`.

Over-correlation is

\[
\frac{1}{28}\sum_{i<j}
\left[\operatorname{corr}(\hat y_i,\hat y_j)
-\operatorname{corr}(y_i,y_j)\right].
\]

A positive value means predictions are, on average, more correlated between
contexts than the observed tracks. Target PCA is fit to observed validation
bins; predictions are projected into that same basis.

Regulatory strata are half-open interval overlaps with the configured master
DHS and consensus H3K27ac peak BEDs. The four mutually exclusive strata are
`DHS_and_H3K27ac_peak`, `DHS_only`, `H3K27ac_peak_only`, and `background`.

Residual BigWigs store

\[
\log(1 + \hat y) - \log(1 + y).
\]

Thus positive residuals are over-predictions and negative residuals are
under-predictions.

## Outputs

The configured report directory contains:

- `index.html`: self-contained quantitative report;
- `metrics.json`: complete metrics and definitions;
- `analysis_config.json`: all parameters, Git commit, package versions, and
  SHA-256 hashes for every quantitative input;
- `per_context_metrics.tsv` and `stratified_metrics.tsv`;
- `representative_loci.tsv` and an IGV-compatible bookmark BED;
- `residuals/`: 16 signed residual BigWigs;
- `igv_session_with_residuals.xml`: observed, predicted, and residual tracks.

Representative loci are deterministic active/variable blocks ranked by their
across-context agreement. They are qualitative bookmarks, not independent
statistical observations.

## Interpretation limits

The current default checkpoint was selected using chr2L validation, so this
report is development/QC evidence rather than an untouched generalization
estimate. Sliding-window contributions overlap, so native bins are not
statistically independent. Correlation measures agreement, not calibration or
causality.
