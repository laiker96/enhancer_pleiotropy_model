# Data contract

Peaks select informative windows; BigWig values define regression targets.
The pipeline never converts peak membership directly into a training label.

All input files must use dm6 coordinates. BigWigs must contain finite,
nonnegative background-TMM-normalized mean signal and chromosome lengths that
exactly match the reference FASTA.

The default input filename contracts are:

```text
<context>.atac.mean.background_tmm.bw
<context>.h3k27ac.mean.background_tmm.bw
<context>_h3k27ac_rep<number>_peaks.broadPeak
```

For contexts with multiple H3K27ac peak replicates, an interval is retained
only when it overlaps a peak in every other replicate. Single-replicate
contexts retain that replicate. Supported intervals are merged into one union
used for sampling.

Every retained central 512-bp target is expanded by 768 bp on each side, giving
a 2,048-bp input. Complete peak-overlapping windows and an equal number of
background windows are retained independently in training and validation.
Blacklisted, ambiguous, duplicate, out-of-chromosome, and cross-split windows
are rejected.

The chromosome split is configured explicitly and validated as disjoint. Test
chromosomes do not enter the prepared training or validation arrays.

Each generated table and profile array has JSON metadata with input hashes,
contexts, shape, target geometry, split counts, and summary statistics.
