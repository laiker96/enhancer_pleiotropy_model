# Last paper figure: source-data replay

`source.json` contains the exact plotted values; `style.json` contains vector
logo glyphs/colors; `manifest.json` pins both and the historical PDF. No full
attribution tensor, model checkpoint or hidden external path is needed to
render the figure. Use the main `figure` command and a new PDF output path.
ReportLab 4.4.9 and the same DejaVu fonts reproduce PDF encoding most closely;
the receipt records font/input/output hashes and whether PDF bytes match.

Historical file: `figure_3bcde_native_enhancer_uniform_filter_grh50.pdf`.
PDF SHA256: `5c4b6c3a800bf12eabbaec76c4d875a5b86886a3c4e07386c4162d9f2924c9a3`.

| Panel | Source and interpretation |
|---|---|
| B | Native-enhancer 50-reference motifs, cumulative degree groups, both signs. Uniform 0.5-bit flank rule with four-of-five informative columns, no extra reclustering. Top four positive and all passing negative motifs; support bars count cluster-assigned enhancers. Fresh JASPAR annotation of revised cores. |
| C | Six illustrative high-contrast enhancer examples, including GAF/Trl, Grh and adjacent cg-like/Trl-like sites. Signed actual IG; original native lengths; motif-oriented examples may be reverse complemented. Examples are not an enrichment test. |
| D | Mean motif importance by exact degree 1–8, all FlyFactorSurvey curves grey, GAF highlighted. It is a motif-occurrence attribution summary, not motif frequency. |
| E | Mutant/WT odds ratios in eight contexts, linear scale, with matched-control ratios near 1. Existing GA, CA/GT and double perturbations are not recalculated by rendering. |

The original figure was called B–D; inserting the example panel made the
current layout B–E. Panel A remains omitted. The feature/attribution source is
the older enhancer-only legacy CNN, **not** the corrected background-trained
classifier used by the new reproduction analysis. The new full-context IG100
outputs must not be relabelled as the source of this figure.

## Code for the historical analysis

The preserved `experiments/classifier_modisco/` modules include:

- `original_gpu.py`: original 50-reference mean-active-logit attribution;
  `classifier_motifs/attribution.py`: kernels and adaptive integration.
- `original_intervals.py`: native-length boundaries and TF-MoDISco adapter.
- `simple_report.py`, `dual_motif_pipeline.py`: original cluster support,
  informative PWM core filters, no post-hoc reclustering.
- `tomtom_atlas.py`: database parsing, matching and alignment.
- `paper_figure3.py`, `paper_figure3_nature.py`, `paper_figure3_examples.py`:
  reusable vector figure builders and example panel.
- `paper_figure3_grh50.py`, `paper_figure3_uniform50.py`: the final historical
  Grh example and uniform-filter assembly, with the original provenance gates.

Those historical driver commands retain their original experiment paths and
need the archived intermediate datasets. They are preserved as provenance,
not presented as portable from-scratch drivers. The new `reproduction` path
is portable for training, context attribution and native motif discovery.
This figure route deliberately separates **re-rendering published source
data** from **rerunning the complete historical statistical analysis**.

To update the plotted scientific results after a new model run, rebuild its
motif summaries, choose examples transparently, recompute motif-importance
summaries and rerun the mutation/control inference. Do not just replace a
checkpoint name in this frozen JSON. That is a new analysis, not exact replay.

`scripts/export_figure3_inputs.py` documents how these compact inputs were
exported from the full original source JSON and filter audit; it refuses to
overwrite an existing bundle. The CPU replay was visually checked against the
original and was pixel-identical at a 1,700-pixel render height.
