"""Reproducible data preparation and default 4x training workflow."""

configfile: "config/default.yaml"

import json
from pathlib import Path


OUTPUT = config["output_directory"]
PYTHON = ".venv/bin/python"
CONTEXTS = config["contexts"]
BIGWIG_DIRECTORY = config["inputs"]["bigwig_directory"]
BROWSER_CONFIG = config["browser_tracks"]
REPORT_CONFIG = config["browser_report"]
SPECIFICITY_CONFIG = config.get("specificity_finetuning", {})
BROWSER_DIRECTORY = f"{OUTPUT}/{BROWSER_CONFIG['subdirectory']}"
REPORT_DIRECTORY = f"{OUTPUT}/{REPORT_CONFIG['subdirectory']}"
BROWSER_TRACKS = [
    f"{BROWSER_DIRECTORY}/{source}.{context}.{assay}.bw"
    for source in ("observed", "predicted")
    for assay in ("atac", "h3k27ac")
    for context in CONTEXTS
]
RESIDUAL_TRACKS = [
    f"{REPORT_DIRECTORY}/residuals/residual.{context}.{assay}.bw"
    for assay in ("atac", "h3k27ac")
    for context in CONTEXTS
]


def bigwigs(assay):
    return [
        f"{BIGWIG_DIRECTORY}/{context}.{assay}.mean.background_tmm.bw"
        for context in CONTEXTS
    ]


rule all:
    input:
        f"{OUTPUT}/model/best_model.pt",
        f"{OUTPUT}/model/metrics.json",
        *(
            [
                f"{OUTPUT}/{SPECIFICITY_CONFIG['model_subdirectory']}/best_model.pt",
                f"{OUTPUT}/{SPECIFICITY_CONFIG['model_subdirectory']}/metrics.json",
            ]
            if SPECIFICITY_CONFIG.get("enabled", False)
            else []
        ),


rule prepared_data:
    input:
        f"{OUTPUT}/data/windows.tsv.gz",
        f"{OUTPUT}/data/windows.metadata.json",
        f"{OUTPUT}/data/profiles/atac/train_profiles.npy",
        f"{OUTPUT}/data/profiles/atac/validation_profiles.npy",
        f"{OUTPUT}/data/profiles/atac/test_profiles.npy",
        f"{OUTPUT}/data/profiles/h3k27ac/train_profiles.npy",
        f"{OUTPUT}/data/profiles/h3k27ac/validation_profiles.npy",
        f"{OUTPUT}/data/profiles/h3k27ac/test_profiles.npy",


rule browser_validation_report:
    input:
        f"{REPORT_DIRECTORY}/index.html",
        f"{REPORT_DIRECTORY}/metrics.json",
        f"{REPORT_DIRECTORY}/igv_session_with_residuals.xml",


rule resolved_config:
    output:
        f"{OUTPUT}/run_config.yaml",
    params:
        payload=json.dumps(config, indent=2, sort_keys=True) + "\n",
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(output[0]).write_text(params.payload, encoding="utf-8")


rule h3k27ac_consensus:
    input:
        peak_directory=config["inputs"]["h3k27ac_peak_directory"],
    output:
        bed=f"{OUTPUT}/data/h3k27ac_consensus_union.bed",
        metadata=f"{OUTPUT}/data/h3k27ac_consensus_union.metadata.json",
    params:
        contexts=" ".join(CONTEXTS),
    log:
        "logs/h3k27ac_consensus.log",
    shell:
        """
        {PYTHON} -m enhancer_pleiotropy_model.preprocessing.h3_peaks \
          --peak-directory {input.peak_directory:q} \
          --output {output.bed:q} \
          --metadata {output.metadata:q} \
          --contexts {params.contexts} 2>&1 | tee {log:q}
        """


rule windows:
    input:
        reference=config["inputs"]["reference_fasta"],
        blacklist=config["inputs"]["blacklist_bed"],
        dhs=config["inputs"]["master_dhs_bed"],
        summits=config["inputs"]["master_dhs_summits_bed"],
        h3=f"{OUTPUT}/data/h3k27ac_consensus_union.bed",
    output:
        table=f"{OUTPUT}/data/windows.tsv.gz",
        metadata=f"{OUTPUT}/data/windows.metadata.json",
    params:
        bigwig_directory=BIGWIG_DIRECTORY,
        contexts=" ".join(CONTEXTS),
        train=" ".join(config["chromosome_splits"]["train"]),
        validation=" ".join(config["chromosome_splits"]["validation"]),
        test=" ".join(config["chromosome_splits"]["test"]),
        train_regions=" ".join(config.get("region_splits", {}).get("train", [])),
        validation_regions=" ".join(config.get("region_splits", {}).get("validation", [])),
        test_regions=" ".join(config.get("region_splits", {}).get("test", [])),
        split_strategy=config.get("split_strategy", "chromosome"),
        target=config["sampling"]["central_target_bp"],
        flank=config["sampling"]["context_flank_bp"],
        train_stride=config["sampling"]["training_stride_bp"],
        validation_stride=config["sampling"]["validation_stride_bp"],
        block=config["sampling"]["block_size_bp"],
        background_ratio=config["sampling"]["background_to_peak_ratio"],
        seed=config["seed"],
    log:
        "logs/windows.log",
    shell:
        """
        {PYTHON} -m enhancer_pleiotropy_model.preprocessing.windows \
          --reference-fasta {input.reference:q} \
          --blacklist-bed {input.blacklist:q} \
          --master-dhs-bed {input.dhs:q} \
          --master-dhs-summits-bed {input.summits:q} \
          --h3k27ac-peaks-bed {input.h3:q} \
          --bigwig-directory {params.bigwig_directory:q} \
          --signal-assays atac h3k27ac \
          --omit-signal-summaries \
          --window-size {params.target} \
          --context-flank-size {params.flank} \
          --stride {params.train_stride} \
          --validation-stride {params.validation_stride} \
          --split-strategy {params.split_strategy} \
          --block-size {params.block} \
          --background-to-peak-ratio {params.background_ratio} \
          --seed {params.seed} \
          --train-chromosomes {params.train} \
          --validation-chromosomes {params.validation} \
          --test-chromosomes {params.test} \
          --train-regions {params.train_regions} \
          --validation-regions {params.validation_regions} \
          --test-regions {params.test_regions} \
          --output {output.table:q} \
          --metadata {output.metadata:q} 2>&1 | tee {log:q}
        """


rule profiles:
    input:
        dataset=f"{OUTPUT}/data/windows.tsv.gz",
        bigwigs=lambda wildcards: bigwigs(wildcards.assay),
    output:
        train=f"{OUTPUT}/data/profiles/{{assay}}/train_profiles.npy",
        validation=f"{OUTPUT}/data/profiles/{{assay}}/validation_profiles.npy",
        test=f"{OUTPUT}/data/profiles/{{assay}}/test_profiles.npy",
        metadata=f"{OUTPUT}/data/profiles/{{assay}}/profiles.metadata.json",
    params:
        bigwig_directory=BIGWIG_DIRECTORY,
        output_directory=lambda wildcards: f"{OUTPUT}/data/profiles/{wildcards.assay}",
        contexts=" ".join(CONTEXTS),
        target=lambda wildcards: (
            config["profiles"]["atac_target_bp"]
            if wildcards.assay == "atac"
            else config["profiles"]["h3k27ac_target_bp"]
        ),
        bin_size=lambda wildcards: config["profiles"]["source_bin_bp"],
    wildcard_constraints:
        assay="atac|h3k27ac",
    log:
        "logs/profiles_{assay}.log",
    shell:
        """
        {PYTHON} -m enhancer_pleiotropy_model.preprocessing.profiles \
          --dataset {input.dataset:q} \
          --bigwig-directory {params.bigwig_directory:q} \
          --output-directory {params.output_directory:q} \
          --assay {wildcards.assay} \
          --target-size {params.target} \
          --bin-size {params.bin_size} \
          --contexts {params.contexts} 2>&1 | tee {log:q}
        """


rule train:
    input:
        config=f"{OUTPUT}/run_config.yaml",
        dataset=f"{OUTPUT}/data/windows.tsv.gz",
        atac_train=f"{OUTPUT}/data/profiles/atac/train_profiles.npy",
        atac_validation=f"{OUTPUT}/data/profiles/atac/validation_profiles.npy",
        atac_test=f"{OUTPUT}/data/profiles/atac/test_profiles.npy",
        atac_metadata=f"{OUTPUT}/data/profiles/atac/profiles.metadata.json",
        h3_train=f"{OUTPUT}/data/profiles/h3k27ac/train_profiles.npy",
        h3_validation=f"{OUTPUT}/data/profiles/h3k27ac/validation_profiles.npy",
        h3_test=f"{OUTPUT}/data/profiles/h3k27ac/test_profiles.npy",
        h3_metadata=f"{OUTPUT}/data/profiles/h3k27ac/profiles.metadata.json",
    output:
        best=f"{OUTPUT}/model/best_model.pt",
        metrics=f"{OUTPUT}/model/metrics.json",
    log:
        f"logs/train_4x.log",
    threads: 4
    resources:
        gpu=1
    shell:
        """
        {PYTHON} -m enhancer_pleiotropy_model.training \
          --config {input.config:q} --stage base --resume 2>&1 | tee {log:q}
        """


rule finetune_specificity:
    input:
        config=f"{OUTPUT}/run_config.yaml",
        base=f"{OUTPUT}/model/best_model.pt",
        base_metrics=f"{OUTPUT}/model/metrics.json",
        dataset=f"{OUTPUT}/data/windows.tsv.gz",
        atac_train=f"{OUTPUT}/data/profiles/atac/train_profiles.npy",
        atac_validation=f"{OUTPUT}/data/profiles/atac/validation_profiles.npy",
        h3_train=f"{OUTPUT}/data/profiles/h3k27ac/train_profiles.npy",
        h3_validation=f"{OUTPUT}/data/profiles/h3k27ac/validation_profiles.npy",
    output:
        best=f"{OUTPUT}/{SPECIFICITY_CONFIG.get('model_subdirectory', 'model_specific_finetune')}/best_model.pt",
        metrics=f"{OUTPUT}/{SPECIFICITY_CONFIG.get('model_subdirectory', 'model_specific_finetune')}/metrics.json",
    log:
        "logs/train_4x_specificity_finetune.log",
    threads: 4
    resources:
        gpu=1
    shell:
        """
        {PYTHON} -m enhancer_pleiotropy_model.training \
          --config {input.config:q} --stage specificity --resume 2>&1 | tee {log:q}
        """


rule browser_tracks:
    input:
        checkpoint=f"{OUTPUT}/model/best_model.pt",
        reference=config["inputs"]["reference_fasta"],
        blacklist=config["inputs"]["blacklist_bed"],
        observed=lambda wildcards: bigwigs("atac") + bigwigs("h3k27ac"),
    output:
        metadata=f"{BROWSER_DIRECTORY}/browser_tracks.metadata.json",
        session=f"{BROWSER_DIRECTORY}/igv_session.xml",
        tracks=BROWSER_TRACKS,
    params:
        output_directory=BROWSER_DIRECTORY,
        bigwig_directory=BIGWIG_DIRECTORY,
        chromosome=BROWSER_CONFIG["chromosome"],
        region_start=BROWSER_CONFIG["region_start"],
        region_end=BROWSER_CONFIG["region_end"],
        stride=BROWSER_CONFIG["stride_bp"],
        batch_size=BROWSER_CONFIG["batch_size"],
        progress_every=BROWSER_CONFIG["progress_every_batches"],
        checkpoint_every=BROWSER_CONFIG["checkpoint_every_batches"],
        mixed_precision=BROWSER_CONFIG["mixed_precision"],
        rc=(
            ""
            if BROWSER_CONFIG["reverse_complement_ensemble"]
            else "--no-reverse-complement-ensemble"
        ),
    log:
        "logs/browser_tracks.log",
    threads: 4
    resources:
        gpu=1
    shell:
        """
        {PYTHON} -m enhancer_pleiotropy_model.browser_tracks \
          --checkpoint {input.checkpoint:q} \
          --reference-fasta {input.reference:q} \
          --blacklist-bed {input.blacklist:q} \
          --observed-bigwig-directory {params.bigwig_directory:q} \
          --output-directory {params.output_directory:q} \
          --chromosome {params.chromosome:q} \
          --region-start {params.region_start} \
          --region-end {params.region_end} \
          --stride {params.stride} \
          --batch-size {params.batch_size} \
          --progress-every-batches {params.progress_every} \
          --checkpoint-every-batches {params.checkpoint_every} \
          --device cuda \
          --mixed-precision {params.mixed_precision:q} \
          {params.rc} 2>&1 | tee {log:q}
        """


rule browser_report:
    input:
        metadata=f"{BROWSER_DIRECTORY}/browser_tracks.metadata.json",
        tracks=BROWSER_TRACKS,
        dhs=config["inputs"]["master_dhs_bed"],
        h3=f"{OUTPUT}/data/h3k27ac_consensus_union.bed",
    output:
        report=f"{REPORT_DIRECTORY}/index.html",
        metrics=f"{REPORT_DIRECTORY}/metrics.json",
        config=f"{REPORT_DIRECTORY}/analysis_config.json",
        per_context=f"{REPORT_DIRECTORY}/per_context_metrics.tsv",
        stratified=f"{REPORT_DIRECTORY}/stratified_metrics.tsv",
        representatives=f"{REPORT_DIRECTORY}/representative_loci.tsv",
        bookmarks=f"{REPORT_DIRECTORY}/representative_loci.bed",
        session=f"{REPORT_DIRECTORY}/igv_session_with_residuals.xml",
        residuals=RESIDUAL_TRACKS,
    params:
        output_directory=REPORT_DIRECTORY,
        active_quantile=REPORT_CONFIG["active_quantile"],
        variable_quantile=REPORT_CONFIG["variable_quantile"],
        representatives=REPORT_CONFIG["representatives_per_class"],
        span=REPORT_CONFIG["representative_span_bp"],
        separation=REPORT_CONFIG["minimum_separation_bp"],
        seed=config["seed"],
    log:
        "logs/browser_report.log",
    threads: 4
    shell:
        """
        {PYTHON} -m enhancer_pleiotropy_model.browser_report \
          --browser-metadata {input.metadata:q} \
          --master-dhs-bed {input.dhs:q} \
          --h3k27ac-peaks-bed {input.h3:q} \
          --output-directory {params.output_directory:q} \
          --active-quantile {params.active_quantile} \
          --variable-quantile {params.variable_quantile} \
          --representatives-per-class {params.representatives} \
          --representative-span-bp {params.span} \
          --minimum-separation-bp {params.separation} \
          --seed {params.seed} 2>&1 | tee {log:q}
        """
