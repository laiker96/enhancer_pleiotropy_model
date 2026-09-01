"""Reproducible data preparation and default 4x training workflow."""

configfile: "config/default.yaml"

import json
from pathlib import Path


OUTPUT = config["output_directory"]
PYTHON = ".venv/bin/python"
CONTEXTS = config["contexts"]
BIGWIG_DIRECTORY = config["inputs"]["bigwig_directory"]


def bigwigs(assay):
    return [
        f"{BIGWIG_DIRECTORY}/{context}.{assay}.mean.background_tmm.bw"
        for context in CONTEXTS
    ]


rule all:
    input:
        f"{OUTPUT}/model/best_model.pt",
        f"{OUTPUT}/model/metrics.json",


rule prepared_data:
    input:
        f"{OUTPUT}/data/windows.tsv.gz",
        f"{OUTPUT}/data/windows.metadata.json",
        f"{OUTPUT}/data/profiles/atac/train_profiles.npy",
        f"{OUTPUT}/data/profiles/atac/validation_profiles.npy",
        f"{OUTPUT}/data/profiles/h3k27ac/train_profiles.npy",
        f"{OUTPUT}/data/profiles/h3k27ac/validation_profiles.npy",


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
        atac_bigwigs=bigwigs("atac"),
        h3_bigwigs=bigwigs("h3k27ac"),
    output:
        table=f"{OUTPUT}/data/windows.tsv.gz",
        metadata=f"{OUTPUT}/data/windows.metadata.json",
    params:
        bigwig_directory=BIGWIG_DIRECTORY,
        contexts=" ".join(CONTEXTS),
        train=" ".join(config["chromosome_splits"]["train"]),
        validation=" ".join(config["chromosome_splits"]["validation"]),
        test=" ".join(config["chromosome_splits"]["test"]),
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
          --window-size {params.target} \
          --context-flank-size {params.flank} \
          --stride {params.train_stride} \
          --validation-stride {params.validation_stride} \
          --split-strategy chromosome \
          --block-size {params.block} \
          --background-to-peak-ratio {params.background_ratio} \
          --seed {params.seed} \
          --train-chromosomes {params.train} \
          --validation-chromosomes {params.validation} \
          --test-chromosomes {params.test} \
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
        atac_metadata=f"{OUTPUT}/data/profiles/atac/profiles.metadata.json",
        h3_train=f"{OUTPUT}/data/profiles/h3k27ac/train_profiles.npy",
        h3_validation=f"{OUTPUT}/data/profiles/h3k27ac/validation_profiles.npy",
        h3_metadata=f"{OUTPUT}/data/profiles/h3k27ac/profiles.metadata.json",
    output:
        best=f"{OUTPUT}/model/best_model.pt",
        last=f"{OUTPUT}/model/last_checkpoint.pt",
        metrics=f"{OUTPUT}/model/metrics.json",
    log:
        f"logs/train_4x.log",
    threads: 4
    resources:
        gpu=1
    shell:
        """
        {PYTHON} -m enhancer_pleiotropy_model.training \
          --config {input.config:q} --resume 2>&1 | tee {log:q}
        """
