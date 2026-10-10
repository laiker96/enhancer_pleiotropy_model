"""End to end on synthetic inputs: prepare data, train, fine-tune, calibrate, predict (CPU)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import repository_config
from enhancer_pleiotropy_model.cli import main
from enhancer_pleiotropy_model.constants import CONTEXTS, SPLITS

ROOT = Path(__file__).resolve().parents[1]
TINY_REGRESSOR = ["training.mixed_precision=none", "training.epochs=2", "training.batch_size=4",
                  "training.eval_batch_size=8", "training.max_batches_per_epoch=3", "training.max_validation_batches=4"]
TINY_CLASSIFIER = ["training.epochs=2", "training.batch_size=8", "training.max_batches_per_epoch=3"]


def overrides(values):
    return [item for value in values for item in ("--set", value)]


def test_prepared_datasets(workspace):
    data = workspace / "data"
    windows = pd.read_csv(data / "regression/windows.tsv.gz", sep="\t")
    assert set(windows.split) == set(SPLITS)
    assert (windows.end - windows.start == 2048).all()
    assert {"genomic_background", "atac_h3k27ac_peak_overlap"} <= set(windows.source)
    assert not ((windows.chrom == "chrA") & (windows.start < 101_000) & (windows.end > 100_000)).any()
    atac = np.load(data / "regression/profiles/atac/train_profiles.npy")
    assert atac.shape == ((windows.split == "train").sum(), 32, 8) and atac.min() >= 0
    enhancers = pd.read_csv(data / "classification/enhancers.tsv.gz", sep="\t")
    catalog = pd.read_csv(workspace / "inputs/master_elements_wide.tsv.gz", sep="\t")
    active = catalog[[f"{c}__h3k27ac_active" for c in CONTEXTS]].any(axis=1)
    expected = catalog[active & (catalog.regulatory_class != "promoter_associated") & (catalog.blacklist_overlap == 0)]
    assert list(enhancers.master_dhs_id) == list(expected.master_dhs_id)
    for split in SPLITS:
        with np.load(data / f"classification/background/{split}.npz") as bg:
            assert bg["sequence"].shape[1] == 2048 and not bg["labels"].any()
            assert np.abs(bg["background_gc"] - bg["enhancer_gc"]).max() <= .05 + 1e-9


@pytest.mark.parametrize("architecture", ["cnn", "attention", "dense", "enhancernet_cnn", "enhancernet_attention"])
def test_train_regressor(workspace, architecture):
    main(["train-regressor", "--config", str(ROOT / "configs/regressor.yaml"), "--architecture", architecture,
          *overrides(repository_config("regression", workspace) + TINY_REGRESSOR)])
    run = workspace / "runs/regressors" / architecture
    assert (run / "best_model.pt").is_file()
    assert len(json.loads((run / "history.json").read_text())) == 2


def test_resume_regressor_is_noop_after_completion(workspace):
    args = ["train-regressor", "--config", str(ROOT / "configs/regressor.yaml"), "--architecture", "cnn", "--resume",
            *overrides(repository_config("regression", workspace) + TINY_REGRESSOR)]
    main(args)
    assert len(json.loads((workspace / "runs/regressors/cnn/history.json").read_text())) == 2


@pytest.mark.parametrize("architecture,init", [("cnn", "pretrained"), ("cnn", "scratch"),
                                               ("dense", "pretrained"), ("enhancernet_attention", "pretrained")])
def test_train_classifier_calibrate_predict(workspace, architecture, init):
    if init == "pretrained" and not (workspace / "runs/regressors" / architecture / "best_model.pt").is_file():
        test_train_regressor(workspace, architecture)
    main(["train-classifier", "--config", str(ROOT / "configs/classifier.yaml"), "--architecture", architecture,
          "--init", init, "--seed", "7",
          *overrides(repository_config("classification", workspace,
                                       regressors=workspace / "runs/regressors") + TINY_CLASSIFIER)])
    run = workspace / "runs/classifiers" / f"{architecture}_{init}_7"
    history = json.loads((run / "history.json").read_text())
    assert len(history) == 2 and "background_metrics" in history[0]
    main(["calibrate", str(run), "--device", "cpu"])
    calibration = json.loads((run / "calibration.json").read_text())
    assert set(calibration) == {"enhancers_only", "enhancers_plus_background_1to1"}
    assert len(calibration["enhancers_only"]["a"]) == 8 and min(calibration["enhancers_only"]["a"]) > 0
    assert (run / "calibration_report.md").is_file()
    bed = workspace / "regions.bed"
    bed.write_text("chrB\t20000\t20300\tregion_1\nchrC\t50000\t50500\tregion_2\n")
    main(["predict", str(run / "best_model.pt"), "--bed", str(bed), "--genome",
          str(workspace / "inputs/dm6.fa"), "--output", str(run / "predictions.tsv"), "--device", "cpu"])
    predictions = pd.read_csv(run / "predictions.tsv", sep="\t")
    assert list(predictions.id) == ["region_1", "region_2"]
    assert predictions[[f"p_{c}" for c in CONTEXTS]].apply(lambda v: v.between(0, 1)).all().all()


def test_manuscript_lists_35_runs(capsys):
    import os
    os.chdir(ROOT)
    main(["manuscript"])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("epm")]
    assert len(lines) == 35
    assert sum("train-regressor" in line for line in lines) == 5


def test_attribution_maps(workspace):
    run = workspace / "runs/classifiers/cnn_pretrained_7"
    if not (run / "calibration.json").is_file():
        test_train_classifier_calibrate_predict(workspace, "cnn", "pretrained")
    output = workspace / "runs/attribution"
    args = ["attribute", "--config", str(ROOT / "configs/attribution.yaml"), "--device", "cpu", "--limit", "1",
            *overrides([f"classifier={run / 'best_model.pt'}", f"data={workspace / 'data/classification'}",
                        f"output={output}", "references=4", "steps=4", "enhancers_per_chunk=3",
                        "reference_block=2", "pair_batch=3", "internal_batch=24"])]
    main(args)
    with np.load(output / "chunks/chunk_000000.npz") as z:
        chunk = dict(z)
    assert chunk["actual"].shape == (3, 8, 2048) and chunk["hypothetical"].shape == (3, 8, 4, 2048)
    projected = np.take_along_axis(chunk["hypothetical"], chunk["sequence"][:, None, None, :], axis=2)[:, :, 0]
    assert np.allclose(projected, chunk["actual"], atol=1e-7)
    assert np.allclose(chunk["actual_reference_halves"].mean(1), chunk["actual"], atol=1e-6)
    assert np.allclose(chunk["actual"].sum(2), chunk["delta"].mean(1) + chunk["target_difference"].mean(1), atol=1e-5)
    assert chunk["reference_hashes"].shape == (3, 4)
    offsets, lengths = chunk["native_offset"], chunk["native_length"]
    assert np.all(offsets >= 0) and np.all(offsets + lengths <= 2048)
    # An interrupted chunk resumes from its saved references and gives the same maps.
    (output / "chunks/chunk_000000.npz").unlink()
    from enhancer_pleiotropy_model import attribution
    original = attribution.integrate
    calls = []
    def interrupt_after_first_block(*a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise TimeoutError
        return original(*a, **k)
    attribution.integrate = interrupt_after_first_block
    try:
        with pytest.raises(TimeoutError):
            main(args)
    finally:
        attribution.integrate = original
    assert (output / "chunks/chunk_000000.state.npz").is_file()
    main(args)
    with np.load(output / "chunks/chunk_000000.npz") as z:
        assert np.allclose(z["actual"], chunk["actual"], atol=1e-6)
