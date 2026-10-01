"""Portable workflow contracts; synthetic inputs and CPU only."""
import copy
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import socket
import sys
from unittest import mock

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT / "scripts")]
from reproduction import common, data, discovery
from classifier_transfer.data import CONTEXTS, SPLITS


def test_matrix_and_configuration_validation(tmp_path):
    cfg = common.config(ROOT / "reproduce/config.yaml")
    entries = list(common.classifiers(cfg))
    assert len(entries) == 60
    assert len({common.run_id(**e) for e in entries}) == 60
    assert {e["readout"] for e in entries if e["architecture"] == "dense"} == {"legacy"}
    assert common.selected_run(cfg).name == "cnn__retained_assay_hidden_v1__background__finetune__20260914"
    assert Path(cfg["paths"]["work"]).is_absolute()
    for field, value in [("seeds", []), ("seeds", [1, 1]), ("populations", ["bad"]),
                         ("readouts", []), ("architectures", ["other"])]:
        raw = yaml.safe_load((ROOT / "reproduce/config.yaml").read_text())
        raw[field] = value
        path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError): common.config(path)


@pytest.mark.parametrize("recipe,n_regressors,n_classifiers", [
    ("best", 1, 6), ("architectures", 3, 18), ("readouts", 2, 24), ("full", 3, 60)])
def test_recipes_preserve_scientific_settings_and_isolate_outputs(recipe, n_regressors, n_classifiers):
    base = common.config(ROOT / "reproduce/config.yaml")
    cfg = common.config(ROOT / "reproduce/config.yaml", recipe=recipe)
    assert len(cfg["architectures"]) == n_regressors
    assert len(list(common.classifiers(cfg))) == n_classifiers
    assert Path(cfg["paths"]["work"]) == Path(base["paths"]["work"]) / recipe
    assert cfg["classifier"] == base["classifier"] and cfg["attribution"] == base["attribution"]
    assert cfg["paths"]["regression_config"] == base["paths"]["regression_config"]
    assert cfg["analysis_classifier"] in list(common.classifiers(cfg))
    if recipe == "architectures":
        dense = [r for r in common.classifiers(cfg) if r["architecture"] == "dense"]
        assert {r["readout"] for r in dense} == {"legacy"}  # Shared dense is already retained.


def test_recipes_reject_unknown_names_and_scientific_overrides(tmp_path):
    for name in ("unknown", "../best"):
        with pytest.raises(ValueError): common.config(ROOT / "reproduce/config.yaml", recipe=name)
    raw = yaml.safe_load((ROOT / "reproduce/config.yaml").read_text())
    raw["recipes"]["best"]["classifier"] = {"epochs": 1}
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="only select"): common.config(path, recipe="best")


def test_guard_requires_real_allocated_host():
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="compute allocation"): common.require_slurm()
    with mock.patch.dict("os.environ", dict(SLURM_JOB_ID="123", SLURM_JOB_NODELIST="gpu[01-02]"), clear=True):
        with mock.patch("subprocess.check_output", return_value="gpu01\ngpu02\n") as check:
            with mock.patch.object(socket, "gethostname", return_value="login.cluster"):
                with pytest.raises(RuntimeError, match="not part"): common.require_slurm()
            with mock.patch.object(socket, "gethostname", return_value="gpu02.cluster"):
                common.require_slurm()
            check.assert_called_with(["scontrol", "show", "hostnames", "gpu[01-02]"], text=True)


def test_scientific_configuration_is_frozen():
    cfg = common.config(ROOT / "reproduce/config.yaml")
    value = data.regression_config(cfg, Path("example_output"))
    assert value["training"]["loss"] == dict(name="alphagenome_profile_cross_context",
        positional_weight=5., cross_context_weight=5., auxiliary_weight=.1)
    assert value["training"]["epochs"] == 40
    assert value["profiles"] == dict(atac_target_bp=512, h3k27ac_target_bp=1536,
                                     source_bin_bp=16, h3k27ac_output_pool_size=4)
    frozen = ROOT / "results/cecar_archives/classifier_suite_20260915/frozen/config/run_config.yaml"
    if frozen.exists():
        original = json.loads(frozen.read_text())
        for key in ("training", "profiles", "model", "sampling", "seed", "contexts", "chromosome_splits", "region_splits"):
            assert value[key] == original[key]


def test_slurm_dry_plan_has_dependencies_and_no_shell_execution():
    spec = importlib.util.spec_from_file_location("reproduction_submit", ROOT / "reproduce/slurm/submit.py")
    submit = importlib.util.module_from_spec(spec); spec.loader.exec_module(submit)
    cfg = common.config(ROOT / "reproduce/config.yaml")
    site = yaml.safe_load((ROOT / "reproduce/slurm/site.example.yaml").read_text())
    cmd = submit.command(cfg, site, "train-classifier", dependency="123")
    assert "--array=0-59%4" in cmd and "--dependency=afterok:123" in cmd and "--gres=gpu:1" in cmd
    assert all("--gres" not in part for part in submit.command(cfg, site, "modisco"))
    assert submit.count(cfg, "attribute") == 80
    chosen = common.config(ROOT / "reproduce/config.yaml", recipe="best")
    cmd = submit.command(chosen, site, "train-classifier", dependency="123", resume=True)
    assert "--array=0-5%4" in cmd
    assert cmd[-3:] == ["--recipe", "best", "--resume"]
    with pytest.raises(ValueError): submit.command(cfg, site, "train-regressor", tasks="3")
    site["extra_sbatch_args"] = ["--wrap=bad"]
    with pytest.raises(ValueError): submit.command(cfg, site, "modisco")


def test_native_target_selection_is_sum_and_checks_order():
    rng = np.random.default_rng(19)
    values = dict(targets=np.asarray(["calibrated_probability_"+c for c in CONTEXTS]+["mean_active_logit"]),
        hypothetical=rng.normal(size=(2,9,4,30)), quality_pass=np.ones((2,9), bool),
        breadth_quality_pass=np.ones(2, bool))
    values["quality_pass"][1, 2] = False
    got, keep = discovery.target_map(values, "calibrated_breadth")
    np.testing.assert_array_equal(got, values["hypothetical"][:, :8].sum(1))
    assert keep.tolist() == [True, False]
    logit, keep = discovery.target_map(values, "mean_active_logit")
    np.testing.assert_array_equal(logit, values["hypothetical"][:, 8])
    assert keep.all()
    values["targets"] = values["targets"][::-1]
    with pytest.raises(ValueError): discovery.target_map(values, "calibrated_breadth")


def test_calibration_can_bind_new_checkpoint_without_relaxing_legacy_default():
    import torch
    from classifier_motifs.calibrated_attribution import CalibratedTargets
    calibration = dict(checkpoint_sha256="a"*64, contexts=list(CONTEXTS), fitting_population="enhancers_only",
        method="sigmoid", probability_clip=1e-7, a=[1.]*8, b=[0.]*8)
    with pytest.raises(ValueError): CalibratedTargets(torch.nn.Identity(), calibration)
    CalibratedTargets(torch.nn.Identity(), calibration, expected_checkpoint_sha256="a"*64)


@pytest.mark.parametrize("architecture,readout", [("cnn","legacy"), ("attention","legacy"),
    ("cnn","retained_assay_hidden_v1"), ("attention","retained_assay_hidden_v1"), ("dense","legacy")])
def test_portable_model_factories_transfer_expected_layers(tmp_path, architecture, readout):
    import torch
    from classifier_transfer.models import make_regressor, initialize_classifier, CNN_NAME, DENSE_NAME
    from enhancer_pleiotropy_model.training import architecture_metadata
    torch.set_num_threads(1)
    reg = make_regressor(architecture)
    meta = architecture_metadata(reg, CONTEXTS)
    if architecture != "attention": meta["name"] = CNN_NAME if architecture == "cnn" else DENSE_NAME
    path = tmp_path / "parent.pt"
    torch.save(dict(architecture=meta, contexts=CONTEXTS, state_dict=reg.state_dict()),path)
    fine = initialize_classifier(architecture,19,np.full(8,.3),path,readout)
    scratch = initialize_classifier(architecture,19,np.full(8,.3),None,readout)
    for key, value in fine.state_dict().items():
        source = key.replace("atac_features.","atac_head.").replace("h3k27ac_features.","h3k27ac_head.")
        expected = scratch.state_dict()[key] if key.startswith("head.") else reg.state_dict()[source]
        torch.testing.assert_close(value,expected,rtol=0,atol=0)
    fine.eval()
    x = torch.nn.functional.one_hot(torch.zeros(1,2048,dtype=torch.long),4).permute(0,2,1).float()
    assert fine(x).shape == (1,8)
    assert all(p.requires_grad for p in fine.parameters())
    assert not torch.cuda.is_initialized()


def test_training_provenance_refuses_changed_source_or_config(tmp_path):
    cfg = common.config(ROOT / "reproduce/config.yaml")
    cfg["paths"]["work"] = str(tmp_path)
    with mock.patch.object(common,"provenance",return_value=dict(signature="one")):
        common.pin_training_source(cfg,"train-regressor",0)
        common.pin_training_source(cfg,"train-regressor",0)
    with mock.patch.object(common,"provenance",return_value=dict(signature="two")):
        with pytest.raises(ValueError): common.pin_training_source(cfg,"train-regressor",0)


def test_regressor_pause_does_not_release_downstream_dependencies(tmp_path):
    from reproduction import training
    cfg = common.config(ROOT / "reproduce/config.yaml")
    cfg["paths"]["work"] = str(tmp_path)
    cfg["architectures"] = ["attention"]
    with (mock.patch("enhancer_pleiotropy_model.training.main"),
          mock.patch("enhancer_pleiotropy_model.training.require_training_node"),
          mock.patch.object(training, "verify_preprocessing")):
        with pytest.raises(SystemExit) as error: training.train_regressor(cfg, 0, False)
    assert error.value.code == 75


def test_rebuilt_test_background_uses_gc_and_excludes_catalog_duplicates(tmp_path):
    rng = np.random.default_rng(37)
    splits = {}
    for s in SPLITS:
        codes = rng.integers(0,4,(12,2048),dtype=np.uint8)
        splits[s] = dict(sequence=codes, ids=np.asarray([s+str(i) for i in range(12)]),
                         chrom=np.full(12, "chr3R"))
        np.savez_compressed(tmp_path / (s+".npz"), **splits[s])
    windows = tmp_path / "windows.tsv.gz"
    header = ["record_id", "split", "source", "chrom", "start", "end", "target_start", "target_end", "sequence"]
    with gzip.open(windows,"wt") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, delimiter="\t"); writer.writeheader()
        for i in range(100):
            codes = (splits["train"]["sequence"][0] if i == 0 else rng.integers(0,4,2048))
            start = 10000+i*256
            writer.writerow(dict(record_id="bg"+str(i), split="test", source="genomic_background", chrom="chr3R",
                start=start,end=start+2048,target_start=start+768,target_end=start+1280,
                sequence="".join(np.asarray(list("ACGT"))[codes])))
    common.write_json(windows.with_name("windows.metadata.json"),
        dict(output=dict(sha256=common.digest(windows)), candidate_and_sampling_counts=dict(test=dict(selected_background=100))))
    with mock.patch.object(data, "load_split", side_effect=lambda root,s:splits[s]):
        data.prepare_test_background(tmp_path, windows, tmp_path / "bg")
    with np.load(tmp_path / "bg/matched_background.npz") as result:
        assert len(result["ids"]) == 12 and "bg0" not in result["ids"]
        assert np.max(np.abs(result["enhancer_gc"]-result["background_gc"])) <= .02
        np.testing.assert_array_equal(result["enhancer_ids"], splits["test"]["ids"][result["enhancer_index"]])


def test_figure_fixture_is_self_contained_and_checksummed():
    root = ROOT / "reproduce/figure3"
    manifest = json.loads((root / "manifest.json").read_text())
    for name, sha in manifest["files"].items(): assert common.digest(root / name) == sha
    source = json.loads((root / "source.json").read_text())
    assert len(source["panel_b"]) == 39
    assert all(k in source for k in ("panel_c", "panel_d", "panel_e"))


def test_readme_best_model_figure_is_shipped_with_matching_provenance():
    from export_reproduction_release import files
    shipped = set(files())
    figure = ROOT / "docs/figures/best_model_architecture"
    for suffix in (".png", ".svg", ".pdf", ".provenance.json"):
        assert figure.with_suffix(suffix) in shipped
    record = json.loads(figure.with_suffix(".provenance.json").read_text())
    assert record["pdf_sha256"] == common.digest(figure.with_suffix(".pdf"))
    snapshot = json.loads((ROOT / "reproduce/benchmarks/models.json").read_text())
    best = next(r for r in snapshot["models"] if r["model_id"] == snapshot["recommended_classifier_id"])
    assert record["model"] == best["model_id"]
    assert record["checkpoint_sha256"] == best["checkpoint_sha256"]
    assert record["parameters"] == best["parameters"] == 1244168
    assert "docs/figures/best_model_architecture.png" in (ROOT / "README.md").read_text()
    assert ROOT / "LICENSES/AlphaGenome-Apache-2.0.txt" in shipped


def test_inventory_includes_planned_models_without_reading_test(tmp_path):
    from reproduction.inventory import refresh
    cfg = common.config(ROOT / "reproduce/config.yaml")
    cfg["paths"]["work"] = str(tmp_path)
    refresh(cfg)
    with (tmp_path / "model_inventory.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 63 and {r["status"] for r in rows} == {"planned"}
    assert sum(r["type"] == "classifier" for r in rows) == 60
    assert all(not r.get("val_macro_ap") for r in rows)
    common.write_json(tmp_path / "regressors/cnn/model/history.json", [
        dict(epoch=1, scientific_composite=.5, learning_rate=1e-4),
        dict(epoch=2, scientific_composite=.4, learning_rate=5e-5)])
    refresh(cfg)
    with (tmp_path / "model_inventory.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    cnn = next(r for r in rows if r["model_id"] == "regressor__cnn")
    assert cnn["status"] == "partial" and cnn["epochs_completed"] == "2" and cnn["selected_epoch"] == "1"
    assert float(cnn["lr_current_encoder"]) == 5e-5


def test_inventory_keeps_selected_epoch_background_metrics_and_lineage(tmp_path):
    from reproduction.inventory import refresh
    cfg = common.config(ROOT / "reproduce/config.yaml", recipe="best")
    cfg["paths"]["work"] = str(tmp_path)
    entry = cfg["analysis_classifier"]; name = common.run_id(**entry)
    path = tmp_path / "classifiers" / name / "history.json"
    common.write_json(path, [dict(epoch=2, selected=True, learning_rates=[8e-6, 8e-5],
        metrics=dict(macro_average_precision=.6, brier=.12, breadth=dict(r2=.3),
                     contexts=dict(ab=dict(average_precision=.7, auroc=.9))),
        background_metrics=dict(enhancers_plus_background=dict(macro_average_precision=.55,
            contexts=dict(ab=dict(average_precision=.65)))),),
        dict(epoch=3, selected=False, learning_rates=[7e-6, 7e-5],
             metrics=dict(macro_average_precision=.59, brier=None))])
    common.write_json(tmp_path / "evaluation/metrics.json", {
        name:dict(enhancer_only=dict(macro_average_precision=.58,
                                   contexts=dict(ab=dict(average_precision=.66))),
                  enhancers_plus_background=dict(macro_average_precision=.53))})
    refresh(cfg)
    with (tmp_path / "model_inventory.tsv").open() as stream:
        row = next(r for r in csv.DictReader(stream, delimiter="\t") if r["model_id"] == name)
    assert row["selected_epoch"] == "2" and row["val_brier"] == "0.12"
    assert float(row["val_enhancers_plus_background_macro_ap"]) == .55
    assert float(row["val_enhancers_plus_background_ab_average_precision"]) == .65
    assert float(row["test_enhancer_only_ab_average_precision"]) == .66
    assert float(row["selected_lr_encoder"]) == 8e-6 and float(row["lr_current_encoder"]) == 7e-6
    assert row["parent_regressor_model_id"] == "regressor__cnn"
    assert row["initialization_checkpoint_path"].endswith("regressors/cnn/model/best_model.pt")
    assert row["initialization_checkpoint_sha256"] == ""  # Not fabricated for a missing checkpoint.


def test_native_discovery_adapter_uses_train_bounds_quality_and_signed_support(tmp_path):
    h5py = pytest.importorskip("h5py")
    cfg = common.config(ROOT / "reproduce/config.yaml")
    cfg["paths"]["work"] = str(tmp_path); cfg["attribution"]["shards"] = 1
    rng = np.random.default_rng(39); n = 6
    cohort = dict(ids=np.asarray(["id"+str(i) for i in range(n)]), sequence=rng.integers(0,4,(n,2048)),
        labels=np.tile(np.array([1,0,0,0,0,0,0,0]),(n,1)), split=np.asarray(["train"]*4+["validation","test"]))
    intervals = dict(ids=cohort["ids"], offset=np.arange(n)+900, length=np.arange(n)+100)
    catalog = tmp_path / "catalog"; catalog.mkdir()
    np.savez_compressed(catalog / "cohort.npz", **cohort)
    np.savez_compressed(catalog / "intervals.npz", **intervals)
    root = tmp_path / "attribution"; (root / "chunks").mkdir(parents=True)
    quality = np.ones((n,9),bool); quality[0,0] = False
    hypothetical = rng.normal(size=(n,9,4,2048))
    path = root / "chunks/chunk_000000.npz"
    np.savez_compressed(path, indices=np.arange(n), ids=cohort["ids"], signature=np.asarray("pinned"),
        targets=np.asarray(["calibrated_probability_"+c for c in CONTEXTS]+["mean_active_logit"]),
        hypothetical=hypothetical, quality_pass=quality, breadth_quality_pass=np.ones(n,bool))
    common.write_json(root / "manifest.json", dict(signature="pinned"))
    common.write_json(root / "shard_0.json", dict(status="complete", signature="pinned", chunks={path.name:common.digest(path)}))
    def discover(seq, hyp, lengths, parameters, output, cap):
        assert len(seq) == 3 and set(lengths) == {101,102,103}
        for row, length in enumerate(lengths):
            i = int(length)-100; start = int(intervals["offset"][i])
            np.testing.assert_allclose(hyp[row,:length], hypothetical[i,:8,:,start:start+length].sum(0).T)
            assert not seq[row,length:].any() and not hyp[row,length:].any()
        with h5py.File(output,"w") as f:
            for prefix, sign in [("pos_patterns",1),("neg_patterns",-1)]:
                g = f.create_group(prefix+"/pattern_0")
                pwm = np.full((20,4),.25); pwm[5:15] = [1,0,0,0]
                g["sequence"] = pwm; g["contrib_scores"] = sign*np.ones((20,4))
                g["seqlets/example_idx"] = [0,0,1]
        return dict(positive=dict(patterns=1),negative=dict(patterns=1))
    with mock.patch.object(discovery,"discover_original",side_effect=discover): discovery.run(cfg,0)
    output = tmp_path / "motifs/calibrated_breadth/specific"
    report = json.loads((output / "report.json").read_text())
    assert report["quality_excluded"] == 1 and set(report["selection"]) == {1,2,3}
    assert len(report["groups"][0]["rows"]) == 2
    for row in report["groups"][0]["rows"]:
        assert row["attribution_support"] == dict(hits=2,n=3,fraction=2/3)
    assert "66.7%" in (output / "report.html").read_text()
    assert json.loads((output / "complete.json").read_text())["status"] == "complete"
