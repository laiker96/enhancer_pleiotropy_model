"""Saved-result integrity and report tests; no inference or private data required."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_paper_benchmarks as report


def snapshot():
    return json.loads(report.SNAPSHOT.read_text())


def test_all_completed_runs_and_readout_strategies_are_preserved():
    data = snapshot()
    rows = report.validate(data)
    assert len(rows) == 94
    assert sum(r["model_type"] == "regressor" for r in rows) == 7
    assert sum(r["model_type"] == "classifier" for r in rows) == 87
    groups = report.groups(rows)
    assert len(groups) == 29 and all(len(g) == 3 for g in groups)
    assert all({r["seed"] for r in g} == {20260914, 20260915, 20260916} for g in groups)
    assert {report.hidden(g[0]) for g in groups} == {
        "Replace assay hidden", "Keep assay hidden", "Keep shared dense"}
    assert sum(r["test_enhancer_macro_ap"] is None for r in rows if r["model_type"] == "classifier") == 12


def test_snapshot_matches_original_inventory_when_available():
    data = snapshot()
    path = ROOT / data["source_inventory"]
    if not path.exists():
        return  # Source-only distributions instead check their frozen snapshot/report.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == data["source_inventory_sha256"]
    source = {r["model_id"]: r for r in csv.DictReader(path.open(), delimiter="\t")}
    for row in data["models"]:
        for key, value in row.items():
            if key not in source[row["model_id"]]:
                continue  # Portable artifact path derived from original machine-specific path.
            original = source[row["model_id"]][key]
            if value is None:
                assert original == ""
            elif isinstance(value, bool):
                assert original == str(value).lower()
            elif isinstance(value, (float, int)):
                assert float(original) == value
            else:
                assert original == value


def test_representative_selection_never_uses_test_metrics():
    data = snapshot()
    group = next(g for g in report.groups(data["models"])
                 if any(r["model_id"] == data["recommended_classifier_id"] for r in g))
    assert report.representative(group)["model_id"] == data["recommended_classifier_id"]
    altered = copy.deepcopy(group)
    for r in altered:
        r["test_combined_macro_ap"] = 0 if r["model_id"] == data["recommended_classifier_id"] else 1
    assert report.representative(altered)["model_id"] == data["recommended_classifier_id"]
    best = report.representative(altered)
    assert best["checkpoint_sha256"] == "7dc8eb721fbd4f292eba95ff2ea44399cfd1aae5a596e8d56e6d7ff25a95628a"
    assert best["selected_epoch"] == 38


def test_null_metrics_not_zero_filled_or_silently_dropped():
    assert report.aggregate([{"v": None}] * 3, "v") == "—"
    assert report.aggregate([{"v": None}, {"v": .6}, {"v": .8}], "v") == "0.7000 ± 0.1414 (n=2/3)"
    assert report.number(None) == "—"


@pytest.mark.parametrize("bad", ["duplicate", "private_path", "absolute_artifact", "wrong_macro", "nan"])
def test_invalid_snapshot_is_rejected(bad):
    data = snapshot()
    row = next(r for r in data["models"] if r["model_type"] == "classifier")
    if bad == "duplicate": data["models"].append(copy.deepcopy(row))
    elif bad == "private_path": row["notes"] = "/home/private/user/file"
    elif bad == "absolute_artifact": row["checkpoint_artifact_path"] = "/secret/model.pt"
    elif bad == "wrong_macro": row["val_macro_ap"] = .12345
    elif bad == "nan": row["val_loss"] = float("nan")
    with pytest.raises(ValueError): report.validate(data)


def test_checked_in_report_is_reproducible_without_models_or_torch():
    text = report.render(snapshot())
    assert report.REPORT.read_text() == text
    assert "15 epochs of the original 40-epoch LR curve" in text
    assert "not a pure input-length comparison" in text
    assert "## Validation AP per context" in text and "## Test AP per context" in text
    assert all(r["checkpoint_sha256"] in text for r in snapshot()["models"])
