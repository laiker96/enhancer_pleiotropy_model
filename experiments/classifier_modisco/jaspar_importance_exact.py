"""Reaggregate frozen per-enhancer motif scores into exact degrees 1-8 on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import digest, write_json
from .jaspar_importance import EXACT_GROUPS, summarize


def main(source, output):
    if source.resolve() == output.resolve():
        raise ValueError("Keep the original scan outputs unchanged")
    receipt = json.loads((source / "complete.json").read_text())
    if receipt["status"] != "complete":
        raise ValueError("Completed scan required")
    hashes = {name: digest(source / name) for name in ("summary.json", "importance.npz")}
    if any(value != receipt["files"][name] for name, value in hashes.items()):
        raise ValueError("Changed scan input")
    original = json.loads((source / "summary.json").read_text())
    if original["profiles"] != 296 or original["split"] != "train" or original["references"] != 50:
        raise ValueError("Expected the all-JASPAR 50-reference training scan")
    with np.load(source / "importance.npz", allow_pickle=False) as saved:
        scores, covered = saved["scores"], saved["covered_bases"]
        metadata = {key: saved[key] for key in ("ids", "breadth", "split")}
        np.testing.assert_array_equal(saved["motif_ids"], [r["id"] for r in original["trajectories"]])
    if len(np.unique(metadata["ids"])) != len(metadata["ids"]):
        raise ValueError("Duplicate enhancer IDs")
    train = metadata["split"] == "train"
    if not np.isin(metadata["breadth"][train], np.arange(1, 9)).all():
        raise ValueError("Training degree outside 1-8")
    if np.isfinite(scores[:, ~train]).any():
        raise ValueError("Unexpected nontraining scores")
    np.testing.assert_array_equal(np.isfinite(scores), covered > 0)
    profiles = original["trajectories"]
    if summarize(scores, metadata, profiles) != profiles:
        raise ValueError("Saved scores do not reproduce the original cumulative summary")
    rows = summarize(scores, metadata, profiles, groups=EXACT_GROUPS)
    if sum(rows[0]["group_enhancers"]) != int(train.sum()):
        raise ValueError("Exact degrees do not partition the training cohort")
    for old, new in zip(profiles, rows):
        for key in ("means", "motif_containing_enhancers", "group_enhancers", "standard_deviation"):
            if old[key][0] != new[key][0] or old[key][-1] != new[key][-1]:
                raise ValueError("Degree-1/8 endpoint mismatch")
    summary = dict(original, groups=EXACT_GROUPS, group_labels=[str(k) for k in range(1, 9)],
                   trajectories=rows, grouping="exact", source_files=hashes,
                   reaggregation_code_sha256=digest(Path(__file__)),
                   scoring_code_sha256=digest(Path(__file__).with_name("jaspar_importance.py")),
                   caveats=["Exact degree groups are disjoint; within each group, motif carriers may overlap"]
                           + original["caveats"][1:])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_json(output / "complete.json", dict(status="complete", profiles=296,
        files={"summary.json": digest(output / "summary.json")}, source_files=hashes,
        inference_rerun=False, discovery_rerun=False, scan_rerun=False,
        checks=["source hashes", "motif alignment", "training-only finite scores", "carrier mask",
                "cumulative reproduction", "disjoint cohort partition", "degree-1/8 endpoints"]))
    print(f"Exact-degree summary complete: {rows[0]['group_enhancers']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
