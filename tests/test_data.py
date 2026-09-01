import gzip

import pytest
import torch

from enhancer_pleiotropy_model.data import (
    EpochShuffleSampler,
    JointProfileCollator,
    read_windows,
)


HEADER = (
    "record_id\tsource\tchrom\tstart\tend\ttarget_start\ttarget_end\t"
    "block_id\tsplit\tsequence\n"
)


def write_windows(path, validation_target_start=1768):
    sequence = "ACGT" * 512
    rows = [
        f"train\tatac_peak_overlap\tchr2R\t0\t2048\t768\t1280\ttrain:chr2R:0\ttrain\t{sequence}\n",
        (
            "validation\tatac_peak_overlap\tchr2L\t1000\t3048\t"
            f"{validation_target_start}\t{validation_target_start + 512}\t"
            f"validation:chr2L:0\tvalidation\t{sequence}\n"
        ),
        f"test\tatac_peak_overlap\tchr3R\t0\t2048\t768\t1280\ttest:chr3R:0\ttest\t{sequence}\n",
    ]
    with gzip.open(path, "wt") as handle:
        handle.write(HEADER)
        handle.writelines(rows)


def test_read_windows_enforces_disjoint_centered_splits(tmp_path):
    path = tmp_path / "windows.tsv.gz"
    write_windows(path)
    records = read_windows(path)
    assert [record.chrom for record in records["train"]] == ["chr2R"]
    assert [record.chrom for record in records["validation"]] == ["chr2L"]
    assert [record.chrom for record in records["test"]] == ["chr3R"]


def test_read_windows_allows_nonoverlapping_regions_on_one_chromosome(tmp_path):
    path = tmp_path / "windows.tsv.gz"
    sequence = "ACGT" * 512
    rows = [
        f"train\tatac_peak_overlap\tchr2L\t4096\t6144\t4864\t5376\ttrain:chr2L:0\ttrain\t{sequence}\n",
        f"validation\tatac_peak_overlap\tchr2L\t0\t2048\t768\t1280\tvalidation:chr2L:0\tvalidation\t{sequence}\n",
        f"test\tatac_peak_overlap\tchr3R\t0\t2048\t768\t1280\ttest:chr3R:0\ttest\t{sequence}\n",
    ]
    with gzip.open(path, "wt") as handle:
        handle.write(HEADER)
        handle.writelines(rows)
    records = read_windows(path)
    assert len(records["train"]) == len(records["validation"]) == 1


def test_read_windows_rejects_cross_split_input_overlap(tmp_path):
    path = tmp_path / "windows.tsv.gz"
    sequence = "ACGT" * 512
    rows = [
        f"train\tatac_peak_overlap\tchr2L\t1000\t3048\t1768\t2280\ttrain:chr2L:0\ttrain\t{sequence}\n",
        f"validation\tatac_peak_overlap\tchr2L\t2000\t4048\t2768\t3280\tvalidation:chr2L:0\tvalidation\t{sequence}\n",
        f"test\tatac_peak_overlap\tchr3R\t0\t2048\t768\t1280\ttest:chr3R:0\ttest\t{sequence}\n",
    ]
    with gzip.open(path, "wt") as handle:
        handle.write(HEADER)
        handle.writelines(rows)
    with pytest.raises(ValueError, match="different splits overlap"):
        read_windows(path)


def test_read_windows_rejects_off_center_target(tmp_path):
    path = tmp_path / "windows.tsv.gz"
    write_windows(path, validation_target_start=1769)
    with pytest.raises(ValueError, match="not centered"):
        read_windows(path)


def test_epoch_sampler_resume_is_exact_tail():
    complete = list(EpochShuffleSampler(10, seed=17, epoch=3))
    resumed = list(EpochShuffleSampler(10, seed=17, epoch=3, start_index=4))
    assert resumed == complete[4:]


def test_collator_builds_centered_target_masks():
    batch = JointProfileCollator()(
        [("A" * 2048, torch.zeros(32, 8), torch.zeros(24, 8), False)]
    )
    assert batch["atac_target_mask"].sum().item() == 512
    assert batch["h3k27ac_target_mask"].sum().item() == 1536
