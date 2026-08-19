from pathlib import Path
import sys

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.audit_can_paired_dataset import audit_paired_dataset


def _demo(data, index, initial_state, reward):
    group = data.create_group(f"demo_{index}")
    group.attrs["num_samples"] = 2
    group.create_dataset("states", data=np.stack([initial_state, initial_state + 1]))
    group.create_dataset("actions", data=np.zeros((2, 2)))
    group.create_dataset("rewards", data=np.asarray([0.0, reward]))


def test_audit_accepts_exact_success_failure_pairs(tmp_path):
    path = tmp_path / "paired.hdf5"
    with h5py.File(path, "w") as dataset:
        data = dataset.create_group("data")
        _demo(data, 0, np.asarray([1.0, 2.0]), 0.0)
        _demo(data, 1, np.asarray([1.0, 2.0]), 1.0)
        _demo(data, 2, np.asarray([3.0, 4.0]), 1.0)
        _demo(data, 3, np.asarray([3.0, 4.0]), 0.0)
        mask = dataset.create_group("mask")
        mask.create_dataset("train", data=np.asarray([b"demo_0", b"demo_1"]))
        mask.create_dataset("valid", data=np.asarray([b"demo_2", b"demo_3"]))

    report = audit_paired_dataset(path)

    assert report["all_pairs_valid"] is True
    assert report["valid_pair_count"] == 2
    assert report["success_count"] == 2
    assert report["split_pair_integrity"] == {"train": True, "valid": True}


def test_audit_rejects_noncomplementary_or_mismatched_pairs(tmp_path):
    path = tmp_path / "invalid.hdf5"
    with h5py.File(path, "w") as dataset:
        data = dataset.create_group("data")
        _demo(data, 0, np.asarray([1.0]), 1.0)
        _demo(data, 1, np.asarray([2.0]), 1.0)

    report = audit_paired_dataset(path)

    assert report["all_pairs_valid"] is False
    assert report["invalid_pair_count"] == 1
    assert report["pairs"][0]["same_initial_state"] is False
    assert report["pairs"][0]["complementary_outcomes"] is False
