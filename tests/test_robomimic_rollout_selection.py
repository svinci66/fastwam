from pathlib import Path
import sys

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.select_can_rollout_demos import select_demos


def test_selects_successful_demos_in_numeric_order_and_split(tmp_path):
    dataset = tmp_path / "rollouts.hdf5"
    with h5py.File(dataset, "w") as output:
        data = output.create_group("data")
        for name, success in (("demo_10", 1), ("demo_2", 1), ("demo_1", 0)):
            group = data.create_group(name)
            group.attrs["success"] = success
        masks = output.create_group("mask")
        masks.create_dataset(
            "train",
            data=np.asarray(["demo_10", "demo_1"], dtype=h5py.string_dtype()),
        )
        masks.create_dataset(
            "valid", data=np.asarray(["demo_2"], dtype=h5py.string_dtype())
        )

    assert select_demos(dataset, require_success=True) == ["demo_2", "demo_10"]
    assert select_demos(dataset, split="train", require_success=True) == ["demo_10"]
