from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.merge_can_deployable_actor_holdout import merge


def _dataset(path, split, demos, values):
    count = len(split)
    np.savez_compressed(
        path,
        source_split=np.asarray(split),
        source_demo=np.asarray(demos),
        source_step=np.arange(count, dtype=np.int32),
        state=np.asarray(values, dtype=np.float32).reshape(count, 1),
        base_action_chunk=np.zeros((count, 1, 1), dtype=np.float32),
        has_improvement=np.zeros(count, dtype=np.uint8),
        residual_scale=np.asarray(0.03, dtype=np.float32),
    )


def test_merge_uses_old_train_and_new_valid_with_namespaced_train_demos(tmp_path):
    training = tmp_path / "training.npz"
    holdout = tmp_path / "holdout.npz"
    _dataset(training, ["train", "valid"], ["demo_0", "demo_1"], [1, 2])
    _dataset(holdout, ["train", "valid"], ["demo_1", "demo_0"], [3, 4])

    report = merge(training, holdout, tmp_path / "merged.npz")

    assert report["train_rows"] == 1
    assert report["valid_rows"] == 1
    with np.load(tmp_path / "merged.npz", allow_pickle=False) as merged:
        assert merged["source_demo"].tolist() == ["frozen_train:demo_0", "demo_0"]
        np.testing.assert_array_equal(merged["state"], [[1], [4]])
