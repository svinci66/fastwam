from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.collect_can_counterfactual_branches import (
    _make_candidate_actions,
    _select_source,
    _set_or_validate_attribute,
)


def test_candidate_actions_perturb_pose_but_preserve_gripper():
    actions = np.zeros((5, 7), dtype=np.float32)
    actions[:, -1] = 1.0
    candidate = _make_candidate_actions(
        actions,
        rng=np.random.default_rng(7),
        sigma=0.2,
        intervention_steps=2,
        perturb_gripper=False,
    )

    assert np.any(candidate[:2, :6] != actions[:2, :6])
    np.testing.assert_array_equal(candidate[2:], actions[2:])
    np.testing.assert_array_equal(candidate[:, -1], actions[:, -1])
    assert np.max(np.abs(candidate)) <= 1.0


def test_source_selection_respects_full_horizon():
    rng = np.random.default_rng(11)
    lengths = {"demo_0": 30}
    for _ in range(100):
        name, step = _select_source(
            rng=rng,
            demo_names=["demo_0"],
            action_lengths=lengths,
            horizon=20,
            late_state_fraction=0.5,
        )
        assert name == "demo_0"
        assert 0 <= step <= 10


def test_resume_attribute_validation_rejects_mixed_configuration(tmp_path):
    import h5py
    import pytest

    path = tmp_path / "collection.hdf5"
    with h5py.File(path, "w") as output:
        _set_or_validate_attribute(output, "horizon", 20)
        _set_or_validate_attribute(output, "horizon", 20)
        with pytest.raises(ValueError, match="horizon"):
            _set_or_validate_attribute(output, "horizon", 40)
