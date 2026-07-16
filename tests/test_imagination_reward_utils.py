import json

import numpy as np
from PIL import Image

from experiments.libero.imagination_reward_utils import (
    apply_action_mode,
    compute_progress_reward,
    save_aligned_transition,
)


def test_progress_reward_is_positive_when_actual_moves_toward_goal():
    metrics = compute_progress_reward(
        current_feature=np.array([1.0, 0.0]),
        actual_feature=np.array([1.0, 1.0]),
        goal_feature=np.array([0.0, 1.0]),
    )
    assert metrics["distance_after"] < metrics["distance_before"]
    assert metrics["imagination_progress"] > 0


def test_zero_action_mode_produces_libero_noop():
    action = np.ones((8, 7), dtype=np.float32)
    result = apply_action_mode(action, mode="zero")
    np.testing.assert_array_equal(result[:, :-1], 0.0)
    np.testing.assert_array_equal(result[:, -1], -1.0)
    np.testing.assert_array_equal(action, 1.0)


def test_noise_action_mode_is_seeded_and_preserves_gripper():
    action = np.zeros((8, 7), dtype=np.float32)
    action[:, -1] = 1.0
    first = apply_action_mode(action, "noise", noise_std=0.2, rng=np.random.default_rng(7))
    second = apply_action_mode(action, "noise", noise_std=0.2, rng=np.random.default_rng(7))
    np.testing.assert_allclose(first, second)
    np.testing.assert_array_equal(first[:, -1], action[:, -1])
    assert not np.allclose(first[:, :-1], action[:, :-1])


def test_save_aligned_transition_writes_lossless_triplet(tmp_path):
    frame = Image.fromarray(np.full((8, 12, 3), 127, dtype=np.uint8))
    metadata_path = save_aligned_transition(
        tmp_path / "transition",
        current_frame=frame,
        predicted_goal_frame=frame,
        actual_frame=frame,
        metadata={"alignment_valid": True},
    )
    assert Image.open(metadata_path.parent / "current.png").size == (12, 8)
    assert Image.open(metadata_path.parent / "predicted_goal.png").size == (12, 8)
    assert Image.open(metadata_path.parent / "actual.png").size == (12, 8)
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["alignment_valid"] is True
