import numpy as np
import pytest

from experiments.robotwin.build_residual_rl_replay import (
    combine_camera_features,
    pad_action_chunk,
    pad_environment_rewards,
    validate_episode_records,
)
from experiments.robotwin.imagination_reward_utils import ROBOTWIN_CAMERA_NAMES


def test_combine_camera_features_uses_all_three_views() -> None:
    features = {
        camera: np.full(4, index + 1, dtype=np.float32)
        for index, camera in enumerate(ROBOTWIN_CAMERA_NAMES)
    }
    combined = combine_camera_features(features)
    assert combined.shape == (12,)
    assert np.linalg.norm(combined) == pytest.approx(1.0)
    assert np.all(combined[:4] < combined[4:8])
    assert np.all(combined[4:8] < combined[8:])


def test_pad_partial_robotwin_chunk_repeats_last_action_and_zeros_rewards() -> None:
    actions = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    padded = pad_action_chunk(actions, 4, name="actions")
    assert padded.tolist() == [[1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]
    rewards = pad_environment_rewards(np.asarray([0.0, 1.0]), 4)
    assert rewards.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_validate_robotwin_episode_requires_contiguous_final_boundary() -> None:
    records = [
        {
            "task_name": "task",
            "behavior": "policy",
            "trial_idx": 0,
            "replan_idx": 0,
            "terminated": False,
            "truncated": False,
        },
        {
            "task_name": "task",
            "behavior": "policy",
            "trial_idx": 0,
            "replan_idx": 1,
            "terminated": True,
            "truncated": False,
        },
    ]
    validate_episode_records(records)
    records[-1]["replan_idx"] = 2
    with pytest.raises(ValueError, match="non-contiguous"):
        validate_episode_records(records)
