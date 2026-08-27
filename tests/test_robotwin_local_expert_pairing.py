import numpy as np

from experiments.robotwin.collect_local_expert_pair_episode import (
    compare_scene_states,
    task_selectors,
)
from experiments.robotwin.validate_local_expert_fastwam_pairs import comparison_metrics


class _FakeTask:
    def __init__(self) -> None:
        self.mug_id = np.int64(7)
        self.mug_name = "039_mug"
        self.unrelated = object()


def test_task_selectors_keeps_replay_critical_ids_and_names() -> None:
    assert task_selectors(_FakeTask()) == {"mug_id": 7, "mug_name": "039_mug"}


def test_scene_state_comparison_rejects_selector_mismatch() -> None:
    base = {
        "selectors": {"mug_id": 7},
        "actors": [
            {"name": "mug", "position": [0, 0, 1], "quaternion": [1, 0, 0, 0]}
        ],
        "robot_joint_vector": [0, 1],
    }
    changed = {**base, "selectors": {"mug_id": 2}}
    assert not compare_scene_states(base, changed)["exact"]


def test_scene_state_comparison_accepts_tiny_pose_noise() -> None:
    left = {
        "selectors": {"mug_id": 7},
        "actors": [
            {"name": "mug", "position": [0, 0, 1], "quaternion": [1, 0, 0, 0]}
        ],
        "robot_joint_vector": [0, 1],
    }
    right = {
        **left,
        "actors": [
            {
                "name": "mug",
                "position": [0, 0, 1.000001],
                "quaternion": [1, 0, 0, 0],
            }
        ],
    }
    assert compare_scene_states(left, right, atol=1e-5)["exact"]


def test_visual_comparison_is_zero_for_identical_images() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert comparison_metrics(image, image)["blurred_mean_abs"] == 0.0
