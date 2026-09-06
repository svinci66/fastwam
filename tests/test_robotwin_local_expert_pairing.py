import numpy as np

from experiments.robotwin.collect_local_expert_pair_episode import (
    compare_scene_states,
    is_infeasible_planning_exception,
    required_stratum_matches,
    task_selectors,
)
from experiments.robotwin.validate_local_expert_fastwam_pairs import comparison_metrics


class _FakeTask:
    def __init__(self) -> None:
        self.mug_id = np.int64(7)
        self.mug_name = "039_mug"
        self.unrelated = object()


class _ArmTag:
    def __str__(self) -> str:
        return "left"


class _FakePlaceTask:
    def __init__(self) -> None:
        self.basket_id = np.int64(0)
        self.basket_name = "110_basket"
        self.arm_tag = _ArmTag()


def test_task_selectors_keeps_replay_critical_ids_and_names() -> None:
    assert task_selectors(_FakeTask()) == {"mug_id": 7, "mug_name": "039_mug"}


def test_task_selectors_records_actual_arm_for_targeted_collection() -> None:
    assert task_selectors(_FakePlaceTask()) == {
        "basket_id": 0,
        "basket_name": "110_basket",
        "arm_tag": "left",
    }


def test_required_stratum_matches_reset_selectors_only() -> None:
    selectors = {"basket_id": 0, "arm_tag": "left"}
    assert required_stratum_matches(selectors, basket_id=0, arm_tag="left")
    assert not required_stratum_matches(selectors, basket_id=1, arm_tag="left")
    assert not required_stratum_matches(selectors, basket_id=0, arm_tag="right")


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


def test_missing_grasp_target_is_a_skippable_planning_failure() -> None:
    error = AssertionError("target_pose cannot be None for move action.")
    assert is_infeasible_planning_exception(error)


def test_unexpected_assertion_remains_fatal() -> None:
    assert not is_infeasible_planning_exception(AssertionError("unexpected bug"))
