import json

import numpy as np
from PIL import Image

from experiments.libero.imagination_reward_utils import (
    apply_action_mode,
    compute_progress_reward,
    save_aligned_transition,
)
from experiments.libero.analyze_imagination_rewards import (
    build_episode_rows,
    select_wrong_goal_record,
    summarize,
)
from experiments.libero.scan_imagination_reward_weights import scan_weights


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


def test_wrong_goal_comes_from_distant_same_mode_episode():
    current = {
        "source_input_dir": "policy",
        "task_suite": "libero_goal",
        "task_id": 3,
        "trial_idx": 0,
        "action_mode": "policy",
        "replan_idx": 1,
        "record_dir": "current",
    }
    same_mode = {
        **current,
        "trial_idx": 1,
        "replan_idx": 8,
        "record_dir": "same-mode",
    }
    other_mode = {
        **current,
        "source_input_dir": "noise",
        "trial_idx": 1,
        "action_mode": "noise",
        "replan_idx": 20,
        "record_dir": "other-mode",
    }
    assert select_wrong_goal_record(current, [current, same_mode, other_mode]) is same_mode


def test_episode_rows_include_return_mean_and_steps():
    common = {
        "source_input_dir": "policy",
        "task_suite": "libero_goal",
        "task_id": 3,
        "trial_idx": 0,
        "action_mode": "policy",
        "success": True,
        "episode_policy_steps": 16,
    }
    episodes = build_episode_rows(
        [
            {**common, "imagination_progress": 0.25},
            {**common, "imagination_progress": -0.05},
        ]
    )
    assert len(episodes) == 1
    assert episodes[0]["episode_imagination_return"] == 0.2
    assert episodes[0]["episode_mean_imagination_progress"] == 0.1
    assert episodes[0]["episode_policy_steps"] == 16


def test_summary_reports_paired_action_quality_order():
    episode_rows = []
    for trial_idx in range(2):
        for mode, progress, success in (
            ("policy", 0.3, True),
            ("noise", 0.2, False),
            ("zero", 0.0, False),
        ):
            episode_rows.append(
                {
                    "task_suite": "libero_goal",
                    "task_id": 3,
                    "trial_idx": trial_idx,
                    "action_mode": mode,
                    "success": success,
                    "episode_policy_steps": 8,
                    "episode_imagination_return": progress,
                    "episode_mean_imagination_progress": progress,
                }
            )
    summary = summarize([], episode_rows)
    assert summary["num_fully_paired_trials"] == 2
    assert summary["paired_policy_gt_noise_gt_zero_fraction"] == 1.0
    assert summary["episode_success_rate_by_action_mode"]["policy"] == 1.0


def test_weight_scan_selects_smallest_passing_match_weight():
    rows = []
    for trial_idx in range(2):
        for mode, progress, distance_after, success in (
            ("policy", 0.30, 0.20, True),
            ("noise", 0.20, 0.30, False),
            ("zero", 0.00, 0.40, False),
        ):
            rows.append(
                {
                    "task_suite": "libero_goal",
                    "task_id": 3,
                    "trial_idx": trial_idx,
                    "action_mode": mode,
                    "success": success,
                    "imagination_progress": progress,
                    "distance_after": distance_after,
                    "wrong_goal_imagination_progress": progress + 0.05,
                    "wrong_goal_distance_after": distance_after + 0.20,
                }
            )

    result = scan_weights(rows, weights=[0.0, 0.5, 1.0], goal_specificity_threshold=0.7)

    assert result["candidates"][0]["correct_goal_beats_wrong_fraction"] == 0.0
    assert result["minimum_passing_candidate"]["match_weight"] == 0.5
    assert result["selected_candidate"]["match_weight"] == 0.5
    assert result["selected_candidate"]["paired_policy_gt_noise_gt_zero_fraction"] == 1.0


def test_weight_scan_uses_fixed_external_zero_reference():
    rows = []
    for mode, progress, distance_after, success in (
        ("policy", 0.30, 0.20, True),
        ("noise", 0.20, 0.30, False),
        ("zero", 0.00, 0.40, False),
    ):
        rows.append(
            {
                "task_suite": "libero_goal",
                "task_id": 3,
                "trial_idx": 0,
                "action_mode": mode,
                "success": success,
                "imagination_progress": progress,
                "distance_after": distance_after,
                "wrong_goal_imagination_progress": progress,
                "wrong_goal_distance_after": distance_after + 0.10,
            }
        )

    result = scan_weights(
        rows,
        weights=[0.05],
        goal_specificity_threshold=0.7,
        fixed_zero_reference=0.45,
    )

    assert result["zero_action_distance_reference_source"] == "fixed_external_calibration"
    assert result["zero_action_distance_reference_by_task"]["libero_goal/task_3"] == 0.45
    assert result["candidates"][0]["paired_trial_rewards"][0][
        "policy_gt_noise_gt_zero"
    ]
