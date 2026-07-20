import json

import numpy as np
from PIL import Image

from experiments.libero.imagination_reward_utils import (
    apply_action_mode,
    build_matched_direction_action_branches,
    compute_delta_alignment_reward,
    compute_progress_reward,
    save_aligned_transition,
    split_horizontal_camera_views,
)
from experiments.libero.analyze_imagination_rewards import (
    build_episode_rows,
    select_wrong_goal_record,
    summarize,
)
from experiments.libero.scan_imagination_reward_weights import scan_weights
from experiments.libero.diagnose_camera_delta_rewards import (
    assign_temporal_phases,
    binary_auc,
    compute_candidate_metrics,
)
from experiments.libero.calibrate_camera_reward_weights import (
    calibrate_camera_scales,
    compute_weighted_reward,
    extract_camera_rewards,
    select_calibration_candidate,
)
from experiments.libero.validate_same_phase_hard_negatives import (
    combine_rewards,
    select_same_phase_hard_negative,
)


def test_progress_reward_is_positive_when_actual_moves_toward_goal():
    metrics = compute_progress_reward(
        current_feature=np.array([1.0, 0.0]),
        actual_feature=np.array([1.0, 1.0]),
        goal_feature=np.array([0.0, 1.0]),
    )
    assert metrics["distance_after"] < metrics["distance_before"]
    assert metrics["imagination_progress"] > 0


def test_horizontal_camera_split_preserves_agent_then_wrist_order():
    agent = np.full((4, 4, 3), 17, dtype=np.uint8)
    wrist = np.full((4, 4, 3), 231, dtype=np.uint8)
    views = split_horizontal_camera_views(np.concatenate([agent, wrist], axis=1))
    np.testing.assert_array_equal(views["agent"], agent)
    np.testing.assert_array_equal(views["wrist"], wrist)


def test_horizontal_camera_split_rejects_unknown_layout():
    malformed = np.zeros((4, 7, 3), dtype=np.uint8)
    try:
        split_horizontal_camera_views(malformed)
    except ValueError as error:
        assert "width == 2 * height" in str(error)
    else:
        raise AssertionError("Expected malformed camera layout to be rejected")


def test_delta_alignment_rewards_matching_change_and_suppresses_noop():
    aligned = compute_delta_alignment_reward(
        current_feature=np.array([0.0, 0.0]),
        actual_feature=np.array([1.0, 0.0]),
        goal_feature=np.array([2.0, 0.0]),
    )
    opposite = compute_delta_alignment_reward(
        current_feature=np.array([0.0, 0.0]),
        actual_feature=np.array([-1.0, 0.0]),
        goal_feature=np.array([2.0, 0.0]),
    )
    noop = compute_delta_alignment_reward(
        current_feature=np.array([0.0, 0.0]),
        actual_feature=np.array([0.0, 0.0]),
        goal_feature=np.array([2.0, 0.0]),
    )
    assert aligned["direction_alignment"] == 1.0
    assert aligned["delta_alignment_reward"] == 0.5
    assert opposite["delta_alignment_reward"] == -0.5
    assert noop["delta_alignment_reward"] == 0.0


def test_candidate_metrics_use_equal_dual_camera_mean_and_fixed_weight():
    features = {
        view: {
            "current": np.array([1.0, 1.0]),
            "actual": np.array([2.0, 1.0]),
            "goal": np.array([3.0, 1.0]),
        }
        for view in ("concat", "agent", "wrist")
    }
    features["wrist"]["actual"] = np.array([0.0, 1.0])
    metrics = compute_candidate_metrics(
        features,
        current_path="current",
        actual_path="actual",
        goal_path="goal",
        direction_weight=0.01,
    )
    assert metrics["candidates"]["agent_delta_alignment"] == 0.5
    assert metrics["candidates"]["dual_delta_alignment"] == 0.0
    assert (
        metrics["candidates"]["concat_progress_plus_dual_delta"]
        == metrics["candidates"]["concat_progress"]
    )


def test_temporal_phases_are_relative_to_each_episode():
    records = []
    for replan_idx in range(7):
        records.append(
            {
                "policy_seed": 42,
                "task_suite": "libero_goal",
                "task_id": 3,
                "trial_idx": 0,
                "action_mode": "policy",
                "replan_idx": replan_idx,
                "record_dir": f"record-{replan_idx}",
            }
        )
    phases = assign_temporal_phases(records)
    assert phases["record-0"] == "early"
    assert phases["record-3"] == "middle"
    assert phases["record-6"] == "late"


def test_binary_auc_is_tie_aware():
    assert binary_auc([False, False, True, True], [0.0, 0.5, 0.5, 1.0]) == 0.875
    assert binary_auc([True, True], [0.0, 1.0]) is None


def test_camera_weight_calibration_recovers_wrist_and_uses_only_calibration_seed():
    def row(seed, mode, agent, wrist):
        return {
            "policy_seed": seed,
            "action_mode": mode,
            "candidate_rewards": {
                "agent_delta_alignment": agent,
                "dual_delta_alignment": 0.5 * (agent + wrist),
            },
        }

    rows = [
        row(42, "policy", 2.0, 1.0),
        row(42, "noise", 1.0, 0.5),
        row(42, "zero", 100.0, 100.0),
        row(1042, "policy", 1000.0, 1000.0),
    ]
    assert extract_camera_rewards(rows[0]) == {"agent": 2.0, "wrist": 1.0}
    scales = calibrate_camera_scales(rows, calibration_seed=42, quantile=1.0)
    assert scales == {"agent": 2.0, "wrist": 1.0}
    reward = compute_weighted_reward(
        rows[0],
        agent_weight=0.6,
        scales=scales,
    )
    assert reward == 1.0


def test_camera_weight_selection_requires_all_calibration_gates():
    def candidate(weight, goal, auc=1.0, order=1.0, success_pairs=1.0):
        return {
            "agent_weight": weight,
            "wrist_weight": 1.0 - weight,
            "correct_goal_beats_wrong_fraction": goal,
            "episode_success_roc_auc": auc,
            "paired_policy_gt_noise_gt_zero_fraction": order,
            "successful_mode_beats_failed_mode_fraction": success_pairs,
        }

    result = select_calibration_candidate(
        [
            candidate(0.4, 0.75, order=0.8),
            candidate(0.5, 0.71),
            candidate(0.6, 0.73),
        ],
        goal_specificity_threshold=0.70,
        success_auc_threshold=0.90,
    )
    assert result["selection_status"] == "passed_calibration_gates"
    assert result["num_passing_candidates"] == 2
    assert result["selected_agent_weight"] == 0.6


def test_same_phase_hard_negative_uses_near_current_but_different_goal():
    def row(trial, phase, current, goal):
        return {
            "policy_seed": 42,
            "task_suite": "libero_goal",
            "task_id": 3,
            "trial_idx": trial,
            "action_mode": "policy",
            "phase": phase,
            "record_dir": f"record-{trial}-{phase}",
            "current_path": current,
            "goal_path": goal,
        }

    target = row(0, "early", "current-0", "goal-0")
    similar_goal = row(1, "early", "current-1", "goal-1")
    different_goal = row(2, "early", "current-2", "goal-2")
    wrong_phase = row(3, "late", "current-3", "goal-3")
    vectors = {
        "current-0": np.array([1.0, 0.0]),
        "current-1": np.array([0.995, 0.1]),
        "current-2": np.array([0.98, 0.2]),
        "current-3": np.array([1.0, 0.0]),
        "goal-0": np.array([0.0, 1.0]),
        "goal-1": np.array([0.1, 0.995]),
        "goal-2": np.array([0.0, -1.0]),
        "goal-3": np.array([0.0, -1.0]),
    }
    features = {camera: vectors for camera in ("agent", "wrist")}
    selected = select_same_phase_hard_negative(
        target,
        [target, similar_goal, different_goal, wrong_phase],
        features,
        nearest_k=2,
    )
    assert selected is not None
    assert selected["record"] is different_goal
    assert selected["candidate_pool_size"] == 2
    assert selected["neighborhood_size"] == 2


def test_frozen_camera_combination_uses_calibrated_scales_and_weights():
    combined = combine_rewards(
        {"agent": 2.0, "wrist": 1.0},
        agent_weight=0.25,
        scales={"agent": 2.0, "wrist": 0.5},
    )
    assert combined["raw_dual"] == 1.5
    assert combined["frozen_normalized"] == 1.75


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


def test_matched_direction_branches_only_reverse_translation_direction():
    action = np.array(
        [
            [0.6, 0.8, 0.0, 0.1, 0.2, 0.3, -1.0],
            [0.03, 0.04, 0.0, -0.1, -0.2, -0.3, 1.0],
        ],
        dtype=np.float32,
    )
    branches = build_matched_direction_action_branches(
        action,
        np.array([0.0, 0.0, -2.0]),
        translation_magnitude_cap=0.25,
    )
    toward = branches["toward_bowl"]
    away = branches["away_from_bowl"]
    np.testing.assert_allclose(toward[:, :3], -away[:, :3], atol=1e-7)
    np.testing.assert_allclose(
        np.linalg.norm(toward[:, :3], axis=1),
        np.array([0.25, 0.05]),
        atol=1e-7,
    )
    np.testing.assert_array_equal(toward[:, 3:], action[:, 3:])
    np.testing.assert_array_equal(away[:, 3:], action[:, 3:])
    np.testing.assert_array_equal(branches["policy"], action)
    np.testing.assert_array_equal(branches["zero"][:, :-1], 0.0)
    np.testing.assert_array_equal(branches["zero"][:, -1], -1.0)


def test_matched_direction_branches_reject_zero_direction():
    try:
        build_matched_direction_action_branches(
            np.zeros((2, 7), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
    except ValueError as error:
        assert "non-zero" in str(error)
    else:
        raise AssertionError("Expected zero direction to be rejected")


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
