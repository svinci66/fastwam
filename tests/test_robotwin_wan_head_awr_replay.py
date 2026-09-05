import numpy as np
import pytest

from experiments.robotwin.build_wan_vae_head_awr_replay import (
    fit_paired_rank_discount_normalization,
    fit_episode_balanced_normalization,
    normalized_score,
    select_reward_tasks,
    validate_reward_payload,
)
from fastwam.rl.rewards import compute_imagination_reward


def test_head_reward_protocol_and_episode_balanced_normalization():
    validate_reward_payload(
        {
            "schema_version": "robotwin_natural_failure_wan_vae_pair_reward_v1",
            "feature_encoder": "wan2.2_vae_single_frame_spatial_latent",
            "trajectory_reference_policy": "frozen_once_per_action_chunk",
            "time_offsets": [0, 4, 8, 12, 16, 20, 24],
            "reward_cameras": ["head"],
            "pairwise_accuracy": 1.0,
        }
    )
    records = [
        {"episode_key": "short", "alignment_valid": True, "wan_head_score": 1.0},
        {"episode_key": "long", "alignment_valid": True, "wan_head_score": 0.0},
        {"episode_key": "long", "alignment_valid": True, "wan_head_score": 0.0},
        {"episode_key": "long", "alignment_valid": True, "wan_head_score": 0.0},
        {"episode_key": "long", "alignment_valid": True, "wan_head_score": 1.0},
        {"episode_key": "long", "alignment_valid": False, "wan_head_score": 99.0},
    ]
    normalization = fit_episode_balanced_normalization(records)
    assert normalization["num_episodes"] == 2
    assert normalization["num_valid_scores"] == 5
    low = normalized_score(0.0, normalization, 0.1)
    high = normalized_score(1.0, normalization, 0.1)
    assert np.isfinite([low, high]).all()
    assert -0.1 <= low < high <= 0.1


def test_multitask_audit_can_use_a_preregistered_ranking_threshold():
    payload = {
        "schema_version": "robotwin_natural_failure_wan_vae_pair_reward_v1",
        "feature_encoder": "wan2.2_vae_single_frame_spatial_latent",
        "trajectory_reference_policy": "frozen_once_per_action_chunk",
        "time_offsets": [0, 4, 8, 12, 16, 20, 24],
        "reward_cameras": ["head"],
        "pairwise_accuracy": 0.75,
    }
    validate_reward_payload(payload, minimum_pairwise_accuracy=0.60)
    with pytest.raises(ValueError, match="pair-ranking threshold"):
        validate_reward_payload(payload)


def test_head_trajectory_reward_cannot_silently_fall_back_to_endpoint_reward():
    features = {"head": np.asarray([1.0, 0.0], dtype=np.float32)}
    with pytest.raises(ValueError, match="precomputed trajectory reward label"):
        compute_imagination_reward(
            features,
            features,
            features,
            reward_type="wan_vae_head_trajectory_global_norm_v1",
        )


def test_task_filter_recomputes_pairwise_quality_before_replay_building():
    payload = {
        "pair_count": 3,
        "correctly_ranked_count": 2,
        "pairwise_accuracy": 2 / 3,
        "mean_success_minus_failure": 0.0,
        "per_task": {"medium": {"pair_count": 2}, "retention": {"pair_count": 1}},
        "pairs": [
            {
                "task": "medium",
                "correctly_ranked": True,
                "success_minus_failure": 0.2,
            },
            {
                "task": "medium",
                "correctly_ranked": True,
                "success_minus_failure": 0.1,
            },
            {
                "task": "retention",
                "correctly_ranked": False,
                "success_minus_failure": -0.3,
            },
        ],
    }
    selected = select_reward_tasks(payload, ["medium"])
    assert selected["pair_count"] == 2
    assert selected["correctly_ranked_count"] == 2
    assert selected["pairwise_accuracy"] == 1.0
    assert selected["mean_success_minus_failure"] == pytest.approx(0.15)
    assert set(selected["per_task"]) == {"medium"}
    assert payload["pair_count"] == 3


def test_task_filter_rejects_missing_or_duplicate_tasks():
    payload = {
        "pairs": [
            {"task": "medium", "correctly_ranked": True, "success_minus_failure": 0.1}
        ]
    }
    with pytest.raises(ValueError, match="absent"):
        select_reward_tasks(payload, ["missing"])
    with pytest.raises(ValueError, match="duplicate"):
        select_reward_tasks(payload, ["medium", "medium"])


def test_paired_rank_discount_normalization_preserves_signed_initial_return(tmp_path):
    payload = {
        "pairs": [
            {
                "task": "place_can_basket",
                "episode_id": 0,
                "success_minus_failure": 0.1,
            }
        ]
    }
    records = []
    for behavior, length in (("expert", 2), ("policy", 4)):
        for replan in range(length):
            records.append(
                {
                    "episode_key": f"place-pair0000-{behavior}",
                    "task_name": "place_can_basket",
                    "pair_episode_id": 0,
                    "behavior": behavior,
                    "replan_idx": replan,
                    "effective_k": 24,
                    "alignment_valid": True,
                    "record_dir": tmp_path / behavior / str(replan),
                }
            )
    shaping, summary = fit_paired_rank_discount_normalization(
        records, payload, gamma=0.99, clip=0.1
    )

    realized = {}
    for behavior, length in (("expert", 2), ("policy", 4)):
        discount = 1.0
        value = 0.0
        for replan in range(length):
            value += discount * shaping[str((tmp_path / behavior / str(replan)).resolve())]
            discount *= 0.99**24
        realized[behavior] = value
    assert realized["expert"] == pytest.approx(0.1 * np.tanh(1.0))
    assert realized["policy"] == pytest.approx(-realized["expert"])
    assert summary["num_episodes"] == 2
    assert summary["positive_pair_count"] == 1


def test_paired_rank_discount_normalization_zeroes_nonpositive_pair(tmp_path):
    payload = {
        "pairs": [
            {"task": "task", "episode_id": 0, "success_minus_failure": 0.2},
            {"task": "task", "episode_id": 1, "success_minus_failure": -0.1},
        ]
    }
    records = []
    for pair in range(2):
        for behavior in ("expert", "policy"):
            records.append(
                {
                    "episode_key": f"task-pair{pair:04d}-{behavior}",
                    "task_name": "task",
                    "pair_episode_id": pair,
                    "behavior": behavior,
                    "replan_idx": 0,
                    "effective_k": 24,
                    "alignment_valid": True,
                    "record_dir": tmp_path / str(pair) / behavior,
                }
            )
    shaping, summary = fit_paired_rank_discount_normalization(
        records, payload, gamma=0.99, clip=0.1
    )
    assert shaping[str((tmp_path / "1" / "expert").resolve())] == 0.0
    assert shaping[str((tmp_path / "1" / "policy").resolve())] == 0.0
    assert summary["zeroed_nonpositive_pair_count"] == 1
