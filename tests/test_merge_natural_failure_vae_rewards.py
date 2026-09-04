from copy import deepcopy
from pathlib import Path

import pytest

from experiments.robotwin.merge_natural_failure_vae_rewards import (
    merge_reward_payloads,
)


def _pair(task: str, seed: int, margin: float):
    failure_head = 0.2
    return {
        "task": task,
        "episode_id": seed,
        "environment_seed": seed,
        "success_minus_failure": margin,
        "correctly_ranked": margin > 0.0,
        "expert_success": {
            "camera_scores": {
                "head": failure_head + margin,
                "left_wrist": 0.3,
                "right_wrist": 0.4,
            }
        },
        "fastwam_failure": {
            "camera_scores": {
                "head": failure_head,
                "left_wrist": 0.2,
                "right_wrist": 0.5,
            }
        },
    }


def _payload(*pairs):
    return {
        "schema_version": "robotwin_natural_failure_wan_vae_pair_reward_v1",
        "feature_encoder": "wan2.2_vae_single_frame_spatial_latent",
        "trajectory_reference_policy": "frozen_once_per_action_chunk",
        "time_offsets": [0, 4, 8, 12, 16, 20, 24],
        "reward_cameras": ["head"],
        "camera_weights": {"head": 1.0, "left_wrist": 0.0, "right_wrist": 0.0},
        "latent_shape": [48, 24, 20],
        "pair_count": len(pairs),
        "pairs": list(pairs),
        "unique_encoded_frames": 10,
    }


def test_merge_filters_tasks_and_recomputes_rankings():
    result = merge_reward_payloads(
        [
            (Path("old.json"), _payload(_pair("a", 1, 0.2), _pair("x", 2, 0.9))),
            (Path("new.json"), _payload(_pair("a", 3, -0.1), _pair("b", 4, 0.3))),
        ],
        tasks=["a", "b"],
    )
    assert result["pair_count"] == 3
    assert result["correctly_ranked_count"] == 2
    assert result["pairwise_accuracy"] == pytest.approx(2 / 3)
    assert result["per_task"]["a"]["pair_count"] == 2
    assert result["per_task"]["b"]["pairwise_accuracy"] == 1.0
    assert result["selected_tasks"] == ["a", "b"]
    assert result["merge_provenance"]["pair_rewards_recomputed"] is False
    assert result["unique_encoded_frames"] is None
    assert [
        (row["task"], row["episode_id"], row["source_episode_id"])
        for row in result["pairs"]
    ] == [("a", 0, 1), ("a", 1, 3), ("b", 0, 4)]


def test_merge_rejects_duplicate_task_seed():
    with pytest.raises(ValueError, match="duplicate task/environment-seed"):
        merge_reward_payloads(
            [
                (Path("old.json"), _payload(_pair("a", 1, 0.2))),
                (Path("new.json"), _payload(_pair("a", 1, 0.1))),
            ],
            tasks=["a"],
        )


def test_merge_rejects_incompatible_protocol():
    incompatible = deepcopy(_payload(_pair("b", 2, 0.1)))
    incompatible["reward_cameras"] = ["left_wrist"]
    with pytest.raises(ValueError, match="incompatible reward protocol"):
        merge_reward_payloads(
            [
                (Path("old.json"), _payload(_pair("a", 1, 0.2))),
                (Path("new.json"), incompatible),
            ],
            tasks=[],
        )
