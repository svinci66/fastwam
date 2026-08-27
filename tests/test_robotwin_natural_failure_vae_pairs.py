import json

import pytest

from experiments.robotwin.export_paired_expert_imagination_trajectories import (
    load_natural_failure_cases,
)
from experiments.robotwin.score_natural_failure_vae_pairs import (
    aggregate_replan_scores,
    select_episode_reward,
)
from experiments.robotwin.summarize_natural_failure_vae_rewards import summarize


def test_case_loader_keeps_only_natural_failures(tmp_path):
    path = tmp_path / "status.jsonl"
    rows = [
        {
            "task": "open_microwave",
            "instruction": "open it",
            "expert_hdf5": "/tmp/expert.hdf5",
            "evaluation_episode_id": 2,
            "environment_seed": 6,
            "decision": "natural_failure",
        },
        {
            "task": "turn_switch",
            "instruction": "turn it",
            "expert_hdf5": "/tmp/expert2.hdf5",
            "evaluation_episode_id": 1,
            "environment_seed": 1,
            "decision": "fastwam_success",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert load_natural_failure_cases(path) == [rows[0]]
    assert load_natural_failure_cases(path, tasks={"open_microwave"}) == [rows[0]]


def test_aggregate_replan_scores_uses_equal_camera_and_time_means():
    rows = [
        {
            "equal_camera_mean": 0.2,
            "camera_scores": {"head": 0.1, "left_wrist": 0.2, "right_wrist": 0.3},
        },
        {
            "equal_camera_mean": 0.4,
            "camera_scores": {"head": 0.3, "left_wrist": 0.4, "right_wrist": 0.5},
        },
    ]
    result = aggregate_replan_scores(rows)
    assert result["mean_reward"] == 0.30000000000000004
    assert result["camera_scores"] == {
        "head": 0.2,
        "left_wrist": 0.30000000000000004,
        "right_wrist": 0.4,
    }


def test_select_episode_reward_uses_only_frozen_camera_subset():
    episode = {
        "camera_scores": {
            "head": 0.9,
            "left_wrist": -0.3,
            "right_wrist": -0.6,
        }
    }
    assert select_episode_reward(episode, ("head",)) == 0.9
    assert select_episode_reward(episode, ("head", "left_wrist")) == pytest.approx(0.3)


def test_summary_filters_task_and_recomputes_camera_rankings(tmp_path):
    def pair(task, seed, expert, failure):
        return {
            "task": task,
            "environment_seed": seed,
            "success_minus_failure": expert - failure,
            "expert_success": {
                "camera_scores": {
                    "head": expert,
                    "left_wrist": expert,
                    "right_wrist": expert,
                }
            },
            "fastwam_failure": {
                "camera_scores": {
                    "head": failure,
                    "left_wrist": failure,
                    "right_wrist": failure,
                }
            },
        }

    path = tmp_path / "reward.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "robotwin_natural_failure_wan_vae_pair_reward_v1",
                "pairs": [
                    pair("open_microwave", 1, 0.4, 0.2),
                    pair("turn_switch", 2, 0.1, 0.3),
                ],
            }
        ),
        encoding="utf-8",
    )
    result = summarize([path], task="open_microwave")
    assert result["pair_count"] == 1
    assert result["pairwise_accuracy"] == 1.0
    assert result["per_camera_pairwise"]["head"]["correctly_ranked_count"] == 1
