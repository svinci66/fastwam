import json

from experiments.robotwin.export_paired_expert_imagination_trajectories import (
    load_natural_failure_cases,
)
from experiments.robotwin.score_natural_failure_vae_pairs import (
    aggregate_replan_scores,
)


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
