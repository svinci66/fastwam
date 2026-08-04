import json

import pytest

from experiments.robotwin.summarize_residual_iql_online_pair import (
    load_episode_initial_hashes,
    parse_log,
)
from experiments.robotwin.summarize_residual_iql_seed_matrix import (
    merge_episode_hashes,
)


def test_parse_online_pair_log_extracts_success_and_residual_metrics(tmp_path):
    log = tmp_path / "eval_task.log"
    log.write_text(
        "\x1b[92mSuccess rate: \x1b[96m4/5\x1b[0m => \x1b[95m80.0%\x1b[0m\n"
        "[fastwam-residual] replan=0 rms=0.010000 max_abs=0.040000 "
        "gripper_max_abs=0.000000 gate_applied=1 q_advantage_min=0.020000 "
        "q_advantage_disagreement=0.010000 gate_approved=1 shadow_mode=1 "
        "circuit_breaker_active=0 circuit_breaker_triggered=0 "
        "support_state_score=0.100000 support_state_threshold=0.200000 "
        "support_action_score=0.200000 support_action_threshold=0.300000 "
        "support_in_distribution=1 support_language_similarity=1.000000 "
        "intervention_allowed=1 intervention_count=1 "
        "intervention_budget_remaining=1 intervention_budget_exhausted=0\n"
        "[fastwam-residual] replan=1 rms=0.020000 max_abs=0.050000 "
        "gripper_max_abs=0.000000 gate_applied=0 q_advantage_min=-0.010000 "
        "q_advantage_disagreement=0.030000 gate_approved=0 shadow_mode=1 "
        "circuit_breaker_active=1 circuit_breaker_triggered=1 "
        "support_state_score=0.400000 support_state_threshold=0.200000 "
        "support_action_score=0.200000 support_action_threshold=0.300000 "
        "support_in_distribution=0 support_language_similarity=1.000000 "
        "intervention_allowed=0 intervention_count=2 "
        "intervention_budget_remaining=0 intervention_budget_exhausted=1\n",
        encoding="utf-8",
    )
    result = parse_log(log)
    assert result["successes"] == 4
    assert result["episodes"] == 5
    assert result["success_rate"] == 0.8
    assert result["num_residual_replans"] == 2
    assert result["residual_rms_mean"] == 0.015
    assert result["residual_max_abs"] == 0.05
    assert result["gripper_residual_max_abs"] == 0.0
    assert result["q_gate_apply_rate"] == 0.5
    assert result["q_advantage_min_mean"] == 0.005
    assert result["q_advantage_disagreement_max"] == 0.03
    assert result["gate_approval_rate"] == 0.5
    assert result["shadow_mode"]
    assert result["circuit_breaker_trigger_count"] == 1
    assert result["support_in_distribution_rate"] == 0.5
    assert result["support_state_score_max"] == 0.4
    assert result["support_action_score_max"] == 0.2
    assert result["intervention_allowed_rate"] == 0.5
    assert result["intervention_count_max"] == 2
    assert result["budget_exhausted_replans"] == 1


def test_parse_online_pair_log_extracts_seed_instruction_and_outcome(tmp_path):
    log = tmp_path / "eval_task.log"
    log.write_text(
        "FASTWAM_ACCEPTED_ENV_SEED episode_id=0 seed=4800000\n"
        "FASTWAM_EVAL_INSTRUCTION episode_id=0 seed=4800000 "
        "instruction='Open the microwave with the left arm.'\n"
        "Success rate: 0/1 => 0.0%\n"
        "FASTWAM_ACCEPTED_ENV_SEED episode_id=1 seed=4800001\n"
        "FASTWAM_EVAL_INSTRUCTION episode_id=1 seed=4800001 "
        "instruction='Open the microwave with the right arm.'\n"
        "Success rate: 1/2 => 50.0%\n",
        encoding="utf-8",
    )

    result = parse_log(log)

    assert result["episode_records"] == [
        {
            "episode_id": 0,
            "seed": 4800000,
            "instruction": "Open the microwave with the left arm.",
            "success": False,
        },
        {
            "episode_id": 1,
            "seed": 4800001,
            "instruction": "Open the microwave with the right arm.",
            "success": True,
        },
    ]


def test_load_episode_initial_hashes_groups_replans(tmp_path):
    root = tmp_path / "task" / "imagination_transitions" / "task" / "policy"
    for trial, digest in ((0, "aaa"), (1, "bbb")):
        for replan in range(2):
            path = root / f"episode_{trial:04d}" / f"replan_{replan:04d}" / "metadata.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "trial_idx": trial,
                        "initial_observation_sha256": digest,
                    }
                )
            )
    assert load_episode_initial_hashes(tmp_path, "task") == {"0": "aaa", "1": "bbb"}


def test_load_episode_initial_hashes_rejects_inconsistent_episode(tmp_path):
    root = tmp_path / "task" / "imagination_transitions" / "task" / "residual"
    for replan, digest in enumerate(("aaa", "bbb")):
        path = root / "episode_0000" / f"replan_{replan:04d}" / "metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"trial_idx": 0, "initial_observation_sha256": digest}
            )
        )
    with pytest.raises(ValueError, match="inconsistent initial hashes"):
        load_episode_initial_hashes(tmp_path, "task")


def test_merge_episode_hashes_requires_complete_disjoint_matrix():
    assert merge_episode_hashes([{"0": "aaa"}, {"1": "bbb"}], 2) == {
        "0": "aaa",
        "1": "bbb",
    }
    with pytest.raises(ValueError, match="conflicting initial hashes"):
        merge_episode_hashes([{"0": "aaa"}, {"0": "bbb"}], 1)
    with pytest.raises(ValueError, match="episode indices differ"):
        merge_episode_hashes([{"1": "bbb"}], 2)
