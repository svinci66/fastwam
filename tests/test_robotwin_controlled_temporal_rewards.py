from experiments.robotwin.validate_controlled_temporal_rewards import (
    validate_temporal_returns,
)


def test_three_replan_return_recovers_delayed_reward_order():
    transitions = []
    episodes = []
    for trial in range(5):
        for behavior in (
            "policy",
            "controlled_corrupt_0.050",
            "controlled_correct_0.050",
        ):
            episodes.append(
                {
                    "task_name": "task",
                    "trial_idx": trial,
                    "behavior": behavior,
                    "success": behavior != "controlled_corrupt_0.050",
                }
            )
            for replan in range(3, 6):
                reward = 0.02
                if behavior == "controlled_corrupt_0.050":
                    reward = 0.03 if replan == 3 else -0.04
                transitions.append(
                    {
                        "task_name": "task",
                        "trial_idx": trial,
                        "behavior": behavior,
                        "replan_idx": replan,
                        "alignment_valid": True,
                        "action_noise_replans": [3],
                        "initial_observation_sha256": f"initial-{trial}",
                        "current_observation_sha256": f"current-{trial}",
                        "baseline_actions_sha256": f"action-{trial}",
                        "environment_seed": 100 + trial,
                        "imagination_reward": reward,
                    }
                )

    summary = validate_temporal_returns(transitions, episodes)

    assert summary["horizon_summary"]["1"]["order_fraction"] == 0.0
    assert summary["horizon_summary"]["3"]["order_fraction"] == 1.0
    assert summary["all_gates_pass"]
