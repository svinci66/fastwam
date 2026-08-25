from experiments.robotwin.validate_controlled_imagination_rewards import validate


def test_controlled_reward_validator_checks_local_causal_order():
    transitions = []
    episodes = []
    behaviors = {
        "policy": 0.06,
        "controlled_corrupt_0.050": -0.02,
        "controlled_correct_0.050": 0.06,
    }
    for trial in range(5):
        for behavior, reward in behaviors.items():
            transitions.append(
                {
                    "task_name": "task",
                    "trial_idx": trial,
                    "behavior": behavior,
                    "replan_idx": 3,
                    "alignment_valid": True,
                    "action_noise_replans": [3],
                    "initial_observation_sha256": f"initial-{trial}",
                    "current_observation_sha256": f"current-{trial}",
                    "baseline_actions_sha256": f"action-{trial}",
                    "environment_seed": 100 + trial,
                    "imagination_reward": reward,
                }
            )
            episodes.append(
                {
                    "task_name": "task",
                    "trial_idx": trial,
                    "behavior": behavior,
                    "success": behavior != "controlled_corrupt_0.050",
                    "mean_imagination_reward": reward,
                    "imagination_return": reward * 3,
                }
            )

    summary = validate(
        transitions,
        episodes,
        corrupt_behavior="controlled_corrupt_0.050",
        correct_behavior="controlled_correct_0.050",
    )

    assert summary["all_gates_pass"]
    assert summary["local_order_fraction"] == 1.0
    assert summary["failed_corrupt_local_order_fraction"] == 1.0
    assert summary["failed_corrupt_episode_mean_order_fraction"] == 1.0
    assert summary["max_clean_correct_abs_gap"] == 0.0
