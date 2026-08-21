from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.evaluate_can_deployable_online_episodes import (
    padded_action_chunk,
    summarize_online_episodes,
)


def test_action_chunk_pads_only_the_short_episode_tail():
    actions = np.arange(10, dtype=np.float32).reshape(5, 2)

    chunk, available = padded_action_chunk(actions, 3, chunk_steps=3)

    assert available == 2
    np.testing.assert_array_equal(chunk, [actions[3], actions[4], actions[4]])


def test_online_summary_counts_success_preservation_and_interventions():
    rows = [
        {
            "reward_delta": 1.0,
            "baseline_success": 1,
            "residual_success": 1,
            "decisions": 10,
            "interventions": 2,
            "restore_linf": 0.0,
            "max_gripper_residual_abs": 0.0,
            "max_residual_component_abs": 0.02,
        },
        {
            "reward_delta": -0.5,
            "baseline_success": 1,
            "residual_success": 0,
            "decisions": 10,
            "interventions": 1,
            "restore_linf": 0.0,
            "max_gripper_residual_abs": 0.0,
            "max_residual_component_abs": 0.01,
        },
    ]

    summary = summarize_online_episodes(rows)

    assert summary["success_losses"] == 1
    assert summary["intervention_rate"] == 3 / 20
    assert summary["reward_delta_mean"] == 0.25
