import json

import numpy as np
import pytest

from fastwam.rl.replay_buffer import ReplayBuffer, ReplayTransition
from fastwam.rl.rewards import (
    CompositeRewardConfig,
    EpisodeShapingBudget,
    compute_composite_reward,
)


def make_transition(
    episode_id: str,
    index: int,
    *,
    terminated: bool = False,
    truncated: bool = False,
    imagination_progress: float = 0.05,
) -> ReplayTransition:
    baseline = np.zeros((8, 7), dtype=np.float32)
    executed = baseline.copy()
    executed[:, 0] = 0.01
    config = CompositeRewardConfig(imitation_weight=0.0, imagination_weight=1.0)
    reward = compute_composite_reward(
        environment_rewards=np.zeros(8, dtype=np.float32),
        success=terminated,
        baseline_actions=baseline,
        executed_actions=executed,
        effective_k=8,
        imagination_progress=imagination_progress,
        alignment_valid=True,
        config=config,
        shaping_budget=EpisodeShapingBudget.from_config(config),
    )
    return ReplayTransition(
        episode_id=episode_id,
        transition_index=index,
        task_suite="libero_goal",
        task_id=3,
        task_description="open the top drawer and put the bowl inside",
        env_seed=42,
        goal_seed=43,
        action_seed=44,
        policy_version="fastwam-release-sha",
        predictor_version="fastwam-release-sha",
        reward_encoder_version="siglip-test-sha",
        behavior_mode="policy",
        action_noise_std=0.0,
        target_k=8,
        effective_k=8,
        goal_frame_index=2,
        goal_tau=8.0,
        terminated=terminated,
        truncated=truncated,
        success=terminated,
        alignment_valid=True,
        observation_feature=np.array([1.0, 0.0], dtype=np.float32),
        next_observation_feature=np.array([0.5, 0.5], dtype=np.float32),
        goal_feature=np.array([0.0, 1.0], dtype=np.float32),
        proprio=np.arange(8, dtype=np.float32),
        next_proprio=np.arange(8, dtype=np.float32) + 0.1,
        baseline_actions=baseline,
        executed_actions=executed,
        environment_rewards=np.zeros(8, dtype=np.float32),
        reward=reward,
    )


def test_replay_round_trip_is_checksummed_and_lossless(tmp_path):
    replay = ReplayBuffer(
        [
            make_transition("episode-0", 0),
            make_transition("episode-0", 1, terminated=True),
        ]
    )
    output = replay.save(tmp_path / "replay")
    loaded = ReplayBuffer.load(output)
    assert len(loaded) == 2
    np.testing.assert_array_equal(
        loaded.transitions[1].executed_actions,
        replay.transitions[1].executed_actions,
    )
    assert loaded.transitions[1].reward == replay.transitions[1].reward
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["num_transitions"] == 2


def test_replay_checksum_detects_tampering(tmp_path):
    output = ReplayBuffer([make_transition("episode-0", 0, terminated=True)]).save(
        tmp_path / "replay"
    )
    with (output / "transitions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ReplayBuffer.load(output)


def test_truncated_episode_requires_explicit_bootstrap():
    replay = ReplayBuffer([make_transition("timeout", 0, truncated=True)])
    with pytest.raises(ValueError, match="explicit bootstrap"):
        replay.monte_carlo_returns(0.99)
    returns = replay.monte_carlo_returns(
        0.99,
        timeout_bootstrap_values={"timeout": 2.0},
    )
    expected = replay.transitions[0].reward.total + (0.99**8) * 2.0
    assert np.isclose(returns[0], expected)


def test_reward_ablation_relabels_identical_transitions_without_mutation():
    replay = ReplayBuffer([make_transition("episode-0", 0, terminated=True)])
    without_imagination, _ = replay.relabel_rewards(
        CompositeRewardConfig(imitation_weight=0.0, imagination_weight=0.0)
    )
    with_imagination, _ = replay.relabel_rewards(
        CompositeRewardConfig(imitation_weight=0.0, imagination_weight=1.0)
    )
    assert with_imagination[0] > without_imagination[0]
    assert replay.transitions[0].reward.imagination_raw == 0.05


def test_replay_rejects_duplicate_episode_transition_identity():
    transition = make_transition("episode-0", 0, terminated=True)
    replay = ReplayBuffer([transition])
    with pytest.raises(ValueError, match="duplicate replay transition identity"):
        replay.append(transition)
