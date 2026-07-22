import numpy as np

from fastwam.rl.rewards import (
    CompositeRewardConfig,
    EpisodeShapingBudget,
    GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE,
    compute_composite_reward,
    compute_imagination_reward,
    compute_imagination_progress,
)


def test_imagination_progress_uses_frozen_camera_weights_and_relative_distance():
    current = {"agent": np.array([1.0, 0.0]), "wrist": np.array([1.0, 0.0])}
    goal = {"agent": np.array([0.0, 1.0]), "wrist": np.array([0.0, 1.0])}
    actual = {"agent": np.array([0.0, 1.0]), "wrist": np.array([1.0, 0.0])}
    result = compute_imagination_progress(
        current,
        actual,
        goal,
        camera_weights={"agent": 0.75, "wrist": 0.25},
        clip_value=1.0,
    )
    assert np.isclose(result.distance_before, 1.0)
    assert np.isclose(result.distance_after, 0.25)
    assert np.isclose(result.raw_progress, 0.75)
    assert result.camera_weights == {"agent": 0.75, "wrist": 0.25}


def test_invalid_temporal_alignment_masks_imagination_shaping():
    features = {"agent": np.array([1.0, 0.0]), "wrist": np.array([1.0, 0.0])}
    goals = {"agent": np.array([0.0, 1.0]), "wrist": np.array([0.0, 1.0])}
    result = compute_imagination_progress(
        features,
        goals,
        goals,
        clip_value=0.1,
        alignment_valid=False,
    )
    assert result.raw_progress > 0.0
    assert result.clipped_progress == 0.0


def test_delta_alignment_rewards_matched_direction_and_suppresses_static_change():
    current = {"agent": np.array([1.0, 0.0]), "wrist": np.array([1.0, 0.0])}
    goal = {"agent": np.array([0.0, 1.0]), "wrist": np.array([0.0, 1.0])}
    matched = compute_imagination_reward(
        current,
        goal,
        goal,
        reward_type="delta_alignment_v1",
        camera_weights={"agent": 0.5, "wrist": 0.5},
        clip_value=1.0,
    )
    static = compute_imagination_reward(
        current,
        current,
        goal,
        reward_type="delta_alignment_v1",
        camera_weights={"agent": 0.5, "wrist": 0.5},
        clip_value=1.0,
    )
    assert np.isclose(matched.raw_progress, 1.0)
    assert static.raw_progress == 0.0


def test_global_camera_normalization_preserves_soft_bounded_differences():
    current = {"agent": np.array([1.0, 0.0]), "wrist": np.array([1.0, 0.0])}
    goal = {"agent": np.array([0.0, 1.0]), "wrist": np.array([0.0, 1.0])}
    actual = {"agent": goal["agent"], "wrist": current["wrist"]}
    result = compute_imagination_reward(
        current,
        actual,
        goal,
        reward_type=GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE,
        camera_weights={"agent": 0.5, "wrist": 0.5},
        camera_normalization={
            "agent": {"center": 0.0, "scale": 1.0},
            "wrist": {"center": 0.0, "scale": 1.0},
        },
        clip_value=0.1,
    )
    assert np.isclose(result.raw_progress, 0.1 * np.tanh(0.5))
    assert 0.0 < result.raw_progress < 0.1
    assert result.clipped_progress == result.raw_progress


def test_global_camera_normalization_requires_frozen_statistics():
    features = {"agent": np.array([1.0, 0.0]), "wrist": np.array([1.0, 0.0])}
    goals = {"agent": np.array([0.0, 1.0]), "wrist": np.array([0.0, 1.0])}
    with np.testing.assert_raises_regex(ValueError, "requires camera_normalization"):
        compute_imagination_reward(
            features,
            goals,
            goals,
            reward_type=GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE,
        )


def test_episode_budget_keeps_total_absolute_imagination_below_half_success_bonus():
    config = CompositeRewardConfig(
        success_bonus=10.0,
        success_weight=1.0,
        imitation_weight=0.0,
        imagination_weight=100.0,
        imagination_clip=1.0,
        max_imagination_to_success_ratio=0.5,
    )
    budget = EpisodeShapingBudget.from_config(config)
    actions = np.zeros((8, 7), dtype=np.float32)
    first = compute_composite_reward(
        environment_rewards=np.zeros(8),
        success=False,
        baseline_actions=actions,
        executed_actions=actions,
        effective_k=8,
        imagination_progress=1.0,
        alignment_valid=True,
        config=config,
        shaping_budget=budget,
    )
    second = compute_composite_reward(
        environment_rewards=np.zeros(8),
        success=False,
        baseline_actions=actions,
        executed_actions=actions,
        effective_k=8,
        imagination_progress=-1.0,
        alignment_valid=True,
        config=config,
        shaping_budget=budget,
    )
    assert first.imagination_applied == 5.0
    assert second.imagination_applied == 0.0
    assert budget.absolute_spent == 5.0
    assert budget.remaining == 0.0


def test_raw_libero_success_reward_is_not_counted_twice_by_default():
    config = CompositeRewardConfig(
        success_bonus=10.0,
        imitation_weight=0.0,
        imagination_weight=0.0,
    )
    actions = np.zeros((8, 7), dtype=np.float32)
    reward = compute_composite_reward(
        environment_rewards=np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
        success=True,
        baseline_actions=actions,
        executed_actions=actions,
        effective_k=8,
        imagination_progress=0.0,
        alignment_valid=True,
        config=config,
        shaping_budget=EpisodeShapingBudget.from_config(config),
    )
    assert reward.environment_return == 1.0
    assert reward.environment_component == 0.0
    assert reward.success_component == 10.0
    assert reward.total == 10.0
