from dataclasses import replace

import numpy as np

from fastwam.rl.paired_advantage import (
    PairedAdvantageTrainingConfig,
    build_paired_advantage_examples,
    build_temporal_context,
    summarize_paired_predictions,
)
from fastwam.rl.replay_buffer import ReplayBuffer
from tests.test_rl_replay_buffer import make_transition


def _episode(episode_id, behavior, seed, success, steps=2):
    rows = []
    for step in range(steps):
        rows.append(
            replace(
                make_transition(
                    episode_id=episode_id,
                    index=step,
                    terminated=step == steps - 1,
                    truncated=False,
                ),
                behavior_mode=behavior,
                env_seed=seed,
                success=success and step == steps - 1,
            )
        )
    return rows


def test_paired_examples_use_only_directional_episode_outcomes():
    replay = ReplayBuffer(
        _episode("train-base-fail", "policy", 0, False)
        + _episode("train-positive", "expert", 0, True)
        + _episode("train-equal", "hold", 0, False)
        + _episode("train-base-success", "policy", 1, True)
        + _episode("train-negative", "hold", 1, False)
        + _episode("val-base-fail", "policy", 4, False)
        + _episode("val-positive", "expert", 4, True)
        + _episode("val-base-success", "policy", 9, True)
        + _episode("val-negative", "hold", 9, False)
    )
    config = PairedAdvantageTrainingConfig()
    train = build_paired_advantage_examples(replay, config, split="train")
    validation = build_paired_advantage_examples(
        replay, config, split="validation"
    )
    assert len(train) == 4
    assert sorted(train.labels.tolist()) == [0.0, 0.0, 1.0, 1.0]
    assert len(validation) == 4
    assert set(validation.labels.tolist()) == {0.0, 1.0}
    assert "train-equal" not in train.episode_ids


def test_paired_prediction_threshold_rejects_validation_negatives():
    examples = type("Examples", (), {})()
    examples.indices = np.arange(4)
    examples.labels = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    examples.episode_ids = ("n0", "n1", "p0", "p1")
    examples.split = "validation"
    examples.__len__ = lambda self: 4
    probabilities = np.asarray(
        [[0.2, 0.3], [0.4, 0.35], [0.8, 0.9], [0.7, 0.75]], dtype=np.float32
    )
    # Use the real dataclass so len() and shape validation are exercised.
    from fastwam.rl.paired_advantage import PairedAdvantageExamples

    real = PairedAdvantageExamples(
        indices=examples.indices,
        labels=examples.labels,
        weights=np.ones(4),
        episode_ids=examples.episode_ids,
        split="validation",
    )
    summary = summarize_paired_predictions(real, probabilities)
    assert summary["transition_false_positive_rate"] == 0.0
    assert summary["transition_true_positive_rate"] == 1.0


def test_residual_equal_outcomes_can_be_strict_non_improvement_negatives():
    replay = ReplayBuffer(
        _episode("train-base-fail", "policy", 0, False)
        + _episode("train-residual-positive", "residual", 0, True)
        + _episode("train-base-fail-two", "policy", 1, False)
        + _episode("train-residual-equal", "residual", 1, False)
        + _episode("train-controlled-equal", "hold", 1, False)
        + _episode("val-base-fail", "policy", 4, False)
        + _episode("val-positive", "expert", 4, True)
        + _episode("val-base-fail-two", "policy", 9, False)
        + _episode("val-residual-equal", "residual", 9, False)
    )
    config = PairedAdvantageTrainingConfig(
        include_residual_equal_outcomes_as_negative=True
    )
    train = build_paired_advantage_examples(replay, config, split="train")
    validation = build_paired_advantage_examples(
        replay, config, split="validation"
    )
    assert set(train.labels.tolist()) == {0.0, 1.0}
    assert "train-residual-equal" in train.episode_ids
    assert "train-controlled-equal" not in train.episode_ids
    assert set(validation.labels.tolist()) == {0.0, 1.0}


def test_paired_prediction_threshold_can_calibrate_above_point_999():
    from fastwam.rl.paired_advantage import PairedAdvantageExamples

    examples = PairedAdvantageExamples(
        indices=np.arange(4),
        labels=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        weights=np.ones(4),
        episode_ids=("n0", "n1", "p0", "p1"),
        split="validation",
    )
    probabilities = np.asarray(
        [
            [0.99905, 0.99910],
            [0.99907, 0.99908],
            [0.99930, 0.99940],
            [0.99920, 0.99925],
        ],
        dtype=np.float32,
    )
    summary = summarize_paired_predictions(examples, probabilities)
    assert 0.999 < summary["recommended_threshold"] < 1.0
    assert summary["transition_false_positive_rate"] == 0.0
    assert summary["transition_true_positive_rate"] == 1.0


def test_paired_examples_can_select_actual_residual_pairs_across_all_seeds():
    replay = ReplayBuffer(
        _episode("base-fail", "policy", 0, False)
        + _episode("residual-rescue", "residual", 0, True)
        + _episode("expert-rescue", "expert", 0, True)
        + _episode("base-success", "policy", 4, True)
        + _episode("residual-regression", "residual", 4, False)
    )
    examples = build_paired_advantage_examples(
        replay,
        PairedAdvantageTrainingConfig(),
        split="all",
        behavior_modes=("residual",),
    )
    assert set(examples.episode_ids) == {
        "residual-rescue",
        "residual-regression",
    }
    assert set(examples.labels.tolist()) == {0.0, 1.0}


def test_temporal_context_uses_deltas_without_crossing_episode_boundaries():
    first = _episode("episode-a", "policy", 0, False)
    second = _episode("episode-b", "policy", 1, False)
    first[0] = replace(
        first[0],
        observation_feature=np.full_like(first[0].observation_feature, 1.0),
        proprio=np.full_like(first[0].proprio, 1.0),
    )
    first[1] = replace(
        first[1],
        observation_feature=np.full_like(first[1].observation_feature, 3.0),
        proprio=np.full_like(first[1].proprio, 3.0),
    )
    second[0] = replace(
        second[0],
        observation_feature=np.full_like(second[0].observation_feature, 10.0),
        proprio=np.full_like(second[0].proprio, 10.0),
    )
    replay = ReplayBuffer(first + second)
    single = build_temporal_context(replay, history_length=1)
    temporal = build_temporal_context(replay, history_length=3)
    width = single.shape[1]
    assert temporal.shape == (len(replay.transitions), width * 3)
    np.testing.assert_allclose(temporal[:, :width], single)
    np.testing.assert_allclose(temporal[0, width:], 0.0)
    np.testing.assert_allclose(temporal[1, width : 2 * width], 2.0)
    np.testing.assert_allclose(temporal[1, 2 * width :], 0.0)
    np.testing.assert_allclose(temporal[2, width:], 0.0)
