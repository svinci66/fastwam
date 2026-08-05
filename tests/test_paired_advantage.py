from dataclasses import replace

import numpy as np

from fastwam.rl.paired_advantage import (
    PairedAdvantageTrainingConfig,
    build_paired_advantage_examples,
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
