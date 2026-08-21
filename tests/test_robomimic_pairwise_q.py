from pathlib import Path
import sys

import numpy as np
import json
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.train_can_pairwise_q import (
    PairwiseQ,
    _auc,
    _initialize_full_from_action_only,
    _metrics,
    _prepare_features,
)
from experiments.robomimic.summarize_can_pairwise_q import summarize


def test_pairwise_metrics_use_candidate_better_as_positive():
    target = np.asarray([0, 0, 1, 1], dtype=np.int8)
    logits = np.asarray([-2.0, -1.0, 1.0, 2.0])

    metrics = _metrics(target, logits)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["auc"] == 1.0


def test_auc_handles_reversed_ranking():
    assert _auc(np.asarray([0, 0, 1, 1]), np.asarray([2.0, 1.0, -1.0, -2.0])) == 0.0


def test_action_only_features_exclude_state():
    state = np.ones((2, 3), dtype=np.float32)
    action = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    kwargs = {
        "state_mean": np.zeros(3),
        "state_std": np.ones(3),
        "action_mean": np.zeros(4),
        "action_std": np.ones(4),
    }

    full = _prepare_features(state, action, state_mode="full", **kwargs)
    action_only = _prepare_features(state, action, state_mode="action_only", **kwargs)

    assert full.shape == (2, 7)
    assert action_only.shape == (2, 4)
    np.testing.assert_array_equal(action_only, action.reshape(2, -1))


def test_full_q_action_initialization_exactly_preserves_action_q():
    torch.manual_seed(7)
    action_q = PairwiseQ(4, (8, 6))
    checkpoint = {
        "model": action_q.state_dict(),
        "input_dim": 4,
        "hidden_dims": [8, 6],
        "state_mode": "action_only",
    }
    full_q = PairwiseQ(7, (8, 6))
    teacher = _initialize_full_from_action_only(
        full_q, checkpoint, state_dim=3, action_dim=4
    )
    state = torch.randn(5, 3)
    action = torch.randn(5, 4)

    torch.testing.assert_close(full_q(torch.cat([state, action], dim=1)), teacher(action))
    torch.testing.assert_close(full_q.network[0].weight[:, :3], torch.zeros(8, 3))


def test_multiseed_summary_requires_state_gain(tmp_path):
    for seed, full_balanced, action_balanced in ((1, 0.72, 0.66), (2, 0.70, 0.65)):
        for mode, balanced in (("full_state", full_balanced), ("action_only", action_balanced)):
            directory = tmp_path / f"{mode}_seed{seed}"
            directory.mkdir()
            (directory / "metrics.json").write_text(
                json.dumps(
                    {
                        "valid": {
                            "balanced_accuracy": balanced,
                            "auc": balanced + 0.05,
                            "accuracy": balanced,
                        }
                    }
                )
            )

    report = summarize(tmp_path)

    assert report["seeds"] == [1, 2]
    assert report["gate"]["passed"] is True
    assert report["aggregate"]["balanced_accuracy_gain"]["mean"] > 0.02
