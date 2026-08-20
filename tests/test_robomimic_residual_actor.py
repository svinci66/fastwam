from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.train_can_residual_actor import ResidualActor, _features
from experiments.robomimic.audit_can_residual_q_support import kth_neighbor_distance
from experiments.robomimic.summarize_can_residual_actor import summarize


def test_residual_actor_is_exactly_zero_initialized_and_bounded():
    actor = ResidualActor(5, 4, (8, 8), residual_scale=0.1)
    features = torch.randn(3, 5)

    initial = actor(features)

    torch.testing.assert_close(initial, torch.zeros_like(initial), rtol=0.0, atol=0.0)
    with torch.no_grad():
        actor.output.bias.fill_(100.0)
    assert torch.max(actor(features)) <= 0.1


def test_actor_features_use_train_normalization():
    state = np.asarray([[1.0, 2.0]], dtype=np.float32)
    action = np.asarray([[[3.0, 4.0]]], dtype=np.float32)
    normalization = {
        "state_mean": np.asarray([1.0, 1.0]),
        "state_std": np.asarray([1.0, 2.0]),
        "action_mean": np.asarray([1.0, 2.0]),
        "action_std": np.asarray([2.0, 1.0]),
    }

    features = _features(state, action, normalization)

    np.testing.assert_allclose(features, np.asarray([[0.0, 0.5, 1.0, 2.0]]))


def test_knn_distance_excludes_self_for_support_threshold():
    points = np.asarray([[0.0], [1.0], [3.0]], dtype=np.float32)

    distance = kth_neighbor_distance(
        points,
        points,
        k=1,
        exclude_identical_index=True,
        chunk_size=2,
    )

    np.testing.assert_allclose(distance, np.asarray([1.0, 1.0, 2.0]))


def test_residual_summary_fails_online_gate_for_unselective_proposals(tmp_path):
    import json

    actor_dir = tmp_path / "actor_seed1"
    actor_dir.mkdir()
    (actor_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": 1,
                "valid": {
                    "positive_cosine_mean": 0.1,
                    "positive_direction_alignment_rate": 0.7,
                    "zero_prediction_norm_mean": 0.01,
                },
            }
        )
    )
    (actor_dir / "gate_audit.json").write_text(
        json.dumps(
            {
                "validation": {
                    "q_advantage_target_auc": 0.52,
                    "intervention_rate_on_improvement_targets": 0.04,
                    "intervention_rate_on_zero_targets": 0.03,
                    "random_action_rejection_rate": 1.0,
                    "joint_in_support_rate": 0.98,
                }
            }
        )
    )

    report = summarize(tmp_path)

    assert report["component_gates"]["ood_random_action_rejection"] is True
    assert report["component_gates"]["q_advantage_discrimination"] is False
    assert report["ready_for_online"] is False
