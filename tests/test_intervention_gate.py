import json
from pathlib import Path

import numpy as np

from fastwam.rl.intervention_gate import (
    audit_intervention_gate_coverage,
    discover_pair_jsonl,
    intervention_decision_metrics,
    load_intervention_gate_examples,
    predict_intervention_gate,
    summarize_intervention_fit,
    train_intervention_gate_ensemble,
)
from fastwam.rl.models import ActionValueCritic, ActionValueCriticConfig


def _write_pair(
    root: Path,
    *,
    task: str,
    seed: int,
    replan: int,
    label: str,
    residual_value: float,
) -> dict:
    record_dir = root / f"{task}_{seed}_{replan}"
    record_dir.mkdir(parents=True)
    np.savez(
        record_dir / "rollout_arrays.npz",
        residual_observation_feature=np.asarray([seed / 10.0, replan], dtype=np.float32),
        proprio=np.asarray([0.25], dtype=np.float32),
        baseline_actions=np.zeros((2, 2), dtype=np.float32),
        candidate_residual_actions=np.full((2, 2), residual_value, dtype=np.float32),
        language_feature=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    (record_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "robotwin_imagination_transition_v1",
                "residual_gate_applied": True,
                "rollout_arrays_file": "rollout_arrays.npz",
            }
        ),
        encoding="utf-8",
    )
    return {
        "status": "accepted",
        "label": label,
        "task_name": task,
        "environment_seed": seed,
        "intervention_replan_idx": replan,
        "residual_record_dir": str(record_dir),
    }


def _pair_file(tmp_path: Path) -> Path:
    rows = [
        _write_pair(
            tmp_path,
            task="task-a",
            seed=1,
            replan=0,
            label="rescue",
            residual_value=0.02,
        ),
        _write_pair(
            tmp_path,
            task="task-a",
            seed=2,
            replan=1,
            label="regression",
            residual_value=-0.02,
        ),
        _write_pair(
            tmp_path,
            task="task-b",
            seed=3,
            replan=2,
            label="terminal_tie_unscored",
            residual_value=0.01,
        ),
    ]
    path = tmp_path / "accepted_pairs.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_intervention_gate_loader_uses_only_the_forced_candidate_replan(tmp_path: Path):
    pair_file = _pair_file(tmp_path)
    examples = load_intervention_gate_examples(
        discover_pair_jsonl((tmp_path,)), non_improving_weight=0.25
    )

    assert len(examples) == 3
    assert examples.context.shape == (3, 3)
    assert examples.baseline_actions.shape == (3, 2, 2)
    assert examples.language_feature.shape == (3, 2)
    np.testing.assert_allclose(examples.candidate_actions[0], 0.02)
    assert examples.labels.tolist() == [1.0, 0.0, 0.0]
    assert examples.pair_ids == ("task-a:1:0", "task-a:2:1", "task-b:3:2")
    assert examples.weights[examples.labels == 1.0].sum() == np.float32(1.5)
    assert examples.weights[examples.labels == 0.0].sum() == np.float32(1.5)

    # Passing both the file and its parent must not duplicate an intervention.
    duplicated_input = load_intervention_gate_examples((pair_file, tmp_path))
    assert len(duplicated_input) == 3


def test_intervention_gate_audit_blocks_single_positive_task_seed(tmp_path: Path):
    examples = load_intervention_gate_examples((_pair_file(tmp_path),))
    audit = audit_intervention_gate_coverage(examples)

    assert audit["positive_examples"] == 1
    assert audit["independent_seed_validation_possible"] is False
    assert audit["deployment_ready"] is False
    assert "two task/seed" in audit["deployment_blocker"]


def test_intervention_gate_training_and_prediction_smoke(tmp_path: Path):
    examples = load_intervention_gate_examples((_pair_file(tmp_path),))
    config = ActionValueCriticConfig(
        context_dim=3,
        action_horizon=2,
        action_dim=2,
        hidden_dims=(8,),
        language_feature_dim=2,
        language_embedding_dim=2,
        baseline_action_embedding_dim=2,
        action_embedding_dim=2,
    )
    models = (ActionValueCritic(config), ActionValueCritic(config))

    histories = train_intervention_gate_ensemble(
        models, examples, device="cpu", epochs=3, batch_size=3, seed=7
    )
    probabilities = predict_intervention_gate(models, examples, device="cpu")
    summary = summarize_intervention_fit(examples, probabilities)

    assert len(histories) == 2
    assert all(len(history) == 3 for history in histories)
    assert probabilities.shape == (3, 2)
    assert np.all(np.isfinite(probabilities))
    assert summary["evaluation_scope"] == "diagnostic_resubstitution_only"
    assert len(summary["pair_rows"]) == 3


def test_intervention_gate_restores_a_verified_terminal_candidate_chunk(tmp_path: Path):
    rescue = _write_pair(
        tmp_path,
        task="task-a",
        seed=1,
        replan=0,
        label="rescue",
        residual_value=0.02,
    )
    residual_dir = Path(rescue["residual_record_dir"])
    np.savez(
        residual_dir / "rollout_arrays.npz",
        residual_observation_feature=np.asarray([0.1, 0.0], dtype=np.float32),
        proprio=np.asarray([0.25], dtype=np.float32),
        baseline_actions=np.zeros((1, 2), dtype=np.float32),
        candidate_residual_actions=np.full((1, 2), 0.02, dtype=np.float32),
        language_feature=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    residual_metadata = json.loads(
        (residual_dir / "metadata.json").read_text(encoding="utf-8")
    )
    residual_metadata.update(
        {
            "terminated": True,
            "target_step": 2,
            "baseline_actions_sha256": "full-baseline-hash",
            "candidate_residual_actions_sha256": "full-candidate-hash",
        }
    )
    (residual_dir / "metadata.json").write_text(
        json.dumps(residual_metadata), encoding="utf-8"
    )

    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    np.savez(
        shadow_dir / "rollout_arrays.npz",
        baseline_actions=np.zeros((2, 2), dtype=np.float32),
        candidate_residual_actions=np.full((2, 2), 0.02, dtype=np.float32),
    )
    (shadow_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "robotwin_imagination_transition_v1",
                "residual_gate_applied": False,
                "rollout_arrays_file": "rollout_arrays.npz",
                "baseline_actions_sha256": "full-baseline-hash",
                "candidate_residual_actions_sha256": "full-candidate-hash",
            }
        ),
        encoding="utf-8",
    )
    rescue["baseline_record_dir"] = str(shadow_dir)
    regression = _write_pair(
        tmp_path,
        task="task-a",
        seed=2,
        replan=1,
        label="regression",
        residual_value=-0.02,
    )
    pair_file = tmp_path / "terminal_pairs.jsonl"
    pair_file.write_text(
        json.dumps(rescue) + "\n" + json.dumps(regression) + "\n",
        encoding="utf-8",
    )

    examples = load_intervention_gate_examples(
        (pair_file,), include_non_improving=False
    )

    assert examples.baseline_actions.shape == (2, 2, 2)
    np.testing.assert_allclose(examples.candidate_actions[0], 0.02)


def test_intervention_decision_metrics_separate_rescue_and_regression_errors():
    metrics = intervention_decision_metrics(
        ("rescue", "rescue", "regression", "regression", "neutral"),
        (True, False, True, False, True),
    )

    assert metrics["rescue_recall"] == 0.5
    assert metrics["regression_false_approval_rate"] == 0.5
