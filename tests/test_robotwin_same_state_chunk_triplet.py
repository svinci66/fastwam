import json
from pathlib import Path

import numpy as np

from experiments.robotwin.audit_open_microwave_chunk_triplet import audit_triplet
from experiments.robotwin.select_open_microwave_chunk_anchor import select_anchor


def test_select_anchor_uses_closest_pre_success_start_ratio() -> None:
    chunks = [
        {"start_open_ratio": 0.1, "executed_actions": 24, "replan_idx": 1},
        {"start_open_ratio": 0.48, "executed_actions": 24, "replan_idx": 8},
        {"start_open_ratio": 0.61, "executed_actions": 5, "replan_idx": 9},
    ]
    result = select_anchor(chunks, 0.5)
    assert result["selected"]["replan_idx"] == 8
    assert result["candidate_count"] == 2


def _write_record(
    root: Path,
    *,
    replan: int,
    end_ratio: float,
    residual: bool,
    actor_override: bool = False,
) -> None:
    record = root / f"replan_{replan:04d}"
    record.mkdir(parents=True)
    baseline_actions = np.zeros((24, 14), dtype=np.float32)
    task_progress = np.zeros((25, 4), dtype=np.float32)
    task_progress[:, 3] = np.linspace(0.4, end_ratio, 25)
    arrays = {
        "proprio": np.zeros(14, dtype=np.float32),
        "baseline_actions": baseline_actions,
        "task_progress": task_progress,
    }
    if residual:
        arrays["candidate_residual_actions"] = np.full(
            (24, 14), 0.01, dtype=np.float32
        )
    np.savez_compressed(record / "rollout_arrays.npz", **arrays)
    metadata = {
        "task_name": "open_microwave",
        "environment_seed": 123,
        "trial_idx": 0,
        "replan_idx": replan,
        "task_description": "Open the microwave.",
        "initial_observation_sha256": "initial",
        "current_observation_sha256": "current",
        "baseline_actions_sha256": "actions",
        "rollout_arrays_file": "rollout_arrays.npz",
        "episode_success": False,
        "residual_gate_applied": residual,
        "residual_actor_override_applied": actor_override,
    }
    (record / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_audit_triplet_accepts_exact_pre_intervention_match(tmp_path: Path) -> None:
    roots = {
        "baseline": tmp_path / "baseline",
        "no_imagination": tmp_path / "no_imagination",
        "imagination": tmp_path / "imagination",
    }
    _write_record(roots["baseline"], replan=2, end_ratio=0.42, residual=False)
    _write_record(roots["no_imagination"], replan=2, end_ratio=0.45, residual=True)
    _write_record(roots["imagination"], replan=2, end_ratio=0.48, residual=True)

    result = audit_triplet(roots, 2)

    assert result["accepted"] is True
    assert result["rejection_reasons"] == []
    effects = result["causal_open_ratio_delta"]
    np.testing.assert_allclose(
        effects["no_imagination_minus_baseline"], 0.03, atol=1e-6
    )
    np.testing.assert_allclose(effects["imagination_minus_baseline"], 0.06, atol=1e-6)
    np.testing.assert_allclose(
        effects["imagination_minus_no_imagination"], 0.03, atol=1e-6
    )


def test_audit_triplet_accepts_shared_residual_prefix(tmp_path: Path) -> None:
    roots = {
        "baseline": tmp_path / "baseline",
        "no_imagination": tmp_path / "no_imagination",
        "imagination": tmp_path / "imagination",
    }
    for replan in (0, 1):
        for root in roots.values():
            _write_record(root, replan=replan, end_ratio=0.4, residual=True)
    _write_record(roots["baseline"], replan=2, end_ratio=0.42, residual=False)
    _write_record(roots["no_imagination"], replan=2, end_ratio=0.45, residual=True)
    _write_record(
        roots["imagination"],
        replan=2,
        end_ratio=0.48,
        residual=True,
        actor_override=True,
    )

    result = audit_triplet(
        roots,
        2,
        prefix_replans=[0, 1],
        require_imagination_override=True,
    )

    assert result["accepted"] is True
    assert result["applied_intervention_replans"] == {
        "baseline": [0, 1],
        "no_imagination": [0, 1, 2],
        "imagination": [0, 1, 2],
    }
    assert result["target_actor_override_applied"] == {
        "baseline": False,
        "no_imagination": False,
        "imagination": True,
    }
