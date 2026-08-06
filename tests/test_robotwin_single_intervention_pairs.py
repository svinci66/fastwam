import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.build_single_intervention_pairs import build_pairs
from experiments.robotwin.fastwam_policy.deploy_policy import (
    _residual_diagnostic_metadata,
)


def _record(
    root: Path,
    *,
    mode: str,
    success: bool,
    gate_applied: bool = False,
    current_hash: str = "same-current",
    proprio_shift: float = 0.0,
) -> dict:
    record_dir = root / mode
    record_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
        record_dir / "current.png"
    )
    arrays = {
        "proprio": np.full(14, proprio_shift, dtype=np.float32),
        "baseline_actions": np.zeros((2, 14), dtype=np.float32),
    }
    if mode == "residual":
        arrays["candidate_residual_actions"] = np.full(
            (2, 14), 0.01, dtype=np.float32
        )
    np.savez_compressed(record_dir / "rollout_arrays.npz", **arrays)
    return {
        "schema_version": "robotwin_imagination_transition_v1",
        "task_name": "hanging_mug",
        "task_description": "Hang the mug.",
        "trial_idx": 0,
        "replan_idx": 5,
        "environment_seed": 101,
        "initial_observation_sha256": "same-initial",
        "current_observation_sha256": current_hash,
        "action_mode": "policy" if mode == "baseline" else "residual",
        "episode_success": success,
        "residual_gate_applied": gate_applied,
        "rollout_arrays_file": "rollout_arrays.npz",
        "record_dir": str(record_dir),
        "metadata_path": str(record_dir / "metadata.json"),
    }


def test_build_pairs_accepts_exact_single_intervention_rescue(tmp_path):
    baseline = _record(tmp_path, mode="baseline", success=False)
    residual = _record(
        tmp_path, mode="residual", success=True, gate_applied=True
    )

    accepted, quarantined = build_pairs(
        [baseline],
        [residual],
        intervention_replans={5},
        rewards={
            str((tmp_path / "baseline").resolve()): 0.02,
            str((tmp_path / "residual").resolve()): 0.08,
        },
    )

    assert not quarantined
    assert len(accepted) == 1
    assert accepted[0]["label"] == "rescue"
    assert accepted[0]["local_progress_delta"] == 0.06


def test_build_pairs_quarantines_pre_intervention_divergence(tmp_path):
    baseline = _record(tmp_path, mode="baseline", success=True)
    residual = _record(
        tmp_path,
        mode="residual",
        success=False,
        gate_applied=True,
        current_hash="different-current",
        proprio_shift=0.1,
    )

    accepted, quarantined = build_pairs(
        [baseline], [residual], intervention_replans={5}
    )

    assert not accepted
    assert quarantined[0]["status"] == "uncertain"
    assert set(quarantined[0]["quarantine_reasons"]) >= {
        "intervention_observation_hash_mismatch",
        "intervention_proprio_mismatch",
    }


class _Support:
    task_name = "hanging_mug"
    language_similarity = 0.9
    state_score = 0.2
    action_score = 0.3
    state_threshold = 1.0
    action_threshold = 1.0
    state_in_support = True
    action_in_support = True
    in_support = True


class _Output:
    gate_applied = False
    gate_approved = False
    shadow_mode = True
    intervention_allowed = True
    intervention_count = 0
    candidate_residual_rms = 0.02
    residual_scale_factor = 1.0
    residual_risk_before = 0.0
    residual_risk_after = 0.0
    circuit_breaker_active = False
    circuit_breaker_triggered = False
    q_advantages = (-0.1, 0.2)
    q_advantage_min = -0.1
    q_advantage_disagreement = 0.3
    q_gate_effective_margin = 0.0
    paired_advantage_probabilities = None
    support_decision = _Support()


def test_residual_diagnostics_keep_rejected_shadow_candidate():
    metadata = _residual_diagnostic_metadata(_Output())

    assert metadata["residual_gate_applied"] is False
    assert metadata["residual_q_advantage_min"] == -0.1
    assert metadata["residual_support_in_distribution"] is True
    json.dumps(metadata)
