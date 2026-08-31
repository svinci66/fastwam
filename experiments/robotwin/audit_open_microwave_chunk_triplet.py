"""Audit and score a deterministic same-state single-chunk triplet.

RoboTwin/SAPIEN does not expose a complete simulator snapshot API.  The
experiment therefore replays each episode from the same seed and accepts a
triplet only when the initial/current observations, proprioception, and frozen
FastWAM action chunk match exactly (within configured numeric tolerances).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "robotwin_open_microwave_same_state_chunk_triplet_v1"


def _load_record(root: Path, replan: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = root / f"replan_{replan:04d}" / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    arrays_path = path.parent / str(metadata["rollout_arrays_file"])
    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    return metadata, arrays


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right), initial=0.0))


def _progress(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    task_progress = np.asarray(arrays["task_progress"], dtype=np.float64)
    if task_progress.ndim != 2 or task_progress.shape[0] < 2 or task_progress.shape[1] != 4:
        raise ValueError(f"invalid task_progress shape: {task_progress.shape}")
    ratio = task_progress[:, 3]
    return {
        "executed_actions": int(ratio.size - 1),
        "start_open_ratio": float(ratio[0]),
        "end_open_ratio": float(ratio[-1]),
        "max_open_ratio": float(np.max(ratio)),
        "min_open_ratio": float(np.min(ratio)),
        "open_ratio_delta": float(ratio[-1] - ratio[0]),
        "open_ratio_trace": ratio.tolist(),
    }


def _residual(arrays: dict[str, np.ndarray]) -> dict[str, Any] | None:
    if "candidate_residual_actions" not in arrays:
        return None
    value = np.asarray(arrays["candidate_residual_actions"], dtype=np.float64)
    return {
        "rms": float(np.sqrt(np.mean(np.square(value)))),
        "left_arm_rms": float(np.sqrt(np.mean(np.square(value[:, :6])))),
        "right_arm_rms": float(np.sqrt(np.mean(np.square(value[:, 7:13])))),
        "gripper_max_abs": float(np.max(np.abs(value[:, [6, 13]]), initial=0.0)),
    }


def _applied_replans(root: Path, *, through_replan: int | None = None) -> list[int]:
    result: list[int] = []
    for path in sorted(root.glob("replan_*/metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        replan = int(metadata["replan_idx"])
        if through_replan is not None and replan > through_replan:
            continue
        if bool(metadata.get("residual_gate_applied", False)):
            result.append(replan)
    return result


def audit_triplet(
    roots: dict[str, Path],
    replan: int,
    *,
    prefix_replans: list[int] | None = None,
    require_imagination_override: bool = False,
    proprio_atol: float = 1e-6,
    action_atol: float = 1e-6,
    progress_atol: float = 1e-6,
) -> dict[str, Any]:
    loaded = {name: _load_record(root, replan) for name, root in roots.items()}
    baseline_meta, baseline_arrays = loaded["baseline"]
    reasons: list[str] = []
    comparisons: dict[str, Any] = {}
    for name, (metadata, arrays) in loaded.items():
        if metadata.get("task_name") != "open_microwave":
            reasons.append(f"{name}_wrong_task")
        if int(metadata.get("replan_idx", -1)) != replan:
            reasons.append(f"{name}_wrong_replan")
        comparison = {
            "initial_observation_match": metadata.get("initial_observation_sha256")
            == baseline_meta.get("initial_observation_sha256"),
            "current_observation_match": metadata.get("current_observation_sha256")
            == baseline_meta.get("current_observation_sha256"),
            "baseline_actions_hash_match": metadata.get("baseline_actions_sha256")
            == baseline_meta.get("baseline_actions_sha256"),
            "instruction_match": metadata.get("task_description")
            == baseline_meta.get("task_description"),
            "proprio_max_abs_difference": _max_abs(
                arrays["proprio"], baseline_arrays["proprio"]
            ),
            "baseline_action_max_abs_difference": _max_abs(
                arrays["baseline_actions"], baseline_arrays["baseline_actions"]
            ),
            "start_progress_max_abs_difference": _max_abs(
                arrays["task_progress"][0], baseline_arrays["task_progress"][0]
            ),
        }
        comparisons[name] = comparison
        for field in (
            "initial_observation_match",
            "current_observation_match",
            "baseline_actions_hash_match",
            "instruction_match",
        ):
            if not comparison[field]:
                reasons.append(f"{name}_{field}_failed")
        if comparison["proprio_max_abs_difference"] > proprio_atol:
            reasons.append(f"{name}_proprio_mismatch")
        if comparison["baseline_action_max_abs_difference"] > action_atol:
            reasons.append(f"{name}_baseline_action_mismatch")
        if comparison["start_progress_max_abs_difference"] > progress_atol:
            reasons.append(f"{name}_start_progress_mismatch")

    interventions = {
        name: _applied_replans(root, through_replan=replan)
        for name, root in roots.items()
    }
    expected_prefix = [] if prefix_replans is None else list(prefix_replans)
    if interventions["baseline"] != expected_prefix:
        reasons.append("baseline_intervention_prefix_mismatch")
    expected_candidate = [*expected_prefix, replan]
    for name in ("no_imagination", "imagination"):
        if interventions[name] != expected_candidate:
            reasons.append(f"{name}_intervention_prefix_mismatch")
    target_override_flags = {
        name: bool(metadata.get("residual_actor_override_applied", False))
        for name, (metadata, _) in loaded.items()
    }
    if require_imagination_override:
        expected_flags = {
            "baseline": False,
            "no_imagination": False,
            "imagination": True,
        }
        for name, expected in expected_flags.items():
            if target_override_flags[name] is not expected:
                reasons.append(f"{name}_actor_override_flag_mismatch")

    branches: dict[str, Any] = {}
    for name, (metadata, arrays) in loaded.items():
        branches[name] = {
            "environment_seed": int(metadata["environment_seed"]),
            "trial_idx": int(metadata["trial_idx"]),
            "episode_success": bool(metadata.get("episode_success", False)),
            "progress": _progress(arrays),
            "candidate_residual": _residual(arrays),
            "record_dir": str(roots[name].resolve() / f"replan_{replan:04d}"),
        }

    seed_values = {branch["environment_seed"] for branch in branches.values()}
    trial_values = {branch["trial_idx"] for branch in branches.values()}
    if len(seed_values) != 1:
        reasons.append("environment_seed_mismatch")
    if len(trial_values) != 1:
        reasons.append("trial_idx_mismatch")
    baseline_delta = branches["baseline"]["progress"]["open_ratio_delta"]
    no_imagination_delta = branches["no_imagination"]["progress"]["open_ratio_delta"]
    imagination_delta = branches["imagination"]["progress"]["open_ratio_delta"]
    return {
        "schema_version": SCHEMA_VERSION,
        "same_state_method": "deterministic_replay_with_exact_pre_intervention_audit",
        "native_simulator_snapshot": False,
        "accepted": not reasons,
        "rejection_reasons": sorted(set(reasons)),
        "intervention_replan": replan,
        "comparisons": comparisons,
        "applied_intervention_replans": interventions,
        "target_actor_override_applied": target_override_flags,
        "branches": branches,
        "causal_open_ratio_delta": {
            "no_imagination_minus_baseline": no_imagination_delta - baseline_delta,
            "imagination_minus_baseline": imagination_delta - baseline_delta,
            "imagination_minus_no_imagination": imagination_delta
            - no_imagination_delta,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--no-imagination-root", type=Path, required=True)
    parser.add_argument("--imagination-root", type=Path, required=True)
    parser.add_argument("--intervention-replan", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--prefix-replans",
        default="none",
        help="Use 'none' or 'all_before_target' for expected applied prefix chunks.",
    )
    parser.add_argument("--require-imagination-override", action="store_true")
    parser.add_argument("--require-accepted", action="store_true")
    args = parser.parse_args()
    roots = {
        "baseline": args.baseline_root.expanduser().resolve(),
        "no_imagination": args.no_imagination_root.expanduser().resolve(),
        "imagination": args.imagination_root.expanduser().resolve(),
    }
    if args.prefix_replans == "none":
        prefix_replans = None
    elif args.prefix_replans == "all_before_target":
        prefix_replans = list(range(args.intervention_replan))
    else:
        raise ValueError("prefix-replans must be 'none' or 'all_before_target'")
    result = audit_triplet(
        roots,
        args.intervention_replan,
        prefix_replans=prefix_replans,
        require_imagination_override=args.require_imagination_override,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compact = {
        "accepted": result["accepted"],
        "rejection_reasons": result["rejection_reasons"],
        "intervention_replan": result["intervention_replan"],
        "causal_open_ratio_delta": result["causal_open_ratio_delta"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    if args.require_accepted and not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
