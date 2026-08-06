"""Validate and export same-seed, single-residual-intervention pairs.

This is the deliberately conservative first version of counterfactual data
construction.  It does not claim that two independent RoboTwin rollouts are an
exact simulator-state fork.  Instead, it accepts a pair only when the initial
observation, the observation at the intervention replan, proprioception, and
the FastWAM baseline action chunk all agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "robotwin_single_intervention_pair_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_arrays(record: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(record["record_dir"]) / str(record["rollout_arrays_file"])
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def discover_records(root: Path, *, action_mode: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(root.expanduser().resolve().rglob("metadata.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "robotwin_imagination_transition_v1":
            continue
        if str(payload.get("action_mode")) != action_mode:
            continue
        payload = dict(payload)
        payload["record_dir"] = str(metadata_path.parent.resolve())
        payload["metadata_path"] = str(metadata_path.resolve())
        records.append(payload)
    if not records:
        raise ValueError(f"No action_mode={action_mode!r} transitions found in {root}")
    return records


def _episode_key(record: dict[str, Any]) -> tuple[str, int, int]:
    seed = record.get("environment_seed")
    if seed is None:
        raise ValueError(
            f"Transition lacks environment_seed: {record.get('metadata_path')}"
        )
    return (
        str(record["task_name"]),
        int(seed),
        int(record["trial_idx"]),
    )


def _pair_key(record: dict[str, Any]) -> tuple[str, int, int, int]:
    return (*_episode_key(record), int(record["replan_idx"]))


def _index_unique(
    records: Iterable[dict[str, Any]],
) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    index: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for record in records:
        key = _pair_key(record)
        if key in index:
            raise ValueError(f"Duplicate transition key: {key}")
        index[key] = record
    return index


def load_reward_rows(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    rewards: dict[str, float] = {}
    for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record_dir = str(Path(row["record_dir"]).resolve())
        rewards[record_dir] = float(row["imagination_reward"])
    return rewards


def _finite_max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(
        np.isfinite(right)
    ):
        return float("inf")
    return float(np.max(np.abs(left - right)))


def _episode_success(record: dict[str, Any]) -> bool:
    return bool(record.get("episode_success", False))


def _outcome_label(
    *,
    baseline_success: bool,
    residual_success: bool,
    local_progress_delta: float | None,
    local_progress_threshold: float,
) -> str:
    if residual_success and not baseline_success:
        return "rescue"
    if baseline_success and not residual_success:
        return "regression"
    if local_progress_delta is None:
        return "terminal_tie_unscored"
    if local_progress_delta > local_progress_threshold:
        return "local_improve"
    if local_progress_delta < -local_progress_threshold:
        return "local_worse"
    return "neutral"


def build_pairs(
    baseline_records: list[dict[str, Any]],
    residual_records: list[dict[str, Any]],
    *,
    intervention_replans: set[int] | None,
    rewards: dict[str, float] | None = None,
    proprio_atol: float = 1e-6,
    action_atol: float = 1e-6,
    local_progress_threshold: float = 0.01,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rewards = rewards or {}
    baseline_index = _index_unique(baseline_records)
    residual_index = _index_unique(residual_records)
    residual_applied: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for record in residual_records:
        if bool(record.get("residual_gate_applied", False)):
            residual_applied[_episode_key(record)].append(int(record["replan_idx"]))

    candidate_keys = sorted(set(baseline_index) & set(residual_index))
    if intervention_replans is not None:
        candidate_keys = [key for key in candidate_keys if key[-1] in intervention_replans]

    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for key in candidate_keys:
        baseline = baseline_index[key]
        residual = residual_index[key]
        reasons: list[str] = []
        episode_key = key[:3]
        applied_replans = sorted(residual_applied.get(episode_key, []))
        if applied_replans != [key[-1]]:
            reasons.append(f"expected_one_applied_intervention_at_{key[-1]}")
        if baseline.get("initial_observation_sha256") != residual.get(
            "initial_observation_sha256"
        ):
            reasons.append("initial_observation_mismatch")
        baseline_current_hash = baseline.get("current_observation_sha256")
        residual_current_hash = residual.get("current_observation_sha256")
        if baseline_current_hash is not None and residual_current_hash is not None:
            if baseline_current_hash != residual_current_hash:
                reasons.append("intervention_observation_hash_mismatch")
        else:
            baseline_current_path = Path(baseline["record_dir"]) / "current.png"
            residual_current_path = Path(residual["record_dir"]) / "current.png"
            if _sha256(baseline_current_path) != _sha256(residual_current_path):
                reasons.append("intervention_image_mismatch")
        if baseline.get("task_description") != residual.get("task_description"):
            reasons.append("instruction_mismatch")

        baseline_arrays = _load_arrays(baseline)
        residual_arrays = _load_arrays(residual)
        proprio_difference = _finite_max_abs_difference(
            baseline_arrays.get("proprio", np.empty(0)),
            residual_arrays.get("proprio", np.empty(0)),
        )
        baseline_action_difference = _finite_max_abs_difference(
            baseline_arrays.get("baseline_actions", np.empty(0)),
            residual_arrays.get("baseline_actions", np.empty(0)),
        )
        if proprio_difference > proprio_atol:
            reasons.append("intervention_proprio_mismatch")
        if baseline_action_difference > action_atol:
            reasons.append("baseline_action_mismatch")
        candidate_residual = residual_arrays.get("candidate_residual_actions")
        if candidate_residual is None:
            reasons.append("missing_candidate_residual_actions")
            candidate_rms = None
        else:
            candidate_rms = float(
                np.sqrt(np.mean(np.square(np.asarray(candidate_residual, dtype=np.float64))))
            )
            if not np.isfinite(candidate_rms) or candidate_rms <= 0.0:
                reasons.append("empty_candidate_residual")

        baseline_reward = rewards.get(str(Path(baseline["record_dir"]).resolve()))
        residual_reward = rewards.get(str(Path(residual["record_dir"]).resolve()))
        local_progress_delta = (
            None
            if baseline_reward is None or residual_reward is None
            else float(residual_reward - baseline_reward)
        )
        common = {
            "schema_version": SCHEMA_VERSION,
            "task_name": key[0],
            "environment_seed": key[1],
            "trial_idx": key[2],
            "intervention_replan_idx": key[3],
            "initial_observation_sha256": baseline.get(
                "initial_observation_sha256"
            ),
            "current_observation_sha256": baseline_current_hash,
            "proprio_max_abs_difference": proprio_difference,
            "baseline_action_max_abs_difference": baseline_action_difference,
            "baseline_record_dir": str(Path(baseline["record_dir"]).resolve()),
            "residual_record_dir": str(Path(residual["record_dir"]).resolve()),
            "baseline_episode_success": _episode_success(baseline),
            "residual_episode_success": _episode_success(residual),
            "baseline_imagination_progress": baseline_reward,
            "residual_imagination_progress": residual_reward,
            "local_progress_delta": local_progress_delta,
            "candidate_residual_rms": candidate_rms,
            "residual_q_advantage_min": residual.get(
                "residual_q_advantage_min"
            ),
            "residual_q_advantage_disagreement": residual.get(
                "residual_q_advantage_disagreement"
            ),
            "residual_support_in_distribution": residual.get(
                "residual_support_in_distribution"
            ),
            "residual_support_state_score": residual.get(
                "residual_support_state_score"
            ),
            "residual_support_action_score": residual.get(
                "residual_support_action_score"
            ),
        }
        if reasons:
            common["status"] = "uncertain"
            common["quarantine_reasons"] = reasons
            quarantined.append(common)
            continue
        common["status"] = "accepted"
        common["label"] = _outcome_label(
            baseline_success=common["baseline_episode_success"],
            residual_success=common["residual_episode_success"],
            local_progress_delta=local_progress_delta,
            local_progress_threshold=local_progress_threshold,
        )
        accepted.append(common)
    return accepted, quarantined


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_replans(value: str) -> set[int] | None:
    if value.strip().lower() == "all":
        return None
    result = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not result or min(result) < 0:
        raise ValueError("intervention replans must be 'all' or non-negative CSV")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--intervention-replans", default="all")
    parser.add_argument("--reward-jsonl", type=Path)
    parser.add_argument("--proprio-atol", type=float, default=1e-6)
    parser.add_argument("--action-atol", type=float, default=1e-6)
    parser.add_argument("--local-progress-threshold", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-accepted", action="store_true")
    args = parser.parse_args()

    baseline = discover_records(args.baseline_dir, action_mode="policy")
    residual = discover_records(args.residual_dir, action_mode="residual")
    accepted, quarantined = build_pairs(
        baseline,
        residual,
        intervention_replans=parse_replans(args.intervention_replans),
        rewards=load_reward_rows(args.reward_jsonl),
        proprio_atol=args.proprio_atol,
        action_atol=args.action_atol,
        local_progress_threshold=args.local_progress_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "accepted_pairs.jsonl", accepted)
    _write_jsonl(args.output_dir / "pairing_quarantine.jsonl", quarantined)
    summary = {
        "schema_version": "robotwin_single_intervention_pair_audit_v1",
        "baseline_dir": str(args.baseline_dir.expanduser().resolve()),
        "residual_dir": str(args.residual_dir.expanduser().resolve()),
        "requested_intervention_replans": args.intervention_replans,
        "accepted_pair_count": len(accepted),
        "quarantined_pair_count": len(quarantined),
        "label_counts": dict(Counter(row["label"] for row in accepted)),
        "quarantine_reason_counts": dict(
            Counter(
                reason
                for row in quarantined
                for reason in row["quarantine_reasons"]
            )
        ),
    }
    (args.output_dir / "pairing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_accepted and not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
