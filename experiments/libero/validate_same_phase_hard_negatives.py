"""Validate frozen camera-aware rewards with same-phase hard negative goals.

For every saved transition, candidates must come from the same seed, task, action
mode, and temporal phase but a different trial.  We keep the K candidates whose
current observations are most similar, then choose the candidate with the most
different future goal.  Selection uses no executed actual frame or reward value.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.calibrate_camera_reward_weights import extract_camera_rewards
from experiments.libero.imagination_reward_utils import (
    compute_delta_alignment_reward,
    cosine_distance,
)


CAMERA_NAMES = ("agent", "wrist")
CANDIDATE_NAMES = ("agent", "wrist", "raw_dual", "frozen_normalized")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--camera-calibration-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-selections-jsonl", default=None)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--goal-specificity-threshold", type=float, default=0.70)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_camera_features(cache_path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(cache_path, allow_pickle=False) as cache:
        paths = [str(path) for path in cache["paths"].tolist()]
        return {
            camera: {
                path: feature for path, feature in zip(paths, np.asarray(cache[camera]))
            }
            for camera in CAMERA_NAMES
        }


def dual_camera_distance(
    features: dict[str, dict[str, np.ndarray]],
    left_path: str,
    right_path: str,
) -> float:
    return float(
        np.mean(
            [
                cosine_distance(features[camera][left_path], features[camera][right_path])
                for camera in CAMERA_NAMES
            ]
        )
    )


def _pool_key(row: dict[str, Any]) -> tuple[int, str, int, str, str]:
    return (
        int(row.get("policy_seed", -1)),
        str(row.get("task_suite", "")),
        int(row.get("task_id", -1)),
        str(row.get("action_mode", "unknown")),
        str(row.get("phase", "unknown")),
    )


def select_same_phase_hard_negative(
    row: dict[str, Any],
    rows: list[dict[str, Any]],
    features: dict[str, dict[str, np.ndarray]],
    *,
    nearest_k: int,
) -> dict[str, Any] | None:
    """Select a future-different goal among current-observation nearest neighbors."""
    if nearest_k <= 0:
        raise ValueError(f"nearest_k must be positive, got {nearest_k}")
    pool = [
        candidate
        for candidate in rows
        if _pool_key(candidate) == _pool_key(row)
        and int(candidate.get("trial_idx", -1)) != int(row.get("trial_idx", -1))
    ]
    if not pool:
        return None

    ranked = sorted(
        (
            dual_camera_distance(
                features,
                str(row["current_path"]),
                str(candidate["current_path"]),
            ),
            str(candidate["record_dir"]),
            candidate,
        )
        for candidate in pool
    )
    neighborhood = ranked[:nearest_k]
    scored: list[tuple[float, float, str, dict[str, Any]]] = []
    for current_distance, record_dir, candidate in neighborhood:
        goal_distance = dual_camera_distance(
            features,
            str(row["goal_path"]),
            str(candidate["goal_path"]),
        )
        scored.append((goal_distance, -current_distance, record_dir, candidate))
    goal_distance, negative_current_distance, _, selected = max(scored)
    return {
        "record": selected,
        "current_distance": float(-negative_current_distance),
        "goal_distance": float(goal_distance),
        "candidate_pool_size": len(pool),
        "neighborhood_size": len(neighborhood),
    }


def compute_camera_rewards_for_goal(
    row: dict[str, Any],
    goal_path: str,
    features: dict[str, dict[str, np.ndarray]],
) -> dict[str, float]:
    return {
        camera: float(
            compute_delta_alignment_reward(
                features[camera][str(row["current_path"])],
                features[camera][str(row["actual_path"])],
                features[camera][goal_path],
            )["delta_alignment_reward"]
        )
        for camera in CAMERA_NAMES
    }


def combine_rewards(
    camera_rewards: dict[str, float],
    *,
    agent_weight: float,
    scales: dict[str, float],
) -> dict[str, float]:
    wrist_weight = 1.0 - agent_weight
    return {
        "agent": float(camera_rewards["agent"]),
        "wrist": float(camera_rewards["wrist"]),
        "raw_dual": float(0.5 * (camera_rewards["agent"] + camera_rewards["wrist"])),
        "frozen_normalized": float(
            agent_weight * camera_rewards["agent"] / scales["agent"]
            + wrist_weight * camera_rewards["wrist"] / scales["wrist"]
        ),
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def summarize_selections(
    selections: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    seed_rows = [row for row in selections if int(row["policy_seed"]) == seed]
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        by_phase[str(row["phase"])].append(row)
        by_mode[str(row["action_mode"])].append(row)
    actionable_rows = [
        row for row in seed_rows if str(row["action_mode"]) in {"policy", "noise"}
    ]
    zero_rows = [row for row in seed_rows if str(row["action_mode"]) == "zero"]

    def reward_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            candidate: {
                "correct_beats_hard_fraction": _mean(
                    [
                        float(row["correct_rewards"][candidate] > row["hard_rewards"][candidate])
                        for row in rows
                    ]
                ),
                "mean_correct_minus_hard_margin": _mean(
                    [
                        float(row["correct_rewards"][candidate] - row["hard_rewards"][candidate])
                        for row in rows
                    ]
                ),
            }
            for candidate in CANDIDATE_NAMES
        }

    old_current_distances = [
        float(row["old_negative_current_distance"])
        for row in seed_rows
        if row.get("old_negative_current_distance") is not None
    ]
    old_goal_distances = [
        float(row["old_negative_goal_distance"])
        for row in seed_rows
        if row.get("old_negative_goal_distance") is not None
    ]
    return {
        "seed": seed,
        "num_selections": len(seed_rows),
        "reward_summary": reward_summary(seed_rows),
        "reward_summary_actionable_policy_noise": reward_summary(actionable_rows),
        "reward_summary_zero_noop": reward_summary(zero_rows),
        "reward_summary_by_phase": {
            phase: reward_summary(rows) for phase, rows in sorted(by_phase.items())
        },
        "reward_summary_by_action_mode": {
            mode: reward_summary(rows) for mode, rows in sorted(by_mode.items())
        },
        "hard_negative_current_distance_mean": _mean(
            [float(row["hard_current_distance"]) for row in seed_rows]
        ),
        "hard_negative_current_distance_median": _median(
            [float(row["hard_current_distance"]) for row in seed_rows]
        ),
        "old_negative_current_distance_mean": _mean(old_current_distances),
        "old_negative_current_distance_median": _median(old_current_distances),
        "hard_is_current_closer_than_old_fraction": _mean(
            [
                float(
                    row.get("old_negative_current_distance") is not None
                    and row["hard_current_distance"] < row["old_negative_current_distance"]
                )
                for row in seed_rows
                if row.get("old_negative_current_distance") is not None
            ]
        ),
        "hard_negative_goal_distance_mean": _mean(
            [float(row["hard_goal_distance"]) for row in seed_rows]
        ),
        "hard_negative_goal_distance_median": _median(
            [float(row["hard_goal_distance"]) for row in seed_rows]
        ),
        "old_negative_goal_distance_mean": _mean(old_goal_distances),
        "old_negative_goal_distance_median": _median(old_goal_distances),
        "candidate_pool_size_mean": _mean(
            [float(row["candidate_pool_size"]) for row in seed_rows]
        ),
    }


def main() -> None:
    args = parse_args()
    if args.nearest_k <= 0:
        raise ValueError("--nearest-k must be positive")
    if not 0 <= args.goal_specificity_threshold <= 1:
        raise ValueError("--goal-specificity-threshold must be in [0, 1]")

    rows = load_jsonl(Path(args.input_jsonl))
    features = load_camera_features(Path(args.feature_cache))
    with Path(args.camera_calibration_json).open("r", encoding="utf-8") as stream:
        calibration = json.load(stream)
    scales = {camera: float(calibration["camera_scales"][camera]) for camera in CAMERA_NAMES}
    agent_weight = float(calibration["selection"]["selected_agent_weight"])
    wrist_weight = float(calibration["selection"]["selected_wrist_weight"])

    rows_by_dir = {str(row["record_dir"]): row for row in rows}
    selections: list[dict[str, Any]] = []
    for row in rows:
        selected = select_same_phase_hard_negative(
            row,
            rows,
            features,
            nearest_k=args.nearest_k,
        )
        if selected is None:
            continue
        hard_record = selected["record"]
        correct_camera = extract_camera_rewards(row)
        hard_camera = compute_camera_rewards_for_goal(
            row,
            str(hard_record["goal_path"]),
            features,
        )
        correct_rewards = combine_rewards(
            correct_camera,
            agent_weight=agent_weight,
            scales=scales,
        )
        hard_rewards = combine_rewards(
            hard_camera,
            agent_weight=agent_weight,
            scales=scales,
        )

        old_record = rows_by_dir.get(str(row.get("wrong_goal_record_dir", "")))
        old_current_distance = None
        old_goal_distance = None
        if old_record is not None:
            old_current_distance = dual_camera_distance(
                features,
                str(row["current_path"]),
                str(old_record["current_path"]),
            )
            old_goal_distance = dual_camera_distance(
                features,
                str(row["goal_path"]),
                str(old_record["goal_path"]),
            )

        selections.append(
            {
                "policy_seed": int(row["policy_seed"]),
                "task_suite": str(row["task_suite"]),
                "task_id": int(row["task_id"]),
                "trial_idx": int(row["trial_idx"]),
                "replan_idx": int(row["replan_idx"]),
                "action_mode": str(row["action_mode"]),
                "phase": str(row["phase"]),
                "record_dir": str(row["record_dir"]),
                "hard_negative_record_dir": str(hard_record["record_dir"]),
                "hard_negative_trial_idx": int(hard_record["trial_idx"]),
                "hard_negative_phase": str(hard_record["phase"]),
                "candidate_pool_size": int(selected["candidate_pool_size"]),
                "neighborhood_size": int(selected["neighborhood_size"]),
                "hard_current_distance": float(selected["current_distance"]),
                "hard_goal_distance": float(selected["goal_distance"]),
                "old_negative_current_distance": old_current_distance,
                "old_negative_goal_distance": old_goal_distance,
                "correct_rewards": correct_rewards,
                "hard_rewards": hard_rewards,
            }
        )

    if not selections:
        raise ValueError("No same-phase hard negatives could be selected.")
    seeds = sorted({int(row["policy_seed"]) for row in selections})
    seed_summaries = {
        str(seed): summarize_selections(selections, seed=seed) for seed in seeds
    }
    validation_seed = int(calibration["protocol"]["validation_seed"])
    validation_fraction = seed_summaries[str(validation_seed)]["reward_summary"][
        "frozen_normalized"
    ]["correct_beats_hard_fraction"]
    validation_actionable_fraction = seed_summaries[str(validation_seed)][
        "reward_summary_actionable_policy_noise"
    ]["frozen_normalized"]["correct_beats_hard_fraction"]
    validation_zero_fraction = seed_summaries[str(validation_seed)][
        "reward_summary_zero_noop"
    ]["frozen_normalized"]["correct_beats_hard_fraction"]
    result = {
        "status": "offline_same_phase_hard_negative_validation",
        "protocol": {
            "pool_constraints": (
                "same seed/task/action-mode/phase, different trial"
            ),
            "nearest_k": args.nearest_k,
            "selection_rule": (
                "among K smallest equal-dual current-feature distances, select maximum "
                "equal-dual correct-goal-to-candidate-goal distance"
            ),
            "selection_uses_actual_frame": False,
            "selection_uses_reward_value": False,
            "goal_specificity_threshold": args.goal_specificity_threshold,
        },
        "frozen_reward": {
            "agent_weight": agent_weight,
            "wrist_weight": wrist_weight,
            "camera_scales": scales,
            "source_calibration_json": str(Path(args.camera_calibration_json).resolve()),
        },
        "num_input_rows": len(rows),
        "num_selected_rows": len(selections),
        "num_rows_without_candidate": len(rows) - len(selections),
        "seed_summaries": seed_summaries,
        "validation_seed": validation_seed,
        "validation_correct_beats_hard_fraction": validation_fraction,
        "validation_actionable_correct_beats_hard_fraction": (
            validation_actionable_fraction
        ),
        "validation_zero_correct_beats_hard_fraction": validation_zero_fraction,
        "validation_passes_goal_specificity_threshold": bool(
            validation_fraction >= args.goal_specificity_threshold
        ),
        "validation_actionable_passes_goal_specificity_threshold": bool(
            validation_actionable_fraction >= args.goal_specificity_threshold
        ),
        "diagnostic_interpretation": (
            "The predeclared all-transition gate is retained. Policy/noise is reported "
            "separately because a no-op has near-zero actual feature change and therefore "
            "cannot identify a goal direction; changing the acceptance gate now would be "
            "post-hoc."
        ),
        "decision": (
            "hard_negative_goal_specificity_passed"
            if validation_fraction >= args.goal_specificity_threshold
            else "do_not_use_for_rl"
        ),
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "feature_cache": str(Path(args.feature_cache).resolve()),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    selections_path = (
        Path(args.output_selections_jsonl)
        if args.output_selections_jsonl
        else output_path.with_name(f"{output_path.stem}_selections.jsonl")
    )
    with selections_path.open("w", encoding="utf-8") as stream:
        for selection in selections:
            stream.write(json.dumps(selection, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
