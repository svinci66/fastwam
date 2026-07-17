"""Calibrate fixed LIBERO camera-reward weights without validation-seed leakage.

The input is Reward V2's transition JSONL.  Camera scales and a fixed agent/wrist
weight are selected using one calibration seed.  The selected values are then frozen
and evaluated once on a different validation seed.  No simulator or neural model is
started by this script.
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

from experiments.libero.diagnose_camera_delta_rewards import binary_auc


DEFAULT_AGENT_WEIGHTS = (0.25, 0.4, 0.5, 0.6, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=1042)
    parser.add_argument(
        "--agent-weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_AGENT_WEIGHTS),
    )
    parser.add_argument("--scale-quantile", type=float, default=0.90)
    parser.add_argument("--goal-specificity-threshold", type=float, default=0.70)
    parser.add_argument("--success-auc-threshold", type=float, default=0.90)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def extract_camera_rewards(
    row: dict[str, Any],
    *,
    wrong_goal: bool = False,
) -> dict[str, float]:
    """Recover agent and wrist rewards from stored agent and equal-dual values."""
    key = "wrong_candidate_rewards" if wrong_goal else "candidate_rewards"
    candidates = row[key]
    agent = float(candidates["agent_delta_alignment"])
    dual = float(candidates["dual_delta_alignment"])
    wrist = float(2.0 * dual - agent)
    return {"agent": agent, "wrist": wrist}


def calibrate_camera_scales(
    rows: list[dict[str, Any]],
    *,
    calibration_seed: int,
    quantile: float = 0.90,
) -> dict[str, float]:
    """Estimate positive per-camera scales using calibration policy/noise only.

    Division by an absolute-value quantile preserves reward sign and a zero no-op
    reference.  Excluding zero transitions prevents abundant no-ops from collapsing
    the scale.  Validation-seed rows are never consulted.
    """
    if not 0 < quantile <= 1:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    calibration_rows = [
        row
        for row in rows
        if int(row.get("policy_seed", -1)) == calibration_seed
        and str(row.get("action_mode", "")) in {"policy", "noise"}
    ]
    if not calibration_rows:
        raise ValueError(
            f"No policy/noise rows found for calibration seed {calibration_seed}."
        )
    scales: dict[str, float] = {}
    for camera in ("agent", "wrist"):
        values = [abs(extract_camera_rewards(row)[camera]) for row in calibration_rows]
        scale = float(np.quantile(values, quantile))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid {camera} camera scale: {scale}")
        scales[camera] = scale
    return scales


def compute_weighted_reward(
    row: dict[str, Any],
    *,
    agent_weight: float,
    scales: dict[str, float],
    wrong_goal: bool = False,
) -> float:
    if not 0 <= agent_weight <= 1:
        raise ValueError(f"agent_weight must be in [0, 1], got {agent_weight}")
    for camera in ("agent", "wrist"):
        if camera not in scales or scales[camera] <= 0:
            raise ValueError(f"A positive {camera!r} scale is required.")
    rewards = extract_camera_rewards(row, wrong_goal=wrong_goal)
    wrist_weight = 1.0 - agent_weight
    return float(
        agent_weight * rewards["agent"] / scales["agent"]
        + wrist_weight * rewards["wrist"] / scales["wrist"]
    )


def _episode_key(row: dict[str, Any]) -> tuple[int, str, int, int, str]:
    return (
        int(row.get("policy_seed", -1)),
        str(row.get("task_suite", "")),
        int(row.get("task_id", -1)),
        int(row.get("trial_idx", -1)),
        str(row.get("action_mode", "unknown")),
    )


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate_weight(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    agent_weight: float,
    scales: dict[str, float],
) -> dict[str, Any]:
    seed_rows = [row for row in rows if int(row.get("policy_seed", -1)) == seed]
    if not seed_rows:
        raise ValueError(f"No rows found for seed {seed}.")

    transition_scores: dict[str, list[float]] = defaultdict(list)
    grouped: dict[tuple[int, str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    correct_beats_wrong: list[bool] = []
    for row in seed_rows:
        reward = compute_weighted_reward(
            row,
            agent_weight=agent_weight,
            scales=scales,
        )
        transition_scores[str(row["action_mode"])].append(reward)
        grouped[_episode_key(row)].append(row)
        if "wrong_candidate_rewards" in row:
            wrong_reward = compute_weighted_reward(
                row,
                agent_weight=agent_weight,
                scales=scales,
                wrong_goal=True,
            )
            correct_beats_wrong.append(reward > wrong_reward)

    episodes: list[dict[str, Any]] = []
    for key, episode_rows in sorted(grouped.items()):
        rewards = [
            compute_weighted_reward(row, agent_weight=agent_weight, scales=scales)
            for row in episode_rows
        ]
        episodes.append(
            {
                "trial_idx": key[3],
                "action_mode": key[4],
                "success": bool(episode_rows[0].get("success", False)),
                "mean_reward": float(np.mean(rewards)),
            }
        )

    episode_by_mode: dict[str, list[float]] = defaultdict(list)
    paired_scores: dict[int, dict[str, float]] = defaultdict(dict)
    paired_success: dict[int, dict[str, bool]] = defaultdict(dict)
    for episode in episodes:
        mode = str(episode["action_mode"])
        trial = int(episode["trial_idx"])
        episode_by_mode[mode].append(float(episode["mean_reward"]))
        paired_scores[trial][mode] = float(episode["mean_reward"])
        paired_success[trial][mode] = bool(episode["success"])

    fully_paired = [
        scores
        for scores in paired_scores.values()
        if {"policy", "noise", "zero"} <= scores.keys()
    ]
    success_failure_pairs: list[bool] = []
    for trial, scores in paired_scores.items():
        modes = sorted(scores)
        for left_index, left_mode in enumerate(modes):
            for right_mode in modes[left_index + 1 :]:
                left_success = paired_success[trial][left_mode]
                right_success = paired_success[trial][right_mode]
                if left_success == right_success:
                    continue
                successful_mode = left_mode if left_success else right_mode
                failed_mode = right_mode if left_success else left_mode
                success_failure_pairs.append(
                    scores[successful_mode] > scores[failed_mode]
                )

    labels = [bool(episode["success"]) for episode in episodes]
    episode_scores = [float(episode["mean_reward"]) for episode in episodes]
    return {
        "seed": seed,
        "agent_weight": float(agent_weight),
        "wrist_weight": float(1.0 - agent_weight),
        "num_transitions": len(seed_rows),
        "num_episodes": len(episodes),
        "mean_transition_reward_by_action_mode": {
            mode: _mean(values) for mode, values in sorted(transition_scores.items())
        },
        "mean_episode_reward_by_action_mode": {
            mode: _mean(values) for mode, values in sorted(episode_by_mode.items())
        },
        "correct_goal_beats_wrong_fraction": _mean(
            [float(value) for value in correct_beats_wrong]
        ),
        "episode_success_roc_auc": binary_auc(labels, episode_scores),
        "num_fully_paired_trials": len(fully_paired),
        "paired_policy_gt_noise_gt_zero_fraction": _mean(
            [
                float(scores["policy"] > scores["noise"] > scores["zero"])
                for scores in fully_paired
            ]
        ),
        "num_success_failure_mode_pairs": len(success_failure_pairs),
        "successful_mode_beats_failed_mode_fraction": _mean(
            [float(value) for value in success_failure_pairs]
        ),
        "paired_trial_rewards": [
            {
                "trial_idx": trial,
                "success_by_mode": paired_success[trial],
                "reward_by_mode": scores,
                "policy_gt_noise_gt_zero": bool(
                    {"policy", "noise", "zero"} <= scores.keys()
                    and scores["policy"] > scores["noise"] > scores["zero"]
                ),
            }
            for trial, scores in sorted(paired_scores.items())
        ],
    }


def candidate_passes(
    result: dict[str, Any],
    *,
    goal_specificity_threshold: float,
    success_auc_threshold: float,
) -> bool:
    return bool(
        result["correct_goal_beats_wrong_fraction"] >= goal_specificity_threshold
        and result["episode_success_roc_auc"] >= success_auc_threshold
        and result["paired_policy_gt_noise_gt_zero_fraction"] == 1.0
        and result["successful_mode_beats_failed_mode_fraction"] == 1.0
    )


def select_calibration_candidate(
    candidates: list[dict[str, Any]],
    *,
    goal_specificity_threshold: float,
    success_auc_threshold: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one candidate is required.")
    passing = [
        candidate
        for candidate in candidates
        if candidate_passes(
            candidate,
            goal_specificity_threshold=goal_specificity_threshold,
            success_auc_threshold=success_auc_threshold,
        )
    ]
    pool = passing or candidates
    selected = max(
        pool,
        key=lambda candidate: (
            float(candidate["correct_goal_beats_wrong_fraction"]),
            float(candidate["paired_policy_gt_noise_gt_zero_fraction"]),
            float(candidate["successful_mode_beats_failed_mode_fraction"]),
            float(candidate["episode_success_roc_auc"]),
            -abs(float(candidate["agent_weight"]) - 0.5),
        ),
    )
    return {
        "selection_status": "passed_calibration_gates" if passing else "no_candidate_passed",
        "num_passing_candidates": len(passing),
        "selected_agent_weight": float(selected["agent_weight"]),
        "selected_wrist_weight": float(selected["wrist_weight"]),
        "selected_calibration_result": selected,
    }


def main() -> None:
    args = parse_args()
    if args.calibration_seed == args.validation_seed:
        raise ValueError("Calibration and validation seeds must differ.")
    if not 0 <= args.goal_specificity_threshold <= 1:
        raise ValueError("--goal-specificity-threshold must be in [0, 1].")
    if not 0 <= args.success_auc_threshold <= 1:
        raise ValueError("--success-auc-threshold must be in [0, 1].")
    weights = [float(weight) for weight in args.agent_weights]
    if not weights or any(weight < 0 or weight > 1 for weight in weights):
        raise ValueError("--agent-weights must contain values in [0, 1].")
    if len(set(weights)) != len(weights):
        raise ValueError("--agent-weights must not contain duplicates.")

    input_path = Path(args.input_jsonl)
    rows = load_rows(input_path)
    scales = calibrate_camera_scales(
        rows,
        calibration_seed=args.calibration_seed,
        quantile=args.scale_quantile,
    )
    calibration_candidates = [
        evaluate_weight(
            rows,
            seed=args.calibration_seed,
            agent_weight=weight,
            scales=scales,
        )
        for weight in weights
    ]
    selection = select_calibration_candidate(
        calibration_candidates,
        goal_specificity_threshold=args.goal_specificity_threshold,
        success_auc_threshold=args.success_auc_threshold,
    )
    selected_weight = float(selection["selected_agent_weight"])
    validation_result = evaluate_weight(
        rows,
        seed=args.validation_seed,
        agent_weight=selected_weight,
        scales=scales,
    )
    validation_passes = candidate_passes(
        validation_result,
        goal_specificity_threshold=args.goal_specificity_threshold,
        success_auc_threshold=args.success_auc_threshold,
    )

    result = {
        "status": "offline_fixed_weight_calibration",
        "protocol": {
            "calibration_seed": args.calibration_seed,
            "validation_seed": args.validation_seed,
            "validation_seed_used_for_selection": False,
            "scale_method": (
                f"seed-{args.calibration_seed} policy/noise absolute reward "
                f"quantile {args.scale_quantile}"
            ),
            "candidate_agent_weights": weights,
            "candidate_wrist_weights": [float(1.0 - weight) for weight in weights],
            "selection_order": (
                "pass all gates, then maximize correct-vs-wrong; tie-break by paired "
                "ordering, success-pair ordering, AUC, and closeness to equal weights"
            ),
            "goal_specificity_threshold": args.goal_specificity_threshold,
            "success_auc_threshold": args.success_auc_threshold,
        },
        "camera_scales": scales,
        "calibration_candidates": calibration_candidates,
        "selection": selection,
        "validation_result": validation_result,
        "validation_passes_all_frozen_gates": validation_passes,
        "decision": (
            "fixed_normalized_weight_passed"
            if selection["selection_status"] == "passed_calibration_gates" and validation_passes
            else "do_not_use_for_rl"
        ),
        "input_jsonl": str(input_path.resolve()),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
