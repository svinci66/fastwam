"""Offline scan for combining progress and absolute imagination matching.

The combined reward is

    progress + match_weight * (zero_action_reference - distance_after)

where the reference is estimated per task from zero-action transitions.  The
centering keeps the zero-action control near zero while preserving the ordering
between correct and wrong goals for the absolute-match term.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_WEIGHTS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--goal-specificity-threshold", type=float, default=0.70)
    parser.add_argument(
        "--selection-margin",
        type=float,
        default=0.02,
        help="Extra goal-specificity margin used to avoid selecting a threshold-edge weight.",
    )
    parser.add_argument(
        "--fixed-zero-reference",
        type=float,
        default=None,
        help=(
            "Use a previously calibrated zero-action distance for single-task validation "
            "instead of estimating it from the input rows."
        ),
    )
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _task_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["task_suite"]), int(row["task_id"])


def _episode_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["task_suite"]),
        int(row["task_id"]),
        int(row["trial_idx"]),
        str(row["action_mode"]),
    )


def estimate_zero_action_references(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], float]:
    """Return a robust per-task no-op distance reference."""
    distances: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("action_mode")) == "zero":
            distances[_task_key(row)].append(float(row["distance_after"]))
    task_keys = {_task_key(row) for row in rows}
    missing = sorted(task_keys - distances.keys())
    if missing:
        raise ValueError(f"Missing zero-action calibration rows for tasks: {missing}")
    return {key: float(np.median(values)) for key, values in distances.items()}


def evaluate_weight(
    rows: list[dict[str, Any]],
    *,
    weight: float,
    zero_references: dict[tuple[str, int], float],
) -> dict[str, Any]:
    if weight < 0:
        raise ValueError(f"weight must be non-negative, got {weight}")

    transition_rewards_by_mode: dict[str, list[float]] = defaultdict(list)
    episode_rewards: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    episode_success: dict[tuple[str, int, int, str], bool] = {}
    correct_beats_wrong: list[float] = []

    for row in rows:
        reference = zero_references[_task_key(row)]
        reward = float(row["imagination_progress"]) + weight * (
            reference - float(row["distance_after"])
        )
        mode = str(row["action_mode"])
        transition_rewards_by_mode[mode].append(reward)
        key = _episode_key(row)
        episode_rewards[key].append(reward)
        episode_success[key] = bool(row["success"])

        if "wrong_goal_imagination_progress" in row and "wrong_goal_distance_after" in row:
            wrong_reward = float(row["wrong_goal_imagination_progress"]) + weight * (
                reference - float(row["wrong_goal_distance_after"])
            )
            correct_beats_wrong.append(float(reward > wrong_reward))

    episode_means = {key: float(np.mean(values)) for key, values in episode_rewards.items()}
    episode_by_mode: dict[str, list[float]] = defaultdict(list)
    episode_by_success: dict[str, list[float]] = defaultdict(list)
    paired_by_trial: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for key, reward in episode_means.items():
        suite, task_id, trial_idx, mode = key
        episode_by_mode[mode].append(reward)
        episode_by_success[str(episode_success[key]).lower()].append(reward)
        paired_by_trial[(suite, task_id, trial_idx)][mode] = reward

    fully_paired = [
        values for values in paired_by_trial.values() if {"policy", "noise", "zero"} <= values.keys()
    ]
    full_order = [values["policy"] > values["noise"] > values["zero"] for values in fully_paired]
    success_mean = _mean(episode_by_success.get("true", []))
    failure_mean = _mean(episode_by_success.get("false", []))
    goal_specificity = _mean(correct_beats_wrong)

    return {
        "match_weight": float(weight),
        "correct_goal_beats_wrong_fraction": goal_specificity,
        "num_wrong_goal_comparisons": len(correct_beats_wrong),
        "mean_transition_reward_by_action_mode": {
            key: _mean(values) for key, values in sorted(transition_rewards_by_mode.items())
        },
        "mean_episode_reward_by_action_mode": {
            key: _mean(values) for key, values in sorted(episode_by_mode.items())
        },
        "mean_episode_reward_by_success": {
            key: _mean(values) for key, values in sorted(episode_by_success.items())
        },
        "num_fully_paired_trials": len(fully_paired),
        "paired_policy_gt_noise_gt_zero_fraction": _mean(
            [float(value) for value in full_order]
        ),
        "paired_trial_rewards": [
            {
                "task_suite": key[0],
                "task_id": key[1],
                "trial_idx": key[2],
                "mean_episode_reward_by_action_mode": {
                    mode: float(reward) for mode, reward in sorted(values.items())
                },
                "policy_gt_noise_gt_zero": bool(
                    values["policy"] > values["noise"] > values["zero"]
                ),
            }
            for key, values in sorted(paired_by_trial.items())
            if {"policy", "noise", "zero"} <= values.keys()
        ],
        "success_mean_exceeds_failure_mean": bool(
            success_mean is not None and failure_mean is not None and success_mean > failure_mean
        ),
    }


def scan_weights(
    rows: list[dict[str, Any]],
    *,
    weights: list[float] | tuple[float, ...],
    goal_specificity_threshold: float,
    selection_margin: float = 0.02,
    fixed_zero_reference: float | None = None,
) -> dict[str, Any]:
    if not 0.0 <= goal_specificity_threshold <= 1.0:
        raise ValueError("goal_specificity_threshold must be in [0, 1]")
    if selection_margin < 0 or goal_specificity_threshold + selection_margin > 1.0:
        raise ValueError("selection_margin must be non-negative and keep the target at most 1")
    if not rows:
        raise ValueError("No reward rows were provided")

    if fixed_zero_reference is None:
        zero_references = estimate_zero_action_references(rows)
        zero_reference_source = "median_of_input_zero_action_rows"
    else:
        if not 0.0 <= fixed_zero_reference <= 2.0:
            raise ValueError("fixed_zero_reference must be a cosine distance in [0, 2]")
        task_keys = sorted({_task_key(row) for row in rows})
        if len(task_keys) != 1:
            raise ValueError(
                "fixed_zero_reference supports one task per scan; "
                f"found {len(task_keys)} tasks"
            )
        zero_references = {task_keys[0]: float(fixed_zero_reference)}
        zero_reference_source = "fixed_external_calibration"
    candidates = [
        evaluate_weight(rows, weight=weight, zero_references=zero_references)
        for weight in sorted(set(float(value) for value in weights))
    ]
    for candidate in candidates:
        goal_fraction = candidate["correct_goal_beats_wrong_fraction"]
        candidate["passes_selection_criteria"] = bool(
            goal_fraction is not None
            and goal_fraction >= goal_specificity_threshold
            and candidate["paired_policy_gt_noise_gt_zero_fraction"] == 1.0
            and candidate["success_mean_exceeds_failure_mean"]
        )
    passing = [candidate for candidate in candidates if candidate["passes_selection_criteria"]]
    minimum_passing = (
        min(passing, key=lambda candidate: candidate["match_weight"]) if passing else None
    )
    recommended_threshold = goal_specificity_threshold + selection_margin
    recommended = [
        candidate
        for candidate in passing
        if candidate["correct_goal_beats_wrong_fraction"] >= recommended_threshold
    ]
    selected = (
        min(recommended, key=lambda candidate: candidate["match_weight"])
        if recommended
        else minimum_passing
    )

    return {
        "reward_definition": (
            "imagination_progress + match_weight * "
            "(zero_action_distance_reference_per_task - distance_after)"
        ),
        "goal_specificity_threshold": float(goal_specificity_threshold),
        "selection_margin": float(selection_margin),
        "recommended_goal_specificity_threshold": float(recommended_threshold),
        "num_transitions": len(rows),
        "zero_action_distance_reference_source": zero_reference_source,
        "zero_action_distance_reference_by_task": {
            f"{suite}/task_{task_id}": value
            for (suite, task_id), value in sorted(zero_references.items())
        },
        "candidates": candidates,
        "minimum_passing_candidate": minimum_passing,
        "selected_candidate": selected,
    }


def main() -> None:
    args = parse_args()
    result = scan_weights(
        load_rows(args.input_jsonl),
        weights=args.weights,
        goal_specificity_threshold=args.goal_specificity_threshold,
        selection_margin=args.selection_margin,
        fixed_zero_reference=args.fixed_zero_reference,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
