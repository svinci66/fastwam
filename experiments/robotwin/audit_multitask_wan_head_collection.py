#!/usr/bin/env python3
"""Audit balanced multi-task RoboTwin collection and Wan-head reward coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_REWARD_SCHEMA = "robotwin_natural_failure_wan_vae_pair_reward_v1"
EXPECTED_TASKS = (
    "open_microwave",
    "hanging_mug",
    "place_can_basket",
    "blocks_ranking_size",
)


def comma_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated list without duplicates"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--reward-json", type=Path, required=True)
    parser.add_argument(
        "--tasks", type=comma_values, default=list(EXPECTED_TASKS)
    )
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--minimum-failures-per-task", type=int, required=True)
    parser.add_argument("--minimum-macro-pairwise-accuracy", type=float, default=0.60)
    parser.add_argument("--minimum-positive-margin-tasks", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--require-training-ready",
        action="store_true",
        help="Exit 3 when collection is complete but the frozen reward stop rule fails.",
    )
    return parser.parse_args()


def require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def main() -> None:
    args = parse_args()
    tasks = list(args.tasks)
    if args.episodes_per_task <= 0:
        raise ValueError("episodes-per-task must be positive")
    if not 0 <= args.minimum_failures_per_task <= args.episodes_per_task:
        raise ValueError("minimum-failures-per-task must be in [0, episodes-per-task]")
    if not 0.0 <= args.minimum_macro_pairwise_accuracy <= 1.0:
        raise ValueError("minimum-macro-pairwise-accuracy must be in [0, 1]")
    if not 0 <= args.minimum_positive_margin_tasks <= len(tasks):
        raise ValueError("minimum-positive-margin-tasks is outside the task count")

    screen = json.loads(args.screen_summary.read_text(encoding="utf-8"))
    reward = json.loads(args.reward_json.read_text(encoding="utf-8"))
    if reward.get("schema_version") != EXPECTED_REWARD_SCHEMA:
        raise ValueError(f"unexpected reward schema: {reward.get('schema_version')!r}")
    if reward.get("reward_cameras") != ["head"]:
        raise ValueError("formal multi-task reward must use only the head camera")

    screen_tasks = require_mapping(screen, "per_task")
    unknown_screen_tasks = sorted(set(screen_tasks) - set(tasks))
    missing_screen_tasks = sorted(set(tasks) - set(screen_tasks))
    pair_rows = reward.get("pairs")
    if not isinstance(pair_rows, list):
        raise ValueError("reward pairs must be a list")

    per_task: dict[str, dict[str, Any]] = {}
    collection_failures: list[str] = []
    pairwise_accuracies: list[float] = []
    positive_margin_tasks = 0
    for task in tasks:
        screen_row = screen_tasks.get(task, {})
        episodes = int(screen_row.get("episodes", 0))
        failures = int(screen_row.get("natural_failures", 0))
        successes = int(screen_row.get("fastwam_successes", 0))
        task_pairs = [row for row in pair_rows if str(row.get("task")) == task]
        margins = [float(row["success_minus_failure"]) for row in task_pairs]
        pair_count = len(task_pairs)
        correctly_ranked = sum(margin > 0.0 for margin in margins)
        accuracy = float(correctly_ranked / pair_count) if pair_count else 0.0
        mean_margin = float(np.mean(margins)) if margins else None
        pairwise_accuracies.append(accuracy)
        if mean_margin is not None and np.isfinite(mean_margin) and mean_margin > 0.0:
            positive_margin_tasks += 1

        if episodes != args.episodes_per_task:
            collection_failures.append(
                f"{task}: episodes={episodes}, expected={args.episodes_per_task}"
            )
        if successes + failures != episodes:
            collection_failures.append(
                f"{task}: successes+failures={successes + failures}, episodes={episodes}"
            )
        if failures < args.minimum_failures_per_task:
            collection_failures.append(
                f"{task}: natural_failures={failures}, minimum={args.minimum_failures_per_task}"
            )
        if pair_count != failures:
            collection_failures.append(
                f"{task}: reward_pairs={pair_count}, natural_failures={failures}"
            )
        per_task[task] = {
            "episodes": episodes,
            "fastwam_successes": successes,
            "natural_failures": failures,
            "reward_pairs": pair_count,
            "correctly_ranked": correctly_ranked,
            "pairwise_accuracy": accuracy,
            "mean_success_minus_failure": mean_margin,
        }

    if unknown_screen_tasks:
        collection_failures.append(f"unexpected screen tasks: {unknown_screen_tasks}")
    if missing_screen_tasks:
        collection_failures.append(f"missing screen tasks: {missing_screen_tasks}")
    unknown_pair_tasks = sorted(
        {str(row.get("task")) for row in pair_rows} - set(tasks)
    )
    if unknown_pair_tasks:
        collection_failures.append(f"unexpected reward tasks: {unknown_pair_tasks}")

    collection_complete = not collection_failures
    macro_accuracy = float(np.mean(pairwise_accuracies))
    reward_validation_pass = bool(
        macro_accuracy >= args.minimum_macro_pairwise_accuracy
        and positive_margin_tasks >= args.minimum_positive_margin_tasks
    )
    training_ready = bool(collection_complete and reward_validation_pass)
    result = {
        "schema_version": "robotwin_multitask_wan_head_collection_audit_v1",
        "tasks": tasks,
        "selection_basis": {
            "open_microwave": "FastWAM candidate-screen success rate 1/5; articulated pull",
            "hanging_mug": "FastWAM candidate-screen success rate 2/5; precision hanging",
            "place_can_basket": "FastWAM candidate-screen success rate 2/5; grasp and place",
            "blocks_ranking_size": "FastWAM candidate-screen success rate 3/5; multi-object ordering",
        },
        "collection_stop_rule": {
            "episodes_per_task": args.episodes_per_task,
            "minimum_natural_failures_per_task": args.minimum_failures_per_task,
            "exact_reward_pair_coverage": True,
        },
        "reward_stop_rule": {
            "minimum_macro_pairwise_accuracy": args.minimum_macro_pairwise_accuracy,
            "minimum_positive_mean_margin_tasks": args.minimum_positive_margin_tasks,
        },
        "per_task": per_task,
        "macro_pairwise_accuracy": macro_accuracy,
        "positive_mean_margin_tasks": positive_margin_tasks,
        "collection_complete": collection_complete,
        "reward_validation_pass": reward_validation_pass,
        "training_ready": training_ready,
        "collection_failures": collection_failures,
        "screen_summary": str(args.screen_summary.resolve()),
        "reward_json": str(args.reward_json.resolve()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if not collection_complete:
        raise SystemExit(2)
    if args.require_training_ready and not training_ready:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
