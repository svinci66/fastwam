#!/usr/bin/env python3
"""Test short-horizon imagination returns on controlled RoboTwin triplets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _single_intervention_replan(row: dict[str, Any]) -> int:
    value = row.get("action_noise_replans")
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(
            "controlled record must contain exactly one intervention replan, "
            f"got {value!r}"
        )
    return int(value[0])


def validate_temporal_returns(
    transition_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    *,
    horizons: tuple[int, ...] = (1, 2, 3),
    primary_horizon: int = 3,
    corrupt_behavior: str = "controlled_corrupt_0.050",
    correct_behavior: str = "controlled_correct_0.050",
) -> dict[str, Any]:
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")
    if primary_horizon not in horizons:
        raise ValueError("primary_horizon must be included in horizons")

    transitions: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for row in transition_rows:
        if not bool(row.get("alignment_valid")):
            continue
        key = (
            str(row["task_name"]),
            int(row["trial_idx"]),
            str(row["behavior"]),
            int(row["replan_idx"]),
        )
        if key in transitions:
            raise ValueError(f"duplicate aligned transition: {key}")
        transitions[key] = row

    episode_success = {
        (str(row["task_name"]), int(row["trial_idx"]), str(row["behavior"])): bool(
            row["success"]
        )
        for row in episode_rows
    }
    trial_keys = sorted(
        {
            (task, trial)
            for task, trial, behavior, _ in transitions
            if behavior == corrupt_behavior
        }
    )
    pairs: list[dict[str, Any]] = []
    behaviors = {
        "clean": "policy",
        "corrupted": corrupt_behavior,
        "corrected": correct_behavior,
    }
    for task, trial in trial_keys:
        corrupt_rows = [
            row
            for (row_task, row_trial, behavior, _), row in transitions.items()
            if row_task == task
            and row_trial == trial
            and behavior == corrupt_behavior
        ]
        intervention_replans = {
            _single_intervention_replan(row) for row in corrupt_rows
        }
        if len(intervention_replans) != 1:
            raise ValueError(
                f"{task}/{trial} has inconsistent interventions: "
                f"{sorted(intervention_replans)}"
            )
        intervention = next(iter(intervention_replans))

        intervention_rows: dict[str, dict[str, Any]] = {}
        for label, behavior in behaviors.items():
            key = (task, trial, behavior, intervention)
            if key not in transitions:
                raise ValueError(f"missing intervention transition: {key}")
            intervention_rows[label] = transitions[key]
        for field in (
            "initial_observation_sha256",
            "current_observation_sha256",
            "baseline_actions_sha256",
        ):
            if len({str(row[field]) for row in intervention_rows.values()}) != 1:
                raise ValueError(f"unpaired {field} for {task}/{trial}")

        horizon_results: dict[str, Any] = {}
        for horizon in horizons:
            returns: dict[str, float] = {}
            for label, behavior in behaviors.items():
                values = []
                for replan in range(intervention, intervention + horizon):
                    key = (task, trial, behavior, replan)
                    if key not in transitions:
                        raise ValueError(
                            f"missing aligned reward for {key}; cannot compute H={horizon}"
                        )
                    values.append(float(transitions[key]["imagination_reward"]))
                returns[label] = float(np.sum(values))
            clean_correct_gap = abs(returns["clean"] - returns["corrected"])
            margin = min(returns["clean"], returns["corrected"]) - returns[
                "corrupted"
            ]
            horizon_results[str(horizon)] = {
                "return": returns,
                "clean_correct_abs_gap": clean_correct_gap,
                "min_clean_correct_minus_corrupted": margin,
                "order_pass": margin > 0.0,
            }
        pairs.append(
            {
                "task_name": task,
                "trial_idx": trial,
                "environment_seed": intervention_rows["clean"].get(
                    "environment_seed"
                ),
                "intervention_replan": intervention,
                "corrupted_success": episode_success[
                    (task, trial, corrupt_behavior)
                ],
                "horizons": horizon_results,
            }
        )

    horizon_summary: dict[str, Any] = {}
    for horizon in horizons:
        key = str(horizon)
        margins = np.asarray(
            [pair["horizons"][key]["min_clean_correct_minus_corrupted"] for pair in pairs],
            dtype=np.float64,
        )
        passes = np.asarray(
            [pair["horizons"][key]["order_pass"] for pair in pairs],
            dtype=np.float64,
        )
        gaps = np.asarray(
            [pair["horizons"][key]["clean_correct_abs_gap"] for pair in pairs],
            dtype=np.float64,
        )
        failed = [pair for pair in pairs if not pair["corrupted_success"]]
        failed_fraction = (
            None
            if not failed
            else float(
                np.mean([pair["horizons"][key]["order_pass"] for pair in failed])
            )
        )
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pair in pairs:
            by_task[str(pair["task_name"])].append(pair)
        horizon_summary[key] = {
            "num_pairs": len(pairs),
            "order_fraction": float(np.mean(passes)),
            "failed_corrupt_order_fraction": failed_fraction,
            "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)),
            "min_margin": float(np.min(margins)),
            "max_clean_correct_abs_gap": float(np.max(gaps)),
            "by_task": {
                task: {
                    "num_pairs": len(values),
                    "order_fraction": float(
                        np.mean([pair["horizons"][key]["order_pass"] for pair in values])
                    ),
                    "mean_margin": float(
                        np.mean(
                            [
                                pair["horizons"][key][
                                    "min_clean_correct_minus_corrupted"
                                ]
                                for pair in values
                            ]
                        )
                    ),
                }
                for task, values in sorted(by_task.items())
            },
        }

    primary = horizon_summary[str(primary_horizon)]
    gates = {
        "five_complete_pairs": len(pairs) == 5,
        "clean_correct_return_identical": (
            primary["max_clean_correct_abs_gap"] <= 1e-7
        ),
        "primary_order_fraction_ge_0_80": primary["order_fraction"] >= 0.8,
        "primary_failed_corrupt_order_eq_1": (
            primary["failed_corrupt_order_fraction"] == 1.0
        ),
        "primary_mean_margin_gt_0": primary["mean_margin"] > 0.0,
    }
    return {
        "schema_version": "controlled_imagination_temporal_validation_v1",
        "primary_horizon": primary_horizon,
        "horizon_summary": horizon_summary,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-rewards", type=Path, required=True)
    parser.add_argument("--episode-rewards", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--horizons", default="1,2,3")
    parser.add_argument("--primary-horizon", type=int, default=3)
    args = parser.parse_args()
    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    summary = validate_temporal_returns(
        _read_jsonl(args.transition_rewards),
        _read_jsonl(args.episode_rewards),
        horizons=horizons,
        primary_horizon=args.primary_horizon,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
