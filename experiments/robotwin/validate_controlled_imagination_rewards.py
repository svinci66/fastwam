#!/usr/bin/env python3
"""Validate imagination reward on controlled clean/corrupt/correct triplets."""

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
            "controlled record must contain exactly one action_noise_replans entry, "
            f"got {value!r}"
        )
    return int(value[0])


def validate(
    transition_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    *,
    corrupt_behavior: str,
    correct_behavior: str,
) -> dict[str, Any]:
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

    episode_index = {
        (str(row["task_name"]), int(row["trial_idx"]), str(row["behavior"])): row
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
    for task, trial in trial_keys:
        corrupt_candidates = [
            row
            for (row_task, row_trial, behavior, _), row in transitions.items()
            if row_task == task
            and row_trial == trial
            and behavior == corrupt_behavior
        ]
        intervention_replans = {
            _single_intervention_replan(row) for row in corrupt_candidates
        }
        if len(intervention_replans) != 1:
            raise ValueError(
                f"{task}/{trial} has inconsistent intervention replans: "
                f"{sorted(intervention_replans)}"
            )
        replan = next(iter(intervention_replans))
        keys = {
            "clean": (task, trial, "policy", replan),
            "corrupted": (task, trial, corrupt_behavior, replan),
            "corrected": (task, trial, correct_behavior, replan),
        }
        if any(key not in transitions for key in keys.values()):
            raise ValueError(f"incomplete reward triplet for {task}/{trial}/replan={replan}")
        rows = {name: transitions[key] for name, key in keys.items()}
        for hash_field in (
            "initial_observation_sha256",
            "current_observation_sha256",
            "baseline_actions_sha256",
        ):
            if len({str(row[hash_field]) for row in rows.values()}) != 1:
                raise ValueError(
                    f"{task}/{trial}/replan={replan} is not paired on {hash_field}"
                )

        rewards = {
            name: float(row["imagination_reward"]) for name, row in rows.items()
        }
        episodes = {
            name: episode_index[(task, trial, str(row["behavior"]))]
            for name, row in rows.items()
        }
        clean_correct_gap = abs(rewards["clean"] - rewards["corrected"])
        corrupt_margin = min(rewards["clean"], rewards["corrected"]) - rewards[
            "corrupted"
        ]
        episode_mean_rewards = {
            name: float(episode["mean_imagination_reward"])
            for name, episode in episodes.items()
        }
        episode_margin = min(
            episode_mean_rewards["clean"], episode_mean_rewards["corrected"]
        ) - episode_mean_rewards["corrupted"]
        pairs.append(
            {
                "task_name": task,
                "trial_idx": trial,
                "environment_seed": rows["clean"].get("environment_seed"),
                "intervention_replan": replan,
                "reward": rewards,
                "clean_correct_abs_gap": clean_correct_gap,
                "min_clean_correct_minus_corrupted": corrupt_margin,
                "local_order_pass": corrupt_margin > 0.0,
                "episode_mean_order_pass": episode_margin > 0.0,
                "min_clean_correct_episode_mean_minus_corrupted": episode_margin,
                "success": {
                    name: bool(episode["success"]) for name, episode in episodes.items()
                },
                "episode_mean_reward": episode_mean_rewards,
                "episode_return": {
                    name: float(episode["imagination_return"])
                    for name, episode in episodes.items()
                },
            }
        )

    if not pairs:
        raise ValueError("no complete controlled reward triplets found")
    margins = np.asarray(
        [pair["min_clean_correct_minus_corrupted"] for pair in pairs],
        dtype=np.float64,
    )
    clean_correct_gaps = np.asarray(
        [pair["clean_correct_abs_gap"] for pair in pairs], dtype=np.float64
    )
    local_passes = np.asarray(
        [bool(pair["local_order_pass"]) for pair in pairs], dtype=np.float64
    )
    failed_corrupt = [
        pair for pair in pairs if not bool(pair["success"]["corrupted"])
    ]
    failed_corrupt_pass_fraction = (
        None
        if not failed_corrupt
        else float(np.mean([pair["local_order_pass"] for pair in failed_corrupt]))
    )
    episode_pass_fraction = float(
        np.mean([pair["episode_mean_order_pass"] for pair in pairs])
    )
    failed_corrupt_episode_pass_fraction = (
        None
        if not failed_corrupt
        else float(
            np.mean([pair["episode_mean_order_pass"] for pair in failed_corrupt])
        )
    )
    gates = {
        "five_complete_pairs": len(pairs) == 5,
        "clean_correct_reward_identical": float(np.max(clean_correct_gaps)) <= 1e-7,
        "local_order_fraction_ge_0_80": float(np.mean(local_passes)) >= 0.8,
        "mean_local_margin_gt_0": float(np.mean(margins)) > 0.0,
        "failed_corrupt_local_order_eq_1": failed_corrupt_pass_fraction == 1.0,
        "failed_corrupt_episode_mean_order_eq_1": (
            failed_corrupt_episode_pass_fraction == 1.0
        ),
    }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_task[str(pair["task_name"])].append(pair)
    return {
        "schema_version": "controlled_imagination_reward_validation_v1",
        "num_pairs": len(pairs),
        "num_corrupted_failures": len(failed_corrupt),
        "local_order_fraction": float(np.mean(local_passes)),
        "mean_local_margin": float(np.mean(margins)),
        "median_local_margin": float(np.median(margins)),
        "min_local_margin": float(np.min(margins)),
        "max_clean_correct_abs_gap": float(np.max(clean_correct_gaps)),
        "failed_corrupt_local_order_fraction": failed_corrupt_pass_fraction,
        "episode_mean_order_fraction": episode_pass_fraction,
        "failed_corrupt_episode_mean_order_fraction": (
            failed_corrupt_episode_pass_fraction
        ),
        "by_task": {
            task: {
                "num_pairs": len(values),
                "local_order_fraction": float(
                    np.mean([value["local_order_pass"] for value in values])
                ),
                "mean_local_margin": float(
                    np.mean(
                        [
                            value["min_clean_correct_minus_corrupted"]
                            for value in values
                        ]
                    )
                ),
            }
            for task, values in sorted(by_task.items())
        },
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-rewards", type=Path, required=True)
    parser.add_argument("--episode-rewards", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--corrupt-behavior", default="controlled_corrupt_0.050")
    parser.add_argument("--correct-behavior", default="controlled_correct_0.050")
    args = parser.parse_args()
    summary = validate(
        _read_jsonl(args.transition_rewards),
        _read_jsonl(args.episode_rewards),
        corrupt_behavior=args.corrupt_behavior,
        correct_behavior=args.correct_behavior,
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
