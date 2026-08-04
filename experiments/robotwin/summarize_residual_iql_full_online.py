"""Summarize a baseline/ungated/Q+OOD RoboTwin run by accepted seed."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from summarize_residual_iql_online_pair import ANSI_ESCAPE, parse_log


SEED_PATTERN = re.compile(r"FASTWAM_ACCEPTED_ENV_SEED episode_id=(\d+) seed=(\d+)")
SUCCESS_PATTERN = re.compile(
    r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([0-9.]+)%"
)
VARIANT_SUFFIXES = {
    "baseline": "ungated_baseline",
    "ungated_residual": "ungated_imagination",
    "qood_residual": "qood_imagination",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-base", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument(
        "--tasks",
        default="adjust_bottle,hanging_mug,open_microwave,place_can_basket",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def episode_outcomes(path: Path, expected_episodes: int) -> list[dict[str, Any]]:
    text = ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    current_seed: int | None = None
    previous_successes = 0
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        seed_match = SEED_PATTERN.search(line)
        if seed_match:
            current_seed = int(seed_match.group(2))
            continue
        success_match = SUCCESS_PATTERN.search(line)
        if not success_match:
            continue
        successes, episodes, _ = success_match.groups()
        successes = int(successes)
        episodes = int(episodes)
        if current_seed is None:
            raise ValueError(f"Success row without accepted seed in {path}")
        if episodes != len(rows) + 1:
            raise ValueError(f"Non-consecutive episode count in {path}: {episodes}")
        outcome = successes - previous_successes
        if outcome not in {0, 1}:
            raise ValueError(f"Invalid success increment in {path}: {outcome}")
        rows.append(
            {
                "episode_id": episodes - 1,
                "seed": current_seed,
                "success": bool(outcome),
            }
        )
        previous_successes = successes
        current_seed = None
    if len(rows) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} complete episodes in {path}, found {len(rows)}"
        )
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate accepted seeds in {path}")
    return rows


def completed_log(run_dir: Path, task: str, expected_episodes: int) -> Path:
    candidates: list[Path] = []
    for path in sorted(run_dir.glob(f"eval_{task}_*.log")):
        try:
            episode_outcomes(path, expected_episodes)
        except ValueError:
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No {expected_episodes}-episode completed log for {task} in {run_dir}"
        )
    return candidates[-1]


def paired_counts(
    baseline: dict[int, bool], candidate: dict[int, bool], seeds: list[int]
) -> dict[str, int]:
    return {
        "improved": sum(not baseline[seed] and candidate[seed] for seed in seeds),
        "regressed": sum(baseline[seed] and not candidate[seed] for seed in seeds),
        "both_success": sum(baseline[seed] and candidate[seed] for seed in seeds),
        "both_failure": sum(
            not baseline[seed] and not candidate[seed] for seed in seeds
        ),
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    variants: dict[str, dict[str, Any]] = {}
    for variant, suffix in VARIANT_SUFFIXES.items():
        run_dir = args.result_base / f"{args.run_prefix}_{suffix}"
        task_rows: dict[str, Any] = {}
        for task in tasks:
            log = completed_log(run_dir, task, args.episodes)
            outcomes = episode_outcomes(log, args.episodes)
            metrics = parse_log(log)
            task_rows[task] = {
                "log": str(log.resolve()),
                "successes": metrics["successes"],
                "episodes": metrics["episodes"],
                "success_rate": metrics["success_rate"],
                "episode_outcomes": outcomes,
                "residual": {
                    key: value
                    for key, value in metrics.items()
                    if key
                    in {
                        "num_residual_replans",
                        "residual_rms_mean",
                        "residual_max_abs",
                        "q_gate_apply_rate",
                        "gate_approval_rate",
                        "support_in_distribution_rate",
                        "circuit_breaker_trigger_count",
                    }
                },
            }
        total_successes = sum(row["successes"] for row in task_rows.values())
        total_episodes = sum(row["episodes"] for row in task_rows.values())
        variants[variant] = {
            "successes": total_successes,
            "episodes": total_episodes,
            "success_rate": total_successes / total_episodes,
            "tasks": task_rows,
        }

    paired_by_task: dict[str, Any] = {}
    aggregate_counts = {
        "ungated_residual": {key: 0 for key in paired_counts({}, {}, [])},
        "qood_residual": {key: 0 for key in paired_counts({}, {}, [])},
    }
    aggregate_successes = {variant: 0 for variant in VARIANT_SUFFIXES}
    aggregate_episodes = 0
    for task in tasks:
        outcome_maps = {
            variant: {
                row["seed"]: row["success"]
                for row in variants[variant]["tasks"][task]["episode_outcomes"]
            }
            for variant in VARIANT_SUFFIXES
        }
        seed_sets = {variant: set(values) for variant, values in outcome_maps.items()}
        common_seeds = sorted(set.intersection(*seed_sets.values()))
        if not common_seeds:
            raise ValueError(f"No common accepted seed for task {task}")
        exact = len({tuple(sorted(values)) for values in seed_sets.values()}) == 1
        matched_successes = {
            variant: sum(outcome_maps[variant][seed] for seed in common_seeds)
            for variant in VARIANT_SUFFIXES
        }
        comparisons = {
            variant: paired_counts(
                outcome_maps["baseline"], outcome_maps[variant], common_seeds
            )
            for variant in ("ungated_residual", "qood_residual")
        }
        for variant, counts in comparisons.items():
            for key, value in counts.items():
                aggregate_counts[variant][key] += value
        for variant, successes in matched_successes.items():
            aggregate_successes[variant] += successes
        aggregate_episodes += len(common_seeds)
        paired_by_task[task] = {
            "exact_seed_match": exact,
            "accepted_seeds": {
                variant: sorted(values) for variant, values in seed_sets.items()
            },
            "common_seeds": common_seeds,
            "matched_successes": matched_successes,
            "matched_episodes": len(common_seeds),
            "matched_success_rates": {
                variant: successes / len(common_seeds)
                for variant, successes in matched_successes.items()
            },
            "vs_baseline": comparisons,
        }

    summary = {
        "format": "fastwam_robotwin_residual_iql_full_online_summary_v1",
        "run_prefix": args.run_prefix,
        "raw": variants,
        "seed_matched": {
            "episodes_per_variant": aggregate_episodes,
            "successes": aggregate_successes,
            "success_rates": {
                variant: successes / aggregate_episodes
                for variant, successes in aggregate_successes.items()
            },
            "vs_baseline": aggregate_counts,
            "tasks": paired_by_task,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
