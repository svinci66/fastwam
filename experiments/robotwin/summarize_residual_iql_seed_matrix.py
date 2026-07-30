"""Summarize a seed-isolated RoboTwin residual-IQL evaluation matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.summarize_residual_iql_online_pair import (
    load_episode_initial_hashes,
    parse_log,
)


TASK_SEEDS = {
    "blocks_ranking_size": [4300000, 4300001, 4300002, 4300003, 4300004],
    "hanging_mug": [4300002, 4300003, 4300006, 4300008, 4300010],
}
VARIANTS = ("baseline", "no_imagination", "imagination")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-base", type=Path, required=True)
    parser.add_argument("--baseline-run-name", required=True)
    parser.add_argument("--segment-run-prefix", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def merge_episode_hashes(
    parts: list[dict[str, str]], expected_episodes: int
) -> dict[str, str]:
    merged: dict[str, str] = {}
    for part in parts:
        for episode, digest in part.items():
            if episode in merged and merged[episode] != digest:
                raise ValueError(
                    f"episode {episode} has conflicting initial hashes: "
                    f"{merged[episode]} != {digest}"
                )
            merged[episode] = digest
    expected = {str(index) for index in range(expected_episodes)}
    if set(merged) != expected:
        raise ValueError(
            f"captured episode indices differ: got {sorted(merged)}, "
            f"expected {sorted(expected)}"
        )
    return dict(sorted(merged.items(), key=lambda item: int(item[0])))


def latest_complete_log(run_dir: Path, task: str) -> Path:
    logs = sorted(run_dir.glob(f"eval_{task}_*.log"))
    for path in reversed(logs):
        try:
            parse_log(path)
        except ValueError:
            continue
        return path
    raise ValueError(f"No complete evaluation log for {task} under {run_dir}")


def aggregate_segments(
    result_base: Path,
    *,
    prefix: str,
    task: str,
    variant: str,
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    hashes: list[dict[str, str]] = []
    segment_logs: list[str] = []
    for episode_index, seed in enumerate(TASK_SEEDS[task]):
        segment_name = f"{prefix}__{task}_episode{episode_index}"
        run_dir = result_base / f"{segment_name}_{variant}"
        log = latest_complete_log(run_dir, task)
        value = parse_log(log)
        if int(value["episodes"]) != 1:
            raise ValueError(f"segment log must contain one episode: {log}")
        captured = load_episode_initial_hashes(run_dir, task)
        if set(captured) != {str(episode_index)}:
            raise ValueError(
                f"segment {segment_name} captured {sorted(captured)}, "
                f"expected episode {episode_index}"
            )
        metrics.append(value)
        hashes.append(captured)
        segment_logs.append(str(log.resolve()))
    residual_replans = sum(int(value["num_residual_replans"]) for value in metrics)
    applied = sum(
        round(float(value.get("q_gate_apply_rate", 0.0)) * int(value["num_residual_replans"]))
        for value in metrics
    )
    return {
        "variant": variant,
        "task": task,
        "status": "complete",
        "successes": sum(int(value["successes"]) for value in metrics),
        "episodes": len(metrics),
        "success_rate": sum(int(value["successes"]) for value in metrics) / len(metrics),
        "num_residual_replans": residual_replans,
        "num_applied_interventions": applied,
        "episode_initial_hashes": merge_episode_hashes(hashes, len(metrics)),
        "environment_seeds": TASK_SEEDS[task],
        "segment_logs": segment_logs,
    }


def main() -> None:
    args = parse_args()
    result_base = args.result_base.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for task, seeds in TASK_SEEDS.items():
        baseline_dir = result_base / f"{args.baseline_run_name}_baseline"
        baseline_log = latest_complete_log(baseline_dir, task)
        baseline = parse_log(baseline_log)
        baseline_hashes = load_episode_initial_hashes(baseline_dir, task)
        if int(baseline["episodes"]) != len(seeds):
            raise ValueError(f"baseline episode count mismatch for {task}")
        rows.append(
            {
                "variant": "baseline",
                "task": task,
                "status": "complete",
                **baseline,
                "episode_initial_hashes": merge_episode_hashes(
                    [baseline_hashes], len(seeds)
                ),
                "environment_seeds": seeds,
            }
        )
        for variant in VARIANTS[1:]:
            rows.append(
                aggregate_segments(
                    result_base,
                    prefix=args.segment_run_prefix,
                    task=task,
                    variant=variant,
                )
            )

    audits: dict[str, Any] = {}
    for task in TASK_SEEDS:
        task_rows = [row for row in rows if row["task"] == task]
        signatures = {
            json.dumps(row["episode_initial_hashes"], sort_keys=True)
            for row in task_rows
        }
        audits[task] = {
            "exact_match": len(task_rows) == len(VARIANTS) and len(signatures) == 1,
            "variants": [row["variant"] for row in task_rows],
            "episodes_per_variant": {
                row["variant"]: len(row["episode_initial_hashes"])
                for row in task_rows
            },
        }

    overall = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        successes = sum(int(row["successes"]) for row in selected)
        episodes = sum(int(row["episodes"]) for row in selected)
        overall[variant] = {
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes,
        }
    payload = {
        "format": "robotwin_residual_iql_seed_matrix_v1",
        "baseline_run_name": args.baseline_run_name,
        "segment_run_prefix": args.segment_run_prefix,
        "initial_state_audit": audits,
        "rows": rows,
        "overall": overall,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
