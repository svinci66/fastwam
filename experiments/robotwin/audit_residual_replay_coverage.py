"""Summarize task, behavior, success, and seed coverage in a residual replay."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes: dict[tuple[str, str, str], dict[str, object]] = {}
    with (args.replay_dir / "transitions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            key = (
                str(record["task_description"]),
                str(record["behavior_mode"]),
                str(record["episode_id"]),
            )
            episode = episodes.setdefault(
                key,
                {
                    "environment_seed": int(record["env_seed"]),
                    "success": False,
                    "transitions": 0,
                },
            )
            episode["success"] = bool(episode["success"] or record["success"])
            episode["transitions"] = int(episode["transitions"]) + 1
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (task, behavior, _episode_id), episode in episodes.items():
        summary = grouped[task][behavior]
        summary["episodes"] = int(summary.get("episodes", 0)) + 1
        summary["successful_episodes"] = int(
            summary.get("successful_episodes", 0)
        ) + int(bool(episode["success"]))
        summary.setdefault("environment_seeds", []).append(
            int(episode["environment_seed"])
        )
    for task in grouped.values():
        for behavior in task.values():
            seeds = sorted(set(behavior["environment_seeds"]))
            behavior["environment_seeds"] = seeds
            behavior["seed_minimum"] = min(seeds)
            behavior["seed_maximum"] = max(seeds)
    payload = {
        "format": "fastwam_robotwin_residual_replay_coverage_v1",
        "num_episodes": len(episodes),
        "tasks": grouped,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
