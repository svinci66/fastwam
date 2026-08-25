"""Summarize a FastWAM medium-task screen and organize rollout videos."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.summarize_residual_iql_online_pair import parse_log


REVIEW_FIELDS = (
    "task",
    "variant",
    "episode_id",
    "seed",
    "instruction",
    "video",
    "first_divergence_replan",
    "failure_stage",
    "failure_source",
    "reward_disagrees_with_video",
    "review_notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    return parser.parse_args()


def copy_video(source: Path, destination: Path) -> str:
    if not source.is_file():
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination.resolve())


def main() -> None:
    args = parse_args()
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    review_rows = []
    for task in tasks:
        logs = sorted(args.run_dir.glob(f"eval_{task}_*.log"))
        if not logs:
            rows.append({"task": task, "status": "missing"})
            continue
        try:
            metrics = parse_log(logs[-1])
        except ValueError as exc:
            rows.append(
                {"task": task, "status": "incomplete", "log": str(logs[-1]), "error": str(exc)}
            )
            continue
        episodes = metrics["episode_records"]
        if len(episodes) != metrics["episodes"]:
            raise ValueError(
                f"{task}: parsed {len(episodes)} episode records for {metrics['episodes']} episodes"
            )
        success_examples = 0
        missing_videos = []
        for record in episodes:
            episode_id = int(record["episode_id"])
            source = args.run_dir / task / f"episode{episode_id}.mp4"
            seed = int(record["seed"])
            if record["success"]:
                if success_examples >= 2:
                    continue
                destination = (
                    args.artifact_dir
                    / "videos"
                    / "success_examples"
                    / task
                    / "baseline"
                    / f"seed{seed}_episode{episode_id}.mp4"
                )
                copied = copy_video(source, destination)
                success_examples += int(bool(copied))
                if not copied:
                    missing_videos.append(str(source))
                continue
            destination = (
                args.artifact_dir
                / "videos"
                / "failures"
                / task
                / "baseline"
                / f"seed{seed}_episode{episode_id}.mp4"
            )
            copied = copy_video(source, destination)
            if not copied:
                missing_videos.append(str(source))
            review_rows.append(
                {
                    "task": task,
                    "variant": "baseline",
                    "episode_id": episode_id,
                    "seed": seed,
                    "instruction": record["instruction"],
                    "video": copied,
                    "first_divergence_replan": "",
                    "failure_stage": "",
                    "failure_source": "",
                    "reward_disagrees_with_video": "",
                    "review_notes": "",
                }
            )
        successes = int(metrics["successes"])
        total = int(metrics["episodes"])
        rows.append(
            {
                "task": task,
                "status": "complete",
                "successes": successes,
                "episodes": total,
                "success_rate": successes / total,
                "provisional_medium_at_5": total == 5 and successes in {2, 3},
                "medium_at_10": total >= 10 and 3 <= successes <= 7,
                "episode_records": episodes,
                "missing_videos": missing_videos,
                "log": metrics["log"],
            }
        )
    payload = {
        "schema_version": "robotwin_imagination_medium_task_screen_v1",
        "run_dir": str(args.run_dir.resolve()),
        "tasks": tasks,
        "rows": rows,
        "provisional_medium_tasks": [
            row["task"] for row in rows if row.get("provisional_medium_at_5")
        ],
        "medium_tasks": [row["task"] for row in rows if row.get("medium_at_10")],
    }
    summary_path = args.artifact_dir / "screen_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (args.artifact_dir / "failure_review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
