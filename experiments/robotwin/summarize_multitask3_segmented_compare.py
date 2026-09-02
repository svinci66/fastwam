"""Summarize a baseline run and crash-isolated residual evaluation segments."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-base", type=Path, required=True)
    parser.add_argument("--baseline-run-name", required=True)
    parser.add_argument("--segment-prefix", required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, required=True)
    parser.add_argument("--crash-marker-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def latest_complete_log(run_dir: Path, task: str) -> Path:
    for path in reversed(sorted(run_dir.glob(f"eval_{task}_*.log"))):
        try:
            parse_log(path)
        except ValueError:
            continue
        return path
    raise ValueError(f"no complete log for {task} under {run_dir}")


def latest_log(run_dir: Path, task: str) -> Path:
    paths = sorted(run_dir.glob(f"eval_{task}_*.log"))
    if not paths:
        raise ValueError(f"no log for {task} under {run_dir}")
    return paths[-1]


def validate_identity(
    *,
    task: str,
    episode: int,
    seed: int,
    instruction: str,
    initial_hash: str,
    manifest_entry: dict[str, Any],
) -> None:
    expected_seed = int(manifest_entry["seeds"][episode])
    expected_instruction = str(manifest_entry["instructions"][episode])
    expected_hash = str(
        manifest_entry["expert_feasibility_records"][episode][
            "initial_observation_sha256"
        ]
    )
    actual = (seed, instruction, initial_hash)
    expected = (expected_seed, expected_instruction, expected_hash)
    if actual != expected:
        raise ValueError(f"identity mismatch for {task} episode {episode}: {actual} != {expected}")


def baseline_row(
    *, result_base: Path, run_name: str, task: str, manifest_entry: dict[str, Any], episodes: int
) -> dict[str, Any]:
    run_dir = result_base / f"{run_name}_baseline"
    metrics = parse_log(latest_complete_log(run_dir, task))
    records = metrics["episode_records"]
    hashes = metrics["episode_initial_hashes"]
    if int(metrics["episodes"]) != episodes or len(records) != episodes:
        raise ValueError(f"baseline episode count mismatch for {task}")
    for episode, record in enumerate(records):
        validate_identity(
            task=task,
            episode=episode,
            seed=int(record["seed"]),
            instruction=str(record["instruction"]),
            initial_hash=str(hashes[str(episode)]),
            manifest_entry=manifest_entry,
        )
    return {
        "variant": "baseline",
        "task": task,
        "successes": int(metrics["successes"]),
        "episodes": episodes,
        "success_rate": int(metrics["successes"]) / episodes,
        "runtime_failures": 0,
        "episode_records": records,
        "episode_initial_hashes": hashes,
        "log": metrics["log"],
    }


def segment_row(
    *,
    result_base: Path,
    segment_prefix: str,
    crash_marker_dir: Path,
    variant: str,
    task: str,
    episode: int,
    max_attempts: int,
    manifest_entry: dict[str, Any],
) -> dict[str, Any]:
    completed: tuple[int, Path, dict[str, Any], dict[str, str]] | None = None
    attempted: list[tuple[int, Path]] = []
    for attempt in range(1, max_attempts + 1):
        segment_name = (
            f"{segment_prefix}__{variant}__{task}__episode{episode}__attempt{attempt}"
        )
        run_dir = result_base / f"{segment_name}_{variant}"
        logs = sorted(run_dir.glob(f"eval_{task}_*.log"))
        if logs:
            attempted.append((attempt, run_dir))
        marker = run_dir / f".{task}_1ep_complete"
        if not marker.is_file():
            continue
        log = latest_complete_log(run_dir, task)
        metrics = parse_log(log)
        hashes = load_episode_initial_hashes(run_dir, task)
        completed = (attempt, run_dir, metrics, hashes)
        break

    if completed is not None:
        attempt, _, metrics, hashes = completed
        records = metrics["episode_records"]
        if int(metrics["episodes"]) != 1 or len(records) != 1:
            raise ValueError(f"segment must contain one episode for {variant}/{task}/{episode}")
        record = records[0]
        if int(record["episode_id"]) != episode or set(hashes) != {str(episode)}:
            raise ValueError(f"segment index mismatch for {variant}/{task}/{episode}")
        initial_hash = str(hashes[str(episode)])
        validate_identity(
            task=task,
            episode=episode,
            seed=int(record["seed"]),
            instruction=str(record["instruction"]),
            initial_hash=initial_hash,
            manifest_entry=manifest_entry,
        )
        return {
            "episode_id": episode,
            "seed": int(record["seed"]),
            "instruction": str(record["instruction"]),
            "success": bool(record["success"]),
            "runtime_status": "complete",
            "attempts": attempt,
            "initial_observation_sha256": initial_hash,
            "log": metrics["log"],
            "num_residual_replans": int(metrics["num_residual_replans"]),
            "residual_rms_mean": metrics.get("residual_rms_mean"),
        }

    crash_marker = crash_marker_dir / f"{variant}__{task}__episode{episode}"
    if not crash_marker.is_file() or not attempted:
        raise ValueError(f"unfinished segment for {variant}/{task}/{episode}")
    attempt, run_dir = attempted[-1]
    log = latest_log(run_dir, task)
    hashes = load_episode_initial_hashes(run_dir, task)
    if set(hashes) != {str(episode)}:
        raise ValueError(f"crashed segment lacks exact initial hash for {variant}/{task}/{episode}")
    seed = int(manifest_entry["seeds"][episode])
    instruction = str(manifest_entry["instructions"][episode])
    initial_hash = str(hashes[str(episode)])
    validate_identity(
        task=task,
        episode=episode,
        seed=seed,
        instruction=instruction,
        initial_hash=initial_hash,
        manifest_entry=manifest_entry,
    )
    return {
        "episode_id": episode,
        "seed": seed,
        "instruction": instruction,
        "success": False,
        "runtime_status": "native_crash_failure",
        "attempts": attempt,
        "initial_observation_sha256": initial_hash,
        "log": str(log.resolve()),
        "num_residual_replans": None,
        "residual_rms_mean": None,
    }


def main() -> None:
    args = parse_args()
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    manifest = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    initial_state_audit: dict[str, Any] = {}
    for task in tasks:
        entry = manifest[task]
        baseline = baseline_row(
            result_base=args.result_base,
            run_name=args.baseline_run_name,
            task=task,
            manifest_entry=entry,
            episodes=args.episodes,
        )
        rows.append(baseline)
        baseline_outcomes = {
            int(record["episode_id"]): bool(record["success"])
            for record in baseline["episode_records"]
        }
        baseline_hashes = baseline["episode_initial_hashes"]
        paired[task] = {}
        hashes_match = True
        for variant in variants:
            records = [
                segment_row(
                    result_base=args.result_base,
                    segment_prefix=args.segment_prefix,
                    crash_marker_dir=args.crash_marker_dir,
                    variant=variant,
                    task=task,
                    episode=episode,
                    max_attempts=args.max_attempts,
                    manifest_entry=entry,
                )
                for episode in range(args.episodes)
            ]
            hashes = {
                str(record["episode_id"]): record["initial_observation_sha256"]
                for record in records
            }
            hashes_match = hashes_match and hashes == baseline_hashes
            successes = sum(int(record["success"]) for record in records)
            rows.append(
                {
                    "variant": variant,
                    "task": task,
                    "successes": successes,
                    "episodes": args.episodes,
                    "success_rate": successes / args.episodes,
                    "runtime_failures": sum(
                        record["runtime_status"] != "complete" for record in records
                    ),
                    "episode_records": records,
                    "episode_initial_hashes": hashes,
                }
            )
            paired[task][variant] = {
                "improved": sum(
                    not baseline_outcomes[record["episode_id"]] and record["success"]
                    for record in records
                ),
                "regressed": sum(
                    baseline_outcomes[record["episode_id"]] and not record["success"]
                    for record in records
                ),
                "both_success": sum(
                    baseline_outcomes[record["episode_id"]] and record["success"]
                    for record in records
                ),
                "both_failure": sum(
                    not baseline_outcomes[record["episode_id"]] and not record["success"]
                    for record in records
                ),
            }
        initial_state_audit[task] = {
            "exact_match": hashes_match,
            "episodes_per_variant": args.episodes,
        }

    all_variants = ["baseline", *variants]
    overall = {}
    for variant in all_variants:
        selected = [row for row in rows if row["variant"] == variant]
        successes = sum(int(row["successes"]) for row in selected)
        episodes = sum(int(row["episodes"]) for row in selected)
        overall[variant] = {
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes,
            "runtime_failures": sum(int(row["runtime_failures"]) for row in selected),
        }
    payload = {
        "schema_version": "robotwin_multitask3_segmented_compare_v1",
        "baseline_run_name": args.baseline_run_name,
        "segment_prefix": args.segment_prefix,
        "tasks": tasks,
        "variants": all_variants,
        "overall": overall,
        "rows": rows,
        "paired_outcomes": paired,
        "initial_state_audit": initial_state_audit,
        "passed_pairing_audit": all(
            value["exact_match"] for value in initial_state_audit.values()
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed_pairing_audit"]:
        raise SystemExit("segmented comparison failed initial-state pairing audit")


if __name__ == "__main__":
    main()
