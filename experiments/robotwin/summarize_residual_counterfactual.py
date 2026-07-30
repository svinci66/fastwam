"""Summarize the fixed-seed RoboTwin residual intervention counterfactual."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
KEY_VALUE = re.compile(r"(\w+)=([^ ]+)")
CONDITIONS = ("shadow", "replan9", "replan11", "replan9_11")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-base", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task", default="blocks_ranking_size")
    parser.add_argument("--expected-seed", type=int, default=4300003)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def latest_complete_log(run_dir: Path, task: str) -> Path:
    logs = sorted(run_dir.glob(f"eval_{task}_*.log"))
    complete = [path for path in logs if "Success rate:" in path.read_text(errors="replace")]
    if not complete:
        raise RuntimeError(f"No complete log for {task} in {run_dir}")
    return complete[-1]


def summarize_condition(
    *, result_base: Path, run_name: str, condition: str, task: str
) -> dict[str, Any]:
    run_dir = result_base / f"{run_name}_{condition}_imagination"
    log_path = latest_complete_log(run_dir, task)
    success = episodes = seed = None
    replans: list[dict[str, str]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        if "[fastwam-residual]" in line:
            replans.append(dict(KEY_VALUE.findall(line)))
        if "Success rate:" in line:
            clean = ANSI_ESCAPE.sub("", line)
            match = re.search(
                r"Success rate: (\d+)/(\d+).*current seed: (\d+)", clean
            )
            if match is None:
                raise RuntimeError(f"Could not parse result line in {log_path}: {clean}")
            success, episodes, seed = (int(value) for value in match.groups())
    metadata_paths = sorted(
        (run_dir / task / "imagination_transitions").rglob("metadata.json")
    )
    if not metadata_paths:
        raise RuntimeError(f"No transition metadata found in {run_dir}")
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    hashes = sorted({str(row["initial_observation_sha256"]) for row in metadata})
    if len(hashes) != 1:
        raise RuntimeError(f"Condition {condition} has multiple initial hashes: {hashes}")
    return {
        "condition": condition,
        "episodes": episodes,
        "successes": success,
        "success": success == 1 and episodes == 1,
        "environment_seed": seed,
        "initial_observation_sha256": hashes[0],
        "num_transitions": len(metadata),
        "num_alignment_valid_transitions": sum(
            bool(row["alignment_valid"]) for row in metadata
        ),
        "num_replans": len(replans),
        "q_and_support_approved_replans": [
            int(row["replan"])
            for row in replans
            if row.get("gate_approved") == "1"
        ],
        "applied_replans": [
            int(row["replan"])
            for row in replans
            if row.get("gate_applied") == "1"
        ],
        "circuit_breaker_trigger_replans": [
            int(row["replan"])
            for row in replans
            if row.get("circuit_breaker_triggered") == "1"
        ],
        "log": str(log_path.resolve()),
        "run_dir": str(run_dir.resolve()),
    }


def main() -> None:
    args = parse_args()
    rows = [
        summarize_condition(
            result_base=args.result_base.resolve(),
            run_name=args.run_name,
            condition=condition,
            task=args.task,
        )
        for condition in CONDITIONS
    ]
    seeds = {row["environment_seed"] for row in rows}
    hashes = {row["initial_observation_sha256"] for row in rows}
    summary = {
        "run_name": args.run_name,
        "task": args.task,
        "expected_seed": args.expected_seed,
        "all_runs_complete": all(row["episodes"] == 1 for row in rows),
        "all_environment_seeds_match": seeds == {args.expected_seed},
        "all_initial_observation_hashes_match": len(hashes) == 1,
        "initial_observation_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
        "rows": rows,
    }
    if not summary["all_environment_seeds_match"]:
        raise RuntimeError(f"Counterfactual seed mismatch: {sorted(seeds)}")
    if not summary["all_initial_observation_hashes_match"]:
        raise RuntimeError(f"Counterfactual initial-state mismatch: {sorted(hashes)}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
