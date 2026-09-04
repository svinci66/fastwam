#!/usr/bin/env python3
"""Merge independently expert-screened single-task manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "robotwin_expert_feasible_heldout_manifest_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_manifests(
    inputs: list[tuple[Path, dict[str, Any]]], tasks: list[str]
) -> dict[str, Any]:
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("tasks must be a non-empty unique ordered list")
    by_task: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, payload in inputs:
        meta = payload.get("_meta", {})
        if meta.get("schema_version") != EXPECTED_SCHEMA:
            raise ValueError(f"unsupported manifest schema in {path}")
        present = [key for key in payload if key != "_meta"]
        if len(present) != 1:
            raise ValueError(f"single-task manifest expected in {path}: {present}")
        task = present[0]
        if task in by_task:
            raise ValueError(f"duplicate task manifest: {task}")
        by_task[task] = (path, payload)
    missing = [task for task in tasks if task not in by_task]
    extras = sorted(set(by_task) - set(tasks))
    if missing or extras:
        raise ValueError(f"manifest task mismatch: missing={missing}, extras={extras}")

    merged: dict[str, Any] = {
        "_meta": {
            "schema_version": "robotwin_multitask_expert_feasible_heldout_manifest_v1",
            "description": "Fresh task-specific held-out states selected only by exact expert feasibility.",
            "selection_uses_policy_outcome": False,
            "expert_validation_mode": "prevalidated_exact_planning_and_replay",
            "task_config": "demo_clean",
            "task_order": tasks,
            "source_manifests": [],
        }
    }
    expected_count: int | None = None
    for task in tasks:
        path, payload = by_task[task]
        entry = payload[task]
        seeds = [int(value) for value in entry.get("seeds", [])]
        instructions = [str(value) for value in entry.get("instructions", [])]
        records = payload["_meta"].get("expert_feasibility_records", [])
        record_seeds = [int(record["seed"]) for record in records]
        if not seeds or len(seeds) != len(instructions) or seeds != record_seeds:
            raise ValueError(f"seed/instruction/attestation mismatch for {task}")
        if expected_count is None:
            expected_count = len(seeds)
        elif len(seeds) != expected_count:
            raise ValueError("all tasks must contain the same number of held-out states")
        merged[task] = {
            "seeds": seeds,
            "instructions": instructions,
            "expert_feasibility_records": records,
            "source_manifest": str(path.resolve()),
            "source_manifest_sha256": sha256(path),
        }
        merged["_meta"]["source_manifests"].append(
            {"task": task, "path": str(path.resolve()), "sha256": sha256(path)}
        )
    merged["_meta"]["episodes_per_task"] = expected_count
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    result = merge_manifests(
        [(path, json.loads(path.read_text(encoding="utf-8"))) for path in args.inputs],
        tasks,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["_meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
