"""Freeze the first strict-valid natural failures without inspecting rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_pairs(
    *,
    status_path: Path,
    strict_audit_path: Path,
    task: str,
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    status_rows = _read_jsonl(status_path)
    strict_audit = json.loads(strict_audit_path.read_text(encoding="utf-8"))
    valid_keys = {
        (str(row["task"]), int(row["episode_index"]), int(row["environment_seed"]))
        for row in strict_audit.get("pairs", [])
        if bool(row.get("valid"))
    }

    candidates = []
    for row in status_rows:
        if str(row.get("task")) != task:
            continue
        if bool(row.get("success")) or row.get("decision") != "natural_failure":
            continue
        key = (task, int(row["episode_index"]), int(row["environment_seed"]))
        if key not in valid_keys:
            continue
        for field in ("expert_hdf5", "fastwam_video", "review_dir"):
            path = Path(str(row[field])).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"{field} is missing for {key}: {path}")
        candidates.append(dict(row))

    candidates.sort(key=lambda row: int(row["episode_index"]))
    if len(candidates) < count:
        raise ValueError(
            f"Need {count} strict-valid natural failures for {task}, "
            f"found {len(candidates)}"
        )
    selected = candidates[:count]
    keys = [
        (str(row["task"]), int(row["episode_index"]), int(row["environment_seed"]))
        for row in selected
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("selected pairs contain duplicate identities")

    audit = {
        "schema_version": "robotwin_natural_failure_selection_audit_v1",
        "task": task,
        "selection_uses_reward": False,
        "selection_rule": (
            "first strict-pair-valid natural FastWAM failures in expert-feasible "
            "candidate order"
        ),
        "available_strict_valid_natural_failures": len(candidates),
        "selected_pair_count": len(selected),
        "selected": [
            {
                "episode_index": int(row["episode_index"]),
                "environment_seed": int(row["environment_seed"]),
                "instruction": str(row["instruction"]),
                "expert_hdf5": str(row["expert_hdf5"]),
                "fastwam_video": str(row["fastwam_video"]),
            }
            for row in selected
        ],
    }
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-jsonl", type=Path, required=True)
    parser.add_argument("--strict-audit", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args()

    selected, audit = select_pairs(
        status_path=args.status_jsonl.expanduser().resolve(),
        strict_audit_path=args.strict_audit.expanduser().resolve(),
        task=args.task,
        count=args.count,
    )
    output_jsonl = args.output_jsonl.expanduser().resolve()
    output_audit = args.output_audit.expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    output_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
