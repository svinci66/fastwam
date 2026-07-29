"""Summarize paired RoboTwin baseline and residual-IQL online evaluation logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
SUCCESS_PATTERN = re.compile(
    r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([0-9.]+)%"
)
RESIDUAL_PATTERN = re.compile(
    r"\[fastwam-residual\]\s+replan=(\d+)\s+rms=([0-9.eE+-]+)\s+"
    r"max_abs=([0-9.eE+-]+)\s+gripper_max_abs=([0-9.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-base", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--variants", default="baseline,no_imagination,imagination")
    parser.add_argument("--tasks", default="adjust_bottle,open_laptop,stack_blocks_two")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def parse_log(path: Path) -> dict[str, Any]:
    text = ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    success_matches = list(SUCCESS_PATTERN.finditer(text))
    if not success_matches:
        raise ValueError(f"No final success rate found in {path}")
    successes, episodes, percent = success_matches[-1].groups()
    residual_rows = [
        {
            "replan": int(match.group(1)),
            "rms": float(match.group(2)),
            "max_abs": float(match.group(3)),
            "gripper_max_abs": float(match.group(4)),
        }
        for match in RESIDUAL_PATTERN.finditer(text)
    ]
    result: dict[str, Any] = {
        "log": str(path.resolve()),
        "successes": int(successes),
        "episodes": int(episodes),
        "success_rate": float(percent) / 100.0,
        "num_residual_replans": len(residual_rows),
    }
    if residual_rows:
        result.update(
            {
                "residual_rms_mean": sum(row["rms"] for row in residual_rows)
                / len(residual_rows),
                "residual_max_abs": max(row["max_abs"] for row in residual_rows),
                "gripper_residual_max_abs": max(
                    row["gripper_max_abs"] for row in residual_rows
                ),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        run_dir = args.result_base / f"{args.run_name}_{variant}"
        for task in tasks:
            logs = sorted(run_dir.glob(f"eval_{task}_*.log"))
            if not logs:
                rows.append(
                    {
                        "variant": variant,
                        "task": task,
                        "status": "missing",
                    }
                )
                continue
            try:
                metrics = parse_log(logs[-1])
            except ValueError as exc:
                rows.append(
                    {
                        "variant": variant,
                        "task": task,
                        "status": "incomplete",
                        "log": str(logs[-1].resolve()),
                        "error": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "variant": variant,
                    "task": task,
                    "status": "complete",
                    **metrics,
                }
            )

    overall: dict[str, Any] = {}
    for variant in variants:
        complete = [
            row
            for row in rows
            if row["variant"] == variant and row["status"] == "complete"
        ]
        successes = sum(int(row["successes"]) for row in complete)
        episodes = sum(int(row["episodes"]) for row in complete)
        overall[variant] = {
            "complete_tasks": len(complete),
            "successes": successes,
            "episodes": episodes,
            "success_rate": None if episodes == 0 else successes / episodes,
        }

    payload = {
        "run_name": args.run_name,
        "result_base": str(args.result_base.resolve()),
        "tasks": tasks,
        "variants": variants,
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
