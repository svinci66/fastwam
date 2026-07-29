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
    r"(?:\s+gate_applied=(\d+)\s+q_advantage_min=([0-9.eE+-]+)"
    r"\s+q_advantage_disagreement=([0-9.eE+-]+))?"
    r"(?:\s+gate_approved=(\d+)\s+shadow_mode=(\d+)"
    r"\s+circuit_breaker_active=(\d+)\s+circuit_breaker_triggered=(\d+)"
    r"(?:\s+support_state_score=([0-9.eE+-]+)"
    r"\s+support_state_threshold=([0-9.eE+-]+)"
    r"\s+support_action_score=([0-9.eE+-]+)"
    r"\s+support_action_threshold=([0-9.eE+-]+)"
    r"\s+support_in_distribution=(\d+)"
    r"\s+support_language_similarity=([0-9.eE+-]+))?)?"
    r"(?:\s+intervention_allowed=(\d+))?"
    r"(?:\s+intervention_count=(\d+)"
    r"\s+intervention_budget_remaining=(None|\d+)"
    r"\s+intervention_budget_exhausted=(\d+))?"
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
            "gate_applied": (
                None if match.group(5) is None else bool(int(match.group(5)))
            ),
            "q_advantage_min": (
                None if match.group(6) is None else float(match.group(6))
            ),
            "q_advantage_disagreement": (
                None if match.group(7) is None else float(match.group(7))
            ),
            "gate_approved": (
                None if match.group(8) is None else bool(int(match.group(8)))
            ),
            "shadow_mode": (
                None if match.group(9) is None else bool(int(match.group(9)))
            ),
            "circuit_breaker_active": (
                None if match.group(10) is None else bool(int(match.group(10)))
            ),
            "circuit_breaker_triggered": (
                None if match.group(11) is None else bool(int(match.group(11)))
            ),
            "support_state_score": (
                None if match.group(12) is None else float(match.group(12))
            ),
            "support_state_threshold": (
                None if match.group(13) is None else float(match.group(13))
            ),
            "support_action_score": (
                None if match.group(14) is None else float(match.group(14))
            ),
            "support_action_threshold": (
                None if match.group(15) is None else float(match.group(15))
            ),
            "support_in_distribution": (
                None if match.group(16) is None else bool(int(match.group(16)))
            ),
            "support_language_similarity": (
                None if match.group(17) is None else float(match.group(17))
            ),
            "intervention_allowed": (
                None if match.group(18) is None else bool(int(match.group(18)))
            ),
            "intervention_count": (
                None if match.group(19) is None else int(match.group(19))
            ),
            "intervention_budget_remaining": (
                None
                if match.group(20) in {None, "None"}
                else int(match.group(20))
            ),
            "intervention_budget_exhausted": (
                None if match.group(21) is None else bool(int(match.group(21)))
            ),
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
        gated_rows = [row for row in residual_rows if row["gate_applied"] is not None]
        if gated_rows:
            result.update(
                {
                    "q_gate_apply_rate": sum(
                        int(row["gate_applied"]) for row in gated_rows
                    )
                    / len(gated_rows),
                    "q_advantage_min_mean": sum(
                        row["q_advantage_min"] for row in gated_rows
                    )
                    / len(gated_rows),
                    "q_advantage_disagreement_max": max(
                        row["q_advantage_disagreement"] for row in gated_rows
                    ),
                }
            )
        approval_rows = [
            row for row in residual_rows if row["gate_approved"] is not None
        ]
        if approval_rows:
            result.update(
                {
                    "gate_approval_rate": sum(
                        int(row["gate_approved"]) for row in approval_rows
                    )
                    / len(approval_rows),
                    "shadow_mode": all(row["shadow_mode"] for row in approval_rows),
                    "circuit_breaker_trigger_count": sum(
                        int(row["circuit_breaker_triggered"])
                        for row in approval_rows
                    ),
                }
            )
        support_rows = [
            row
            for row in residual_rows
            if row["support_in_distribution"] is not None
        ]
        if support_rows:
            result.update(
                {
                    "support_in_distribution_rate": sum(
                        int(row["support_in_distribution"])
                        for row in support_rows
                    )
                    / len(support_rows),
                    "support_state_score_max": max(
                        row["support_state_score"] for row in support_rows
                    ),
                    "support_action_score_max": max(
                        row["support_action_score"] for row in support_rows
                    ),
                }
            )
        intervention_rows = [
            row
            for row in residual_rows
            if row["intervention_allowed"] is not None
        ]
        if intervention_rows:
            result["intervention_allowed_rate"] = sum(
                int(row["intervention_allowed"]) for row in intervention_rows
            ) / len(intervention_rows)
        budget_rows = [
            row
            for row in residual_rows
            if row["intervention_count"] is not None
        ]
        if budget_rows:
            result.update(
                {
                    "intervention_count_max": max(
                        row["intervention_count"] for row in budget_rows
                    ),
                    "budget_exhausted_replans": sum(
                        int(row["intervention_budget_exhausted"])
                        for row in budget_rows
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
