"""Audit a paired online weight candidate and apply the fixed stopping rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--retry-on-tie", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def keyed(records: list[dict[str, Any]]) -> dict[tuple[int, str], bool]:
    return {
        (int(record["seed"]), str(record["instruction"])): bool(record["success"])
        for record in records
    }


def decide(payload: dict[str, Any], *, weight: float, retry_on_tie: bool) -> dict[str, Any]:
    task = "open_microwave"
    if not payload["initial_state_audit"][task]["exact_match"]:
        raise ValueError("candidate/control initial observations are not exactly paired")
    if not payload["protocol_pairing_audit"][task]["exact_seed_and_instruction_match"]:
        raise ValueError("candidate/control seeds or official instructions differ")
    rows = {
        str(row["variant"]): row
        for row in payload["rows"]
        if row.get("task") == task and row.get("status") == "complete"
    }
    if set(rows) != {"no_imagination", "imagination"}:
        raise ValueError(f"expected exactly two complete variants, got {sorted(rows)}")
    control = keyed(rows["no_imagination"]["episode_records"])
    candidate = keyed(rows["imagination"]["episode_records"])
    if set(control) != set(candidate) or not control:
        raise ValueError("candidate/control episode records are not exactly paired")
    wins = sum(candidate[key] and not control[key] for key in control)
    losses = sum(control[key] and not candidate[key] for key in control)
    both_success = sum(control[key] and candidate[key] for key in control)
    both_failure = sum(not control[key] and not candidate[key] for key in control)
    control_successes = sum(control.values())
    candidate_successes = sum(candidate.values())
    if candidate_successes > control_successes and losses == 0:
        decision = "promote_to_new_heldout"
        reason = "candidate improves paired development success without regression"
    elif retry_on_tie and candidate_successes == control_successes:
        decision = "retry_lower_weight"
        reason = "candidate ties control; one pre-registered lower-weight retry remains"
    else:
        decision = "redesign_reward"
        reason = "candidate does not produce a clean paired improvement"
    return {
        "schema_version": "robotwin_wan_head_weight_decision_v1",
        "weight": float(weight),
        "pairs": len(control),
        "control_successes": control_successes,
        "candidate_successes": candidate_successes,
        "paired_wins": wins,
        "paired_losses": losses,
        "both_success": both_success,
        "both_failure": both_failure,
        "decision": decision,
        "reason": reason,
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    output = decide(payload, weight=args.weight, retry_on_tie=args.retry_on_tie)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
