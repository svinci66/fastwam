"""Audit the pre-registered no-imagination versus Wan-head held-out comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.decide_wan_head_weight_candidate import keyed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def audit(payload: dict) -> dict:
    task = "open_microwave"
    if not payload["initial_state_audit"][task]["exact_match"]:
        raise ValueError("held-out initial observations are not exactly paired")
    if not payload["protocol_pairing_audit"][task]["exact_seed_and_instruction_match"]:
        raise ValueError("held-out seeds or official instructions differ")
    rows = {
        str(row["variant"]): row
        for row in payload["rows"]
        if row.get("task") == task and row.get("status") == "complete"
    }
    if set(rows) != {"no_imagination", "imagination"}:
        raise ValueError(f"expected two complete variants, got {sorted(rows)}")
    control = keyed(rows["no_imagination"]["episode_records"])
    candidate = keyed(rows["imagination"]["episode_records"])
    if set(control) != set(candidate) or len(control) != 10:
        raise ValueError("held-out comparison must contain exactly ten paired episodes")
    wins = sum(candidate[key] and not control[key] for key in control)
    losses = sum(control[key] and not candidate[key] for key in control)
    control_successes = sum(control.values())
    candidate_successes = sum(candidate.values())
    if candidate_successes > control_successes and wins > losses:
        decision = "confirmed"
    elif candidate_successes < control_successes or losses > wins:
        decision = "rejected"
    else:
        decision = "inconclusive"
    return {
        "schema_version": "robotwin_wan_head_heldout_pair_audit_v1",
        "pairs": len(control),
        "control_successes": control_successes,
        "candidate_successes": candidate_successes,
        "paired_wins": wins,
        "paired_losses": losses,
        "both_success": sum(control[key] and candidate[key] for key in control),
        "both_failure": sum(not control[key] and not candidate[key] for key in control),
        "decision": decision,
        "strong_confirmation": bool(wins >= 2 and losses == 0),
        "rule": "confirmed iff candidate successes are higher and paired wins exceed losses",
    }


def main() -> None:
    args = parse_args()
    result = audit(json.loads(args.summary.read_text(encoding="utf-8")))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
