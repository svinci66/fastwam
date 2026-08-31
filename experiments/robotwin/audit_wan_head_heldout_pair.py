"""Audit the pre-registered no-imagination versus Wan-head held-out comparison."""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--expected-pairs", type=int, default=10)
    return parser.parse_args()


def audit(payload: dict, *, expected_pairs: int = 10) -> dict:
    if expected_pairs <= 0:
        raise ValueError("expected_pairs must be positive")
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
    required = {"no_imagination", "imagination"}
    allowed = required | {"baseline"}
    if not required.issubset(rows) or not set(rows).issubset(allowed):
        raise ValueError(
            "expected complete no_imagination/imagination variants and an optional "
            f"baseline, got {sorted(rows)}"
        )
    control = keyed(rows["no_imagination"]["episode_records"])
    candidate = keyed(rows["imagination"]["episode_records"])
    if set(control) != set(candidate) or len(control) != expected_pairs:
        raise ValueError(
            f"held-out comparison must contain exactly {expected_pairs} paired episodes"
        )
    wins = sum(candidate[key] and not control[key] for key in control)
    losses = sum(control[key] and not candidate[key] for key in control)
    control_successes = sum(control.values())
    candidate_successes = sum(candidate.values())
    discordant = wins + losses
    exact_one_sided_p = (
        1.0
        if discordant == 0
        else sum(math.comb(discordant, value) for value in range(wins, discordant + 1))
        / (2**discordant)
    )
    if candidate_successes > control_successes and wins > losses:
        decision = "confirmed"
    elif candidate_successes < control_successes or losses > wins:
        decision = "rejected"
    else:
        decision = "inconclusive"
    result = {
        "schema_version": "robotwin_wan_head_heldout_pair_audit_v1",
        "pairs": len(control),
        "control_successes": control_successes,
        "candidate_successes": candidate_successes,
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_discordant": discordant,
        "paired_exact_one_sided_p": exact_one_sided_p,
        "both_success": sum(control[key] and candidate[key] for key in control),
        "both_failure": sum(not control[key] and not candidate[key] for key in control),
        "decision": decision,
        "strong_confirmation": bool(wins >= 2 and losses == 0),
        "statistical_confirmation": bool(
            candidate_successes > control_successes
            and wins > losses
            and exact_one_sided_p <= 0.05
        ),
        "rule": "confirmed iff candidate successes are higher and paired wins exceed losses",
    }
    if "baseline" in rows:
        baseline = keyed(rows["baseline"]["episode_records"])
        if set(baseline) != set(control):
            raise ValueError("baseline episodes do not match the residual pair")

        def compare(reference: dict, contender: dict) -> dict:
            contender_wins = sum(
                contender[key] and not reference[key] for key in reference
            )
            contender_losses = sum(
                reference[key] and not contender[key] for key in reference
            )
            return {
                "reference_successes": sum(reference.values()),
                "contender_successes": sum(contender.values()),
                "paired_wins": contender_wins,
                "paired_losses": contender_losses,
                "both_success": sum(
                    reference[key] and contender[key] for key in reference
                ),
                "both_failure": sum(
                    not reference[key] and not contender[key] for key in reference
                ),
            }

        result["baseline_successes"] = sum(baseline.values())
        result["comparisons_to_baseline"] = {
            "no_imagination": compare(baseline, control),
            "imagination": compare(baseline, candidate),
        }
    return result


def main() -> None:
    args = parse_args()
    result = audit(
        json.loads(args.summary.read_text(encoding="utf-8")),
        expected_pairs=args.expected_pairs,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
