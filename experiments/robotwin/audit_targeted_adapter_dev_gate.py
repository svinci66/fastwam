#!/usr/bin/env python3
"""Apply the frozen three-state stop gate to a paired adapter evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (4800282, 4800283, 4800286)


def audit_dev_gate(summary: dict[str, Any]) -> dict[str, Any]:
    by_variant: dict[str, dict[int, bool]] = {}
    for row in summary.get("rows", []):
        variant = str(row.get("variant"))
        if variant not in {"no_imagination", "imagination"}:
            continue
        by_variant[variant] = {
            int(record["seed"]): bool(record["success"])
            for record in row.get("episode_records", [])
        }

    expected = set(EXPECTED_SEEDS)
    control = by_variant.get("no_imagination", {})
    treatment = by_variant.get("imagination", {})
    control_successes = sum(control.get(seed, False) for seed in EXPECTED_SEEDS)
    treatment_successes = sum(treatment.get(seed, False) for seed in EXPECTED_SEEDS)
    initial = summary.get("initial_state_audit", {}).get("place_can_basket", {})
    protocol = summary.get("protocol_pairing_audit", {}).get(
        "place_can_basket", {}
    )
    checks = {
        "exact_expected_control_seeds": set(control) == expected,
        "exact_expected_treatment_seeds": set(treatment) == expected,
        "exact_initial_state_pairing": bool(initial.get("exact_match")),
        "exact_seed_instruction_pairing": bool(
            protocol.get("exact_seed_and_instruction_match")
        ),
        "treatment_preserves_all_three": treatment_successes == 3,
        "treatment_exceeds_same_data_control": (
            treatment_successes > control_successes
        ),
    }
    return {
        "schema_version": "robotwin_targeted_adapter_dev_gate_v1",
        "diagnostic_only": True,
        "not_a_final_evaluation": True,
        "passed": all(checks.values()),
        "checks": checks,
        "expected_seeds": list(EXPECTED_SEEDS),
        "successes": {
            "no_imagination": control_successes,
            "imagination": treatment_successes,
        },
        "outcomes": {
            variant: {str(seed): values.get(seed) for seed in EXPECTED_SEEDS}
            for variant, values in (
                ("no_imagination", control),
                ("imagination", treatment),
            )
        },
        "failure_action": (
            None
            if all(checks.values())
            else "Reject the current reward/training candidate. Do not collect more of the same stratum; revise temporal credit assignment or the objective."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    report = audit_dev_gate(summary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
