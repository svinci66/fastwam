"""Summarize paired RoboTwin baseline and residual-IQL online evaluation logs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
SUCCESS_PATTERN = re.compile(
    r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([0-9.]+)%"
)
SEED_PATTERN = re.compile(r"FASTWAM_ACCEPTED_ENV_SEED episode_id=(\d+) seed=(\d+)")
INSTRUCTION_PATTERN = re.compile(
    r"FASTWAM_EVAL_INSTRUCTION episode_id=(\d+) seed=(\d+) instruction=(.+)$"
)
INITIAL_HASH_PATTERN = re.compile(
    r"FASTWAM_INITIAL_OBSERVATION episode_id=(\d+) sha256=([0-9a-f]{64})"
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
    r"(?:\s+residual_language_canonicalized=\d+)?"
    r"(?:\s+intervention_allowed=(\d+))?"
    r"(?:\s+intervention_count=(\d+)"
    r"\s+intervention_budget_remaining=(None|\d+)"
    r"\s+intervention_budget_exhausted=(\d+))?"
    r"(?:\s+q_gate_effective_margin=(None|[0-9.eE+-]+)"
    r"\s+candidate_residual_rms=([0-9.eE+-]+)"
    r"\s+residual_risk_before=([0-9.eE+-]+)"
    r"\s+residual_risk_after=([0-9.eE+-]+))?"
    r"(?:\s+residual_scale_factor=([0-9.eE+-]+)"
    r"\s+q_scale_confidence=([0-9.eE+-]+)"
    r"\s+support_scale_confidence=([0-9.eE+-]+)"
    r"\s+outcome_confirmation_pending=(\d+)"
    r"\s+last_outcome_progress=(None|[0-9.eE+-]+)"
    r"\s+last_outcome_confirmed=(None|True|False)"
    r"\s+outcome_reanchor_remaining=(\d+)"
    r"\s+outcome_blocked=(\d+))?"
)
OUTCOME_PATTERN = re.compile(
    r"\[fastwam-residual-outcome\]\s+replan=(\d+)\s+"
    r"gate_applied=(\d+)\s+imagination_progress=([0-9.eE+-]+)"
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
            "q_gate_effective_margin": (
                None
                if match.group(22) in {None, "None"}
                else float(match.group(22))
            ),
            "candidate_residual_rms": (
                None if match.group(23) is None else float(match.group(23))
            ),
            "residual_risk_before": (
                None if match.group(24) is None else float(match.group(24))
            ),
            "residual_risk_after": (
                None if match.group(25) is None else float(match.group(25))
            ),
            "residual_scale_factor": (
                None if match.group(26) is None else float(match.group(26))
            ),
            "q_scale_confidence": (
                None if match.group(27) is None else float(match.group(27))
            ),
            "support_scale_confidence": (
                None if match.group(28) is None else float(match.group(28))
            ),
            "outcome_confirmation_pending": (
                None if match.group(29) is None else bool(int(match.group(29)))
            ),
            "last_outcome_progress": (
                None
                if match.group(30) in {None, "None"}
                else float(match.group(30))
            ),
            "last_outcome_confirmed": (
                None
                if match.group(31) in {None, "None"}
                else match.group(31) == "True"
            ),
            "outcome_reanchor_remaining": (
                None if match.group(32) is None else int(match.group(32))
            ),
            "outcome_blocked": (
                None if match.group(33) is None else bool(int(match.group(33)))
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
        "episode_records": _episode_records(text, path),
        "episode_initial_hashes": _initial_hashes(text, path),
    }
    outcome_rows = [
        {
            "replan": int(match.group(1)),
            "gate_applied": bool(int(match.group(2))),
            "imagination_progress": float(match.group(3)),
        }
        for match in OUTCOME_PATTERN.finditer(text)
    ]
    if outcome_rows:
        applied_outcomes = [row for row in outcome_rows if row["gate_applied"]]
        result["outcome_feedback_replans"] = len(outcome_rows)
        result["applied_outcome_feedback_replans"] = len(applied_outcomes)
        if applied_outcomes:
            result["applied_outcome_progress_mean"] = sum(
                row["imagination_progress"] for row in applied_outcomes
            ) / len(applied_outcomes)
            result["applied_outcome_positive_rate"] = sum(
                int(row["imagination_progress"] >= 0.0)
                for row in applied_outcomes
            ) / len(applied_outcomes)
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
            risk_rows = [
                row for row in gated_rows if row["residual_risk_after"] is not None
            ]
            if risk_rows:
                result["candidate_residual_rms_max"] = max(
                    row["candidate_residual_rms"] for row in risk_rows
                )
                result["residual_risk_max"] = max(
                    row["residual_risk_after"] for row in risk_rows
                )
                q_margin_rows = [
                    row
                    for row in risk_rows
                    if row["q_gate_effective_margin"] is not None
                ]
                if q_margin_rows:
                    result["q_gate_effective_margin_max"] = max(
                        row["q_gate_effective_margin"] for row in q_margin_rows
                    )
            scale_rows = [
                row
                for row in gated_rows
                if row["residual_scale_factor"] is not None
            ]
            if scale_rows:
                result["residual_scale_factor_mean"] = sum(
                    row["residual_scale_factor"] for row in scale_rows
                ) / len(scale_rows)
                applied_scale_rows = [row for row in scale_rows if row["gate_applied"]]
                if applied_scale_rows:
                    result["applied_residual_scale_factor_mean"] = sum(
                        row["residual_scale_factor"] for row in applied_scale_rows
                    ) / len(applied_scale_rows)
                result["outcome_blocked_replans"] = sum(
                    int(row["outcome_blocked"]) for row in scale_rows
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


def _episode_records(text: str, path: Path) -> list[dict[str, Any]]:
    current_seed: tuple[int, int] | None = None
    current_instruction: tuple[int, int, str] | None = None
    previous_successes = 0
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        seed_match = SEED_PATTERN.search(line)
        if seed_match:
            current_seed = (int(seed_match.group(1)), int(seed_match.group(2)))
            continue
        instruction_match = INSTRUCTION_PATTERN.search(line)
        if instruction_match:
            try:
                instruction = ast.literal_eval(instruction_match.group(3))
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Invalid instruction record in {path}: {line}") from exc
            current_instruction = (
                int(instruction_match.group(1)),
                int(instruction_match.group(2)),
                str(instruction),
            )
            continue
        success_match = SUCCESS_PATTERN.search(line)
        if not success_match:
            continue
        successes, episodes, _ = success_match.groups()
        successes = int(successes)
        episodes = int(episodes)
        if current_seed is None or current_instruction is None:
            # Legacy logs predate exact instruction provenance.
            return []
        episode_id, seed = current_seed
        instruction_episode_id, instruction_seed, instruction = current_instruction
        if (instruction_episode_id, instruction_seed) != (episode_id, seed):
            raise ValueError(f"Seed/instruction mismatch in {path}")
        outcome = successes - previous_successes
        if outcome not in {0, 1} or episodes != len(rows) + 1:
            raise ValueError(f"Invalid episode success sequence in {path}")
        rows.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "instruction": instruction,
                "success": bool(outcome),
            }
        )
        previous_successes = successes
        current_seed = None
        current_instruction = None
    return rows


def _initial_hashes(text: str, path: Path) -> dict[str, str]:
    grouped: dict[int, set[str]] = {}
    for match in INITIAL_HASH_PATTERN.finditer(text):
        grouped.setdefault(int(match.group(1)), set()).add(match.group(2))
    inconsistent = {
        episode: sorted(hashes)
        for episode, hashes in grouped.items()
        if len(hashes) != 1
    }
    if inconsistent:
        raise ValueError(f"Inconsistent initial hashes in {path}: {inconsistent}")
    return {
        str(episode): next(iter(hashes))
        for episode, hashes in sorted(grouped.items())
    }


def load_episode_initial_hashes(run_dir: Path, task: str) -> dict[str, str]:
    """Load one audited initial-observation hash from metadata or logs."""

    root = run_dir / task / "imagination_transitions" / task
    grouped: dict[int, set[str]] = {}
    for path in root.rglob("metadata.json") if root.is_dir() else ():
        payload = json.loads(path.read_text(encoding="utf-8"))
        trial = int(payload["trial_idx"])
        grouped.setdefault(trial, set()).add(
            str(payload["initial_observation_sha256"])
        )
    inconsistent = {
        trial: sorted(hashes) for trial, hashes in grouped.items() if len(hashes) != 1
    }
    if inconsistent:
        raise ValueError(
            f"Captured episodes contain inconsistent initial hashes: {inconsistent}"
        )
    captured = {
        str(trial): next(iter(hashes)) for trial, hashes in sorted(grouped.items())
    }
    if captured:
        return captured
    for log_path in reversed(sorted(run_dir.glob(f"eval_{task}_*.log"))):
        logged = _initial_hashes(
            ANSI_ESCAPE.sub(
                "", log_path.read_text(encoding="utf-8", errors="replace")
            ),
            log_path,
        )
        if logged:
            return logged
    return {}


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
                    "episode_initial_hashes": load_episode_initial_hashes(
                        run_dir, task
                    ),
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

    initial_state_audit: dict[str, Any] = {}
    protocol_pairing_audit: dict[str, Any] = {}
    paired_outcomes: dict[str, Any] = {}
    for task in tasks:
        task_rows = {
            str(row["variant"]): row
            for row in rows
            if row["task"] == task and row["status"] == "complete"
        }
        captured = {
            variant: row["episode_initial_hashes"]
            for variant, row in task_rows.items()
            if row["episode_initial_hashes"]
        }
        signatures = {
            json.dumps(hashes, sort_keys=True) for hashes in captured.values()
        }
        initial_state_audit[task] = {
            "captured_variants": sorted(captured),
            "episodes_per_variant": {
                variant: len(hashes) for variant, hashes in sorted(captured.items())
            },
            "exact_match": len(captured) == len(variants) and len(signatures) == 1,
        }
        records = {
            variant: row.get("episode_records", [])
            for variant, row in task_rows.items()
        }
        protocol_signatures = {
            variant: [
                (record["seed"], record["instruction"])
                for record in variant_records
            ]
            for variant, variant_records in records.items()
        }
        exact_protocol_pair = (
            len(protocol_signatures) == len(variants)
            and bool(protocol_signatures)
            and len(
                {
                    json.dumps(signature, ensure_ascii=False)
                    for signature in protocol_signatures.values()
                }
            )
            == 1
        )
        protocol_pairing_audit[task] = {
            "exact_seed_and_instruction_match": exact_protocol_pair,
            "records_per_variant": {
                variant: len(signature)
                for variant, signature in sorted(protocol_signatures.items())
            },
        }
        if "baseline" in records:
            baseline_by_pair = {
                (record["seed"], record["instruction"]): record["success"]
                for record in records["baseline"]
            }
            paired_outcomes[task] = {}
            for variant, variant_records in records.items():
                if variant == "baseline":
                    continue
                candidate_by_pair = {
                    (record["seed"], record["instruction"]): record["success"]
                    for record in variant_records
                }
                common = sorted(set(baseline_by_pair) & set(candidate_by_pair))
                paired_outcomes[task][variant] = {
                    "pairs": len(common),
                    "improved": sum(
                        not baseline_by_pair[key] and candidate_by_pair[key]
                        for key in common
                    ),
                    "regressed": sum(
                        baseline_by_pair[key] and not candidate_by_pair[key]
                        for key in common
                    ),
                    "both_success": sum(
                        baseline_by_pair[key] and candidate_by_pair[key]
                        for key in common
                    ),
                    "both_failure": sum(
                        not baseline_by_pair[key] and not candidate_by_pair[key]
                        for key in common
                    ),
                }

    payload = {
        "run_name": args.run_name,
        "result_base": str(args.result_base.resolve()),
        "tasks": tasks,
        "variants": variants,
        "rows": rows,
        "overall": overall,
        "initial_state_audit": initial_state_audit,
        "protocol_pairing_audit": protocol_pairing_audit,
        "paired_outcomes": paired_outcomes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
