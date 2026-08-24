"""Require exact closed-loop equivalence between FastWAM and residual shadow logs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
SEED_RE = re.compile(r"FASTWAM_ACCEPTED_ENV_SEED episode_id=(\d+) seed=(\d+)")
INSTRUCTION_RE = re.compile(
    r"FASTWAM_EVAL_INSTRUCTION episode_id=(\d+) seed=(\d+) instruction=(.+)$"
)
INITIAL_RE = re.compile(
    r"FASTWAM_INITIAL_OBSERVATION episode_id=(\d+) sha256=([0-9a-f]{64})"
)
REPLAN_RE = re.compile(
    r"FASTWAM_REPLAN_AUDIT "
    r"episode_id=(\d+) seed=(\d+) replan=(\d+) "
    r"instruction_sha256=([0-9a-f]{64}) "
    r"current_observation_sha256=([0-9a-f]{64}) "
    r"baseline_actions_sha256=([0-9a-f]{64}) "
    r"executed_actions_sha256=([0-9a-f]{64}) n_exec=(\d+)"
)
SUCCESS_RE = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>")
PROTOCOL_RE = re.compile(r"FASTWAM_EVAL_PROTOCOL (\{.+\})")


def _latest_log(run_dir: Path, task: str) -> Path:
    paths = sorted(run_dir.glob(f"eval_{task}_*.log"))
    if not paths:
        raise FileNotFoundError(f"No evaluation log for {task} under {run_dir}")
    return paths[-1]


def _parse_log(path: Path) -> dict[str, Any]:
    text = ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    protocol_matches = list(PROTOCOL_RE.finditer(text))
    if len(protocol_matches) != 1:
        raise ValueError(f"Expected one protocol record in {path}")
    protocol = json.loads(protocol_matches[0].group(1))

    seeds = {
        int(match.group(1)): int(match.group(2)) for match in SEED_RE.finditer(text)
    }
    instructions: dict[int, dict[str, Any]] = {}
    for match in INSTRUCTION_RE.finditer(text):
        episode_id = int(match.group(1))
        instructions[episode_id] = {
            "seed": int(match.group(2)),
            "instruction": str(ast.literal_eval(match.group(3))),
        }
    initial_hashes = {
        int(match.group(1)): match.group(2) for match in INITIAL_RE.finditer(text)
    }

    replans: dict[tuple[int, int], dict[str, Any]] = {}
    for match in REPLAN_RE.finditer(text):
        episode_id = int(match.group(1))
        replan = int(match.group(3))
        key = (episode_id, replan)
        if key in replans:
            raise ValueError(f"Duplicate replan record {key} in {path}")
        replans[key] = {
            "episode_id": episode_id,
            "seed": int(match.group(2)),
            "replan": replan,
            "instruction_sha256": match.group(4),
            "current_observation_sha256": match.group(5),
            "baseline_actions_sha256": match.group(6),
            "executed_actions_sha256": match.group(7),
            "n_exec": int(match.group(8)),
        }

    outcomes: dict[int, bool] = {}
    previous_successes = 0
    for match in SUCCESS_RE.finditer(text):
        successes = int(match.group(1))
        episodes = int(match.group(2))
        episode_id = episodes - 1
        delta = successes - previous_successes
        if delta not in (0, 1):
            raise ValueError(f"Invalid cumulative success sequence in {path}")
        outcomes[episode_id] = bool(delta)
        previous_successes = successes

    episode_ids = set(seeds)
    if not episode_ids:
        raise ValueError(f"No accepted episodes in {path}")
    if set(instructions) != episode_ids or set(initial_hashes) != episode_ids:
        raise ValueError(f"Incomplete episode provenance in {path}")
    if set(outcomes) != episode_ids:
        raise ValueError(f"Incomplete episode outcomes in {path}")
    if not replans:
        raise ValueError(f"No replan audit records in {path}")
    for episode_id, seed in seeds.items():
        if instructions[episode_id]["seed"] != seed:
            raise ValueError(f"Seed/instruction mismatch in {path}, episode {episode_id}")
    return {
        "path": str(path.resolve()),
        "protocol": protocol,
        "seeds": seeds,
        "instructions": instructions,
        "initial_hashes": initial_hashes,
        "replans": replans,
        "outcomes": outcomes,
    }


def _core_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "action_mode",
        "residual_checkpoint",
        "residual_shadow_mode",
        "save_imagination_transitions",
        "output_dir",
    }
    return {key: value for key, value in protocol.items() if key not in ignored}


def audit(baseline_dir: Path, shadow_dir: Path, tasks: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for task in tasks:
        baseline = _parse_log(_latest_log(baseline_dir, task))
        shadow = _parse_log(_latest_log(shadow_dir, task))
        reasons: list[str] = []
        if _core_protocol(baseline["protocol"]) != _core_protocol(shadow["protocol"]):
            reasons.append("protocol_mismatch")
        for field in ("seeds", "instructions", "initial_hashes", "outcomes"):
            if baseline[field] != shadow[field]:
                reasons.append(f"{field}_mismatch")
        if set(baseline["replans"]) != set(shadow["replans"]):
            reasons.append("replan_keys_mismatch")
        differing_replans: list[dict[str, Any]] = []
        for key in sorted(set(baseline["replans"]) | set(shadow["replans"])):
            base_row = baseline["replans"].get(key)
            shadow_row = shadow["replans"].get(key)
            if base_row != shadow_row:
                differing_replans.append(
                    {"episode_id": key[0], "replan": key[1]}
                )
        if differing_replans:
            reasons.append("replan_records_mismatch")
        baseline_internal = [
            {"episode_id": key[0], "replan": key[1]}
            for key, value in baseline["replans"].items()
            if value["baseline_actions_sha256"] != value["executed_actions_sha256"]
        ]
        shadow_internal = [
            {"episode_id": key[0], "replan": key[1]}
            for key, value in shadow["replans"].items()
            if value["baseline_actions_sha256"] != value["executed_actions_sha256"]
        ]
        if baseline_internal:
            reasons.append("baseline_did_not_execute_fastwam_actions")
        if shadow_internal:
            reasons.append("shadow_did_not_execute_fastwam_actions")
        row = {
            "task": task,
            "status": "exact" if not reasons else "invalid",
            "episodes": len(baseline["seeds"]),
            "replans": len(baseline["replans"]),
            "reasons": reasons,
            "differing_replans": differing_replans,
            "baseline_internal_mismatches": baseline_internal,
            "shadow_internal_mismatches": shadow_internal,
            "baseline_log": baseline["path"],
            "shadow_log": shadow["path"],
        }
        rows.append(row)
        if reasons:
            invalid.append(row)
    total_replans = sum(row["replans"] for row in rows)
    return {
        "schema_version": "robotwin_zero_residual_equivalence_audit_v1",
        "baseline_dir": str(baseline_dir.resolve()),
        "shadow_dir": str(shadow_dir.resolve()),
        "tasks": rows,
        "task_count": len(rows),
        "episode_count": sum(row["episodes"] for row in rows),
        "replan_count": total_replans,
        "exact_task_count": len(rows) - len(invalid),
        "invalid_task_count": len(invalid),
        "all_exact": bool(rows) and not invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--shadow-dir", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.baseline_dir,
        args.shadow_dir,
        [item.strip() for item in args.tasks.split(",") if item.strip()],
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_exact and not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
