"""Audit exact initial-state pairing in RoboTwin corruption collections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODES = ("hold_0.250", "hold_0.750", "gripper_delay_024")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_terminal(episode_dir: Path) -> bool:
    metadata_paths = sorted(episode_dir.rglob("metadata.json"))
    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "episode_success" in payload or bool(payload.get("truncated", False)):
            return True
    return False


def audit_collection(root: Path, modes: Iterable[str] = DEFAULT_MODES) -> dict[str, Any]:
    root = root.resolve()
    mode_names = tuple(modes)
    records: list[dict[str, Any]] = []
    task_summaries: dict[str, dict[str, int]] = {}

    for policy_dir in sorted(root.glob("*/imagination_transitions/*/policy")):
        task_name = policy_dir.parent.name
        task_summary = task_summaries.setdefault(
            task_name,
            {"policy_episodes": 0, "expected_pairs": 0, "exact_pairs": 0},
        )
        for policy_episode in sorted(policy_dir.glob("episode_*")):
            task_summary["policy_episodes"] += 1
            policy_current = policy_episode / "replan_0000" / "current.png"
            policy_terminal = _is_terminal(policy_episode)
            policy_hash = _sha256(policy_current) if policy_current.is_file() else None
            for mode in mode_names:
                task_summary["expected_pairs"] += 1
                candidate = policy_dir.parent / mode / policy_episode.name
                candidate_current = candidate / "replan_0000" / "current.png"
                candidate_terminal = _is_terminal(candidate) if candidate.is_dir() else False
                candidate_hash = (
                    _sha256(candidate_current) if candidate_current.is_file() else None
                )
                exact = bool(
                    policy_terminal
                    and candidate_terminal
                    and policy_hash is not None
                    and policy_hash == candidate_hash
                )
                if exact:
                    task_summary["exact_pairs"] += 1
                records.append(
                    {
                        "task": task_name,
                        "trial": policy_episode.name,
                        "mode": mode,
                        "status": "exact" if exact else "invalid",
                        "policy_terminal": policy_terminal,
                        "candidate_terminal": candidate_terminal,
                        "policy_sha256": policy_hash,
                        "candidate_sha256": candidate_hash,
                    }
                )

    expected_pairs = sum(item["expected_pairs"] for item in task_summaries.values())
    exact_pairs = sum(item["exact_pairs"] for item in task_summaries.values())
    return {
        "schema_version": "robotwin_exact_pairing_audit_v1",
        "root": str(root),
        "modes": list(mode_names),
        "task_count": len(task_summaries),
        "policy_episode_count": sum(
            item["policy_episodes"] for item in task_summaries.values()
        ),
        "expected_pair_count": expected_pairs,
        "exact_pair_count": exact_pairs,
        "invalid_pair_count": expected_pairs - exact_pairs,
        "all_exact": expected_pairs > 0 and expected_pairs == exact_pairs,
        "tasks": task_summaries,
        "invalid_pairs": [row for row in records if row["status"] != "exact"],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RoboTwin Exact Pairing Audit",
        "",
        f"- Root: `{report['root']}`",
        f"- Exact pairs: {report['exact_pair_count']}/{report['expected_pair_count']}",
        f"- Result: {'PASS' if report['all_exact'] else 'FAIL'}",
        "",
        "| Task | Policy episodes | Exact pairs | Expected pairs |",
        "|---|---:|---:|---:|",
    ]
    for task, values in sorted(report["tasks"].items()):
        lines.append(
            f"| {task} | {values['policy_episodes']} | "
            f"{values['exact_pairs']} | {values['expected_pairs']} |"
        )
    if report["invalid_pairs"]:
        lines.extend(["", "## Invalid pairs", ""])
        for row in report["invalid_pairs"]:
            lines.append(f"- `{row['task']}/{row['mode']}/{row['trial']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    report = audit_collection(args.input_dir)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    if args.require_exact and not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
