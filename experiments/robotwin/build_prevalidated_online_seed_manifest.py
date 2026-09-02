"""Freeze expert-validated baseline seeds for deterministic paired policy evaluation.

The source baseline must be a completed strict, paper-aligned run with expert
checking enabled.  Successful task execution is deliberately ignored: only the
expert-feasibility acceptance, selected official instruction, and initial-state
hash are copied.  Residual variants can then recreate each scene without
rerunning RoboTwin's nondeterministic expert motion planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.summarize_residual_iql_online_pair import parse_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_single_latest(paths: list[Path], *, label: str) -> Path:
    if not paths:
        raise ValueError(f"missing {label}")
    return sorted(paths)[-1]


def build_manifest(
    *,
    baseline_run_dir: Path,
    source_manifest_path: Path,
    tasks: list[str],
    episodes: int,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output: dict[str, Any] = {
        "_meta": {
            "schema_version": "robotwin_multitask_online_prevalidated_manifest_v1",
            "description": (
                "Task-specific seeds and official instructions frozen from a completed "
                "strict FastWAM baseline after exact RoboTwin expert validation."
            ),
            "selection_uses_policy_outcome": False,
            "expert_validation_mode": "accepted_by_strict_baseline_expert_check",
            "source_manifest": str(source_manifest_path.resolve()),
            "source_manifest_sha256": sha256(source_manifest_path),
            "source_baseline_run_dir": str(baseline_run_dir.resolve()),
            "episodes_per_task": episodes,
        }
    }
    for task in tasks:
        marker = baseline_run_dir / f".{task}_{episodes}ep_complete"
        if not marker.is_file():
            raise ValueError(f"baseline completion marker missing for {task}: {marker}")
        source_entry = source.get(task)
        if not isinstance(source_entry, dict) or "seeds" not in source_entry:
            raise ValueError(f"source manifest has no seed entry for {task}")
        expected_seeds = [int(value) for value in source_entry["seeds"][:episodes]]
        if len(expected_seeds) != episodes:
            raise ValueError(f"source manifest has fewer than {episodes} seeds for {task}")

        log_path = require_single_latest(
            list(baseline_run_dir.glob(f"eval_{task}_*.log")),
            label=f"baseline log for {task}",
        )
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "FASTWAM_PREVALIDATED_ENV_SEED" in text:
            raise ValueError(f"source baseline for {task} was already prevalidated")
        metrics = parse_log(log_path)
        records = metrics["episode_records"]
        if int(metrics["episodes"]) != episodes or len(records) != episodes:
            raise ValueError(f"baseline log for {task} is not a complete {episodes}-episode run")
        actual_seeds = [int(record["seed"]) for record in records]
        if actual_seeds != expected_seeds:
            raise ValueError(
                f"baseline/source seed mismatch for {task}: {actual_seeds} != {expected_seeds}"
            )
        initial_hashes = metrics["episode_initial_hashes"]
        if sorted(int(key) for key in initial_hashes) != list(range(episodes)):
            raise ValueError(f"baseline initial-state hashes incomplete for {task}")

        config_path = baseline_run_dir / f"eval_config_{task}.yaml"
        protocol_path = baseline_run_dir / f"eval_protocol_{task}.json"
        saved_config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(saved_config.EVALUATION, resolve=False)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"invalid evaluation config for {task}")
        evaluation = config
        checks = {
            "action_mode_policy": evaluation["action_mode"] == "policy",
            "expert_check": bool(evaluation["expert_check"]),
            "paper_aligned": bool(evaluation["paper_aligned"]),
            "strict_paired": bool(evaluation["strict_paired"]),
            "protocol_task": protocol["task"] == task,
            "protocol_episodes": int(protocol["episodes"]) == episodes,
            "protocol_paper_aligned": bool(protocol["paper_aligned"]),
            "protocol_strict_paired": bool(protocol["strict_paired"]),
        }
        if not all(checks.values()):
            raise ValueError(f"baseline protocol audit failed for {task}: {checks}")

        attestations = []
        for record in records:
            episode_id = int(record["episode_id"])
            attestations.append(
                {
                    "episode_id": episode_id,
                    "seed": int(record["seed"]),
                    "initial_observation_sha256": initial_hashes[str(episode_id)],
                    "source_log_sha256": sha256(log_path),
                    "evidence": "FASTWAM_ACCEPTED_ENV_SEED after live expert check",
                }
            )
        output[task] = {
            "seeds": actual_seeds,
            "instructions": [str(record["instruction"]) for record in records],
            "expert_feasibility_records": attestations,
            "source_log": str(log_path.resolve()),
            "source_log_sha256": sha256(log_path),
            "source_config_sha256": sha256(config_path),
            "source_protocol_sha256": sha256(protocol_path),
        }
    return output


def main() -> None:
    args = parse_args()
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    payload = build_manifest(
        baseline_run_dir=args.baseline_run_dir.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        tasks=tasks,
        episodes=args.episodes,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
