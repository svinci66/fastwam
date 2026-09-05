#!/usr/bin/env python3
"""Audit whether a reward ablation improves paired AWR return targets.

Episode-level Wan-VAE ranking is not sufficient for AWR: the actor is weighted
from transition-level discounted returns after normalization and reward
composition.  This audit relabels one immutable replay with control/treatment
configs, computes the exact returns consumed by training, and checks whether
the expert-minus-failure return gap is preserved for every task and a required
diagnostic seed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.robotwin.audit_awr_training_pair import (
    ALLOWED_CONFIG_DIFFERENCES,
    differing_paths,
)
from fastwam.rl.replay_buffer import ReplayBuffer
from fastwam.rl.rewards import CompositeRewardConfig


EPISODE_PATTERN = re.compile(r"-pair(?P<pair_id>\d+)-(?P<behavior>expert|policy)$")


def summarize_return_gap_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no paired return-gap rows")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    per_task = {}
    for task, task_rows in sorted(grouped.items()):
        changes = np.asarray(
            [float(row["expert_failure_gap_change"]) for row in task_rows],
            dtype=np.float64,
        )
        per_task[task] = {
            "pair_count": len(task_rows),
            "nonshrinking_count": int(np.sum(changes >= -1e-8)),
            "strictly_improved_count": int(np.sum(changes > 1e-8)),
            "nonshrinking_fraction": float(np.mean(changes >= -1e-8)),
            "mean_gap_change": float(np.mean(changes)),
            "minimum_gap_change": float(np.min(changes)),
        }
    changes = np.asarray(
        [float(row["expert_failure_gap_change"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "pair_count": len(rows),
        "nonshrinking_count": int(np.sum(changes >= -1e-8)),
        "strictly_improved_count": int(np.sum(changes > 1e-8)),
        "nonshrinking_fraction": float(np.mean(changes >= -1e-8)),
        "mean_gap_change": float(np.mean(changes)),
        "minimum_gap_change": float(np.min(changes)),
        "per_task": per_task,
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"top-level config must be a mapping: {path}")
    return payload


def audit_reward_credit(
    replay: ReplayBuffer,
    replay_manifest: dict[str, Any],
    control_config: dict[str, Any],
    treatment_config: dict[str, Any],
    *,
    minimum_nonshrinking_fraction: float,
    required_task: str,
    required_env_seed: int,
) -> dict[str, Any]:
    if not 0.0 <= minimum_nonshrinking_fraction <= 1.0:
        raise ValueError("minimum_nonshrinking_fraction must be in [0, 1]")
    differences = differing_paths(control_config, treatment_config)
    if differences != ALLOWED_CONFIG_DIFFERENCES:
        raise ValueError(
            "control/treatment are not a strict reward ablation: "
            f"{sorted('.'.join(path) for path in differences)}"
        )
    gamma = float(control_config["awr"]["gamma"])
    if gamma != float(treatment_config["awr"]["gamma"]):
        raise ValueError("control/treatment gamma values differ")
    control_reward = CompositeRewardConfig(**control_config["reward"])
    treatment_reward = CompositeRewardConfig(**treatment_config["reward"])
    imitation_scales = control_config.get("imitation_dimension_scales")
    imitation_scales_array = (
        None
        if imitation_scales is None
        else np.asarray(imitation_scales, dtype=np.float32)
    )
    control_values, control_breakdowns = replay.relabel_rewards(
        control_reward, imitation_dimension_scales=imitation_scales_array
    )
    treatment_values, treatment_breakdowns = replay.relabel_rewards(
        treatment_reward, imitation_dimension_scales=imitation_scales_array
    )
    timeout_bootstrap = {
        transition.episode_id: 0.0
        for transition in replay.transitions
        if transition.truncated
    }
    control_returns = replay.monte_carlo_returns(
        gamma,
        timeout_bootstrap_values=timeout_bootstrap,
        transition_rewards=control_values,
    )
    treatment_returns = replay.monte_carlo_returns(
        gamma,
        timeout_bootstrap_values=timeout_bootstrap,
        transition_rewards=treatment_values,
    )

    task_id_map = replay_manifest.get("provenance", {}).get("task_id_map", {})
    id_to_task = {int(task_id): str(task) for task, task_id in task_id_map.items()}
    if set(id_to_task) != {transition.task_id for transition in replay.transitions}:
        raise ValueError("replay provenance task_id_map is incomplete")
    episodes: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    episode_indices: dict[str, list[int]] = defaultdict(list)
    for index, transition in enumerate(replay.transitions):
        episode_indices[transition.episode_id].append(index)
    for episode_id, indices in episode_indices.items():
        match = EPISODE_PATTERN.search(episode_id)
        if match is None:
            raise ValueError(f"cannot recover pair identity from {episode_id!r}")
        indices.sort(key=lambda index: replay.transitions[index].transition_index)
        first = replay.transitions[indices[0]]
        behavior = match.group("behavior")
        if behavior != first.behavior_mode:
            raise ValueError(f"episode behavior mismatch for {episode_id!r}")
        task = id_to_task[first.task_id]
        pair_id = int(match.group("pair_id"))
        identity = (task, pair_id)
        if behavior in episodes[identity]:
            raise ValueError(f"duplicate {behavior} episode for {identity}")
        episodes[identity][behavior] = {
            "episode_id": episode_id,
            "environment_seed": int(first.env_seed),
            "initial_control_return": float(control_returns[indices[0]]),
            "initial_treatment_return": float(treatment_returns[indices[0]]),
            "control_reward_sum": float(np.sum(control_values[indices])),
            "treatment_reward_sum": float(np.sum(treatment_values[indices])),
            "treatment_imagination_applied_sum": float(
                np.sum(
                    [
                        treatment_breakdowns[index].imagination_applied
                        for index in indices
                    ]
                )
            ),
        }
    rows = []
    for (task, pair_id), behaviors in sorted(episodes.items()):
        if set(behaviors) != {"expert", "policy"}:
            raise ValueError(f"pair {(task, pair_id)} lacks expert/policy episodes")
        expert = behaviors["expert"]
        policy = behaviors["policy"]
        if expert["environment_seed"] != policy["environment_seed"]:
            raise ValueError(f"paired environment seeds differ for {(task, pair_id)}")
        control_gap = (
            expert["initial_control_return"] - policy["initial_control_return"]
        )
        treatment_gap = (
            expert["initial_treatment_return"] - policy["initial_treatment_return"]
        )
        rows.append(
            {
                "task": task,
                "pair_id": pair_id,
                "environment_seed": expert["environment_seed"],
                "control_expert_failure_gap": control_gap,
                "treatment_expert_failure_gap": treatment_gap,
                "expert_failure_gap_change": treatment_gap - control_gap,
                "expert": expert,
                "policy": policy,
            }
        )
    summary = summarize_return_gap_rows(rows)
    task_gates = {
        task: values["nonshrinking_fraction"]
        >= minimum_nonshrinking_fraction
        for task, values in summary["per_task"].items()
    }
    required_rows = [
        row
        for row in rows
        if row["task"] == required_task
        and int(row["environment_seed"]) == required_env_seed
    ]
    if len(required_rows) != 1:
        raise ValueError(
            f"required diagnostic pair {(required_task, required_env_seed)} not unique"
        )
    required_seed_gate = required_rows[0]["expert_failure_gap_change"] > 1e-8
    passed = all(task_gates.values()) and required_seed_gate
    return {
        "schema_version": "robotwin_awr_reward_credit_audit_v1",
        "passed": passed,
        "config_differences": [".".join(path) for path in sorted(differences)],
        "gamma": gamma,
        "minimum_nonshrinking_fraction": minimum_nonshrinking_fraction,
        "task_gates": task_gates,
        "required_pair": {
            "task": required_task,
            "environment_seed": required_env_seed,
            "passed": required_seed_gate,
            "row": required_rows[0],
        },
        "summary": summary,
        "pairs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--treatment-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--minimum-nonshrinking-fraction", type=float, default=0.9)
    parser.add_argument("--required-task", default="place_can_basket")
    parser.add_argument("--required-env-seed", type=int, default=619)
    parser.add_argument("--require-passed", action="store_true")
    args = parser.parse_args()
    replay = ReplayBuffer.load(args.replay_dir)
    replay_manifest = json.loads(
        (args.replay_dir / "manifest.json").read_text(encoding="utf-8")
    )
    result = audit_reward_credit(
        replay,
        replay_manifest,
        _load_config(args.control_config),
        _load_config(args.treatment_config),
        minimum_nonshrinking_fraction=args.minimum_nonshrinking_fraction,
        required_task=args.required_task,
        required_env_seed=args.required_env_seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compact = {
        "passed": result["passed"],
        "task_gates": result["task_gates"],
        "required_pair": result["required_pair"],
        "summary": result["summary"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    if args.require_passed and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
