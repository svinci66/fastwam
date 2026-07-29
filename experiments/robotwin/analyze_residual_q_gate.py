"""Audit conservative dual-Q residual gates on a frozen RoboTwin replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.rl.online_policy import (
    load_iql_q_critics,
    load_residual_actor_checkpoint,
)
from fastwam.rl.replay_buffer import ReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--critic-source", choices=("target", "online"), default="target")
    parser.add_argument("--margins", default="0,0.0025,0.005,0.01,0.02,0.05")
    parser.add_argument("--max-disagreements", default="0.01,0.02,0.05,0.1")
    return parser.parse_args()


def _csv_floats(value: str, *, name: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(not np.isfinite(item) or item < 0.0 for item in values):
        raise ValueError(f"{name} must contain finite non-negative values")
    return values


def masked_mse(
    left: np.ndarray,
    right: np.ndarray,
    effective_k: np.ndarray,
    selection: np.ndarray,
) -> float:
    indices = np.flatnonzero(selection)
    if indices.size == 0:
        raise ValueError("masked_mse selection is empty")
    values = [
        np.square(left[index, : effective_k[index]] - right[index, : effective_k[index]])
        for index in indices
    ]
    return float(np.mean(np.concatenate([value.reshape(-1) for value in values])))


def _rates_by_label(labels: list[str], gate: np.ndarray) -> dict[str, float]:
    return {
        label: float(np.mean(gate[np.asarray([item == label for item in labels])]))
        for label in sorted(set(labels))
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    margins = _csv_floats(args.margins, name="margins")
    max_disagreements = _csv_floats(
        args.max_disagreements, name="max-disagreements"
    )
    device = torch.device(args.device)
    replay = ReplayBuffer.load(args.replay_dir)
    arrays = replay.arrays()
    actor, payload = load_residual_actor_checkpoint(args.checkpoint, device=device)
    q_critics = load_iql_q_critics(
        payload, device=device, source=args.critic_source
    )
    candidates: list[np.ndarray] = []
    advantages: list[np.ndarray] = []
    count = len(replay)
    with torch.inference_mode():
        for start in range(0, count, args.batch_size):
            end = min(start + args.batch_size, count)
            context = torch.from_numpy(
                np.concatenate(
                    [
                        arrays["observation_feature"][start:end],
                        arrays["proprio"][start:end],
                    ],
                    axis=1,
                )
            ).to(device)
            baseline = torch.from_numpy(
                arrays["baseline_actions"][start:end]
            ).to(device)
            language = (
                None
                if "language_feature" not in arrays
                else torch.from_numpy(arrays["language_feature"][start:end]).to(device)
            )
            candidate = actor(context, baseline, language_feature=language)
            batch_advantages = []
            for critic in q_critics:
                baseline_q = critic(context, baseline, baseline, language)
                candidate_q = critic(context, baseline, candidate, language)
                batch_advantages.append((candidate_q - baseline_q).cpu().numpy())
            candidates.append(candidate.cpu().numpy())
            advantages.append(np.stack(batch_advantages, axis=1))

    candidate_actions = np.concatenate(candidates, axis=0)
    q_advantages = np.concatenate(advantages, axis=0)
    q_min = np.min(q_advantages, axis=1)
    q_disagreement = np.abs(q_advantages[:, 0] - q_advantages[:, 1])
    effective_k = arrays["effective_k"].astype(np.int64)
    expert = np.asarray(
        [transition.behavior_mode == "expert" for transition in replay.transitions]
    )
    baseline_expert_mse = masked_mse(
        arrays["baseline_actions"], arrays["executed_actions"], effective_k, expert
    )
    candidate_expert_mse = masked_mse(
        candidate_actions, arrays["executed_actions"], effective_k, expert
    )
    behaviors = [transition.behavior_mode for transition in replay.transitions]
    tasks = [transition.task_description for transition in replay.transitions]

    grid: list[dict[str, Any]] = []
    for margin in margins:
        for max_disagreement in max_disagreements:
            gate = (q_min >= margin) & (q_disagreement <= max_disagreement)
            gated_actions = arrays["baseline_actions"].copy()
            gated_actions[gate] = candidate_actions[gate]
            gated_expert_mse = masked_mse(
                gated_actions, arrays["executed_actions"], effective_k, expert
            )
            grid.append(
                {
                    "margin": margin,
                    "max_disagreement": max_disagreement,
                    "apply_rate": float(np.mean(gate)),
                    "expert_apply_rate": float(np.mean(gate[expert])),
                    "apply_rate_by_behavior": _rates_by_label(behaviors, gate),
                    "apply_rate_by_task": _rates_by_label(tasks, gate),
                    "expert_action_mse": gated_expert_mse,
                    "expert_mse_reduction_vs_frozen_baseline_fraction": float(
                        1.0 - gated_expert_mse / baseline_expert_mse
                    ),
                }
            )

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "critic_source": args.critic_source,
        "num_transitions": count,
        "num_expert_transitions": int(expert.sum()),
        "frozen_baseline_expert_action_mse": baseline_expert_mse,
        "ungated_candidate_expert_action_mse": candidate_expert_mse,
        "q_advantage_min_quantiles": {
            str(quantile): float(np.quantile(q_min, quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "q_advantage_disagreement_quantiles": {
            str(quantile): float(np.quantile(q_disagreement, quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "grid": grid,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
