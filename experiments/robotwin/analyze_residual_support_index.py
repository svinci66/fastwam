"""Audit a residual support index on labeled replay episodes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.rl.online_policy import load_residual_actor_checkpoint
from fastwam.rl.replay_buffer import ReplayBuffer
from fastwam.rl.support_gate import ResidualSupportIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--support-index", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        label: {
            "count": len(values),
            "state_accept_rate": float(
                np.mean([value["state_in_support"] for value in values])
            ),
            "action_accept_rate": float(
                np.mean([value["action_in_support"] for value in values])
            ),
            "joint_accept_rate": float(
                np.mean([value["in_support"] for value in values])
            ),
            "state_score_median": float(
                np.median([value["state_score"] for value in values])
            ),
            "action_score_median": float(
                np.median([value["action_score"] for value in values])
            ),
        }
        for label, values in sorted(grouped.items())
    }


def main() -> None:
    args = parse_args()
    replay = ReplayBuffer.load(args.replay_dir)
    arrays = replay.arrays()
    if "language_feature" not in arrays:
        raise ValueError("support audit requires language-conditioned replay data")
    actor, _ = load_residual_actor_checkpoint(args.checkpoint, device=args.device)
    support = ResidualSupportIndex.load(args.support_index)
    episode_success: dict[str, bool] = defaultdict(bool)
    for transition in replay.transitions:
        episode_success[transition.episode_id] |= bool(transition.success)

    rows: list[dict[str, Any]] = []
    device = torch.device(args.device)
    with torch.inference_mode():
        for index, transition in enumerate(replay.transitions):
            baseline = arrays["baseline_actions"][index]
            context = np.concatenate(
                [arrays["observation_feature"][index], arrays["proprio"][index]]
            ).astype(np.float32)
            candidate = actor(
                torch.from_numpy(context).unsqueeze(0).to(device),
                torch.from_numpy(baseline).unsqueeze(0).to(device),
                language_feature=torch.from_numpy(
                    arrays["language_feature"][index]
                ).unsqueeze(0).to(device),
            )[0].cpu().numpy()
            decision = support.evaluate(
                observation_feature=arrays["observation_feature"][index],
                proprio=arrays["proprio"][index],
                baseline_actions=baseline,
                candidate_residual_actions=candidate - baseline,
                language_feature=arrays["language_feature"][index],
            )
            rows.append(
                {
                    "episode_id": transition.episode_id,
                    "episode_success": episode_success[transition.episode_id],
                    "task": transition.task_description,
                    "behavior": transition.behavior_mode,
                    "state_score": decision.state_score,
                    "action_score": decision.action_score,
                    "state_in_support": decision.state_in_support,
                    "action_in_support": decision.action_in_support,
                    "in_support": decision.in_support,
                }
            )

    summary = {
        "num_transitions": len(rows),
        "state_threshold": support.state_threshold,
        "action_threshold": support.action_threshold,
        "overall_joint_accept_rate": float(
            np.mean([row["in_support"] for row in rows])
        ),
        "by_episode_success": _group_summary(rows, "episode_success"),
        "by_behavior": _group_summary(rows, "behavior"),
        "by_task": _group_summary(rows, "task"),
        "support_index": str(args.support_index.resolve()),
        "replay_dir": str(args.replay_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
