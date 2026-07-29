"""Audit matched RoboTwin residual-IQL checkpoints on their frozen replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.rl.models import ResidualActor, ResidualActorConfig
from fastwam.rl.replay_buffer import ReplayBuffer
from fastwam.rl.rewards import CompositeRewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--no-imagination-checkpoint", type=Path, required=True)
    parser.add_argument("--imagination-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def _actor_config(payload: dict[str, Any]) -> ResidualActorConfig:
    values = dict(payload["actor_config"])
    for key in ("hidden_dims", "residual_scale", "action_low", "action_high"):
        values[key] = tuple(values[key])
    return ResidualActorConfig(**values)


def load_actor(path: Path, device: torch.device) -> tuple[ResidualActor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "fastwam_residual_iql_v1":
        raise ValueError(f"unexpected checkpoint format in {path}: {payload.get('format')}")
    actor = ResidualActor(_actor_config(payload))
    actor.load_state_dict(payload["actor"], strict=True)
    return actor.to(device).eval(), payload


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def predict_actions(
    actor: ResidualActor,
    arrays: dict[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    count = arrays["observation_feature"].shape[0]
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            context = np.concatenate(
                [arrays["observation_feature"][start:end], arrays["proprio"][start:end]],
                axis=1,
            )
            language = arrays.get("language_feature")
            predicted = actor(
                torch.from_numpy(context).to(device),
                torch.from_numpy(arrays["baseline_actions"][start:end]).to(device),
                language_feature=(
                    None
                    if language is None
                    else torch.from_numpy(language[start:end]).to(device)
                ),
            )
            outputs.append(predicted.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def masked_mse(
    left: np.ndarray,
    right: np.ndarray,
    effective_k: np.ndarray,
    selection: np.ndarray,
) -> float:
    indices = np.flatnonzero(selection)
    if indices.size == 0:
        raise ValueError("masked_mse selection is empty")
    squared: list[np.ndarray] = []
    for index in indices:
        squared.append(np.square(left[index, : effective_k[index]] - right[index, : effective_k[index]]))
    return float(np.mean(np.concatenate([value.reshape(-1) for value in squared])))


def residual_rms(
    corrected: np.ndarray,
    baseline: np.ndarray,
    effective_k: np.ndarray,
) -> float:
    squared = [
        np.square(corrected[index, : effective_k[index]] - baseline[index, : effective_k[index]])
        for index in range(corrected.shape[0])
    ]
    flattened = np.concatenate([value.reshape(-1) for value in squared])
    return float(np.sqrt(np.mean(flattened)))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    replay = ReplayBuffer.load(args.replay_dir)
    arrays = replay.arrays()
    no_actor, no_payload = load_actor(args.no_imagination_checkpoint, device)
    imagination_actor, imagination_payload = load_actor(
        args.imagination_checkpoint, device
    )
    no_actions = predict_actions(
        no_actor, arrays, device=device, batch_size=args.batch_size
    )
    imagination_actions = predict_actions(
        imagination_actor, arrays, device=device, batch_size=args.batch_size
    )
    effective_k = arrays["effective_k"].astype(np.int64)
    expert = np.asarray(
        [transition.behavior_mode == "expert" for transition in replay.transitions]
    )
    baseline_expert_mse = masked_mse(
        arrays["baseline_actions"], arrays["executed_actions"], effective_k, expert
    )

    def checkpoint_metrics(
        corrected: np.ndarray, payload: dict[str, Any]
    ) -> dict[str, Any]:
        expert_mse = masked_mse(
            corrected, arrays["executed_actions"], effective_k, expert
        )
        reward_config = CompositeRewardConfig(**payload["reward_config"])
        rewards, _ = replay.relabel_rewards(reward_config)
        residual = corrected - arrays["baseline_actions"]
        return {
            "actor_state_sha256": state_dict_sha256(payload["actor"]),
            "reward_mean": float(np.mean(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "predicted_residual_rms": residual_rms(
                corrected, arrays["baseline_actions"], effective_k
            ),
            "predicted_residual_max_abs": float(np.max(np.abs(residual))),
            "gripper_residual_max_abs": float(
                np.max(np.abs(residual[..., [6, 13]]))
            ),
            "expert_action_mse": expert_mse,
            "expert_mse_reduction_vs_frozen_baseline_fraction": float(
                1.0 - expert_mse / baseline_expert_mse
            ),
        }

    difference = imagination_actions - no_actions
    summary = {
        "num_transitions": len(replay),
        "num_expert_transitions": int(expert.sum()),
        "frozen_baseline_expert_action_mse": baseline_expert_mse,
        "no_imagination": checkpoint_metrics(no_actions, no_payload),
        "imagination": checkpoint_metrics(imagination_actions, imagination_payload),
        "matched_initialization": (
            no_payload["summary"]["initialization_sha256"]
            == imagination_payload["summary"]["initialization_sha256"]
        ),
        "actor_output_difference_rms": float(np.sqrt(np.mean(np.square(difference)))),
        "actor_output_difference_max_abs": float(np.max(np.abs(difference))),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
