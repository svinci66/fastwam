"""Audit a residual-AWR training run before it may advance to the next stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--expected-max-residual-scale", type=float, default=0.1)
    parser.add_argument("--maximum-critic-loss-ratio", type=float, default=1.25)
    parser.add_argument("--maximum-saturation-fraction", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def finite_history(history: list[dict[str, Any]]) -> bool:
    return bool(history) and all(
        math.isfinite(float(value))
        for row in history
        for value in row.values()
        if isinstance(value, (int, float))
    )


def residual_statistics(
    checkpoint: dict[str, Any], replay: ReplayBuffer, *, batch_size: int = 256
) -> dict[str, float]:
    actor = ResidualActor(ResidualActorConfig(**checkpoint["actor_config"]))
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    arrays = replay.arrays()
    use_goal = bool(checkpoint["awr_config"]["use_goal_conditioning"])
    residuals: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(replay), batch_size):
            stop = min(start + batch_size, len(replay))
            context_parts = [
                torch.from_numpy(arrays["observation_feature"][start:stop]),
                torch.from_numpy(arrays["proprio"][start:stop]),
            ]
            if use_goal:
                context_parts.append(torch.from_numpy(arrays["goal_feature"][start:stop]))
            language = (
                torch.from_numpy(arrays["language_feature"][start:stop])
                if "language_feature" in arrays
                else None
            )
            residual = actor.residual(
                torch.cat(context_parts, dim=-1),
                baseline_actions=torch.from_numpy(arrays["baseline_actions"][start:stop]),
                language_feature=language,
            )
            residuals.append(residual.cpu().numpy())
    values = np.concatenate(residuals, axis=0)
    scale = np.asarray(checkpoint["actor_config"]["residual_scale"], dtype=np.float32)
    active = scale > 0
    normalized = np.abs(values[..., active]) / scale[active]
    frozen_max = float(np.max(np.abs(values[..., ~active]))) if np.any(~active) else 0.0
    return {
        "residual_rms": float(np.sqrt(np.mean(np.square(values[..., active])))),
        "residual_abs_max": float(np.max(np.abs(values[..., active]))),
        "normalized_abs_mean": float(np.mean(normalized)),
        "saturation_fraction": float(np.mean(normalized >= 0.95)),
        "frozen_dimension_abs_max": frozen_max,
    }


def main() -> None:
    args = parse_args()
    training_dir = args.training_dir.expanduser().resolve()
    replay_dir = args.replay_dir.expanduser().resolve()
    expected_tasks = sorted(
        value.strip() for value in args.expected_tasks.split(",") if value.strip()
    )
    checkpoint = torch.load(
        training_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    history = json.loads((training_dir / "history.json").read_text(encoding="utf-8"))
    manifest = json.loads((replay_dir / "manifest.json").read_text(encoding="utf-8"))
    replay = ReplayBuffer.load(replay_dir)
    provenance = manifest.get("provenance", {})
    actual_tasks = sorted(provenance.get("selected_tasks", []))
    task_pair_counts = provenance.get("task_pair_counts", {})
    critic_initial = float(history[0]["critic_loss"])
    critic_final = float(history[-1]["critic_loss"])
    actor_initial = str(checkpoint["summary"]["initialization_sha256"]["actor"])
    actor_final = state_dict_sha256(checkpoint["actor"])
    scales = np.asarray(checkpoint["actor_config"]["residual_scale"], dtype=np.float64)
    stats = residual_statistics(checkpoint, replay)

    checks = {
        "checkpoint_format": checkpoint.get("format") == "fastwam_residual_awr_v2",
        "expected_epoch_count": len(history) == args.expected_epochs,
        "finite_history": finite_history(history),
        "critic_not_diverged": (
            math.isfinite(critic_initial)
            and math.isfinite(critic_final)
            and critic_final <= critic_initial * args.maximum_critic_loss_ratio
        ),
        "actor_moved_from_zero_initialization": actor_final != actor_initial,
        "zero_initialized_configuration": bool(
            checkpoint["actor_config"].get("zero_init_output", False)
        ),
        "expected_residual_scale": bool(
            np.isclose(scales.max(), args.expected_max_residual_scale)
            and np.count_nonzero(scales == 0.0) == 2
        ),
        "frozen_action_dimensions_exact": stats["frozen_dimension_abs_max"] == 0.0,
        "residual_not_saturated": (
            stats["saturation_fraction"] <= args.maximum_saturation_fraction
        ),
        "nonzero_training_targets": float(
            checkpoint["summary"].get("executed_residual_rms", 0.0)
        )
        > 1e-8,
        "expected_tasks_only": actual_tasks == expected_tasks,
        "every_task_has_pairs": all(int(task_pair_counts.get(task, 0)) > 0 for task in expected_tasks),
        "task_balancing_enabled": bool(checkpoint["awr_config"].get("balance_tasks", False)),
        "head_wan_reward": (
            checkpoint["reward_config"].get("imagination_reward_type")
            == "wan_vae_head_trajectory_global_norm_v1"
            and float(checkpoint["reward_config"].get("imagination_weight", 0.0)) > 0.0
        ),
    }
    payload = {
        "schema_version": "robotwin_residual_awr_training_health_v1",
        "training_dir": str(training_dir),
        "replay_dir": str(replay_dir),
        "expected_tasks": expected_tasks,
        "actual_tasks": actual_tasks,
        "task_pair_counts": task_pair_counts,
        "num_transitions": len(replay),
        "num_epochs": len(history),
        "critic_loss_initial": critic_initial,
        "critic_loss_final": critic_final,
        "critic_loss_ratio": critic_final / max(critic_initial, 1e-12),
        "actor_loss_initial": float(history[0]["actor_loss"]),
        "actor_loss_final": float(history[-1]["actor_loss"]),
        "actor_initialization_sha256": actor_initial,
        "actor_final_sha256": actor_final,
        "residual_statistics": stats,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit("residual AWR training health audit failed")


if __name__ == "__main__":
    main()
