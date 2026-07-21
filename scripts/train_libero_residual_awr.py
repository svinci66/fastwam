"""Train a lightweight residual AWR learner from a frozen FastWAM replay shard.

This entrypoint never instantiates FastWAM or LIBERO.  Use a collector process to
produce a replay shard, release the large inference model, and then run this learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parents[1]
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from fastwam.rl.awr_trainer import AWRConfig, train_residual_awr
from fastwam.rl.models import ResidualActor, ResidualActorConfig, ValueCritic, ValueCriticConfig
from fastwam.rl.replay_buffer import ReplayBuffer
from fastwam.rl.rewards import CompositeRewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timeout-bootstrap-json",
        type=Path,
        default=None,
        help="JSON mapping truncated episode_id to an explicit bootstrap value.",
    )
    parser.add_argument(
        "--timeout-bootstrap-value",
        type=float,
        default=None,
        help="Explicit shared bootstrap for every truncated episode (for example 0.0).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate replay/config/model shapes without optimizer steps or output writes.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_training_process(seed: int) -> None:
    """Seed every learner RNG before actor or critic construction."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _state_dict_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.timeout_bootstrap_json is not None and args.timeout_bootstrap_value is not None:
        raise ValueError("pass only one timeout bootstrap option")
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("top-level training config must be a mapping")
    reward_config = CompositeRewardConfig(**cfg["reward"])
    awr_config = AWRConfig(**cfg["awr"])
    reward_config.validate()
    awr_config.validate()
    seed_training_process(awr_config.seed)

    replay = ReplayBuffer.load(args.replay_dir)
    replay_manifest = json.loads((args.replay_dir / "manifest.json").read_text())
    imitation_scales = cfg.get("imitation_dimension_scales")
    imitation_scales_array = (
        None if imitation_scales is None else np.asarray(imitation_scales, dtype=np.float32)
    )
    transition_rewards, _ = replay.relabel_rewards(
        reward_config,
        imitation_dimension_scales=imitation_scales_array,
    )
    timeout_bootstrap = None
    if args.timeout_bootstrap_json is not None:
        timeout_bootstrap = {
            str(key): float(value)
            for key, value in json.loads(args.timeout_bootstrap_json.read_text()).items()
        }
    elif args.timeout_bootstrap_value is not None:
        timeout_bootstrap = {
            transition.episode_id: float(args.timeout_bootstrap_value)
            for transition in replay.transitions
            if transition.truncated
        }
    returns = replay.monte_carlo_returns(
        awr_config.gamma,
        timeout_bootstrap_values=timeout_bootstrap,
        transition_rewards=transition_rewards,
    )

    arrays = replay.arrays()
    feature_dim = int(arrays["observation_feature"].shape[1])
    proprio_dim = int(arrays["proprio"].shape[1])
    action_horizon = int(arrays["baseline_actions"].shape[1])
    action_dim = int(arrays["baseline_actions"].shape[2])
    language_feature_dim = (
        int(arrays["language_feature"].shape[1])
        if "language_feature" in arrays
        else 0
    )
    action_mask = (
        np.arange(action_horizon)[None, :] < arrays["effective_k"][:, None]
    )[..., None]
    residual_values = arrays["executed_actions"] - arrays["baseline_actions"]
    executed_residual_rms = float(
        np.sqrt(
            np.sum(np.square(residual_values) * action_mask)
            / (np.sum(action_mask) * action_dim)
        )
    )
    if executed_residual_rms <= 1e-8:
        raise ValueError(
            "replay has no executed action variation relative to frozen FastWAM; "
            "collect policy and controlled-noise behavior before residual AWR"
        )
    context_dim = feature_dim + proprio_dim
    if awr_config.use_goal_conditioning:
        context_dim += feature_dim
    model_cfg = dict(cfg["model"])
    language_embedding_dim = int(model_cfg.get("language_embedding_dim", 0))
    if language_embedding_dim > 0 and language_feature_dim == 0:
        raise ValueError(
            "model requests language conditioning but replay has no language_feature array"
        )
    model_cfg.update(
        context_dim=context_dim,
        action_horizon=action_horizon,
        action_dim=action_dim,
        language_feature_dim=(language_feature_dim if language_embedding_dim > 0 else 0),
    )
    actor_config = ResidualActorConfig(**model_cfg)
    actor = ResidualActor(actor_config)
    critic = ValueCritic(
        ValueCriticConfig(
            context_dim=context_dim,
            hidden_dims=tuple(cfg["critic_hidden_dims"]),
            action_horizon=action_horizon,
            action_dim=action_dim,
            language_feature_dim=actor_config.language_feature_dim,
            language_embedding_dim=actor_config.language_embedding_dim,
            baseline_action_embedding_dim=actor_config.baseline_action_embedding_dim,
        )
    )
    initialization_sha256 = {
        "actor": _state_dict_sha256(actor),
        "critic": _state_dict_sha256(critic),
    }

    summary = {
        "num_transitions": len(replay),
        "feature_dim": feature_dim,
        "proprio_dim": proprio_dim,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "language_feature_dim": language_feature_dim,
        "language_conditioning": actor_config.language_feature_dim > 0,
        "baseline_action_conditioning": actor_config.baseline_action_embedding_dim > 0,
        "executed_residual_rms": executed_residual_rms,
        "behavior_mode_counts": {
            mode: sum(transition.behavior_mode == mode for transition in replay.transitions)
            for mode in sorted({transition.behavior_mode for transition in replay.transitions})
        },
        "task_transition_counts": {
            f"{suite}/task{task_id}": sum(
                transition.task_suite == suite and transition.task_id == task_id
                for transition in replay.transitions
            )
            for suite, task_id in sorted(
                {(transition.task_suite, transition.task_id) for transition in replay.transitions}
            )
        },
        "task_balancing": awr_config.balance_tasks,
        "context_dim": context_dim,
        "reward_mean": float(np.mean(transition_rewards)),
        "reward_min": float(np.min(transition_rewards)),
        "reward_max": float(np.max(transition_rewards)),
        "return_mean": float(np.mean(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "actor_parameters": sum(parameter.numel() for parameter in actor.parameters()),
        "critic_parameters": sum(parameter.numel() for parameter in critic.parameters()),
        "timeout_bootstrap_mode": (
            "per_episode_json"
            if args.timeout_bootstrap_json is not None
            else "shared_explicit_value"
            if args.timeout_bootstrap_value is not None
            else "none"
        ),
        "timeout_bootstrap_value": args.timeout_bootstrap_value,
        "training_seed": awr_config.seed,
        "initialization_sha256": initialization_sha256,
        "reward_encoder_version": replay_manifest["reward_encoder_version"],
        "language_encoder_version": replay_manifest.get("language_encoder_version"),
        "imagination_reward_type": replay_manifest.get(
            "imagination_reward_type", "progress_v1"
        ),
        "replay_schema_version": replay_manifest["schema_version"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.validate_only:
        return

    device_name = str(cfg.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Refusing an accidental CPU training run."
        )
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    history = train_residual_awr(
        actor,
        critic,
        replay,
        returns,
        awr_config,
        device=device_name,
    )
    checkpoint = {
        "format": "fastwam_residual_awr_v2",
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_config": actor.export_config(),
        "critic_config": critic.export_config(),
        "awr_config": asdict(awr_config),
        "reward_config": asdict(reward_config),
        "replay_manifest_sha256": _sha256(args.replay_dir / "manifest.json"),
        "replay_provenance": replay_manifest.get("provenance", {}),
        "summary": summary,
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
