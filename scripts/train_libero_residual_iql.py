#!/usr/bin/env python3
"""Train a bounded FastWAM residual actor with offline Implicit Q-Learning."""

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

from fastwam.rl.iql_trainer import IQLConfig, train_residual_iql
from fastwam.rl.models import (
    ActionValueCritic,
    ActionValueCriticConfig,
    ResidualActor,
    ResidualActorConfig,
    ValueCritic,
    ValueCriticConfig,
)
from fastwam.rl.replay_buffer import ReplayBuffer
from fastwam.rl.rewards import CompositeRewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Warm-start actor, both online Q critics, and the value critic from "
            "an existing fastwam_residual_iql_v1 checkpoint. Optimizer state is "
            "intentionally reset for replay-scale changes."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate replay/config/model shapes without optimizer steps or output writes.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override the config device (useful for a CPU-only smoke test).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override IQL epochs without changing the frozen experiment config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the IQL seed while preserving every other experiment setting.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _seed_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _warm_start_models(
    checkpoint_path: Path,
    actor: ResidualActor,
    q_critics: tuple[ActionValueCritic, ActionValueCritic],
    value_critic: ValueCritic,
) -> dict[str, str]:
    """Strictly restore trainable IQL modules while resetting optimizers."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"initialization checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "fastwam_residual_iql_v1":
        raise ValueError(
            "initialization checkpoint must use format fastwam_residual_iql_v1; "
            f"got {checkpoint.get('format')!r}"
        )
    saved_q_critics = checkpoint.get("q_critics")
    if not isinstance(saved_q_critics, (list, tuple)) or len(saved_q_critics) != 2:
        raise ValueError("initialization checkpoint must contain exactly two Q critics")

    expected_configs = {
        "actor_config": actor.export_config(),
        "q_critic_config": q_critics[0].export_config(),
        "value_critic_config": value_critic.export_config(),
    }
    for key, expected in expected_configs.items():
        actual = checkpoint.get(key)
        # zero_init_output changes only fresh initialization; it is absent from
        # checkpoints produced before that safety option was added and has no
        # effect once a state dict is restored.
        comparable_actual = dict(actual) if isinstance(actual, dict) else actual
        comparable_expected = dict(expected)
        if key == "actor_config" and isinstance(comparable_actual, dict):
            comparable_actual.pop("zero_init_output", None)
            comparable_expected.pop("zero_init_output", None)
        if comparable_actual != comparable_expected:
            raise ValueError(
                f"initialization checkpoint {key} does not match the current model: "
                f"saved={actual!r}, current={expected!r}"
            )

    actor.load_state_dict(checkpoint["actor"], strict=True)
    for critic, state_dict in zip(q_critics, saved_q_critics):
        critic.load_state_dict(state_dict, strict=True)
    value_critic.load_state_dict(checkpoint["value_critic"], strict=True)
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": _sha256(checkpoint_path),
        "format": str(checkpoint["format"]),
    }


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("top-level training config must be a mapping")
    iql_values = dict(cfg["iql"])
    if args.epochs is not None:
        iql_values["epochs"] = args.epochs
    if args.seed is not None:
        iql_values["seed"] = args.seed
    iql_config = IQLConfig(**iql_values)
    reward_config = CompositeRewardConfig(**cfg["reward"])
    iql_config.validate()
    reward_config.validate()
    _seed_process(iql_config.seed)

    replay = ReplayBuffer.load(args.replay_dir)
    replay_manifest = json.loads((args.replay_dir / "manifest.json").read_text())
    imitation_scales = cfg.get("imitation_dimension_scales")
    imitation_scales_array = (
        None if imitation_scales is None
        else np.asarray(imitation_scales, dtype=np.float32)
    )
    transition_rewards, _ = replay.relabel_rewards(
        reward_config,
        imitation_dimension_scales=imitation_scales_array,
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
    context_dim = feature_dim + proprio_dim
    if iql_config.use_goal_conditioning:
        context_dim += feature_dim

    model_cfg = dict(cfg["model"])
    language_embedding_dim = int(model_cfg.get("language_embedding_dim", 0))
    if language_embedding_dim > 0 and language_feature_dim == 0:
        raise ValueError(
            "model requests language conditioning but replay has no language_feature"
        )
    model_cfg.update(
        context_dim=context_dim,
        action_horizon=action_horizon,
        action_dim=action_dim,
        language_feature_dim=(
            language_feature_dim if language_embedding_dim > 0 else 0
        ),
    )
    actor_config = ResidualActorConfig(**model_cfg)
    actor = ResidualActor(actor_config)

    q_cfg = dict(cfg["q_critic"])
    q_cfg.update(
        context_dim=context_dim,
        action_horizon=action_horizon,
        action_dim=action_dim,
        language_feature_dim=actor_config.language_feature_dim,
        language_embedding_dim=actor_config.language_embedding_dim,
    )
    q_config = ActionValueCriticConfig(**q_cfg)
    q_critics = (ActionValueCritic(q_config), ActionValueCritic(q_config))

    value_cfg = dict(cfg["value_critic"])
    value_cfg.update(
        context_dim=context_dim,
        action_horizon=action_horizon,
        action_dim=action_dim,
        language_feature_dim=actor_config.language_feature_dim,
        language_embedding_dim=actor_config.language_embedding_dim,
        baseline_action_embedding_dim=actor_config.baseline_action_embedding_dim,
    )
    value_config = ValueCriticConfig(**value_cfg)
    value_critic = ValueCritic(value_config)
    initialization_source = None
    if args.init_checkpoint is not None:
        initialization_source = _warm_start_models(
            args.init_checkpoint,
            actor,
            q_critics,
            value_critic,
        )
    initialization_sha256 = {
        "actor": _state_dict_sha256(actor),
        "q1": _state_dict_sha256(q_critics[0]),
        "q2": _state_dict_sha256(q_critics[1]),
        "value": _state_dict_sha256(value_critic),
    }
    task_keys = sorted(
        {(transition.task_suite, transition.task_id) for transition in replay.transitions}
    )
    summary = {
        "num_transitions": len(replay),
        "feature_dim": feature_dim,
        "proprio_dim": proprio_dim,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "language_feature_dim": language_feature_dim,
        "language_conditioning": actor_config.language_feature_dim > 0,
        "baseline_action_conditioning": actor_config.baseline_action_embedding_dim > 0,
        "task_balancing": iql_config.balance_tasks,
        "behavior_balancing": iql_config.balance_behaviors,
        "num_tasks": len(task_keys),
        "context_dim": context_dim,
        "reward_mean": float(np.mean(transition_rewards)),
        "reward_min": float(np.min(transition_rewards)),
        "reward_max": float(np.max(transition_rewards)),
        "terminated_transitions": int(arrays["terminated"].sum()),
        "truncated_transitions": int(arrays["truncated"].sum()),
        "bootstrap_timeouts": iql_config.bootstrap_timeouts,
        "actor_parameters": sum(parameter.numel() for parameter in actor.parameters()),
        "q_parameters_each": sum(
            parameter.numel() for parameter in q_critics[0].parameters()
        ),
        "value_parameters": sum(
            parameter.numel() for parameter in value_critic.parameters()
        ),
        "training_seed": iql_config.seed,
        "initialization_sha256": initialization_sha256,
        "initialization_source": initialization_source,
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

    device_name = str(args.device or cfg.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Pass --device cpu only for a smoke test."
        )
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    history, target_q_critics = train_residual_iql(
        actor,
        q_critics,
        value_critic,
        replay,
        transition_rewards,
        iql_config,
        device=device_name,
    )
    checkpoint = {
        "format": "fastwam_residual_iql_v1",
        "actor": actor.state_dict(),
        "q_critics": [critic.state_dict() for critic in q_critics],
        "target_q_critics": [critic.state_dict() for critic in target_q_critics],
        "value_critic": value_critic.state_dict(),
        "actor_config": actor.export_config(),
        "q_critic_config": q_critics[0].export_config(),
        "value_critic_config": value_critic.export_config(),
        "iql_config": asdict(iql_config),
        "reward_config": asdict(reward_config),
        "replay_manifest_sha256": _sha256(args.replay_dir / "manifest.json"),
        "replay_provenance": replay_manifest.get("provenance", {}),
        "summary": summary,
        "initialization_source": initialization_source,
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_config = dict(cfg)
    run_config["iql"] = asdict(iql_config)
    run_config["resolved_device"] = device_name
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
