#!/usr/bin/env python3
"""Train a bounded adapter on top of a frozen RoboTwin residual actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.rl.adapter_trainer import train_residual_adapter_awr
from fastwam.rl.awr_trainer import AWRConfig, build_context
from fastwam.rl.models import (
    FrozenResidualAdapterActor,
    FrozenSpatialResidualAdapterActor,
    ResidualAdapter,
    ResidualAdapterConfig,
    SpatialResidualAdapter,
    SpatialResidualAdapterConfig,
    ValueCritic,
    ValueCriticConfig,
)
from fastwam.rl.online_policy import load_residual_actor_checkpoint
from fastwam.rl.replay_buffer import ReplayBuffer
from fastwam.rl.rewards import CompositeRewardConfig
from fastwam.models.wan22.fastwam import (
    FASTWAM_VIDEO_EXPERT_HEAD_SPATIAL_VERSION,
)
from experiments.robotwin.build_wan_vae_head_awr_replay import (
    discover_records,
    select_reward_tasks,
)


CHECKPOINT_FORMAT = "fastwam_residual_adapter_awr_v1"
SPATIAL_CHECKPOINT_FORMAT = "fastwam_residual_spatial_adapter_awr_v1"
PAIR_PATTERN = re.compile(r"-pair(?P<pair_id>\d+)-(?P<behavior>expert|policy)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout-bootstrap-value", type=float, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--spatial-reward-json",
        type=Path,
        help="Reward payload whose source records contain aligned spatial features.",
    )
    parser.add_argument("--tasks", default="")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def seed_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _config(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("top-level adapter config must be a mapping")
    return payload


def zero_equivalence_audit(
    actor: FrozenResidualAdapterActor,
    arrays: dict[str, np.ndarray],
    *,
    use_goal_conditioning: bool,
    batch_size: int = 128,
    spatial_features: np.ndarray | None = None,
) -> dict[str, Any]:
    actor.eval()
    exact = True
    maximum = 0.0
    adapter_maximum = 0.0
    with torch.inference_mode():
        for start in range(0, len(arrays["observation_feature"]), batch_size):
            stop = min(start + batch_size, len(arrays["observation_feature"]))
            observation = torch.from_numpy(arrays["observation_feature"][start:stop])
            proprio = torch.from_numpy(arrays["proprio"][start:stop])
            goal = torch.from_numpy(arrays["goal_feature"][start:stop])
            baseline = torch.from_numpy(arrays["baseline_actions"][start:stop])
            language = (
                None
                if "language_feature" not in arrays
                else torch.from_numpy(arrays["language_feature"][start:stop])
            )
            context = build_context(
                observation,
                proprio,
                goal,
                use_goal_conditioning=use_goal_conditioning,
            )
            base = actor.base_actor(context, baseline, language)
            if isinstance(actor, FrozenSpatialResidualAdapterActor):
                if spatial_features is None:
                    raise ValueError("spatial zero audit requires spatial_features")
                spatial = torch.from_numpy(spatial_features[start:stop])
                composed = actor(context, baseline, spatial, language)
                _, _, adapter_residual = actor.components(
                    context, baseline, spatial, language
                )
            else:
                if spatial_features is not None:
                    raise ValueError("mean adapter zero audit received spatial features")
                composed = actor(context, baseline, language)
                _, _, adapter_residual = actor.components(context, baseline, language)
            exact = exact and torch.equal(base, composed)
            maximum = max(maximum, float(torch.max(torch.abs(base - composed))))
            adapter_maximum = max(
                adapter_maximum, float(torch.max(torch.abs(adapter_residual)))
            )
    return {
        "exact": exact,
        "maximum_action_difference": maximum,
        "maximum_adapter_residual": adapter_maximum,
        "transitions_checked": int(len(arrays["observation_feature"])),
    }


def adapter_output_audit(
    actor: FrozenResidualAdapterActor,
    arrays: dict[str, np.ndarray],
    *,
    use_goal_conditioning: bool,
    device: torch.device,
    maximum_ratio: float,
    batch_size: int = 128,
    spatial_features: np.ndarray | None = None,
) -> dict[str, Any]:
    actor.eval()
    adapter_square = 0.0
    base_square = 0.0
    elements = 0
    adapter_maximum = 0.0
    gripper_maximum = 0.0
    with torch.inference_mode():
        for start in range(0, len(arrays["observation_feature"]), batch_size):
            stop = min(start + batch_size, len(arrays["observation_feature"]))
            observation = torch.from_numpy(arrays["observation_feature"][start:stop]).to(device)
            proprio = torch.from_numpy(arrays["proprio"][start:stop]).to(device)
            goal = torch.from_numpy(arrays["goal_feature"][start:stop]).to(device)
            baseline = torch.from_numpy(arrays["baseline_actions"][start:stop]).to(device)
            language = (
                None
                if "language_feature" not in arrays
                else torch.from_numpy(arrays["language_feature"][start:stop]).to(device)
            )
            context = build_context(
                observation,
                proprio,
                goal,
                use_goal_conditioning=use_goal_conditioning,
            )
            if isinstance(actor, FrozenSpatialResidualAdapterActor):
                if spatial_features is None:
                    raise ValueError("spatial output audit requires spatial_features")
                spatial = torch.from_numpy(spatial_features[start:stop]).to(device)
                _, ordinary, adapter = actor.components(
                    context, baseline, spatial, language
                )
            else:
                _, ordinary, adapter = actor.components(context, baseline, language)
            adapter_square += float(torch.sum(torch.square(adapter)))
            base_square += float(torch.sum(torch.square(ordinary)))
            elements += adapter.numel()
            adapter_maximum = max(
                adapter_maximum, float(torch.max(torch.abs(adapter)))
            )
            gripper_maximum = max(
                gripper_maximum,
                float(torch.max(torch.abs(adapter[..., [6, 13]]))),
            )
    adapter_rms = float(np.sqrt(adapter_square / elements))
    base_rms = float(np.sqrt(base_square / elements))
    ratio = adapter_rms / max(base_rms, 1e-12)
    passed = (
        ratio <= maximum_ratio
        and gripper_maximum == 0.0
        and adapter_maximum
        <= float(torch.max(actor.adapter.adapter_scale).detach().cpu()) + 1e-7
    )
    return {
        "passed": passed,
        "adapter_rms": adapter_rms,
        "base_residual_rms": base_rms,
        "adapter_to_base_rms_ratio": ratio,
        "maximum_allowed_ratio": maximum_ratio,
        "adapter_max_abs": adapter_maximum,
        "gripper_adapter_max_abs": gripper_maximum,
        "transitions_checked": int(len(arrays["observation_feature"])),
    }


def load_spatial_features(
    reward_json: Path,
    replay: ReplayBuffer,
    replay_manifest: dict[str, Any],
    *,
    tasks: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(reward_json.read_text(encoding="utf-8"))
    payload = select_reward_tasks(payload, tasks)
    records = discover_records(payload)
    record_lookup = {}
    checkpoint_hashes = set()
    for record in records:
        key = (
            str(record["task_name"]),
            int(record["pair_episode_id"]),
            str(record["behavior"]),
            int(record["replan_idx"]),
        )
        if key in record_lookup:
            raise ValueError(f"duplicate spatial source record {key}")
        record_lookup[key] = record
    task_id_map = {
        int(task_id): str(task)
        for task, task_id in replay_manifest["provenance"]["task_id_map"].items()
    }
    features = []
    source_paths = []
    expected_shape = None
    for transition in replay.transitions:
        match = PAIR_PATTERN.search(transition.episode_id)
        if match is None:
            raise ValueError(f"cannot recover pair id from {transition.episode_id!r}")
        key = (
            task_id_map[transition.task_id],
            int(match.group("pair_id")),
            match.group("behavior"),
            int(transition.transition_index),
        )
        record = record_lookup.get(key)
        if record is None:
            raise ValueError(f"spatial source record is missing for {key}")
        record_dir = Path(record["record_dir"])
        metadata = json.loads((record_dir / "metadata.json").read_text())
        if (
            metadata.get("video_expert_head_spatial_feature_version")
            != FASTWAM_VIDEO_EXPERT_HEAD_SPATIAL_VERSION
        ):
            raise ValueError(f"missing spatial feature provenance in {record_dir}")
        checkpoint_hash = str(
            metadata.get("video_expert_checkpoint_sha256", "")
        ).strip()
        if len(checkpoint_hash) != 64:
            raise ValueError(f"missing FastWAM checkpoint hash in {record_dir}")
        checkpoint_hashes.add(checkpoint_hash)
        with np.load(
            record_dir / str(record["rollout_arrays_file"]), allow_pickle=False
        ) as arrays:
            if "video_expert_head_spatial_feature" not in arrays.files:
                raise ValueError(f"spatial feature array missing in {record_dir}")
            feature = np.asarray(
                arrays["video_expert_head_spatial_feature"], dtype=np.float32
            )
        if feature.ndim != 2 or feature.shape[0] != 3 or not np.all(
            np.isfinite(feature)
        ):
            raise ValueError(f"invalid spatial feature {feature.shape} in {record_dir}")
        if expected_shape is None:
            expected_shape = tuple(int(value) for value in feature.shape)
        elif tuple(feature.shape) != expected_shape:
            raise ValueError(
                f"mixed spatial feature shapes: {expected_shape} and {feature.shape}"
            )
        features.append(feature)
        source_paths.append(str(record_dir.resolve()))
    if len(checkpoint_hashes) != 1:
        raise ValueError(f"mixed spatial FastWAM checkpoints: {checkpoint_hashes}")
    result = np.stack(features).astype(np.float32)
    return result, {
        "version": FASTWAM_VIDEO_EXPERT_HEAD_SPATIAL_VERSION,
        "shape": list(result.shape),
        "fastwam_checkpoint_sha256": next(iter(checkpoint_hashes)),
        "reward_json": str(reward_json.resolve()),
        "reward_json_sha256": sha256(reward_json),
        "source_record_count": len(source_paths),
    }


def main() -> None:
    args = parse_args()
    cfg = _config(args.config)
    reward_config = CompositeRewardConfig(**cfg["reward"])
    awr_config = AWRConfig(**cfg["awr"])
    if args.seed is not None:
        awr_config = replace(awr_config, seed=int(args.seed))
        cfg["awr"]["seed"] = int(args.seed)
    reward_config.validate()
    awr_config.validate()
    if cfg.get("sampler", {}).get("type") != "pair_behavior_balanced_v1":
        raise ValueError("adapter training requires pair_behavior_balanced_v1")
    seed_process(awr_config.seed)

    replay = ReplayBuffer.load(args.replay_dir)
    replay_manifest = json.loads((args.replay_dir / "manifest.json").read_text())
    if replay_manifest.get("imagination_reward_type") != reward_config.imagination_reward_type:
        raise ValueError("adapter reward type does not match replay")
    arrays = replay.arrays()
    adapter_type = str(cfg["adapter"].get("type", "mean")).strip()
    if adapter_type not in {"mean", "head_spatial_contrast"}:
        raise ValueError(f"unsupported adapter type: {adapter_type!r}")
    spatial_features = None
    spatial_provenance = None
    if adapter_type == "head_spatial_contrast":
        if args.spatial_reward_json is None:
            raise ValueError("spatial adapter requires --spatial-reward-json")
        selected_tasks = [
            value.strip() for value in args.tasks.split(",") if value.strip()
        ]
        spatial_features, spatial_provenance = load_spatial_features(
            args.spatial_reward_json,
            replay,
            replay_manifest,
            tasks=selected_tasks,
        )
    elif args.spatial_reward_json is not None:
        raise ValueError("mean adapter must not receive --spatial-reward-json")
    base_actor, base_payload = load_residual_actor_checkpoint(
        args.base_checkpoint, device="cpu"
    )
    if base_payload.get("format") != "fastwam_residual_awr_v2":
        raise ValueError("base checkpoint must be an ordinary residual-AWR actor")
    base_summary = base_payload.get("summary", {})
    feature_dim = int(arrays["observation_feature"].shape[1])
    proprio_dim = int(arrays["proprio"].shape[1])
    language_dim = int(arrays["language_feature"].shape[1])
    context_dim = feature_dim + proprio_dim
    if awr_config.use_goal_conditioning:
        context_dim += feature_dim
    if base_actor.config.context_dim != context_dim:
        raise ValueError("base actor context dimension does not match replay")
    if base_actor.config.action_horizon != arrays["baseline_actions"].shape[1]:
        raise ValueError("base actor action horizon does not match replay")
    if base_actor.config.action_dim != arrays["baseline_actions"].shape[2]:
        raise ValueError("base actor action dimension does not match replay")
    if base_actor.config.language_feature_dim != language_dim:
        raise ValueError("base actor language dimension does not match replay")

    adapter_payload = dict(cfg["adapter"])
    adapter_payload.pop("type", None)
    for key in ("hidden_dims", "adapter_scale"):
        adapter_payload[key] = tuple(adapter_payload[key])
    adapter_payload.update(
        context_dim=context_dim,
        action_horizon=base_actor.config.action_horizon,
        action_dim=base_actor.config.action_dim,
        language_feature_dim=language_dim,
    )
    if adapter_type == "head_spatial_contrast":
        if spatial_features is None:
            raise RuntimeError("spatial feature loading was skipped")
        adapter_payload.update(
            spatial_token_count=int(spatial_features.shape[1]),
            spatial_token_dim=int(spatial_features.shape[2]),
        )
        adapter = SpatialResidualAdapter(
            SpatialResidualAdapterConfig(**adapter_payload)
        )
        actor = FrozenSpatialResidualAdapterActor(base_actor, adapter)
    else:
        adapter = ResidualAdapter(ResidualAdapterConfig(**adapter_payload))
        actor = FrozenResidualAdapterActor(base_actor, adapter)
    zero_audit = zero_equivalence_audit(
        actor,
        arrays,
        use_goal_conditioning=awr_config.use_goal_conditioning,
        spatial_features=spatial_features,
    )
    if not zero_audit["exact"] or zero_audit["maximum_adapter_residual"] != 0.0:
        raise RuntimeError(f"zero adapter equivalence failed: {zero_audit}")

    imitation_scales = np.asarray(
        cfg["imitation_dimension_scales"], dtype=np.float32
    )
    transition_rewards, _ = replay.relabel_rewards(
        reward_config, imitation_dimension_scales=imitation_scales
    )
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
    critic = ValueCritic(
        ValueCriticConfig(
            context_dim=context_dim,
            hidden_dims=tuple(cfg["critic_hidden_dims"]),
            action_horizon=base_actor.config.action_horizon,
            action_dim=base_actor.config.action_dim,
            language_feature_dim=language_dim,
            language_embedding_dim=base_actor.config.language_embedding_dim,
            baseline_action_embedding_dim=base_actor.config.baseline_action_embedding_dim,
        )
    )
    initialization = {
        "adapter": state_dict_sha256(adapter.state_dict()),
        "critic": state_dict_sha256(critic.state_dict()),
    }
    summary = {
        "num_transitions": len(replay),
        "feature_dim": feature_dim,
        "proprio_dim": proprio_dim,
        "language_feature_dim": language_dim,
        "action_horizon": base_actor.config.action_horizon,
        "action_dim": base_actor.config.action_dim,
        "context_dim": context_dim,
        "training_seed": awr_config.seed,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "base_actor_sha256": state_dict_sha256(base_actor.state_dict()),
        "initialization_sha256": initialization,
        "zero_equivalence_audit": zero_audit,
        "reward_mean": float(np.mean(transition_rewards)),
        "return_mean": float(np.mean(returns)),
        "imagination_reward_type": reward_config.imagination_reward_type,
        "sampler_type": cfg["sampler"]["type"],
        "adapter_type": adapter_type,
        "spatial_feature_provenance": spatial_provenance,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.validate_only:
        return
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(str(cfg.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the adapter config")
    history, sampler_audit = train_residual_adapter_awr(
        actor,
        critic,
        replay,
        returns,
        awr_config,
        anchor_weight=float(cfg["loss"]["anchor_weight"]),
        smoothness_weight=float(cfg["loss"]["smoothness_weight"]),
        device=device,
        spatial_features=spatial_features,
    )
    output_audit = adapter_output_audit(
        actor,
        arrays,
        use_goal_conditioning=awr_config.use_goal_conditioning,
        device=device,
        maximum_ratio=float(cfg["audit"]["max_adapter_to_base_rms_ratio"]),
        spatial_features=spatial_features,
    )
    summary["sampler_audit"] = sampler_audit
    summary["trained_adapter_audit"] = output_audit
    summary["adapter_parameters"] = sum(
        parameter.numel() for parameter in actor.adapter.parameters()
    )
    checkpoint = {
        "format": (
            SPATIAL_CHECKPOINT_FORMAT
            if adapter_type == "head_spatial_contrast"
            else CHECKPOINT_FORMAT
        ),
        "actor": actor.base_actor.state_dict(),
        "actor_config": actor.base_actor.export_config(),
        "adapter": actor.adapter.state_dict(),
        "adapter_config": actor.adapter.export_config(),
        "critic": critic.state_dict(),
        "critic_config": critic.export_config(),
        "awr_config": asdict(awr_config),
        "reward_config": asdict(reward_config),
        "adapter_loss_config": dict(cfg["loss"]),
        "replay_manifest_sha256": sha256(args.replay_dir / "manifest.json"),
        "replay_provenance": replay_manifest.get("provenance", {}),
        "base_checkpoint_sha256": summary["base_checkpoint_sha256"],
        "summary": summary,
        "trained_epochs": awr_config.epochs,
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "adapter_audit.json").write_text(
        json.dumps(
            {
                "zero_equivalence": zero_audit,
                "sampler": sampler_audit,
                "trained_adapter": output_audit,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"trained_adapter_audit": output_audit}, indent=2))
    if not output_audit["passed"]:
        raise SystemExit("trained adapter failed trust-region audit")


if __name__ == "__main__":
    main()
