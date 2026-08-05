"""Build a compact residual-IQL replay from RoboTwin transition captures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.robotwin.analyze_imagination_rewards import (
    discover_records,
    encode_record_images,
    fit_task_balanced_camera_normalization,
    resolve_encoder_dtype,
)
from experiments.robotwin.imagination_reward_utils import ROBOTWIN_CAMERA_NAMES
from fastwam.rl.replay_buffer import ReplayBuffer, ReplayTransition
from fastwam.rl.rewards import (
    CompositeRewardConfig,
    EpisodeShapingBudget,
    GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE,
    compute_composite_reward,
    compute_imagination_reward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--reward-encoder-version", required=True)
    parser.add_argument(
        "--reward-config",
        type=Path,
        default=PROJECT_ROOT / "configs/rl/robotwin_residual_iql_smoke.yaml",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--encoder-dtype",
        default="auto",
        choices=("auto", "fp32", "bf16", "fp16"),
        help="SigLIP precision; auto matches online bf16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--min-trial-index",
        type=int,
        help="Keep only records whose trial index is at least this value.",
    )
    parser.add_argument(
        "--max-trial-index",
        type=int,
        help="Keep only records whose trial index is at most this value.",
    )
    parser.add_argument(
        "--env-seed-override",
        action="append",
        default=[],
        metavar="INPUT_DIR=SEED",
        help=(
            "Override the raw trial index used as env_seed for every record from "
            "an input directory. Repeat once per input directory when several "
            "single-seed captures are merged."
        ),
    )
    parser.add_argument(
        "--camera-normalization-manifest",
        type=Path,
        help=(
            "Reuse camera normalization from an existing replay manifest instead "
            "of fitting it on the new shard. Required for merge-compatible "
            "incremental replay construction."
        ),
    )
    return parser.parse_args()


def parse_env_seed_overrides(values: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"--env-seed-override must have INPUT_DIR=SEED format, got {value!r}"
            )
        raw_path, raw_seed = value.rsplit("=", 1)
        path = Path(raw_path).expanduser().resolve()
        try:
            seed = int(raw_seed)
        except ValueError as exc:
            raise ValueError(
                f"--env-seed-override seed must be an integer, got {raw_seed!r}"
            ) from exc
        if str(path) in overrides:
            raise ValueError(f"duplicate environment-seed override for {path}")
        overrides[str(path)] = seed
    return overrides


def discover_sourced_records(
    input_dirs: list[Path],
    *,
    env_seed_overrides: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    env_seed_overrides = env_seed_overrides or {}
    for source_index, input_dir in enumerate(input_dirs):
        source_id = f"source{source_index:03d}-{input_dir.resolve().name}"
        resolved_input_dir = input_dir.expanduser().resolve()
        override = env_seed_overrides.get(str(resolved_input_dir))
        for record in discover_records([input_dir]):
            record = dict(record)
            record["source_id"] = source_id
            if override is not None:
                record["raw_trial_idx"] = int(record["trial_idx"])
                record["trial_idx"] = int(override)
            records.append(record)
    return records


def filter_records_by_trial_range(
    records: list[dict[str, Any]],
    *,
    min_trial_index: int | None,
    max_trial_index: int | None,
) -> list[dict[str, Any]]:
    if (
        min_trial_index is not None
        and max_trial_index is not None
        and min_trial_index > max_trial_index
    ):
        raise ValueError("min-trial-index must not exceed max-trial-index")
    filtered = [
        record
        for record in records
        if (min_trial_index is None or int(record["trial_idx"]) >= min_trial_index)
        and (max_trial_index is None or int(record["trial_idx"]) <= max_trial_index)
    ]
    if not filtered:
        raise ValueError(
            "trial-index filtering removed every transition: "
            f"min={min_trial_index}, max={max_trial_index}"
        )
    return filtered


def load_camera_normalization_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    normalization = payload.get("provenance", {}).get("camera_normalization")
    if not isinstance(normalization, dict):
        raise ValueError(f"Replay manifest has no camera normalization: {path}")
    cameras = normalization.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != set(ROBOTWIN_CAMERA_NAMES):
        raise ValueError(
            "Camera normalization manifest must contain exactly "
            f"{ROBOTWIN_CAMERA_NAMES}, got {None if not isinstance(cameras, dict) else sorted(cameras)}"
        )
    for camera, settings in cameras.items():
        center = float(settings.get("center", np.nan))
        scale = float(settings.get("scale", np.nan))
        if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid camera normalization for {camera}: {settings}")
    return normalization


def combine_camera_features(camera_features: dict[str, np.ndarray]) -> np.ndarray:
    if set(camera_features) != set(ROBOTWIN_CAMERA_NAMES):
        raise ValueError(
            f"expected camera features {ROBOTWIN_CAMERA_NAMES}, "
            f"got {sorted(camera_features)}"
        )
    # Keep replay features byte-for-byte aligned with
    # ``OnlineResidualPolicy.combine_normalized_camera_features``: normalize
    # each camera independently before concatenating the camera embeddings,
    # then normalize the fused vector once more.  A single normalization after
    # concatenation changes the relative camera weights and makes an offline
    # replay disagree with the online gate.
    normalized = []
    feature_dims: set[int] = set()
    for camera in ROBOTWIN_CAMERA_NAMES:
        feature = np.asarray(camera_features[camera], dtype=np.float32).reshape(-1)
        if feature.size == 0 or not np.all(np.isfinite(feature)):
            raise ValueError(f"camera feature {camera!r} must be finite and non-empty")
        norm = float(np.linalg.norm(feature))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"camera feature {camera!r} has invalid norm {norm}")
        normalized.append(feature / norm)
        feature_dims.add(int(feature.size))
    if len(feature_dims) != 1:
        raise ValueError(f"camera feature dimensions differ: {sorted(feature_dims)}")
    combined = np.concatenate(normalized).astype(np.float32)
    norm = float(np.linalg.norm(combined))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"combined camera feature has invalid norm {norm}")
    return np.ascontiguousarray(combined / norm)


def pad_action_chunk(value: np.ndarray, target_k: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[0] > target_k:
        raise ValueError(
            f"{name} must have shape [1..{target_k},D], got {array.shape}"
        )
    if array.shape[0] == target_k:
        return np.ascontiguousarray(array)
    suffix = np.repeat(array[-1:], target_k - array.shape[0], axis=0)
    return np.ascontiguousarray(np.concatenate([array, suffix], axis=0))


def pad_environment_rewards(value: np.ndarray, target_k: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size <= 0 or array.size > target_k:
        raise ValueError(
            f"environment_rewards must contain 1..{target_k} values, got {array.shape}"
        )
    return np.pad(array, (0, target_k - array.size)).astype(np.float32, copy=False)


def validate_episode_records(records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["task_name"]),
                str(record["behavior"]),
                str(record.get("source_id", "default")),
                int(record["trial_idx"]),
            )
        ].append(record)
    for key, values in grouped.items():
        values.sort(key=lambda item: int(item["replan_idx"]))
        indices = [int(value["replan_idx"]) for value in values]
        if indices != list(range(len(indices))):
            raise ValueError(f"episode {key} has non-contiguous replans: {indices}")
        terminal = [
            bool(value["terminated"]) or bool(value["truncated"]) for value in values
        ]
        if any(terminal[:-1]) or not terminal[-1]:
            raise ValueError(f"episode {key} does not have exactly one final boundary")


def build_replay(
    records: list[dict[str, Any]],
    encoded: list[dict[str, dict[str, np.ndarray]]],
    *,
    reward_config: CompositeRewardConfig,
    reward_encoder_version: str,
    camera_normalization: dict[str, Any],
    imitation_dimension_scales: np.ndarray | None,
) -> ReplayBuffer:
    if len(records) != len(encoded):
        raise ValueError("record and encoded-feature counts differ")
    task_ids = {
        task_name: index
        for index, task_name in enumerate(
            sorted({str(record["task_name"]) for record in records})
        )
    }
    camera_norm = {
        camera: {
            "center": float(settings["center"]),
            "scale": float(settings["scale"]),
        }
        for camera, settings in camera_normalization["cameras"].items()
    }
    budgets: dict[str, EpisodeShapingBudget] = {}
    replay = ReplayBuffer()
    ordered = sorted(
        zip(records, encoded),
        key=lambda item: (
            str(item[0]["task_name"]),
            str(item[0]["behavior"]),
            int(item[0]["trial_idx"]),
            int(item[0]["replan_idx"]),
        ),
    )
    for record, features in ordered:
        task_name = str(record["task_name"])
        behavior = str(record["behavior"])
        episode_id = (
            f"robotwin2.0-{str(record.get('source_id', 'default'))}-"
            f"{task_name}-{behavior}-"
            f"trial{int(record['trial_idx']):06d}"
        )
        target_k = int(record["target_step"])
        effective_k = int(record["effective_k"])
        arrays_path = Path(record["record_dir"]) / str(record["rollout_arrays_file"])
        with np.load(arrays_path, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        baseline_actions = pad_action_chunk(
            arrays["baseline_actions"], target_k, name="baseline_actions"
        )
        executed_actions = pad_action_chunk(
            arrays["executed_actions"], target_k, name="executed_actions"
        )
        environment_rewards = pad_environment_rewards(
            arrays["environment_rewards"], target_k
        )
        progress = compute_imagination_reward(
            features["current"],
            features["actual"],
            features["predicted_goal"],
            reward_type=reward_config.imagination_reward_type,
            camera_weights={camera: 1.0 for camera in ROBOTWIN_CAMERA_NAMES},
            camera_normalization=camera_norm,
            clip_value=reward_config.imagination_clip,
            alignment_valid=bool(record["alignment_valid"]),
        )
        budget = budgets.setdefault(
            episode_id, EpisodeShapingBudget.from_config(reward_config)
        )
        reward = compute_composite_reward(
            environment_rewards=environment_rewards,
            success=bool(record["transition_success"]),
            baseline_actions=baseline_actions,
            executed_actions=executed_actions,
            effective_k=effective_k,
            imagination_progress=progress.raw_progress,
            alignment_valid=bool(record["alignment_valid"]),
            config=reward_config,
            shaping_budget=budget,
            imitation_dimension_scales=imitation_dimension_scales,
        )
        action_seed = int(
            record.get("action_corruption_seed", record.get("action_noise_seed", 0))
        )
        replay.append(
            ReplayTransition(
                episode_id=episode_id,
                transition_index=int(record["replan_idx"]),
                task_suite=str(record["task_suite"]),
                task_id=task_ids[task_name],
                task_description=str(record["task_description"]),
                env_seed=int(record["trial_idx"]),
                goal_seed=0,
                action_seed=action_seed,
                policy_version=str(record["policy_version"]),
                predictor_version=str(record["predictor_version"]),
                reward_encoder_version=reward_encoder_version,
                behavior_mode=str(record["action_mode"]),
                action_noise_std=float(record.get("action_noise_std", 0.0)),
                target_k=target_k,
                effective_k=effective_k,
                goal_frame_index=int(record["goal_frame_index"]),
                goal_tau=float(record["goal_tau"]),
                terminated=bool(record["terminated"]),
                truncated=bool(record["truncated"]),
                success=bool(record["transition_success"]),
                alignment_valid=bool(record["alignment_valid"]),
                observation_feature=combine_camera_features(features["current"]),
                next_observation_feature=combine_camera_features(features["actual"]),
                goal_feature=combine_camera_features(features["predicted_goal"]),
                proprio=np.asarray(arrays["proprio"], dtype=np.float32),
                next_proprio=np.asarray(arrays["next_proprio"], dtype=np.float32),
                baseline_actions=baseline_actions,
                executed_actions=executed_actions,
                environment_rewards=environment_rewards,
                reward=reward,
                imagination_reward_type=reward_config.imagination_reward_type,
                language_feature=np.asarray(
                    arrays["language_feature"], dtype=np.float32
                ),
                language_encoder_version=str(record["language_encoder_version"]),
            )
        )
    return replay


def main() -> None:
    args = parse_args()
    if not args.reward_encoder_version.strip():
        raise ValueError("reward_encoder_version must not be empty")
    cfg = OmegaConf.to_container(OmegaConf.load(args.reward_config), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("top-level reward config must be a mapping")
    reward_config = CompositeRewardConfig(**cfg["reward"])
    if reward_config.imagination_reward_type != GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE:
        raise ValueError(
            "RoboTwin replay builder requires task-balanced global camera normalization"
        )
    imitation_scales = cfg.get("imitation_dimension_scales")
    imitation_scales_array = (
        None
        if imitation_scales is None
        else np.asarray(imitation_scales, dtype=np.float32)
    )
    env_seed_overrides = parse_env_seed_overrides(args.env_seed_override)
    encoder_dtype = resolve_encoder_dtype(args.encoder_dtype, device=args.device)
    records = filter_records_by_trial_range(
        discover_sourced_records(
            args.input_dir,
            env_seed_overrides=env_seed_overrides,
        ),
        min_trial_index=args.min_trial_index,
        max_trial_index=args.max_trial_index,
    )
    validate_episode_records(records)
    behavior_counts = Counter(str(record["behavior"]) for record in records)
    task_behavior_counts = Counter(
        (str(record["task_name"]), str(record["behavior"])) for record in records
    )
    encoded = encode_record_images(
        records,
        encoder_path=args.encoder_path,
        device=args.device,
        batch_size=args.batch_size,
        encoder_dtype=encoder_dtype,
    )
    normalization = (
        fit_task_balanced_camera_normalization(records, encoded)
        if args.camera_normalization_manifest is None
        else load_camera_normalization_manifest(
            args.camera_normalization_manifest
        )
    )
    replay = build_replay(
        records,
        encoded,
        reward_config=reward_config,
        reward_encoder_version=args.reward_encoder_version,
        camera_normalization=normalization,
        imitation_dimension_scales=imitation_scales_array,
    )
    language_versions = {
        str(record["language_encoder_version"]) for record in records
    }
    if len(language_versions) != 1:
        raise ValueError(
            f"RoboTwin replay mixes language encoders: {sorted(language_versions)}"
        )
    output = replay.save(
        args.output_dir,
        provenance={
            "reward_encoder_version": args.reward_encoder_version,
            "imagination_reward_type": reward_config.imagination_reward_type,
            "camera_names": list(ROBOTWIN_CAMERA_NAMES),
            "camera_weights": {
                camera: 1.0 for camera in ROBOTWIN_CAMERA_NAMES
            },
            "camera_normalization": normalization,
            "camera_image_size": 224,
            "encoder_dtype": str(encoder_dtype).removeprefix("torch."),
            "feature_fusion": "per_camera_l2_then_head_left_right_concat_l2_v1",
            "language_encoder_version": next(iter(language_versions)),
            "language_pooling": "fastwam_umt5_masked_mean_v1",
            "source_schema": "robotwin_imagination_transition_v1",
            "seed_fields": "trial_index_and_action_corruption_seed_v1",
            "input_dirs": [str(path.resolve()) for path in args.input_dir],
            "input_sources": sorted(
                {str(record["source_id"]) for record in records}
            ),
            "env_seed_overrides": {
                key: value
                for key, value in sorted(env_seed_overrides.items())
                if "/" in key
            },
            "trial_index_filter": {
                "minimum": args.min_trial_index,
                "maximum": args.max_trial_index,
            },
            "camera_normalization_source_manifest": (
                None
                if args.camera_normalization_manifest is None
                else str(args.camera_normalization_manifest.resolve())
            ),
            "behavior_transition_counts": dict(sorted(behavior_counts.items())),
            "task_behavior_transition_counts": {
                f"{task}/{behavior}": count
                for (task, behavior), count in sorted(task_behavior_counts.items())
            },
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "num_transitions": len(replay),
                "num_tasks": normalization["num_tasks"],
                "behavior_transition_counts": dict(sorted(behavior_counts.items())),
                "camera_normalization": normalization,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
