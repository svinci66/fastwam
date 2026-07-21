"""Convert saved LIBERO action chunks into a compact, versioned RL replay shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.imagination_reward_utils import split_horizontal_camera_views
from fastwam.rl.replay_buffer import ReplayBuffer, ReplayTransition
from fastwam.rl.rewards import (
    CompositeRewardConfig,
    EpisodeShapingBudget,
    compute_composite_reward,
    compute_imagination_reward,
)


CAMERAS = ("agent", "wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        required=True,
        help="Raw transition root. Repeat for policy/noise collectors sharing one replay.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument(
        "--reward-encoder-version",
        required=True,
        help="Immutable version or checksum recorded in every replay transition.",
    )
    parser.add_argument(
        "--reward-config",
        type=Path,
        default=project_root / "configs/rl/libero_residual_awr_mvp.yaml",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--agent-weight", type=float, default=0.5)
    parser.add_argument("--wrist-weight", type=float, default=0.5)
    return parser.parse_args()


def discover_records(input_dirs: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        for metadata_path in sorted(input_dir.resolve().rglob("metadata.json")):
            metadata = json.loads(metadata_path.read_text())
            array_name = metadata.get("rollout_arrays_file")
            if not array_name:
                continue
            record_dir = metadata_path.parent
            record = dict(metadata)
            record["record_dir"] = record_dir
            record["source_input_dir"] = str(input_dir.resolve())
            record["arrays_path"] = record_dir / str(array_name)
            for name in ("current", "predicted_goal", "actual"):
                path = record_dir / f"{name}.png"
                if not path.exists():
                    raise FileNotFoundError(path)
                record[f"{name}_path"] = path
            records.append(record)
    if not records:
        raise ValueError(f"no transition records with rollout arrays found under {input_dirs}")
    records.sort(
        key=lambda record: (
            str(record["task_suite"]),
            int(record["task_id"]),
            int(record["env_seed"]),
            str(record["action_mode"]),
            float(record.get("action_noise_std", 0.0)),
            int(record["trial_idx"]),
            int(record["replan_idx"]),
        )
    )
    return records


def _camera_images(
    records: list[dict[str, Any]],
) -> tuple[list[Image.Image], list[tuple[int, str, str]]]:
    images: list[Image.Image] = []
    keys: list[tuple[int, str, str]] = []
    for record_index, record in enumerate(records):
        for phase in ("current", "actual", "predicted_goal"):
            with Image.open(record[f"{phase}_path"]) as image:
                views = split_horizontal_camera_views(image.convert("RGB"))
            for camera in CAMERAS:
                images.append(Image.fromarray(views[camera]))
                keys.append((record_index, phase, camera))
    return images, keys


def encode_record_images(
    records: list[dict[str, Any]],
    *,
    encoder_path: Path,
    device: str,
    batch_size: int,
) -> list[dict[str, dict[str, np.ndarray]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    from transformers import SiglipImageProcessor, SiglipVisionModel

    processor = SiglipImageProcessor.from_pretrained(encoder_path, local_files_only=True)
    model = SiglipVisionModel.from_pretrained(
        encoder_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    images, keys = _camera_images(records)
    features: list[dict[str, dict[str, np.ndarray]]] = [
        {phase: {} for phase in ("current", "actual", "predicted_goal")}
        for _ in records
    ]
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            inputs = processor(images=batch_images, return_tensors="pt")
            output = model(pixel_values=inputs["pixel_values"].to(device)).pooler_output
            output = torch.nn.functional.normalize(output.float(), dim=-1).cpu().numpy()
            for key, feature in zip(keys[start : start + batch_size], output):
                record_index, phase, camera = key
                features[record_index][phase][camera] = feature.astype(np.float32, copy=False)
    return features


def _combined_feature(camera_features: dict[str, np.ndarray]) -> np.ndarray:
    if set(camera_features) != set(CAMERAS):
        raise ValueError(f"expected camera features {CAMERAS}, got {sorted(camera_features)}")
    combined = np.concatenate([camera_features[camera] for camera in CAMERAS]).astype(np.float32)
    norm = float(np.linalg.norm(combined))
    if norm <= 0.0:
        raise ValueError("combined feature has zero norm")
    return combined / norm


def build_replay(
    records: list[dict[str, Any]],
    encoded: list[dict[str, dict[str, np.ndarray]]],
    *,
    reward_encoder_version: str,
    reward_config: CompositeRewardConfig,
    camera_weights: dict[str, float],
    imitation_dimension_scales: np.ndarray | None,
) -> ReplayBuffer:
    if len(records) != len(encoded):
        raise ValueError("record and feature counts differ")
    replay = ReplayBuffer()
    budgets: dict[str, EpisodeShapingBudget] = {}
    for record, features in zip(records, encoded):
        behavior_mode = str(record["action_mode"])
        action_noise_std = float(record.get("action_noise_std", 0.0))
        behavior_tag = (
            f"noise{action_noise_std:.6f}" if behavior_mode == "noise" else behavior_mode
        )
        episode_id = (
            f"{record['task_suite']}-task{int(record['task_id']):02d}-"
            f"seed{int(record['env_seed'])}-{behavior_tag}-"
            f"trial{int(record['trial_idx']):06d}"
        )
        budget = budgets.setdefault(episode_id, EpisodeShapingBudget.from_config(reward_config))
        progress = compute_imagination_reward(
            features["current"],
            features["actual"],
            features["predicted_goal"],
            reward_type=reward_config.imagination_reward_type,
            camera_weights=camera_weights,
            clip_value=reward_config.imagination_clip,
            alignment_valid=bool(record["alignment_valid"]),
        )
        with np.load(record["arrays_path"], allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        target_k = int(record["target_step"])
        effective_k = int(record["effective_k"])
        breakdown = compute_composite_reward(
            environment_rewards=arrays["environment_rewards"],
            success=bool(record["transition_success"]),
            baseline_actions=arrays["baseline_actions"],
            executed_actions=arrays["executed_actions"],
            effective_k=effective_k,
            imagination_progress=progress.raw_progress,
            alignment_valid=bool(record["alignment_valid"]),
            config=reward_config,
            shaping_budget=budget,
            imitation_dimension_scales=imitation_dimension_scales,
        )
        replay.append(
            ReplayTransition(
                episode_id=episode_id,
                transition_index=int(record["replan_idx"]),
                task_suite=str(record["task_suite"]),
                task_id=int(record["task_id"]),
                task_description=str(record["task_description"]),
                env_seed=int(record["env_seed"]),
                goal_seed=int(record["goal_seed"]),
                action_seed=int(record["action_seed"]),
                policy_version=str(record["policy_version"]),
                predictor_version=str(record["predictor_version"]),
                reward_encoder_version=reward_encoder_version,
                behavior_mode=behavior_mode,
                action_noise_std=action_noise_std,
                target_k=target_k,
                effective_k=effective_k,
                goal_frame_index=int(record["goal_frame_index"]),
                goal_tau=float(record["goal_tau"]),
                terminated=bool(record["terminated"]),
                truncated=bool(record["truncated"]),
                success=bool(record["transition_success"]),
                alignment_valid=bool(record["alignment_valid"]),
                observation_feature=_combined_feature(features["current"]),
                next_observation_feature=_combined_feature(features["actual"]),
                goal_feature=_combined_feature(features["predicted_goal"]),
                proprio=arrays["proprio"],
                next_proprio=arrays["next_proprio"],
                baseline_actions=arrays["baseline_actions"],
                executed_actions=arrays["executed_actions"],
                environment_rewards=arrays["environment_rewards"],
                reward=breakdown,
                imagination_reward_type=reward_config.imagination_reward_type,
            )
        )
    return replay


def main() -> None:
    args = parse_args()
    if not args.reward_encoder_version.strip():
        raise ValueError("reward_encoder_version must not be empty")
    cfg = OmegaConf.to_container(OmegaConf.load(args.reward_config), resolve=True)
    reward_config = CompositeRewardConfig(**cfg["reward"])
    imitation_scales = cfg.get("imitation_dimension_scales")
    imitation_scales_array = (
        None if imitation_scales is None else np.asarray(imitation_scales, dtype=np.float32)
    )
    records = discover_records(args.input_dir)
    encoded = encode_record_images(
        records,
        encoder_path=args.encoder_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    replay = build_replay(
        records,
        encoded,
        reward_encoder_version=args.reward_encoder_version,
        reward_config=reward_config,
        camera_weights={"agent": args.agent_weight, "wrist": args.wrist_weight},
        imitation_dimension_scales=imitation_scales_array,
    )
    output = replay.save(
        args.output_dir,
        provenance={
            "reward_encoder_version": args.reward_encoder_version,
            "imagination_reward_type": reward_config.imagination_reward_type,
            "camera_names": list(CAMERAS),
            "camera_weights": {
                "agent": float(args.agent_weight),
                "wrist": float(args.wrist_weight),
            },
            "camera_image_size": 224,
            "feature_fusion": "per_camera_l2_then_agent_wrist_concat_l2_v1",
        },
    )
    print(json.dumps({"output_dir": str(output), "num_transitions": len(replay)}, indent=2))


if __name__ == "__main__":
    main()
