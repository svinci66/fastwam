"""Build a residual-AWR replay from audited head-camera Wan-VAE pair rewards.

The preferred actor observation is the frozen FastWAM Video Expert feature
captured by the same inference call that produced the baseline action.  Legacy
SigLIP observations remain available only for reproducing prior ablations.
Only the immutable reward label comes from head-camera Wan-VAE trajectory
agreement.  Successful expert and natural FastWAM-failure trajectories remain
separate episodes so Monte-Carlo returns never cross behavior boundaries.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
    encode_record_images,
    resolve_encoder_dtype,
)
from experiments.robotwin.build_residual_rl_replay import (
    combine_camera_features,
    pad_action_chunk,
    pad_environment_rewards,
    validate_episode_records,
)
from experiments.robotwin.imagination_reward_utils import ROBOTWIN_CAMERA_NAMES
from fastwam.rl.replay_buffer import ReplayBuffer, ReplayTransition
from fastwam.rl.rewards import (
    CompositeRewardConfig,
    EpisodeShapingBudget,
    WAN_VAE_HEAD_TRAJECTORY_REWARD_TYPE,
    compute_composite_reward,
)
from fastwam.models.wan22.fastwam import FASTWAM_VIDEO_EXPERT_FEATURE_VERSION

EXPECTED_SCHEMA = "robotwin_natural_failure_wan_vae_pair_reward_v1"
EXPECTED_FEATURE_ENCODER = "wan2.2_vae_single_frame_spatial_latent"
EXPECTED_REFERENCE_POLICY = "frozen_once_per_action_chunk"
EXPECTED_TIME_OFFSETS = [0, 4, 8, 12, 16, 20, 24]
SOURCE_SCHEMA = "robotwin_imagination_trajectory_v2"
FEATURE_FUSION = "per_camera_l2_then_head_left_right_concat_l2_v1"
VIDEO_EXPERT_FUSION = "final_layer_token_mean_l2_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--observation-encoder-version", required=True)
    parser.add_argument(
        "--actor-observation-source",
        choices=("fastwam_video_expert", "siglip"),
        default="fastwam_video_expert",
        help=(
            "Use native features saved by FastWAM inference (default), or the "
            "legacy independently encoded three-camera SigLIP observation."
        ),
    )
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--encoder-dtype", default="auto", choices=("auto", "fp32", "bf16", "fp16")
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--minimum-pairwise-accuracy",
        type=float,
        default=1.0,
        help=(
            "Fail below this success-vs-failure reward-ranking accuracy. Keep the "
            "default for the audited single-task pipeline; a pre-registered "
            "multi-task audit may pass a lower threshold explicitly."
        ),
    )
    parser.add_argument(
        "--tasks",
        default="",
        help=(
            "Optional comma-separated task allowlist. Filtering happens before "
            "the pairwise-accuracy gate and replay normalization so excluded tasks "
            "cannot affect training labels or task balancing."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or values.size == 0:
        raise ValueError("values and weights must be non-empty matching vectors")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("values must be finite and weights finite and positive")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    positions = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / np.sum(
        ordered_weights
    )
    return float(
        np.interp(quantile, positions, ordered_values, left=ordered_values[0], right=ordered_values[-1])
    )


def fit_episode_balanced_normalization(records: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record["alignment_valid"]:
            grouped[str(record["episode_key"])].append(float(record["wan_head_score"]))
    if not grouped:
        raise ValueError("no aligned Wan-VAE scores are available")
    values: list[float] = []
    weights: list[float] = []
    for scores in grouped.values():
        values.extend(scores)
        weights.extend([1.0 / len(scores)] * len(scores))
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    q25 = weighted_quantile(values_array, weights_array, 0.25)
    center = weighted_quantile(values_array, weights_array, 0.5)
    q75 = weighted_quantile(values_array, weights_array, 0.75)
    scale = q75 - q25
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f"Wan-VAE score IQR is degenerate: q25={q25}, q75={q75}")
    return {
        "method": "episode_balanced_weighted_median_iqr_tanh_v1",
        "center": center,
        "scale": scale,
        "q25": q25,
        "q75": q75,
        "num_valid_scores": int(values_array.size),
        "num_episodes": len(grouped),
    }


def normalized_score(raw_score: float, normalization: dict[str, float], clip: float) -> float:
    return float(clip * np.tanh((float(raw_score) - normalization["center"]) / normalization["scale"]))


def validate_reward_payload(
    payload: dict[str, Any], *, minimum_pairwise_accuracy: float = 1.0
) -> None:
    if not 0.0 <= minimum_pairwise_accuracy <= 1.0:
        raise ValueError("minimum_pairwise_accuracy must be in [0, 1]")
    expected = {
        "schema_version": EXPECTED_SCHEMA,
        "feature_encoder": EXPECTED_FEATURE_ENCODER,
        "trajectory_reference_policy": EXPECTED_REFERENCE_POLICY,
        "time_offsets": EXPECTED_TIME_OFFSETS,
        "reward_cameras": ["head"],
    }
    mismatches = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"reward payload is not the audited head-only protocol: {mismatches}")
    actual_accuracy = float(payload.get("pairwise_accuracy", -1.0))
    if actual_accuracy < minimum_pairwise_accuracy:
        raise ValueError(
            "reward payload failed the pair-ranking threshold: "
            f"actual={actual_accuracy}, minimum={minimum_pairwise_accuracy}"
        )


def select_reward_tasks(
    payload: dict[str, Any], tasks: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    """Return a reward payload restricted to an explicit task set.

    Aggregate ranking fields are recomputed from the selected pairs.  This keeps
    the replay builder's quality gate honest when a high-success retention task
    is intentionally excluded from the residual learner.
    """

    requested = [str(task).strip() for task in tasks if str(task).strip()]
    if not requested:
        return payload
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate task in --tasks: {requested}")
    available = {str(pair["task"]) for pair in payload.get("pairs", [])}
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"requested tasks are absent from reward payload: {missing}")
    selected_pairs = [
        pair for pair in payload["pairs"] if str(pair["task"]) in set(requested)
    ]
    if not selected_pairs:
        raise ValueError("task selection produced no reward pairs")
    correctly_ranked = sum(bool(pair["correctly_ranked"]) for pair in selected_pairs)
    margins = np.asarray(
        [float(pair["success_minus_failure"]) for pair in selected_pairs],
        dtype=np.float64,
    )
    selected = copy.deepcopy(payload)
    selected["pairs"] = selected_pairs
    selected["pair_count"] = len(selected_pairs)
    selected["correctly_ranked_count"] = correctly_ranked
    selected["pairwise_accuracy"] = correctly_ranked / len(selected_pairs)
    selected["mean_success_minus_failure"] = float(np.mean(margins))
    if isinstance(payload.get("per_task"), dict):
        selected["per_task"] = {
            task: copy.deepcopy(payload["per_task"][task])
            for task in requested
            if task in payload["per_task"]
        }
    selected["selected_tasks"] = requested
    return selected


def load_episode_records(
    *, root: Path, score_rows: list[dict[str, Any]], pair: dict[str, Any], behavior: str
) -> list[dict[str, Any]]:
    task = str(pair["task"])
    score_lookup = {
        int(row["replan_idx"]): float(row["camera_scores"]["head"])
        for row in score_rows
    }
    if len(score_lookup) != len(score_rows):
        raise ValueError(f"duplicate reward replan index in {root}")
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(root.glob("replan_*/metadata.json")):
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        if record.get("schema_version") != SOURCE_SCHEMA:
            raise ValueError(f"unexpected source schema in {metadata_path}")
        if str(record.get("task_name")) != task:
            raise ValueError(
                f"task mismatch in {metadata_path}: "
                f"metadata={record.get('task_name')!r}, pair={task!r}"
            )
        replan_idx = int(record["replan_idx"])
        alignment_valid = bool(record["alignment_valid"])
        trajectory_valid = bool(record.get("trajectory_alignment_valid", False))
        has_score = replan_idx in score_lookup
        if has_score != (alignment_valid and trajectory_valid):
            raise ValueError(
                f"score/alignment mismatch in {metadata_path}: score={has_score}, "
                f"alignment={alignment_valid}, trajectory={trajectory_valid}"
            )
        if record.get("trajectory_reference_policy") != EXPECTED_REFERENCE_POLICY:
            raise ValueError(f"trajectory was not frozen within action chunk: {metadata_path}")
        if list(record.get("trajectory_expected_action_offsets", [])) != EXPECTED_TIME_OFFSETS:
            raise ValueError(f"unexpected trajectory offsets: {metadata_path}")
        if str(record.get("action_mode")) != behavior:
            raise ValueError(f"behavior mismatch in {metadata_path}")
        enriched = dict(record)
        enriched.update(
            {
                "record_dir": str(metadata_path.parent),
                "behavior": behavior,
                "task_name": task,
                "environment_seed": int(pair["environment_seed"]),
                "pair_episode_id": int(pair["episode_id"]),
                "episode_key": (
                    f"{task}-pair{int(pair['episode_id']):04d}-{behavior}"
                ),
                "wan_head_score": score_lookup.get(replan_idx, 0.0),
            }
        )
        for phase in ("current", "actual", "predicted_goal"):
            image_path = metadata_path.parent / f"{phase}.png"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            enriched[f"{phase}_path"] = str(image_path)
        records.append(enriched)
    if not records:
        raise ValueError(f"no trajectory records in {root}")
    if set(score_lookup) != {
        int(record["replan_idx"]) for record in records if record["alignment_valid"]
    }:
        raise ValueError(f"reward rows do not exactly cover aligned records in {root}")
    return records


def discover_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair in payload["pairs"]:
        records.extend(
            load_episode_records(
                root=Path(pair["expert_success"]["root"]),
                score_rows=pair["expert_success"]["per_replan"],
                pair=pair,
                behavior="expert",
            )
        )
        records.extend(
            load_episode_records(
                root=Path(pair["fastwam_failure"]["root"]),
                score_rows=pair["fastwam_failure"]["per_replan"],
                pair=pair,
                behavior="policy",
            )
        )
    validate_episode_records(records)
    return records


def load_video_expert_record_features(
    records: list[dict[str, Any]],
    *,
    expected_version: str = FASTWAM_VIDEO_EXPERT_FEATURE_VERSION,
) -> list[dict[str, dict[str, np.ndarray]]]:
    """Load native FastWAM observations captured with each baseline action.

    AWR is intentionally configured with ``use_goal_conditioning=false``.  The
    replay schema still requires next/goal feature slots, so both are explicit
    copies of the current native feature and provenance records that they are
    placeholders.  They must not be used by a goal-conditioned learner.
    """

    encoded: list[dict[str, dict[str, np.ndarray]]] = []
    feature_dim: int | None = None
    for record in records:
        version = str(record.get("video_expert_feature_version", "")).strip()
        if version != expected_version:
            raise ValueError(
                "Missing or incompatible FastWAM Video Expert feature in "
                f"{record['record_dir']}: actual={version!r}, expected={expected_version!r}. "
                "Re-export/backfill the transition with the current FastWAM checkpoint."
            )
        checkpoint_sha256 = str(
            record.get("video_expert_checkpoint_sha256", "")
        ).strip()
        if len(checkpoint_sha256) != 64:
            raise ValueError(
                f"Missing FastWAM checkpoint SHA-256 in {record['record_dir']}"
            )
        arrays_path = Path(record["record_dir"]) / str(record["rollout_arrays_file"])
        with np.load(arrays_path, allow_pickle=False) as payload:
            if "video_expert_feature" not in payload.files:
                raise ValueError(
                    f"{arrays_path} lacks video_expert_feature; re-export/backfill it"
                )
            feature = np.asarray(
                payload["video_expert_feature"], dtype=np.float32
            ).reshape(-1)
        if feature.size == 0 or not np.all(np.isfinite(feature)):
            raise ValueError(f"Invalid Video Expert feature in {arrays_path}")
        declared_dim = int(record.get("video_expert_feature_dim", -1))
        if declared_dim != feature.size:
            raise ValueError(
                f"Video Expert feature dimension mismatch in {arrays_path}: "
                f"metadata={declared_dim}, array={feature.size}"
            )
        if feature_dim is None:
            feature_dim = int(feature.size)
        elif feature.size != feature_dim:
            raise ValueError(
                f"Mixed Video Expert feature dimensions: {feature_dim} and {feature.size}"
            )
        encoded.append(
            {
                "current": {"native": feature},
                "actual": {"native": feature.copy()},
                "predicted_goal": {"native": feature.copy()},
            }
        )
    return encoded


def collapse_record_feature(
    features: dict[str, np.ndarray], *, observation_source: str
) -> np.ndarray:
    if observation_source == "fastwam_video_expert":
        if set(features) != {"native"}:
            raise ValueError(f"Expected one native Video Expert feature, got {sorted(features)}")
        return np.asarray(features["native"], dtype=np.float32).reshape(-1)
    if observation_source == "siglip":
        return combine_camera_features(features)
    raise ValueError(f"Unsupported actor observation source: {observation_source!r}")


def build_replay(
    records: list[dict[str, Any]],
    encoded: list[dict[str, dict[str, np.ndarray]]],
    *,
    reward_config: CompositeRewardConfig,
    observation_source: str,
    normalization: dict[str, float],
    imitation_dimension_scales: np.ndarray | None,
) -> ReplayBuffer:
    if len(records) != len(encoded):
        raise ValueError("record and encoded-feature counts differ")
    replay = ReplayBuffer()
    task_ids = {
        task: index
        for index, task in enumerate(
            sorted({str(record["task_name"]) for record in records})
        )
    }
    budgets: dict[str, EpisodeShapingBudget] = {}
    ordered = sorted(
        zip(records, encoded),
        key=lambda item: (item[0]["episode_key"], int(item[0]["replan_idx"])),
    )
    for record, features in ordered:
        task_name = str(record["task_name"])
        episode_id = f"robotwin2.0-{task_name}-{record['episode_key']}"
        target_k = int(record["target_step"])
        effective_k = int(record["effective_k"])
        arrays_path = Path(record["record_dir"]) / str(record["rollout_arrays_file"])
        with np.load(arrays_path, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        baseline_actions = pad_action_chunk(arrays["baseline_actions"], target_k, name="baseline_actions")
        executed_actions = pad_action_chunk(arrays["executed_actions"], target_k, name="executed_actions")
        environment_rewards = pad_environment_rewards(arrays["environment_rewards"], target_k)
        alignment_valid = bool(record["alignment_valid"])
        shaping = (
            normalized_score(record["wan_head_score"], normalization, reward_config.imagination_clip)
            if alignment_valid
            else 0.0
        )
        reward = compute_composite_reward(
            environment_rewards=environment_rewards,
            success=bool(record["transition_success"]),
            baseline_actions=baseline_actions,
            executed_actions=executed_actions,
            effective_k=effective_k,
            imagination_progress=shaping,
            alignment_valid=alignment_valid,
            config=reward_config,
            shaping_budget=budgets.setdefault(
                episode_id, EpisodeShapingBudget.from_config(reward_config)
            ),
            imitation_dimension_scales=imitation_dimension_scales,
        )
        replay.append(
            ReplayTransition(
                episode_id=episode_id,
                transition_index=int(record["replan_idx"]),
                task_suite=str(record["task_suite"]),
                task_id=task_ids[task_name],
                task_description=str(record["task_description"]),
                env_seed=int(record["environment_seed"]),
                goal_seed=0,
                action_seed=int(record.get("action_corruption_seed", 0)),
                policy_version=str(record["policy_version"]),
                predictor_version=str(record["predictor_version"]),
                reward_encoder_version=EXPECTED_FEATURE_ENCODER,
                behavior_mode=str(record["action_mode"]),
                action_noise_std=float(record.get("action_noise_std", 0.0)),
                target_k=target_k,
                effective_k=effective_k,
                goal_frame_index=int(record["goal_frame_index"]),
                goal_tau=float(record["goal_tau"]),
                terminated=bool(record["terminated"]),
                truncated=bool(record["truncated"]),
                success=bool(record["transition_success"]),
                alignment_valid=alignment_valid,
                observation_feature=collapse_record_feature(
                    features["current"], observation_source=observation_source
                ),
                next_observation_feature=collapse_record_feature(
                    features["actual"], observation_source=observation_source
                ),
                goal_feature=collapse_record_feature(
                    features["predicted_goal"], observation_source=observation_source
                ),
                proprio=np.asarray(arrays["proprio"], dtype=np.float32),
                next_proprio=np.asarray(arrays["next_proprio"], dtype=np.float32),
                baseline_actions=baseline_actions,
                executed_actions=executed_actions,
                environment_rewards=environment_rewards,
                reward=reward,
                imagination_reward_type=WAN_VAE_HEAD_TRAJECTORY_REWARD_TYPE,
                language_feature=np.asarray(arrays["language_feature"], dtype=np.float32),
                language_encoder_version=str(record["language_encoder_version"]),
            )
        )
    return replay


def main() -> None:
    args = parse_args()
    reward_path = args.reward_json.expanduser().resolve()
    source_payload = json.loads(reward_path.read_text(encoding="utf-8"))
    selected_tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    payload = select_reward_tasks(source_payload, selected_tasks)
    validate_reward_payload(
        payload, minimum_pairwise_accuracy=args.minimum_pairwise_accuracy
    )
    config_payload = OmegaConf.to_container(OmegaConf.load(args.reward_config), resolve=True)
    if not isinstance(config_payload, dict):
        raise ValueError("top-level reward config must be a mapping")
    reward_config = CompositeRewardConfig(**config_payload["reward"])
    reward_config.validate()
    if reward_config.imagination_reward_type != WAN_VAE_HEAD_TRAJECTORY_REWARD_TYPE:
        raise ValueError("reward config must select the head-camera Wan-VAE reward type")
    if args.actor_observation_source == "fastwam_video_expert" and bool(
        config_payload.get("awr", {}).get("use_goal_conditioning", False)
    ):
        raise ValueError(
            "FastWAM-native replay currently captures the current Video Expert "
            "feature only and requires awr.use_goal_conditioning=false"
        )
    imitation_scales = config_payload.get("imitation_dimension_scales")
    imitation_scales_array = None if imitation_scales is None else np.asarray(imitation_scales, dtype=np.float32)
    records = discover_records(payload)
    normalization = fit_episode_balanced_normalization(records)
    if args.actor_observation_source == "fastwam_video_expert":
        if args.encoder_path is not None:
            raise ValueError(
                "--encoder-path must be omitted for FastWAM-native observations"
            )
        if args.observation_encoder_version != FASTWAM_VIDEO_EXPERT_FEATURE_VERSION:
            raise ValueError(
                "FastWAM-native observations require --observation-encoder-version="
                f"{FASTWAM_VIDEO_EXPERT_FEATURE_VERSION}"
            )
        encoder_dtype = None
        encoded = load_video_expert_record_features(records)
    else:
        if args.encoder_path is None:
            raise ValueError("legacy SigLIP observations require --encoder-path")
        encoder_dtype = resolve_encoder_dtype(args.encoder_dtype, device=args.device)
        encoded = encode_record_images(
            records,
            encoder_path=args.encoder_path,
            device=args.device,
            batch_size=args.batch_size,
            encoder_dtype=encoder_dtype,
        )
    replay = build_replay(
        records,
        encoded,
        reward_config=reward_config,
        observation_source=args.actor_observation_source,
        normalization=normalization,
        imitation_dimension_scales=imitation_scales_array,
    )
    behavior_counts = Counter(str(record["behavior"]) for record in records)
    task_transition_counts = Counter(str(record["task_name"]) for record in records)
    task_pair_counts = Counter(str(pair["task"]) for pair in payload["pairs"])
    valid_counts = Counter(
        f"{record['behavior']}/{'valid' if record['alignment_valid'] else 'partial_terminal'}"
        for record in records
    )
    language_versions = {str(record["language_encoder_version"]) for record in records}
    if len(language_versions) != 1:
        raise ValueError(f"mixed language encoder versions: {sorted(language_versions)}")
    video_expert_checkpoint_hashes = {
        str(record.get("video_expert_checkpoint_sha256", "")).strip()
        for record in records
        if args.actor_observation_source == "fastwam_video_expert"
    }
    if args.actor_observation_source == "fastwam_video_expert" and len(
        video_expert_checkpoint_hashes
    ) != 1:
        raise ValueError(
            "FastWAM-native replay must use exactly one base checkpoint, got "
            f"{sorted(video_expert_checkpoint_hashes)}"
        )
    output = replay.save(
        args.output_dir,
        provenance={
            "reward_encoder_version": EXPECTED_FEATURE_ENCODER,
            "observation_encoder_version": args.observation_encoder_version,
            "observation_encoder_path": (
                None
                if args.encoder_path is None
                else str(args.encoder_path.expanduser().resolve())
            ),
            "observation_feature_dim": int(
                replay.transitions[0].observation_feature.size
            ),
            "actor_observation_source": args.actor_observation_source,
            "fastwam_checkpoint_sha256": (
                next(iter(video_expert_checkpoint_hashes))
                if video_expert_checkpoint_hashes
                else None
            ),
            "wan_reward_encoder": EXPECTED_FEATURE_ENCODER,
            "wan_reward_source_json": str(reward_path),
            "wan_reward_source_sha256": sha256(reward_path),
            "reward_cameras": ["head"],
            "reward_time_offsets": EXPECTED_TIME_OFFSETS,
            "trajectory_reference_policy": EXPECTED_REFERENCE_POLICY,
            "reward_normalization": normalization,
            "camera_names": (
                list(ROBOTWIN_CAMERA_NAMES)
                if args.actor_observation_source == "siglip"
                else ["robotwin_composite_head_left_right"]
            ),
            "camera_image_size": (
                224 if args.actor_observation_source == "siglip" else [384, 320]
            ),
            "encoder_dtype": (
                None
                if encoder_dtype is None
                else str(encoder_dtype).removeprefix("torch.")
            ),
            "feature_fusion": (
                FEATURE_FUSION
                if args.actor_observation_source == "siglip"
                else VIDEO_EXPERT_FUSION
            ),
            "unused_feature_fields": (
                None
                if args.actor_observation_source == "siglip"
                else "next_and_goal_copy_current_use_goal_conditioning_false"
            ),
            "language_encoder_version": next(iter(language_versions)),
            "language_pooling": "fastwam_umt5_masked_mean_v1",
            "source_schema": SOURCE_SCHEMA,
            "behavior_transition_counts": dict(sorted(behavior_counts.items())),
            "alignment_transition_counts": dict(sorted(valid_counts.items())),
            "environment_seeds": sorted({int(record["environment_seed"]) for record in records}),
            "pair_episode_ids": sorted({int(record["pair_episode_id"]) for record in records}),
            "task_transition_counts": dict(sorted(task_transition_counts.items())),
            "task_pair_counts": dict(sorted(task_pair_counts.items())),
            "task_id_map": {
                task: index for index, task in enumerate(sorted(task_transition_counts))
            },
            "selected_tasks": sorted(task_transition_counts),
        },
    )
    raw_shaping = np.asarray([transition.reward.imagination_raw for transition in replay.transitions])
    summary = {
        "output_dir": str(output),
        "num_transitions": len(replay),
        "num_episodes": len({transition.episode_id for transition in replay.transitions}),
        "behavior_transition_counts": dict(sorted(behavior_counts.items())),
        "alignment_transition_counts": dict(sorted(valid_counts.items())),
        "task_transition_counts": dict(sorted(task_transition_counts.items())),
        "task_pair_counts": dict(sorted(task_pair_counts.items())),
        "selected_tasks": sorted(task_transition_counts),
        "selected_pairwise_accuracy": float(payload["pairwise_accuracy"]),
        "normalization": normalization,
        "normalized_shaping_min": float(raw_shaping.min()),
        "normalized_shaping_mean": float(raw_shaping.mean()),
        "normalized_shaping_max": float(raw_shaping.max()),
    }
    (output / "build_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
