"""Frozen-feature reward composition for chunk-level LIBERO RL.

The primary imagination term is relative progress toward a predicted future:

    d(current, goal) - d(actual_after_K, goal)

All feature inputs must come from the same frozen encoder version.  The code does
not own that encoder and never propagates gradients through reward features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE = "delta_alignment_global_camera_norm_v1"
WAN_VAE_HEAD_TRAJECTORY_REWARD_TYPE = "wan_vae_head_trajectory_global_norm_v1"
WAN_VAE_HEAD_PAIRED_RANK_REWARD_TYPE = (
    "wan_vae_head_trajectory_paired_rank_discount_norm_v1"
)
IMAGINATION_REWARD_TYPES = (
    "progress_v1",
    "delta_alignment_v1",
    GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE,
    WAN_VAE_HEAD_TRAJECTORY_REWARD_TYPE,
    WAN_VAE_HEAD_PAIRED_RANK_REWARD_TYPE,
)


def _as_finite_vector(value: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def cosine_distance(
    left: np.ndarray | Sequence[float],
    right: np.ndarray | Sequence[float],
    *,
    eps: float = 1e-8,
) -> float:
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    left_vector = _as_finite_vector(left, "left")
    right_vector = _as_finite_vector(right, "right")
    if left_vector.shape != right_vector.shape:
        raise ValueError(
            f"feature shapes must match, got {left_vector.shape} and {right_vector.shape}"
        )
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator <= eps:
        raise ValueError("cosine distance is undefined for a zero-norm feature")
    similarity = float(np.dot(left_vector, right_vector) / denominator)
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def _normalized_camera_weights(
    cameras: Sequence[str],
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    camera_names = tuple(str(camera) for camera in cameras)
    if not camera_names:
        raise ValueError("at least one camera is required")
    if len(set(camera_names)) != len(camera_names):
        raise ValueError(f"camera names must be unique, got {camera_names}")

    if weights is None:
        return {camera: 1.0 / len(camera_names) for camera in camera_names}
    missing = set(camera_names) - set(weights)
    extra = set(weights) - set(camera_names)
    if missing or extra:
        raise ValueError(f"camera weight keys mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    values = {camera: float(weights[camera]) for camera in camera_names}
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError(f"camera weights must be finite and non-negative, got {values}")
    total = float(sum(values.values()))
    if total <= 0.0:
        raise ValueError("camera weights must have a positive sum")
    return {camera: value / total for camera, value in values.items()}


@dataclass(frozen=True)
class ImaginationProgress:
    reward_type: str
    distance_before: float
    distance_after: float
    raw_progress: float
    clipped_progress: float
    per_camera: dict[str, dict[str, float]]
    camera_weights: dict[str, float]
    alignment_valid: bool


def compute_imagination_progress(
    current_features: Mapping[str, np.ndarray | Sequence[float]],
    actual_features: Mapping[str, np.ndarray | Sequence[float]],
    goal_features: Mapping[str, np.ndarray | Sequence[float]],
    *,
    camera_weights: Mapping[str, float] | None = None,
    clip_value: float = 0.1,
    alignment_valid: bool = True,
) -> ImaginationProgress:
    """Compute camera-weighted progress toward a time-aligned imagined goal.

    Invalid temporal alignment is represented explicitly and returns zero shaping;
    it is never silently treated as a valid transition.
    """

    if not np.isfinite(clip_value) or clip_value <= 0.0:
        raise ValueError(f"clip_value must be finite and positive, got {clip_value}")
    cameras = tuple(sorted(current_features))
    if set(actual_features) != set(cameras) or set(goal_features) != set(cameras):
        raise ValueError(
            "current, actual, and goal feature dictionaries must have identical camera keys"
        )
    normalized_weights = _normalized_camera_weights(cameras, camera_weights)

    per_camera: dict[str, dict[str, float]] = {}
    weighted_before = 0.0
    weighted_after = 0.0
    for camera in cameras:
        before = cosine_distance(current_features[camera], goal_features[camera])
        after = cosine_distance(actual_features[camera], goal_features[camera])
        progress = before - after
        per_camera[camera] = {
            "distance_before": before,
            "distance_after": after,
            "progress": progress,
        }
        weighted_before += normalized_weights[camera] * before
        weighted_after += normalized_weights[camera] * after

    raw_progress = float(weighted_before - weighted_after)
    clipped = float(np.clip(raw_progress, -clip_value, clip_value)) if alignment_valid else 0.0
    return ImaginationProgress(
        reward_type="progress_v1",
        distance_before=float(weighted_before),
        distance_after=float(weighted_after),
        raw_progress=raw_progress,
        clipped_progress=clipped,
        per_camera=per_camera,
        camera_weights=normalized_weights,
        alignment_valid=bool(alignment_valid),
    )


def _delta_alignment_reward(
    current: np.ndarray | Sequence[float],
    actual: np.ndarray | Sequence[float],
    goal: np.ndarray | Sequence[float],
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Return direction alignment scaled down for visually static transitions."""

    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    current_vector = _as_finite_vector(current, "current")
    actual_vector = _as_finite_vector(actual, "actual")
    goal_vector = _as_finite_vector(goal, "goal")
    if not (current_vector.shape == actual_vector.shape == goal_vector.shape):
        raise ValueError(
            "current, actual, and goal feature shapes must match, got "
            f"{current_vector.shape}, {actual_vector.shape}, and {goal_vector.shape}"
        )
    actual_delta = actual_vector - current_vector
    imagined_delta = goal_vector - current_vector
    actual_norm = float(np.linalg.norm(actual_delta))
    imagined_norm = float(np.linalg.norm(imagined_delta))
    denominator = actual_norm * imagined_norm
    direction_alignment = (
        0.0
        if denominator <= eps
        else float(np.clip(np.dot(actual_delta, imagined_delta) / denominator, -1.0, 1.0))
    )
    magnitude_ratio = float(min(actual_norm / max(imagined_norm, eps), 1.0))
    return {
        "actual_change_norm": actual_norm,
        "imagined_change_norm": imagined_norm,
        "direction_alignment": direction_alignment,
        "magnitude_ratio": magnitude_ratio,
        "delta_alignment_reward": direction_alignment * magnitude_ratio,
    }


def compute_imagination_reward(
    current_features: Mapping[str, np.ndarray | Sequence[float]],
    actual_features: Mapping[str, np.ndarray | Sequence[float]],
    goal_features: Mapping[str, np.ndarray | Sequence[float]],
    *,
    reward_type: str,
    camera_weights: Mapping[str, float] | None = None,
    camera_normalization: Mapping[str, Mapping[str, float]] | None = None,
    clip_value: float = 0.1,
    alignment_valid: bool = True,
) -> ImaginationProgress:
    """Compute one explicitly versioned camera-aware imagination signal."""

    reward_type = str(reward_type).strip()
    if reward_type not in IMAGINATION_REWARD_TYPES:
        raise ValueError(
            f"unsupported imagination reward type {reward_type!r}; "
            f"expected one of {IMAGINATION_REWARD_TYPES}"
        )
    if reward_type in {
        WAN_VAE_HEAD_TRAJECTORY_REWARD_TYPE,
        WAN_VAE_HEAD_PAIRED_RANK_REWARD_TYPE,
    }:
        raise ValueError(
            f"{reward_type} is a precomputed trajectory "
            "reward label; build it with build_wan_vae_head_awr_replay.py instead "
            "of recomputing it from three endpoint feature dictionaries"
        )
    if reward_type == "progress_v1":
        if camera_normalization is not None:
            raise ValueError(
                "camera_normalization may only be used with "
                f"{GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE}"
            )
        return compute_imagination_progress(
            current_features,
            actual_features,
            goal_features,
            camera_weights=camera_weights,
            clip_value=clip_value,
            alignment_valid=alignment_valid,
        )

    if not np.isfinite(clip_value) or clip_value <= 0.0:
        raise ValueError(f"clip_value must be finite and positive, got {clip_value}")
    cameras = tuple(sorted(current_features))
    if set(actual_features) != set(cameras) or set(goal_features) != set(cameras):
        raise ValueError(
            "current, actual, and goal feature dictionaries must have identical camera keys"
        )
    normalized_weights = _normalized_camera_weights(cameras, camera_weights)
    per_camera: dict[str, dict[str, float]] = {}
    weighted_before = 0.0
    weighted_after = 0.0
    weighted_reward = 0.0
    normalized_reward = 0.0
    if reward_type == GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE:
        if camera_normalization is None:
            raise ValueError(
                f"{GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE} requires camera_normalization"
            )
        missing = set(cameras) - set(camera_normalization)
        extra = set(camera_normalization) - set(cameras)
        if missing or extra:
            raise ValueError(
                "camera normalization keys mismatch: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
    elif camera_normalization is not None:
        raise ValueError(
            "camera_normalization may only be used with "
            f"{GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE}"
        )
    for camera in cameras:
        before = cosine_distance(current_features[camera], goal_features[camera])
        after = cosine_distance(actual_features[camera], goal_features[camera])
        delta = _delta_alignment_reward(
            current_features[camera], actual_features[camera], goal_features[camera]
        )
        per_camera[camera] = {
            "distance_before": before,
            "distance_after": after,
            "progress": before - after,
            **delta,
        }
        weighted_before += normalized_weights[camera] * before
        weighted_after += normalized_weights[camera] * after
        camera_reward = delta["delta_alignment_reward"]
        weighted_reward += normalized_weights[camera] * camera_reward
        if camera_normalization is not None:
            settings = camera_normalization[camera]
            center = float(settings.get("center", np.nan))
            scale = float(settings.get("scale", np.nan))
            if not np.isfinite(center):
                raise ValueError(f"camera normalization center must be finite for {camera}")
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"camera normalization scale must be positive for {camera}")
            normalized = (camera_reward - center) / scale
            per_camera[camera].update(
                normalization_center=center,
                normalization_scale=scale,
                normalized_delta_alignment=normalized,
            )
            normalized_reward += normalized_weights[camera] * normalized

    raw_reward = (
        float(clip_value * np.tanh(normalized_reward))
        if reward_type == GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE
        else float(weighted_reward)
    )
    clipped = float(np.clip(raw_reward, -clip_value, clip_value)) if alignment_valid else 0.0
    return ImaginationProgress(
        reward_type=reward_type,
        distance_before=float(weighted_before),
        distance_after=float(weighted_after),
        raw_progress=raw_reward,
        clipped_progress=clipped,
        per_camera=per_camera,
        camera_weights=normalized_weights,
        alignment_valid=bool(alignment_valid),
    )


@dataclass(frozen=True)
class CompositeRewardConfig:
    """Frozen coefficients for the first residual-RL comparison.

    `success_bonus` is intentionally separate from `success_weight`.  The maximum
    cumulative absolute imagination contribution is constrained to
    `success_bonus * max_imagination_to_success_ratio` by EpisodeShapingBudget.
    """

    success_bonus: float = 10.0
    environment_weight: float = 0.0
    success_weight: float = 1.0
    imitation_weight: float = 0.1
    imagination_weight: float = 1.0
    imagination_reward_type: str = "progress_v1"
    step_penalty: float = 0.0
    imagination_clip: float = 0.1
    max_imagination_to_success_ratio: float = 0.5

    def validate(self) -> None:
        numeric = {
            key: value
            for key, value in asdict(self).items()
            if key != "imagination_reward_type"
        }
        if any(not np.isfinite(float(value)) for value in numeric.values()):
            raise ValueError(f"all reward settings must be finite, got {numeric}")
        if self.imagination_reward_type not in IMAGINATION_REWARD_TYPES:
            raise ValueError(
                "imagination_reward_type must be one of "
                f"{IMAGINATION_REWARD_TYPES}, got {self.imagination_reward_type!r}"
            )
        if self.success_bonus <= 0.0:
            raise ValueError("success_bonus must be positive")
        if self.environment_weight < 0.0:
            raise ValueError("environment_weight must be non-negative")
        if self.success_weight < 0.0 or self.imitation_weight < 0.0:
            raise ValueError("success_weight and imitation_weight must be non-negative")
        if self.imagination_weight < 0.0:
            raise ValueError("imagination_weight must be non-negative")
        if self.step_penalty < 0.0:
            raise ValueError("step_penalty must be non-negative")
        if self.imagination_clip <= 0.0:
            raise ValueError("imagination_clip must be positive")
        if not 0.0 <= self.max_imagination_to_success_ratio <= 0.5:
            raise ValueError(
                "max_imagination_to_success_ratio must be in [0, 0.5] so success remains dominant"
            )


class EpisodeShapingBudget:
    """Cap cumulative absolute imagination shaping within one episode."""

    def __init__(self, maximum_absolute_total: float):
        maximum_absolute_total = float(maximum_absolute_total)
        if not np.isfinite(maximum_absolute_total) or maximum_absolute_total < 0.0:
            raise ValueError("maximum_absolute_total must be finite and non-negative")
        self.maximum_absolute_total = maximum_absolute_total
        self.absolute_spent = 0.0

    @classmethod
    def from_config(cls, config: CompositeRewardConfig) -> "EpisodeShapingBudget":
        config.validate()
        return cls(
            config.success_bonus
            * config.success_weight
            * config.max_imagination_to_success_ratio
        )

    @property
    def remaining(self) -> float:
        return max(self.maximum_absolute_total - self.absolute_spent, 0.0)

    def apply(self, proposed_value: float) -> float:
        proposed_value = float(proposed_value)
        if not np.isfinite(proposed_value):
            raise ValueError(f"proposed shaping must be finite, got {proposed_value}")
        applied = float(np.clip(proposed_value, -self.remaining, self.remaining))
        self.absolute_spent += abs(applied)
        return applied


def compute_imitation_reward(
    baseline_actions: np.ndarray,
    executed_actions: np.ndarray,
    *,
    effective_k: int,
    dimension_scales: Sequence[float] | np.ndarray | None = None,
) -> float:
    baseline = np.asarray(baseline_actions, dtype=np.float32)
    executed = np.asarray(executed_actions, dtype=np.float32)
    if baseline.shape != executed.shape or baseline.ndim != 2:
        raise ValueError(
            "baseline_actions and executed_actions must have identical [H, action_dim] shapes"
        )
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(executed)):
        raise ValueError("actions must contain only finite values")
    if effective_k <= 0 or effective_k > baseline.shape[0]:
        raise ValueError(f"effective_k must be in [1, {baseline.shape[0]}], got {effective_k}")
    difference = executed[:effective_k] - baseline[:effective_k]
    if dimension_scales is not None:
        scales = np.asarray(dimension_scales, dtype=np.float32).reshape(-1)
        if scales.shape != (baseline.shape[1],):
            raise ValueError(
                f"dimension_scales must have shape {(baseline.shape[1],)}, got {scales.shape}"
            )
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("dimension_scales must be finite and positive")
        difference = difference / scales
    return -float(np.mean(np.square(difference)))


@dataclass(frozen=True)
class RewardBreakdown:
    environment_return: float
    environment_component: float
    success_component: float
    imitation_raw: float
    imitation_component: float
    imagination_raw: float
    imagination_proposed: float
    imagination_applied: float
    step_component: float
    total: float
    alignment_valid: bool


def compute_composite_reward(
    *,
    environment_rewards: Sequence[float] | np.ndarray,
    success: bool,
    baseline_actions: np.ndarray,
    executed_actions: np.ndarray,
    effective_k: int,
    imagination_progress: float,
    alignment_valid: bool,
    config: CompositeRewardConfig,
    shaping_budget: EpisodeShapingBudget,
    imitation_dimension_scales: Sequence[float] | np.ndarray | None = None,
) -> RewardBreakdown:
    config.validate()
    rewards = np.asarray(environment_rewards, dtype=np.float32).reshape(-1)
    if rewards.size < effective_k:
        raise ValueError(
            f"environment_rewards has {rewards.size} entries but effective_k={effective_k}"
        )
    if np.any(~np.isfinite(rewards)):
        raise ValueError("environment_rewards must contain only finite values")
    environment_return = float(np.sum(rewards[:effective_k]))
    environment_component = config.environment_weight * environment_return
    success_component = config.success_weight * config.success_bonus * float(bool(success))
    imitation_raw = compute_imitation_reward(
        baseline_actions,
        executed_actions,
        effective_k=effective_k,
        dimension_scales=imitation_dimension_scales,
    )
    imitation_component = config.imitation_weight * imitation_raw
    bounded_progress = (
        float(np.clip(imagination_progress, -config.imagination_clip, config.imagination_clip))
        if alignment_valid
        else 0.0
    )
    imagination_proposed = config.imagination_weight * bounded_progress
    imagination_applied = shaping_budget.apply(imagination_proposed)
    step_component = -config.step_penalty * effective_k
    total = (
        environment_component
        + success_component
        + imitation_component
        + imagination_applied
        + step_component
    )
    return RewardBreakdown(
        environment_return=environment_return,
        environment_component=environment_component,
        success_component=success_component,
        imitation_raw=imitation_raw,
        imitation_component=imitation_component,
        imagination_raw=float(imagination_progress),
        imagination_proposed=imagination_proposed,
        imagination_applied=imagination_applied,
        step_component=step_component,
        total=float(total),
        alignment_valid=bool(alignment_valid),
    )
