"""Small, model-independent helpers for LIBERO imagination-reward validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


ACTION_MODES = {"policy", "noise", "zero"}
LIBERO_CAMERA_NAMES = ("agent", "wrist")


def build_matched_direction_action_branches(
    action: np.ndarray,
    direction: np.ndarray,
    *,
    translation_magnitude_cap: float = 0.25,
) -> dict[str, np.ndarray]:
    """Build policy/toward/away/zero branches for a controlled direction test.

    ``toward`` and ``away`` have identical per-step translation magnitudes,
    rotations, and gripper commands.  Their translation vectors are exact
    opposites along the supplied world-space direction.  Capping (rather than
    rescaling upward) keeps the counterfactual local when the reference policy
    emits a saturated translation command.
    """
    base = np.asarray(action, dtype=np.float32)
    unit_direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    if base.ndim != 2 or base.shape[1] != 7:
        raise ValueError(f"Expected LIBERO action chunk [T, 7], got {base.shape}")
    if unit_direction.shape != (3,):
        raise ValueError(f"Expected a three-dimensional direction, got {unit_direction.shape}")
    direction_norm = float(np.linalg.norm(unit_direction))
    if direction_norm <= 0.0:
        raise ValueError("Direction must be non-zero")
    if not 0.0 < float(translation_magnitude_cap) <= 1.0:
        raise ValueError("translation_magnitude_cap must be in (0, 1]")

    unit_direction = unit_direction / direction_norm
    reference_magnitudes = np.linalg.norm(base[:, :3], axis=1)
    matched_magnitudes = np.minimum(reference_magnitudes, translation_magnitude_cap)

    toward = base.copy()
    away = base.copy()
    toward[:, :3] = matched_magnitudes[:, None] * unit_direction[None, :]
    away[:, :3] = -toward[:, :3]

    zero = np.zeros_like(base)
    zero[:, -1] = -1.0
    return {
        "policy": base.copy(),
        "toward_bowl": toward,
        "away_from_bowl": away,
        "zero": zero,
    }


def build_shared_direction_action_branches(
    action: np.ndarray,
    *,
    noise_stds: tuple[float, ...] = (0.075, 0.15, 0.30),
    rng: Optional[np.random.Generator] = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build exact-state branches using one shared Gaussian noise direction.

    Only non-gripper dimensions are perturbed.  Scaling the same unit-variance
    sample at every noise level makes the ordering a genuine severity test
    instead of comparing unrelated random actions.
    """
    base = np.asarray(action, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] < 2:
        raise ValueError(f"Expected action chunk [T, D>=2], got {base.shape}")
    if not noise_stds or any(float(std) <= 0.0 for std in noise_stds):
        raise ValueError(f"noise_stds must be positive, got {noise_stds}")
    if tuple(sorted(float(std) for std in noise_stds)) != tuple(
        float(std) for std in noise_stds
    ):
        raise ValueError(f"noise_stds must be increasing, got {noise_stds}")
    if rng is None:
        rng = np.random.default_rng()

    epsilon = rng.normal(size=base[:, :-1].shape).astype(np.float32)
    branches = {"policy": base.copy()}
    for std in noise_stds:
        branch = base.copy()
        branch[:, :-1] = np.clip(
            branch[:, :-1] + float(std) * epsilon,
            -1.0,
            1.0,
        )
        branches[f"noise_{float(std):.3f}"] = branch
    zero = np.zeros_like(base)
    zero[:, -1] = -1.0
    branches["zero"] = zero
    return branches, epsilon


def apply_action_mode(
    action: np.ndarray,
    mode: str,
    noise_std: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return the action chunk to execute without modifying the input array."""
    mode = str(mode).strip().lower()
    if mode not in ACTION_MODES:
        raise ValueError(f"Unsupported action_mode={mode!r}; expected one of {sorted(ACTION_MODES)}")

    result = np.asarray(action, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] < 2:
        raise ValueError(f"Expected action chunk [T, D>=2], got {result.shape}")

    if mode == "policy":
        return result
    if mode == "zero":
        result.fill(0.0)
        result[:, -1] = -1.0  # LIBERO's standard no-op gripper value.
        return result

    if noise_std < 0:
        raise ValueError(f"action_noise_std must be non-negative, got {noise_std}")
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=float(noise_std), size=result[:, :-1].shape)
    result[:, :-1] = np.clip(result[:, :-1] + noise.astype(np.float32), -1.0, 1.0)
    return result


def frame_to_rgb_array(frame: Any) -> np.ndarray:
    """Convert a predicted PIL frame or a LIBERO multi-camera dict to RGB."""
    if isinstance(frame, dict):
        images = [frame_to_rgb_array(value) for value in frame.values()]
        if not images:
            raise ValueError("Cannot convert an empty camera dictionary.")
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an RGB image [H,W,3], got {array.shape}")
    return array.astype(np.uint8, copy=False)


def split_horizontal_camera_views(frame: Any) -> dict[str, np.ndarray]:
    """Split FastWAM's LIBERO ``agent | wrist`` image without resizing it.

    The released two-camera LIBERO configuration concatenates two square views
    horizontally.  Failing loudly on a different layout avoids silently assigning
    image content to the wrong camera in reward diagnostics.
    """
    array = frame_to_rgb_array(frame)
    height, width = array.shape[:2]
    if width != 2 * height:
        raise ValueError(
            "Expected two square LIBERO views concatenated horizontally "
            f"with width == 2 * height, got {array.shape}."
        )
    midpoint = width // 2
    return {
        LIBERO_CAMERA_NAMES[0]: array[:, :midpoint],
        LIBERO_CAMERA_NAMES[1]: array[:, midpoint:],
    }


def _resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    target_h, target_w = reference.shape[:2]
    if image.shape[:2] == (target_h, target_w):
        return image
    return np.asarray(
        Image.fromarray(image).resize((target_w, target_h), resample=Image.BILINEAR),
        dtype=np.uint8,
    )


def save_aligned_transition(
    output_dir: Path,
    *,
    current_frame: Any,
    predicted_goal_frame: Any,
    actual_frame: Any,
    metadata: dict[str, Any],
    rollout_arrays: Optional[dict[str, np.ndarray]] = None,
) -> Path:
    """Save one lossless current/goal/actual triplet and optional raw RL arrays."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    goal = frame_to_rgb_array(predicted_goal_frame)
    current = _resize_like(frame_to_rgb_array(current_frame), goal)
    actual = _resize_like(frame_to_rgb_array(actual_frame), goal)

    Image.fromarray(current).save(output_dir / "current.png")
    Image.fromarray(goal).save(output_dir / "predicted_goal.png")
    Image.fromarray(actual).save(output_dir / "actual.png")

    serializable_metadata = dict(metadata)
    if rollout_arrays is not None:
        arrays: dict[str, np.ndarray] = {}
        for name, value in rollout_arrays.items():
            array = np.asarray(value)
            if array.size == 0:
                raise ValueError(f"rollout array {name!r} must not be empty")
            if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                raise ValueError(f"rollout array {name!r} contains non-finite values")
            arrays[name] = array
        array_path = output_dir / "rollout_arrays.npz"
        with array_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        serializable_metadata["rollout_arrays_file"] = array_path.name

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(serializable_metadata, stream, ensure_ascii=False, indent=2)
    return metadata_path


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        raise ValueError("Cosine distance requires non-zero feature vectors.")
    similarity = float(np.dot(left, right) / denominator)
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def compute_progress_reward(
    current_feature: np.ndarray,
    actual_feature: np.ndarray,
    goal_feature: np.ndarray,
    clip_value: Optional[float] = None,
) -> dict[str, float]:
    distance_before = cosine_distance(current_feature, goal_feature)
    distance_after = cosine_distance(actual_feature, goal_feature)
    progress = distance_before - distance_after
    if clip_value is not None:
        if clip_value <= 0:
            raise ValueError(f"clip_value must be positive, got {clip_value}")
        progress = float(np.clip(progress, -clip_value, clip_value))
    return {
        "distance_before": distance_before,
        "distance_after": distance_after,
        "imagination_progress": float(progress),
    }


def compute_delta_alignment_reward(
    current_feature: np.ndarray,
    actual_feature: np.ndarray,
    goal_feature: np.ndarray,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compare the observed and imagined feature *changes* from one state.

    Let ``actual_delta = actual - current`` and
    ``imagined_delta = goal - current``.  Their cosine measures whether the two
    changes point in the same direction.  The returned reward also multiplies that
    cosine by ``min(||actual_delta|| / ||imagined_delta||, 1)``.  Consequently a
    visually static transition receives approximately zero reward even if numerical
    noise happens to point in the same direction; no dataset-tuned no-op threshold is
    required.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    current = np.asarray(current_feature, dtype=np.float32).reshape(-1)
    actual = np.asarray(actual_feature, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal_feature, dtype=np.float32).reshape(-1)
    if not (current.shape == actual.shape == goal.shape):
        raise ValueError(
            "current, actual, and goal features must have the same flattened shape, "
            f"got {current.shape}, {actual.shape}, and {goal.shape}."
        )

    actual_delta = actual - current
    imagined_delta = goal - current
    actual_norm = float(np.linalg.norm(actual_delta))
    imagined_norm = float(np.linalg.norm(imagined_delta))
    denominator = actual_norm * imagined_norm
    if denominator <= eps:
        alignment = 0.0
    else:
        alignment = float(
            np.clip(np.dot(actual_delta, imagined_delta) / denominator, -1.0, 1.0)
        )
    magnitude_ratio = float(min(actual_norm / max(imagined_norm, eps), 1.0))
    return {
        "actual_change_norm": actual_norm,
        "imagined_change_norm": imagined_norm,
        "direction_alignment": alignment,
        "magnitude_ratio": magnitude_ratio,
        "delta_alignment_reward": float(alignment * magnitude_ratio),
    }
