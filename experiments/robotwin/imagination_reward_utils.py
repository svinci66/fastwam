"""Helpers for temporally aligned RoboTwin imagination-reward collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROBOTWIN_CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
ROBOTWIN_GRIPPER_INDICES = (6, 13)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {array.shape}")
    return array.astype(np.uint8, copy=False)


def split_robotwin_camera_views(frame: Any) -> dict[str, np.ndarray]:
    """Split FastWAM's 384x320 ``head / (left | right)`` composite."""

    array = frame_to_rgb_array(frame)
    height, width = array.shape[:2]
    if (height, width) != (384, 320):
        raise ValueError(
            "Expected RoboTwin composite with shape [384,320,3], "
            f"got {array.shape}."
        )
    return {
        "head": array[:256, :],
        "left_wrist": array[256:, :160],
        "right_wrist": array[256:, 160:],
    }


def apply_normalized_action_noise(
    action: np.ndarray,
    *,
    noise_std: float,
    rng: np.random.Generator,
    gripper_indices: tuple[int, ...] = ROBOTWIN_GRIPPER_INDICES,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb non-gripper dimensions in normalized action space.

    Returning the unit Gaussian direction makes mild and strong collectors
    auditable: when they share a seed, they use the same perturbation direction.
    """

    baseline = np.asarray(action, dtype=np.float32)
    if baseline.ndim != 2:
        raise ValueError(f"Expected normalized actions [T,D], got {baseline.shape}")
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError(f"noise_std must be finite and non-negative, got {noise_std}")
    if any(index < 0 or index >= baseline.shape[1] for index in gripper_indices):
        raise ValueError(
            f"gripper indices {gripper_indices} are invalid for action dim {baseline.shape[1]}"
        )

    epsilon = rng.normal(size=baseline.shape).astype(np.float32)
    epsilon[:, list(gripper_indices)] = 0.0
    perturbed = np.clip(baseline + float(noise_std) * epsilon, -5.0, 5.0)
    return perturbed.astype(np.float32, copy=False), epsilon


def save_aligned_transition(
    output_dir: Path,
    *,
    current_frame: Any,
    predicted_goal_frame: Any,
    actual_frame: Any,
    metadata: dict[str, Any],
    rollout_arrays: dict[str, np.ndarray],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        "current": frame_to_rgb_array(current_frame),
        "predicted_goal": frame_to_rgb_array(predicted_goal_frame),
        "actual": frame_to_rgb_array(actual_frame),
    }
    expected_shape = frames["predicted_goal"].shape
    for name, frame in tuple(frames.items()):
        if frame.shape != expected_shape:
            target_h, target_w = expected_shape[:2]
            frames[name] = np.asarray(
                Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        Image.fromarray(frames[name]).save(output_dir / f"{name}.png")

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

    serializable = dict(metadata)
    serializable["rollout_arrays_file"] = array_path.name
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata_path


def update_episode_success(metadata_paths: list[Path], success: bool) -> None:
    """Backfill an episode-level label after RoboTwin reports terminal success."""

    for metadata_path in metadata_paths:
        payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        payload["episode_success"] = bool(success)
        Path(metadata_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
