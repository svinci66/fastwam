#!/usr/bin/env python3
"""Validate multi-time RoboTwin imagination alignment with frozen Wan VAE features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from fastwam.models.wan22.helpers.loader import _load_registered_model


CAMERA_NAMES = ("head", "left_wrist", "right_wrist")


def _cosine(left: np.ndarray, right: np.ndarray, eps: float = 1e-8) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= eps:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def compute_trajectory_alignment(
    predicted: dict[str, list[np.ndarray]],
    actual: dict[str, list[np.ndarray]],
) -> dict[str, Any]:
    if set(predicted) != set(CAMERA_NAMES) or set(actual) != set(CAMERA_NAMES):
        raise ValueError(f"Expected camera names {CAMERA_NAMES}")
    camera_scores: dict[str, float] = {}
    camera_time_scores: dict[str, list[float | None]] = {}
    for camera in CAMERA_NAMES:
        predicted_frames = predicted[camera]
        actual_frames = actual[camera]
        if len(predicted_frames) != len(actual_frames) or len(predicted_frames) < 2:
            raise ValueError(
                f"{camera} requires equal trajectories with at least two frames"
            )
        time_scores = [
            _cosine(
                predicted_frame - predicted_frames[0],
                actual_frame - actual_frames[0],
            )
            for predicted_frame, actual_frame in zip(
                predicted_frames[1:], actual_frames[1:]
            )
        ]
        finite = [score for score in time_scores if np.isfinite(score)]
        if not finite:
            raise ValueError(f"{camera} has no non-degenerate trajectory deltas")
        camera_scores[camera] = float(np.mean(finite))
        camera_time_scores[camera] = [
            None if not np.isfinite(score) else float(score) for score in time_scores
        ]
    return {
        "equal_camera_mean": float(np.mean(list(camera_scores.values()))),
        "camera_scores": camera_scores,
        "camera_time_scores": camera_time_scores,
    }


class WanVaeFrameEncoder:
    def __init__(self, *, vae_path: Path, device: str, dtype: torch.dtype) -> None:
        self.device = str(device)
        self.dtype = dtype
        self.vae_path = Path(vae_path)
        self.vae = _load_registered_model(
            str(self.vae_path),
            "wan_video_vae",
            torch_dtype=dtype,
            device=self.device,
        ).eval()
        self.cache: dict[str, dict[str, np.ndarray]] = {}
        self.latent_shape: tuple[int, ...] | None = None

    @torch.no_grad()
    def encode(self, path: Path) -> dict[str, np.ndarray]:
        content = path.read_bytes()
        key = hashlib.sha256(content).hexdigest()
        if key in self.cache:
            return self.cache[key]
        frame = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if frame.shape != (384, 320, 3):
            raise ValueError(f"Expected RoboTwin composite [384,320,3], got {frame.shape}")
        tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).to(
            device=self.device, dtype=self.dtype
        )
        tensor = (tensor * (2.0 / 255.0) - 1.0).unsqueeze(1)
        encoded = self.vae.encode([tensor], device=self.device, tiled=False)
        latent = encoded[0] if isinstance(encoded, list) else encoded[0]
        if latent.ndim != 4 or latent.shape[1] != 1:
            raise ValueError(f"Expected single-frame VAE latent [C,1,H,W], got {latent.shape}")
        latent = latent[:, 0].detach().to(device="cpu", dtype=torch.float32).numpy()
        _, height, width = latent.shape
        head_end = height * 2 // 3
        wrist_mid = width // 2
        if head_end * 3 != height * 2 or wrist_mid * 2 != width:
            raise ValueError(
                f"VAE latent shape {latent.shape} does not preserve the camera grid"
            )
        features = {
            "head": latent[:, :head_end, :].copy(),
            "left_wrist": latent[:, head_end:, :wrist_mid].copy(),
            "right_wrist": latent[:, head_end:, wrist_mid:].copy(),
        }
        self.latent_shape = tuple(int(value) for value in latent.shape)
        self.cache[key] = features
        return features


def _load_metadata(root: Path, replan: int) -> tuple[Path, dict[str, Any]]:
    record_dir = root / f"replan_{replan:04d}"
    path = record_dir / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "robotwin_imagination_trajectory_v2":
        raise ValueError(f"Unsupported trajectory schema in {path}")
    if payload.get("trajectory_alignment_valid") is not True:
        raise ValueError(f"Incomplete trajectory in {path}")
    return record_dir, payload


def _encode_trajectory(
    encoder: WanVaeFrameEncoder,
    *,
    record_dir: Path,
    files: list[str],
) -> dict[str, list[np.ndarray]]:
    output = {name: [] for name in CAMERA_NAMES}
    for relative_path in files:
        features = encoder.encode(record_dir / relative_path)
        for name in CAMERA_NAMES:
            output[name].append(features[name])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--corrupt-root", type=Path, required=True)
    parser.add_argument("--correct-root", type=Path, required=True)
    parser.add_argument("--replan", type=int, required=True)
    parser.add_argument("--shuffle-replan", type=int, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    encoder = WanVaeFrameEncoder(
        vae_path=args.vae_path, device=args.device, dtype=dtype
    )
    clean_dir, clean_metadata = _load_metadata(args.clean_root, args.replan)
    shuffle_dir, shuffle_metadata = _load_metadata(
        args.clean_root, args.shuffle_replan
    )
    matched_prediction = _encode_trajectory(
        encoder,
        record_dir=clean_dir,
        files=clean_metadata["predicted_trajectory_files"],
    )
    shuffled_prediction = _encode_trajectory(
        encoder,
        record_dir=shuffle_dir,
        files=shuffle_metadata["predicted_trajectory_files"],
    )

    branches: dict[str, Any] = {}
    for name, root in (
        ("clean", args.clean_root),
        ("corrupt", args.corrupt_root),
        ("correct", args.correct_root),
    ):
        record_dir, metadata = _load_metadata(root, args.replan)
        actual = _encode_trajectory(
            encoder,
            record_dir=record_dir,
            files=metadata["actual_trajectory_files"],
        )
        matched = compute_trajectory_alignment(matched_prediction, actual)
        shuffled = compute_trajectory_alignment(shuffled_prediction, actual)
        branches[name] = {
            "episode_success": bool(metadata["episode_success"]),
            "matched": matched,
            "shuffled": shuffled,
            "matched_minus_shuffled": (
                matched["equal_camera_mean"] - shuffled["equal_camera_mean"]
            ),
        }

    clean_score = branches["clean"]["matched"]["equal_camera_mean"]
    corrupt_score = branches["corrupt"]["matched"]["equal_camera_mean"]
    correct_score = branches["correct"]["matched"]["equal_camera_mean"]
    result = {
        "schema_version": "robotwin_frozen_plan_vae_reward_validation_v1",
        "status": "diagnostic",
        "feature_encoder": "wan2.2_vae_single_frame_spatial_latent",
        "vae_path": str(args.vae_path.resolve()),
        "dtype": args.dtype,
        "latent_shape": encoder.latent_shape,
        "camera_pooling": "flattened_spatial_region",
        "camera_weights": {name: 1.0 / 3.0 for name in CAMERA_NAMES},
        "time_offsets": clean_metadata["trajectory_expected_action_offsets"],
        "replan": args.replan,
        "shuffle_replan": args.shuffle_replan,
        "branches": branches,
        "checks": {
            "clean_correct_equal": bool(abs(clean_score - correct_score) <= 1e-7),
            "clean_above_corrupt": bool(clean_score > corrupt_score),
            "clean_matched_above_shuffled": bool(
                branches["clean"]["matched_minus_shuffled"] > 0.0
            ),
            "correct_matched_above_shuffled": bool(
                branches["correct"]["matched_minus_shuffled"] > 0.0
            ),
        },
        "unique_encoded_frames": len(encoder.cache),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
