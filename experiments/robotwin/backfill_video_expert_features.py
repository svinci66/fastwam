#!/usr/bin/env python3
"""Backfill frozen FastWAM Video Expert observations into audited transitions.

The script reuses each record's saved composite ``current.png``, instruction,
and proprioception.  It runs only FastWAM's first-frame VAE + Video Expert
prefill, then stores the versioned 3072-D feature beside the existing rollout
arrays.  Baseline actions, outcomes, and Wan-VAE reward labels are not changed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.robotwin.build_wan_vae_head_awr_replay import (
    discover_records,
    select_reward_tasks,
)
from experiments.robotwin.export_expert_imagination_transitions import _make_policy
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.fastwam import FASTWAM_VIDEO_EXPERT_FEATURE_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument(
        "--sim-config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/sim_robotwin.yaml",
    )
    parser.add_argument("--sim-task", default="robotwin_uncond_3cam_384_1e-4")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.video-expert.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.video-expert.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _already_complete(
    metadata: dict[str, Any], arrays: dict[str, np.ndarray], *, checkpoint_sha256: str
) -> bool:
    feature = arrays.get("video_expert_feature")
    return bool(
        metadata.get("video_expert_feature_version")
        == FASTWAM_VIDEO_EXPERT_FEATURE_VERSION
        and metadata.get("video_expert_checkpoint_sha256") == checkpoint_sha256
        and feature is not None
        and np.asarray(feature).ndim == 1
        and np.asarray(feature).size == int(metadata.get("video_expert_feature_dim", -1))
        and np.all(np.isfinite(feature))
    )


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    for path in (
        args.reward_json,
        args.checkpoint,
        args.dataset_stats,
        args.model_base_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_base_path.resolve())
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"

    reward_payload = json.loads(args.reward_json.read_text(encoding="utf-8"))
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    reward_payload = select_reward_tasks(reward_payload, tasks)
    records = discover_records(reward_payload)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError("No transition records selected")

    # _make_policy loads exactly the same released FastWAM config/checkpoint as
    # RoboTwin online evaluation.  The task field does not constrain inference.
    policy_args = argparse.Namespace(**vars(args))
    policy = _make_policy(policy_args, str(records[0]["task_name"]))
    encoded = 0
    skipped = 0
    feature_dim: int | None = None
    checkpoint_sha256: str | None = None
    try:
        checkpoint_sha256 = policy._fastwam_checkpoint_sha256
        if not checkpoint_sha256:
            raise RuntimeError("FastWAM checkpoint hash was not initialized")
        for index, record in enumerate(records, 1):
            record_dir = Path(record["record_dir"])
            metadata_path = record_dir / "metadata.json"
            arrays_path = record_dir / str(record["rollout_arrays_file"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(arrays_path, allow_pickle=False) as payload:
                arrays = {key: payload[key] for key in payload.files}
            if _already_complete(
                metadata, arrays, checkpoint_sha256=checkpoint_sha256
            ):
                completed_dim = int(np.asarray(arrays["video_expert_feature"]).size)
                if feature_dim is None:
                    feature_dim = completed_dim
                elif completed_dim != feature_dim:
                    raise RuntimeError(
                        "Existing Video Expert feature dimension changed: "
                        f"{feature_dim} -> {completed_dim}"
                    )
                skipped += 1
                continue

            image = np.asarray(
                Image.open(record["current_path"]).convert("RGB"), dtype=np.uint8
            )
            image_tensor = (
                torch.from_numpy(image.copy())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=policy.model.device, dtype=policy.model.torch_dtype)
            )
            image_tensor = image_tensor * (2.0 / 255.0) - 1.0
            proprio = policy._normalize_state(
                np.asarray(arrays["proprio"], dtype=np.float32)
            )
            prompt = DEFAULT_PROMPT.format(task=str(record["task_description"]))
            output = policy.model.encode_video_expert_feature(
                prompt=prompt,
                input_image=image_tensor,
                proprio=proprio,
                tiled=policy.tiled,
            )
            version = str(output["video_expert_feature_version"])
            if version != FASTWAM_VIDEO_EXPERT_FEATURE_VERSION:
                raise RuntimeError(
                    f"Unexpected Video Expert feature version: {version!r}"
                )
            feature = (
                output["video_expert_feature"]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            if feature_dim is None:
                feature_dim = int(feature.size)
            elif feature.size != feature_dim:
                raise RuntimeError(
                    f"Video Expert feature dimension changed: {feature_dim} -> {feature.size}"
                )
            arrays["video_expert_feature"] = feature
            metadata["video_expert_feature_version"] = version
            metadata["video_expert_feature_dim"] = int(feature.size)
            metadata["video_expert_checkpoint_sha256"] = checkpoint_sha256
            _atomic_save_npz(arrays_path, arrays)
            _atomic_save_json(metadata_path, metadata)
            encoded += 1
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": len(records),
                        "task": record["task_name"],
                        "record_dir": str(record_dir),
                        "feature_dim": int(feature.size),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "schema_version": "fastwam_video_expert_feature_backfill_v1",
        "feature_version": FASTWAM_VIDEO_EXPERT_FEATURE_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "selected_records": len(records),
        "encoded_records": encoded,
        "skipped_records": skipped,
        "feature_dim": feature_dim,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "video_expert_backfill_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
