#!/usr/bin/env python3
"""Score strict expert-success/FastWAM-failure pairs with frozen Wan VAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.robotwin.export_paired_expert_imagination_trajectories import (
    load_natural_failure_cases,
)
from experiments.robotwin.validate_frozen_plan_vae_reward import (
    CAMERA_NAMES,
    WanVaeFrameEncoder,
    _encode_trajectory,
    compute_trajectory_alignment,
)


def aggregate_replan_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("episode has no valid trajectory replans")
    return {
        "valid_replans": len(rows),
        "mean_reward": float(np.mean([row["equal_camera_mean"] for row in rows])),
        "camera_scores": {
            camera: float(np.mean([row["camera_scores"][camera] for row in rows]))
            for camera in CAMERA_NAMES
        },
        "per_replan": rows,
    }


def select_episode_reward(
    episode: dict[str, Any], reward_cameras: tuple[str, ...]
) -> float:
    return float(
        np.mean([float(episode["camera_scores"][camera]) for camera in reward_cameras])
    )


def score_episode(
    root: Path, encoder: WanVaeFrameEncoder
) -> tuple[dict[str, Any], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for metadata_path in sorted(root.glob("replan_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != "robotwin_imagination_trajectory_v2":
            raise ValueError(f"Unsupported schema in {metadata_path}")
        if metadata.get("trajectory_alignment_valid") is not True:
            skipped += 1
            continue
        record_dir = metadata_path.parent
        predicted = _encode_trajectory(
            encoder,
            record_dir=record_dir,
            files=metadata["predicted_trajectory_files"],
        )
        actual = _encode_trajectory(
            encoder,
            record_dir=record_dir,
            files=metadata["actual_trajectory_files"],
        )
        alignment = compute_trajectory_alignment(predicted, actual)
        rows.append(
            {
                "replan_idx": int(metadata["replan_idx"]),
                "equal_camera_mean": alignment["equal_camera_mean"],
                "camera_scores": alignment["camera_scores"],
                "camera_time_scores": alignment["camera_time_scores"],
            }
        )
    result = aggregate_replan_scores(rows)
    result["skipped_incomplete_replans"] = skipped
    result["root"] = str(root.resolve())
    return result, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task filter.",
    )
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--fastwam-run-dir", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--reward-cameras",
        default=",".join(CAMERA_NAMES),
        help=(
            "Comma-separated camera regions used for the primary pair reward. "
            "All camera diagnostics are still reported."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    reward_cameras = tuple(
        value.strip() for value in args.reward_cameras.split(",") if value.strip()
    )
    if not reward_cameras:
        raise ValueError("--reward-cameras must select at least one camera")
    unknown_cameras = sorted(set(reward_cameras) - set(CAMERA_NAMES))
    if unknown_cameras:
        raise ValueError(
            f"Unknown reward cameras {unknown_cameras}; expected a subset of {CAMERA_NAMES}"
        )
    if len(reward_cameras) != len(set(reward_cameras)):
        raise ValueError("--reward-cameras contains duplicate camera names")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    encoder = WanVaeFrameEncoder(
        vae_path=args.vae_path, device=args.device, dtype=dtype
    )
    task_filter = {value.strip() for value in args.tasks.split(",") if value.strip()}
    pairs = []
    for case in load_natural_failure_cases(
        args.cases_jsonl, tasks=(task_filter if task_filter else None)
    ):
        task = str(case["task"])
        episode_id = int(case["evaluation_episode_id"])
        expert_root = (
            args.expert_root / task / "expert" / f"episode_{episode_id:04d}"
        )
        failure_root = (
            args.fastwam_run_dir
            / task
            / "imagination_transitions"
            / task
            / "policy"
            / f"episode_{episode_id:04d}"
        )
        expert, _ = score_episode(expert_root, encoder)
        failure, _ = score_episode(failure_root, encoder)
        expert_primary_reward = select_episode_reward(expert, reward_cameras)
        failure_primary_reward = select_episode_reward(failure, reward_cameras)
        expert["primary_reward"] = expert_primary_reward
        failure["primary_reward"] = failure_primary_reward
        margin = float(expert_primary_reward - failure_primary_reward)
        pairs.append(
            {
                "task": task,
                "episode_id": episode_id,
                "environment_seed": int(case["environment_seed"]),
                "instruction": str(case["instruction"]),
                "expert_success": expert,
                "fastwam_failure": failure,
                "success_minus_failure": margin,
                "correctly_ranked": bool(margin > 0.0),
            }
        )
    margins = [float(pair["success_minus_failure"]) for pair in pairs]
    camera_pairwise = {}
    for camera in CAMERA_NAMES:
        camera_margins = [
            float(
                pair["expert_success"]["camera_scores"][camera]
                - pair["fastwam_failure"]["camera_scores"][camera]
            )
            for pair in pairs
        ]
        camera_pairwise[camera] = {
            "correctly_ranked_count": sum(margin > 0.0 for margin in camera_margins),
            "pairwise_accuracy": float(np.mean([margin > 0.0 for margin in camera_margins])),
            "mean_success_minus_failure": float(np.mean(camera_margins)),
        }
    result = {
        "schema_version": "robotwin_natural_failure_wan_vae_pair_reward_v1",
        "feature_encoder": "wan2.2_vae_single_frame_spatial_latent",
        "trajectory_reference_policy": "frozen_once_per_action_chunk",
        "time_offsets": [0, 4, 8, 12, 16, 20, 24],
        "reward_cameras": list(reward_cameras),
        "camera_weights": {
            camera: (1.0 / len(reward_cameras) if camera in reward_cameras else 0.0)
            for camera in CAMERA_NAMES
        },
        "pair_count": len(pairs),
        "correctly_ranked_count": sum(bool(pair["correctly_ranked"]) for pair in pairs),
        "pairwise_accuracy": float(
            np.mean([bool(pair["correctly_ranked"]) for pair in pairs])
        ),
        "mean_success_minus_failure": float(np.mean(margins)),
        "per_camera_pairwise": camera_pairwise,
        "pairs": pairs,
        "latent_shape": encoder.latent_shape,
        "unique_encoded_frames": len(encoder.cache),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
