#!/usr/bin/env python3
"""Export local expert HDF5 cases with frozen FastWAM trajectory imagination."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from experiments.robotwin.export_expert_imagination_transitions import (
    _language_feature,
    _make_policy,
    expert_action_chunk,
    read_observation,
)
from experiments.robotwin.imagination_reward_utils import (
    array_sha256,
    save_aligned_transition,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


def load_natural_failure_cases(
    path: Path, *, tasks: set[str] | None = None
) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("decision") != "natural_failure":
            continue
        if tasks is not None and str(row.get("task")) not in tasks:
            continue
        for key in ("task", "instruction", "expert_hdf5", "evaluation_episode_id"):
            if key not in row:
                raise ValueError(f"{path}:{line_number} missing {key}")
        cases.append(row)
    identities = [
        (str(row["task"]), int(row["evaluation_episode_id"])) for row in cases
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("natural-failure cases contain duplicate task/episode identities")
    if not cases:
        raise ValueError(f"No natural-failure cases in {path}")
    return cases


def completed_case(
    output_dir: Path,
    *,
    task: str,
    episode_id: int,
    instruction: str,
    expert_hdf5: Path,
) -> int | None:
    episode_dir = output_dir / task / "expert" / f"episode_{episode_id:04d}"
    paths = sorted(episode_dir.glob("replan_*/metadata.json"))
    if not paths:
        return None
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if [int(row.get("replan_idx", -1)) for row in rows] != list(range(len(rows))):
        return None
    if any(row.get("schema_version") != "robotwin_imagination_trajectory_v2" for row in rows):
        return None
    if any(row.get("task_description") != instruction for row in rows):
        return None
    if any(Path(row.get("expert_source", "")) != expert_hdf5.resolve() for row in rows):
        return None
    if any(bool(row.get("terminated")) for row in rows[:-1]):
        return None
    if not bool(rows[-1].get("terminated")):
        return None
    return len(rows)


def export_case(
    policy: Any,
    *,
    task: str,
    instruction: str,
    expert_hdf5: Path,
    episode_id: int,
    output_dir: Path,
    replan_steps: int,
) -> int:
    with h5py.File(expert_hdf5, "r") as handle:
        states = np.asarray(handle["joint_action/vector"][:], dtype=np.float32)
        if states.shape[0] < 2:
            raise ValueError(f"expert episode is too short: {expert_hdf5}")
        first = read_observation(handle, 0)
        initial_image = policy._build_robotwin_image(first)
        initial_state = np.asarray(first["joint_action"]["vector"], dtype=np.float32)
        initial_hash = array_sha256(
            np.concatenate([initial_image.reshape(-1), initial_state])
        )
        language_feature = _language_feature(policy, instruction)
        starts = list(range(0, states.shape[0] - 1, replan_steps))
        for replan_idx, start in enumerate(starts):
            current = read_observation(handle, start)
            expert_actions, effective_k, end = expert_action_chunk(
                states, start, replan_steps
            )
            (
                baseline_actions,
                _corrupted_actions,
                _target_residual,
                _executed_actions,
                predicted_frames,
                current_image,
                _noise,
                _corruption_mask,
                _residual_output,
            ) = policy._infer_action_chunk(current, instruction)
            if predicted_frames is None:
                raise RuntimeError("FastWAM did not return a frozen imagined trajectory")

            frame_stride = int(policy.action_video_freq_ratio)
            predicted_offsets = list(range(0, replan_steps + 1, frame_stride))
            if len(predicted_frames) != len(predicted_offsets):
                raise ValueError(
                    f"predicted frame count {len(predicted_frames)} does not match "
                    f"offsets {predicted_offsets}"
                )
            actual_offsets = list(range(0, effective_k + 1, frame_stride))
            actual_frames = [
                policy._build_robotwin_image(read_observation(handle, start + offset))
                for offset in actual_offsets
            ]
            actual = read_observation(handle, end)
            terminal = end == states.shape[0] - 1
            trajectory_valid = bool(
                effective_k == replan_steps and actual_offsets == predicted_offsets
            )
            record_dir = (
                output_dir
                / task
                / "expert"
                / f"episode_{episode_id:04d}"
                / f"replan_{replan_idx:04d}"
            )
            environment_rewards = np.zeros(effective_k, dtype=np.float32)
            if terminal:
                environment_rewards[-1] = 1.0
            save_aligned_transition(
                record_dir,
                current_frame=current_image,
                predicted_goal_frame=predicted_frames[-1],
                actual_frame=policy._build_robotwin_image(actual),
                metadata={
                    "schema_version": "robotwin_imagination_trajectory_v2",
                    "task_suite": "robotwin2.0",
                    "task_name": task,
                    "task_description": instruction,
                    "trial_idx": episode_id,
                    "replan_idx": replan_idx,
                    "action_mode": "expert",
                    "behavior_tag": "expert",
                    "action_noise_std": 0.0,
                    "action_noise_seed": 0,
                    "action_corruption_seed": 0,
                    "initial_observation_sha256": initial_hash,
                    "current_observation_sha256": array_sha256(
                        np.concatenate(
                            [
                                current_image.reshape(-1),
                                np.asarray(
                                    current["joint_action"]["vector"], dtype=np.float32
                                ),
                            ]
                        )
                    ),
                    "baseline_actions_sha256": array_sha256(
                        baseline_actions[:effective_k]
                    ),
                    "target_step": replan_steps,
                    "effective_k": effective_k,
                    "goal_frame_index": replan_steps // frame_stride,
                    "goal_tau": float(replan_steps),
                    "terminated": terminal,
                    "truncated": False,
                    "transition_success": terminal,
                    "episode_success": True,
                    "alignment_valid": effective_k == replan_steps,
                    "trajectory_alignment_valid": trajectory_valid,
                    "trajectory_expected_action_offsets": predicted_offsets,
                    "trajectory_num_predicted_frames": len(predicted_frames),
                    "trajectory_num_actual_frames": len(actual_frames),
                    "trajectory_reference_policy": "frozen_once_per_action_chunk",
                    "camera_layout": "head_256x320_over_left_right_128x160_v1",
                    "policy_version": "fastwam_infer_action_frozen",
                    "predictor_version": "fastwam_infer_joint_frozen",
                    "language_encoder_version": "fastwam_umt5_masked_mean_v1",
                    "language_prompt_template": DEFAULT_PROMPT,
                    "expert_source": str(expert_hdf5.resolve()),
                    "expert_action_alignment": "next_qpos_t_plus_1_v1",
                },
                rollout_arrays={
                    "proprio": np.asarray(
                        current["joint_action"]["vector"], dtype=np.float32
                    ),
                    "next_proprio": np.asarray(
                        actual["joint_action"]["vector"], dtype=np.float32
                    ),
                    "baseline_actions": baseline_actions[:effective_k],
                    "planned_actions": expert_actions,
                    "executed_actions": expert_actions,
                    "environment_rewards": environment_rewards,
                    "language_feature": language_feature,
                },
                predicted_trajectory_frames=predicted_frames,
                predicted_trajectory_action_offsets=predicted_offsets,
                actual_trajectory_frames=actual_frames,
                actual_trajectory_action_offsets=actual_offsets,
            )
    return len(starts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task filter.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument(
        "--sim-config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/sim_robotwin.yaml",
    )
    parser.add_argument("--sim-task", default="robotwin_uncond_3cam_384_1e-4")
    args = parser.parse_args()

    if not args.model_base_path.is_dir():
        raise FileNotFoundError(args.model_base_path)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_base_path.resolve())
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    tasks = {value.strip() for value in args.tasks.split(",") if value.strip()}
    cases = load_natural_failure_cases(
        args.cases_jsonl, tasks=(tasks if tasks else None)
    )
    summary: dict[str, Any] = {"cases": [], "total_transitions": 0}
    for task in sorted({str(row["task"]) for row in cases}):
        policy_args = argparse.Namespace(**vars(args))
        policy = _make_policy(policy_args, task)
        try:
            for row in (case for case in cases if str(case["task"]) == task):
                episode_id = int(row["evaluation_episode_id"])
                instruction = str(row["instruction"])
                expert_hdf5 = Path(row["expert_hdf5"]).resolve()
                count = completed_case(
                    args.output_dir,
                    task=task,
                    episode_id=episode_id,
                    instruction=instruction,
                    expert_hdf5=expert_hdf5,
                )
                if count is None:
                    count = export_case(
                        policy,
                        task=task,
                        instruction=instruction,
                        expert_hdf5=expert_hdf5,
                        episode_id=episode_id,
                        output_dir=args.output_dir,
                        replan_steps=args.replan_steps,
                    )
                summary["cases"].append(
                    {
                        "task": task,
                        "episode_id": episode_id,
                        "environment_seed": int(row["environment_seed"]),
                        "transitions": count,
                    }
                )
                summary["total_transitions"] += count
                print(json.dumps(summary["cases"][-1], sort_keys=True), flush=True)
        finally:
            del policy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "expert_export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
