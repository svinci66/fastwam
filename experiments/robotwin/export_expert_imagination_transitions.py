"""Export a small RoboTwin expert dataset as aligned residual-RL transitions.

The upstream RoboTwin ``clean_50`` archives contain successful scripted qpos
trajectories.  At every replan boundary this tool freezes the expert observation,
queries the frozen FastWAM checkpoint for its baseline action chunk and imagined
future, and stores the expert's next qpos targets as the executed action chunk.
The resulting records share the same schema as online controlled-corruption
captures, so both sources can be combined by ``build_residual_rl_replay.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.robotwin.fastwam_policy.deploy_policy import get_model
from experiments.robotwin.imagination_reward_utils import (
    array_sha256,
    save_aligned_transition,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


DEFAULT_INSTRUCTIONS = {
    "adjust_bottle": "Pick up the bottle from the table and keep it upright.",
    "open_laptop": "Open the laptop completely.",
    "stack_blocks_two": (
        "Move the red and green blocks to the center and stack the green block "
        "on the red block."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument(
        "--model-base-path",
        type=Path,
        required=True,
        help="Local DiffSynth/Wan component root; network downloads are disabled.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_INSTRUCTIONS),
        help="Comma-separated task names.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument(
        "--sim-config",
        type=Path,
        default=PROJECT_ROOT / "configs/sim_robotwin.yaml",
    )
    parser.add_argument(
        "--sim-task", default="robotwin_uncond_3cam_384_1e-4"
    )
    return parser.parse_args()


def episode_index(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("episode"):
        raise ValueError(f"unexpected expert episode filename: {path.name}")
    return int(stem.removeprefix("episode"))


def resolve_task_data_dir(dataset_root: Path, task_name: str) -> Path:
    """Find the single HDF5 ``data`` directory inside one extracted task zip."""

    task_root = dataset_root / task_name
    direct = task_root / "data"
    if any(direct.glob("episode*.hdf5")):
        return direct
    matches = sorted(
        path
        for path in task_root.rglob("data")
        if path.is_dir() and any(path.glob("episode*.hdf5"))
    )
    if len(matches) != 1:
        raise ValueError(
            f"{task_name}: expected one expert HDF5 data directory below "
            f"{task_root}, found {matches}"
        )
    return matches[0]


def completed_episode_transition_count(
    output_dir: Path, task_name: str, episode_id: int
) -> int | None:
    episode_dir = output_dir / task_name / "expert" / f"episode_{episode_id:04d}"
    metadata_paths = sorted(episode_dir.glob("replan_*/metadata.json"))
    if not metadata_paths:
        return None
    records = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    indices = [int(record["replan_idx"]) for record in records]
    if indices != list(range(len(indices))):
        return None
    boundaries = [
        bool(record["terminated"]) or bool(record["truncated"]) for record in records
    ]
    if any(boundaries[:-1]) or not boundaries[-1]:
        return None
    return len(records)


def decode_jpeg(value: Any) -> np.ndarray:
    """Decode an upstream RoboTwin HDF5 JPEG value as RGB uint8."""

    payload = bytes(value)
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode RoboTwin JPEG payload")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_observation(handle: h5py.File, index: int) -> dict[str, Any]:
    observation: dict[str, Any] = {"observation": {}, "joint_action": {}}
    for camera in ("head_camera", "left_camera", "right_camera"):
        value = handle[f"observation/{camera}/rgb"][index]
        observation["observation"][camera] = {"rgb": decode_jpeg(value)}
    observation["joint_action"]["vector"] = np.asarray(
        handle["joint_action/vector"][index], dtype=np.float32
    )
    return observation


def expert_action_chunk(
    states: np.ndarray, start: int, target_k: int
) -> tuple[np.ndarray, int, int]:
    """Return qpos targets after ``start``, including a terminal partial chunk."""

    if states.ndim != 2 or states.shape[1] != 14:
        raise ValueError(f"expected RoboTwin qpos [T,14], got {states.shape}")
    if start < 0 or start >= states.shape[0] - 1:
        raise ValueError(f"start must leave a future state, got {start}")
    end = min(start + target_k, states.shape[0] - 1)
    actions = np.asarray(states[start + 1 : end + 1], dtype=np.float32)
    return np.ascontiguousarray(actions), int(actions.shape[0]), end


def _language_feature(policy: Any, instruction: str) -> np.ndarray:
    cached = policy._language_feature_cache.get(instruction)
    if cached is None:
        prompt = DEFAULT_PROMPT.format(task=instruction)
        with torch.inference_mode():
            pooled = policy.model.encode_prompt_pooled([prompt])
        cached = pooled[0].detach().float().cpu().numpy().astype(np.float32)
        policy._language_feature_cache[instruction] = cached
    return cached


def _make_policy(args: argparse.Namespace, task_name: str) -> Any:
    return get_model(
        {
            "sim_cfg_path": str(args.sim_config.resolve()),
            "sim_task": args.sim_task,
            "ckpt_setting": str(args.checkpoint.resolve()),
            "dataset_stats_path": str(args.dataset_stats.resolve()),
            "device": args.device,
            "mixed_precision": args.mixed_precision,
            "action_horizon": args.action_horizon,
            "replan_steps": args.replan_steps,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "rand_device": "cpu",
            "action_mode": "policy",
            "save_imagination_transitions": True,
            "imagination_transition_dir": str(args.output_dir.resolve()),
            "task_name": task_name,
            "fixed_instruction": DEFAULT_INSTRUCTIONS[task_name],
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
        }
    )


def export_episode(
    policy: Any,
    *,
    task_name: str,
    instruction: str,
    episode_path: Path,
    episode_id: int,
    output_dir: Path,
    replan_steps: int,
) -> int:
    with h5py.File(episode_path, "r") as handle:
        states = np.asarray(handle["joint_action/vector"][:], dtype=np.float32)
        if states.shape[0] < 2:
            raise ValueError(f"expert episode is too short: {episode_path}")
        first = read_observation(handle, 0)
        initial_image = policy._build_robotwin_image(first)
        initial_state = np.asarray(first["joint_action"]["vector"], dtype=np.float32)
        initial_hash = array_sha256(
            np.concatenate([initial_image.reshape(-1), initial_state])
        )
        language_feature = _language_feature(policy, instruction)
        starts = list(range(0, states.shape[0] - 1, replan_steps))
        metadata_paths: list[Path] = []
        for replan_idx, start in enumerate(starts):
            current = read_observation(handle, start)
            expert_actions, effective_k, end = expert_action_chunk(
                states, start, replan_steps
            )
            actual = read_observation(handle, end)
            (
                baseline_actions,
                _unused_executed,
                predicted_frames,
                current_image,
                _noise,
                _corruption_mask,
            ) = policy._infer_action_chunk(current, instruction)
            if predicted_frames is None:
                raise RuntimeError("FastWAM did not return an imagined future")
            terminal = end == states.shape[0] - 1
            record_dir = (
                output_dir
                / task_name
                / "expert"
                / f"episode_{episode_id:04d}"
                / f"replan_{replan_idx:04d}"
            )
            environment_rewards = np.zeros(effective_k, dtype=np.float32)
            if terminal:
                environment_rewards[-1] = 1.0
            metadata_paths.append(
                save_aligned_transition(
                    record_dir,
                    current_frame=current_image,
                    predicted_goal_frame=predicted_frames[-1],
                    actual_frame=policy._build_robotwin_image(actual),
                    metadata={
                        "schema_version": "robotwin_imagination_transition_v1",
                        "task_suite": "robotwin2.0",
                        "task_name": task_name,
                        "task_description": instruction,
                        "trial_idx": episode_id,
                        "replan_idx": replan_idx,
                        "action_mode": "expert",
                        "behavior_tag": "expert",
                        "action_noise_std": 0.0,
                        "action_noise_seed": 0,
                        "action_corruption_seed": 0,
                        "initial_observation_sha256": initial_hash,
                        "target_step": replan_steps,
                        "effective_k": effective_k,
                        "goal_frame_index": replan_steps // policy.action_video_freq_ratio,
                        "goal_tau": float(replan_steps),
                        "terminated": terminal,
                        "truncated": False,
                        "transition_success": terminal,
                        "episode_success": True,
                        "alignment_valid": effective_k == replan_steps,
                        "camera_layout": "head_256x320_over_left_right_128x160_v1",
                        "policy_version": "fastwam_infer_action_frozen",
                        "predictor_version": "fastwam_infer_joint_frozen",
                        "language_encoder_version": "fastwam_umt5_masked_mean_v1",
                        "language_prompt_template": DEFAULT_PROMPT,
                        "expert_source": str(episode_path.resolve()),
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
                )
            )
    return len(metadata_paths)


def main() -> None:
    args = parse_args()
    if not args.model_base_path.is_dir():
        raise FileNotFoundError(args.model_base_path)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_base_path.resolve())
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    if args.episodes_per_task <= 0:
        raise ValueError("episodes-per-task must be positive")
    if args.replan_steps <= 0 or args.replan_steps > args.action_horizon:
        raise ValueError("replan-steps must be in [1, action-horizon]")
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = sorted(set(tasks) - set(DEFAULT_INSTRUCTIONS))
    if unknown:
        raise ValueError(f"missing fixed instructions for tasks: {unknown}")
    summary: dict[str, Any] = {"tasks": {}, "total_transitions": 0}
    policy = _make_policy(args, tasks[0])
    for task_name in tasks:
        data_dir = resolve_task_data_dir(args.dataset_root, task_name)
        episodes = sorted(data_dir.glob("episode*.hdf5"), key=episode_index)
        selected = episodes[: args.episodes_per_task]
        if len(selected) != args.episodes_per_task:
            raise ValueError(
                f"{task_name}: expected {args.episodes_per_task} episodes under "
                f"{data_dir}, found {len(selected)}"
            )
        task_transitions = 0
        for path in selected:
            index = episode_index(path)
            count = completed_episode_transition_count(
                args.output_dir, task_name, index
            )
            if count is None:
                count = export_episode(
                    policy,
                    task_name=task_name,
                    instruction=DEFAULT_INSTRUCTIONS[task_name],
                    episode_path=path,
                    episode_id=index,
                    output_dir=args.output_dir,
                    replan_steps=args.replan_steps,
                )
            task_transitions += count
            print(
                json.dumps(
                    {
                        "task": task_name,
                        "episode": index,
                        "transitions": count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        summary["tasks"][task_name] = {
            "episodes": len(selected),
            "transitions": task_transitions,
        }
        summary["total_transitions"] += task_transitions
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "expert_export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
