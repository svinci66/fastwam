"""Collect exact-state LIBERO counterfactuals for Reward V2 validation.

For each initial state this script performs exactly one FastWAM inference, then
restores the same MuJoCo anchor state before executing policy, shared-direction
noise, and zero-action branches.  It does not train or update any model.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.eval_libero_single import (  # noqa: E402
    _center_crop_resize,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _predict_action_chunk,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
)
from experiments.libero.imagination_reward_utils import (  # noqa: E402
    build_shared_direction_action_branches,
    frame_to_rgb_array,
    split_horizontal_camera_views,
)
from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def _create_deterministic_env(task: Any, *, seed: int):
    """Construct identical MuJoCo models, not merely identically seeded rollouts."""
    random.seed(seed)
    np.random.seed(seed)
    env = get_libero_env(
        task,
        resolution=LIBERO_ENV_RESOLUTION,
        seed=seed,
    )[0]
    # LIBERO defaults to hard resets, which reload the XML and changes model-level
    # state that is absent from flattened MuJoCo state. A soft reset still reloads
    # controllers and observables while preserving the exact camera / geometry model.
    env.env.hard_reset = False
    return env


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _resize_camera_images(images: dict[str, np.ndarray], size: int = 224) -> dict[str, np.ndarray]:
    return {
        "agent": _center_crop_resize(images["image"], width=size, height=size),
        "wrist": _center_crop_resize(images["wrist_image"], width=size, height=size),
    }


def _save_camera_images(output_dir: Path, prefix: str, images: dict[str, np.ndarray]) -> None:
    for camera, image in sorted(images.items()):
        _save_rgb(output_dir / f"{prefix}_{camera}.png", image)


def _render_endpoint_ensemble(
    env: Any,
    state: np.ndarray,
    *,
    repeats: int,
    output_dir: Path,
    prefix: str = "actual",
) -> list[dict[str, str]]:
    paths: list[dict[str, str]] = []
    for repeat_index in range(repeats):
        obs = env.set_init_state(state.copy())
        cameras = _resize_camera_images(get_libero_image(obs))
        repeat_paths = {}
        for camera, image in sorted(cameras.items()):
            path = output_dir / f"{prefix}_repeat{repeat_index:02d}_{camera}.png"
            _save_rgb(path, image)
            repeat_paths[camera] = str(path.resolve())
        paths.append(repeat_paths)
    return paths


def _execute_actions_from_current_state(env: Any, actions: np.ndarray) -> dict[str, Any]:
    rewards: list[float] = []
    dones: list[bool] = []
    for action in actions:
        obs, reward, done, _ = env.step(action.copy())
        rewards.append(float(reward))
        dones.append(bool(done))
    return {
        "final_state": env.get_sim_state().copy(),
        "rewards": rewards,
        "dones": dones,
        "success": bool(env.check_success()),
    }


def _reset_controller_without_resetting_model(env: Any) -> None:
    """Reset controller / episode caches while preserving LIBERO's MuJoCo model."""
    inner = env.env
    for robot in inner.robots:
        robot.reset(deterministic=True)
    inner.cur_time = 0
    inner.timestep = 0
    inner.done = False
    inner._obs_cache = {}
    for observable in inner._observables.values():
        observable.reset()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None")
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError("Exact-state collection requires EVALUATION.visualize_future_video=true")
    if not bool(cfg.EVALUATION.get("imagination_use_direct_action", False)):
        raise ValueError("Exact-state collection requires direct policy actions")

    seed = int(cfg.get("seed", 2042))
    num_states = int(cfg.EVALUATION.get("exact_state_num_states", 10))
    exec_steps = int(cfg.EVALUATION.get("exact_state_exec_steps", 8))
    render_repeats = int(cfg.EVALUATION.get("exact_state_render_repeats", 8))
    noise_stds = tuple(
        float(value)
        for value in cfg.EVALUATION.get("exact_state_noise_stds", [0.075, 0.15, 0.30])
    )
    wait_steps = int(cfg.EVALUATION.get("num_steps_wait", 30))
    if min(num_states, exec_steps, render_repeats) <= 0:
        raise ValueError("num_states, exec_steps, and render_repeats must be positive")
    set_global_seed(seed, get_worker_init_fn=False)

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    input_h, input_w = (int(value) for value in cfg.data.train.video_size)

    task_suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    if num_states > len(initial_states):
        raise ValueError(f"Requested {num_states} states, only {len(initial_states)} available")

    output_root = Path(cfg.EVALUATION.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    env = _create_deterministic_env(task, seed=seed)
    task_description = task.language
    anchors: list[dict[str, Any]] = []
    try:
        for state_index in range(num_states):
            anchor_start = time.time()
            env.reset()
            obs = env.set_init_state(initial_states[state_index])
            for _ in range(wait_steps):
                obs, _, _, _ = env.step(get_libero_dummy_action())

            anchor_state = env.get_sim_state().copy()
            action_chunk, current_images_raw, predicted_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            if predicted_frames is None or len(predicted_frames) < 2:
                raise ValueError("FastWAM did not return a usable future goal")
            if len(action_chunk) < exec_steps:
                raise ValueError(
                    f"Action chunk has {len(action_chunk)} steps, expected at least {exec_steps}"
                )

            base_actions = np.asarray(action_chunk[:exec_steps], dtype=np.float32)
            rng = np.random.default_rng(seed + state_index * 100_003)
            branches, epsilon = build_shared_direction_action_branches(
                base_actions,
                noise_stds=noise_stds,
                rng=rng,
            )

            anchor_dir = output_root / f"anchor{state_index:02d}"
            anchor_dir.mkdir(parents=True, exist_ok=True)
            np.save(anchor_dir / "anchor_state.npy", anchor_state)
            np.save(anchor_dir / "base_actions.npy", base_actions)
            np.save(anchor_dir / "shared_noise_epsilon.npy", epsilon)

            current_cameras = _resize_camera_images(current_images_raw)
            predicted_goal_concat = frame_to_rgb_array(predicted_frames[-1])
            predicted_goal_cameras = split_horizontal_camera_views(predicted_goal_concat)
            _save_camera_images(anchor_dir, "current", current_cameras)
            _save_camera_images(anchor_dir, "predicted_goal", predicted_goal_cameras)
            _save_rgb(anchor_dir / "predicted_goal_concat.png", predicted_goal_concat)
            reward_current_paths = _render_endpoint_ensemble(
                env,
                anchor_state,
                repeats=render_repeats,
                output_dir=anchor_dir,
                prefix="reward_current",
            )

            branch_records = []
            for branch_name, actions in branches.items():
                branch_dir = anchor_dir / branch_name
                branch_dir.mkdir(parents=True, exist_ok=True)
                np.save(branch_dir / "actions.npy", actions)
                _reset_controller_without_resetting_model(env)
                branch_obs = env.set_init_state(initial_states[state_index])
                for _ in range(wait_steps):
                    branch_obs, _, _, _ = env.step(get_libero_dummy_action())
                replayed_anchor_state = env.get_sim_state().copy()
                wait_replay_state_max_abs_diff = float(
                    np.max(np.abs(replayed_anchor_state - anchor_state))
                )
                env.set_init_state(anchor_state.copy())
                reconstructed_anchor_state = env.get_sim_state().copy()
                reconstructed_state_max_abs_diff = float(
                    np.max(np.abs(reconstructed_anchor_state - anchor_state))
                )
                if reconstructed_state_max_abs_diff > 1e-10:
                    raise RuntimeError(
                        f"Independent branch failed to reconstruct anchor {state_index}: "
                        f"max_abs_diff={reconstructed_state_max_abs_diff}"
                    )
                branch_current_paths = _render_endpoint_ensemble(
                    env,
                    reconstructed_anchor_state,
                    repeats=render_repeats,
                    output_dir=branch_dir,
                    prefix="branch_current",
                )
                outcome = _execute_actions_from_current_state(env, actions)
                np.save(branch_dir / "final_state.npy", outcome["final_state"])
                actual_paths = _render_endpoint_ensemble(
                    env,
                    outcome["final_state"],
                    repeats=render_repeats,
                    output_dir=branch_dir,
                )
                branch_records.append(
                    {
                        "name": branch_name,
                        "noise_std": (
                            None
                            if branch_name in {"policy", "zero"}
                            else float(branch_name.removeprefix("noise_"))
                        ),
                        "quality": (
                            None
                            if branch_name == "zero"
                            else -(
                                0.0
                                if branch_name == "policy"
                                else float(branch_name.removeprefix("noise_"))
                            )
                        ),
                        "success": outcome["success"],
                        "sim_rewards": outcome["rewards"],
                        "dones": outcome["dones"],
                        "reconstructed_anchor_state_max_abs_diff": (
                            reconstructed_state_max_abs_diff
                        ),
                        "wait_replay_state_max_abs_diff_before_exact_restore": (
                            wait_replay_state_max_abs_diff
                        ),
                        "current_paths": branch_current_paths,
                        "actions_path": str((branch_dir / "actions.npy").resolve()),
                        "final_state_path": str((branch_dir / "final_state.npy").resolve()),
                        "actual_paths": actual_paths,
                    }
                )

            anchor_record = {
                "anchor_index": state_index,
                "inference_seed": seed,
                "noise_seed": seed + state_index * 100_003,
                "anchor_state_path": str((anchor_dir / "anchor_state.npy").resolve()),
                "base_actions_path": str((anchor_dir / "base_actions.npy").resolve()),
                "shared_noise_path": str((anchor_dir / "shared_noise_epsilon.npy").resolve()),
                "current_paths": {
                    camera: str((anchor_dir / f"current_{camera}.png").resolve())
                    for camera in ("agent", "wrist")
                },
                "reward_current_paths": reward_current_paths,
                "predicted_goal_paths": {
                    camera: str((anchor_dir / f"predicted_goal_{camera}.png").resolve())
                    for camera in ("agent", "wrist")
                },
                "branches": branch_records,
                "duration_seconds": time.time() - anchor_start,
            }
            anchors.append(anchor_record)
            (anchor_dir / "metadata.json").write_text(
                json.dumps(anchor_record, indent=2) + "\n"
            )
            print(
                f"anchor {state_index + 1}/{num_states} complete "
                f"({anchor_record['duration_seconds']:.2f}s)"
            )
    finally:
        env.close()

    manifest = {
        "schema_version": 1,
        "experiment": "exact_state_counterfactual_reward_v2",
        "task_suite": task_suite_name,
        "task_id": task_id,
        "task_description": task_description,
        "seed": seed,
        "num_states": num_states,
        "wait_steps": wait_steps,
        "exec_steps": exec_steps,
        "render_repeats": render_repeats,
        "noise_stds": noise_stds,
        "action_source": "direct_infer_action",
        "goal_source": "infer_joint_last_aligned_frame",
        "branch_initialization": (
            "same_model_controller_reset_plus_initial_state_wait_replay_exact_restore"
        ),
        "checkpoint": str(Path(str(cfg.ckpt)).resolve()),
        "dataset_stats_path": str(dataset_stats_path.resolve()),
        "total_model_inferences": num_states,
        "total_short_rollouts": num_states * 5,
        "total_simulator_action_steps": num_states * 5 * exec_steps,
        "duration_seconds": time.time() - start_time,
        "anchors": anchors,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logging.info("Wrote exact-state manifest to %s", output_root / "manifest.json")
    print(f"collection complete in {manifest['duration_seconds']:.2f}s")


if __name__ == "__main__":
    main()
