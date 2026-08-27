"""Collect one locally reproducible RoboTwin expert episode for strict pairing."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


CAMERAS = ("head_camera", "left_camera", "right_camera")


def _json_value(value: Any) -> Any | None:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (str, int, float, bool, np.generic)) for item in value
    ):
        return [_json_value(item) for item in value]
    return None


def task_selectors(task_env: Any) -> dict[str, Any]:
    selectors: dict[str, Any] = {}
    for name, value in vars(task_env).items():
        if not name.endswith(("_id", "_name")):
            continue
        converted = _json_value(value)
        if converted is not None:
            selectors[name] = converted
    return selectors


def scene_state(task_env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    actors = []
    for index, actor in enumerate(task_env.scene.get_all_actors()):
        pose = actor.get_pose()
        actors.append(
            {
                "index": index,
                "name": str(actor.get_name()),
                "position": np.asarray(pose.p, dtype=np.float64).round(8).tolist(),
                "quaternion": np.asarray(pose.q, dtype=np.float64).round(8).tolist(),
            }
        )
    vector = observation.get("joint_action", {}).get("vector", [])
    return {
        "selectors": task_selectors(task_env),
        "actors": actors,
        "robot_joint_vector": np.asarray(vector, dtype=np.float64).round(8).tolist(),
    }


def compare_scene_states(
    planning: dict[str, Any], replay: dict[str, Any], *, atol: float = 1e-5
) -> dict[str, Any]:
    selector_match = planning["selectors"] == replay["selectors"]
    planning_actors = planning["actors"]
    replay_actors = replay["actors"]
    actor_identity_match = [row["name"] for row in planning_actors] == [
        row["name"] for row in replay_actors
    ]
    pose_differences: list[float] = []
    if actor_identity_match:
        for left, right in zip(planning_actors, replay_actors):
            for field in ("position", "quaternion"):
                pose_differences.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(left[field], dtype=np.float64)
                                - np.asarray(right[field], dtype=np.float64)
                            )
                        )
                    )
                )
    actor_pose_max_abs = max(pose_differences, default=float("inf"))
    planning_joints = np.asarray(planning["robot_joint_vector"], dtype=np.float64)
    replay_joints = np.asarray(replay["robot_joint_vector"], dtype=np.float64)
    joint_shape_match = planning_joints.shape == replay_joints.shape
    joint_max_abs = (
        float(np.max(np.abs(planning_joints - replay_joints)))
        if joint_shape_match and planning_joints.size
        else (0.0 if joint_shape_match else float("inf"))
    )
    exact = bool(
        selector_match
        and actor_identity_match
        and actor_pose_max_abs <= atol
        and joint_max_abs <= atol
    )
    return {
        "exact": exact,
        "selector_match": selector_match,
        "actor_identity_match": actor_identity_match,
        "actor_pose_max_abs": actor_pose_max_abs,
        "robot_joint_shape_match": joint_shape_match,
        "robot_joint_max_abs": joint_max_abs,
        "atol": atol,
    }


def rgb_observations(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    values = observation["observation"]
    return {
        camera: np.ascontiguousarray(values[camera]["rgb"], dtype=np.uint8)
        for camera in CAMERAS
    }


def compose_three_camera(images: dict[str, np.ndarray]) -> np.ndarray:
    head = cv2.resize(images["head_camera"], (320, 256))
    left = cv2.resize(images["left_camera"], (160, 128))
    right = cv2.resize(images["right_camera"], (160, 128))
    return np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)


def visual_metrics(
    planning: dict[str, np.ndarray], replay: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for camera in CAMERAS:
        left = cv2.GaussianBlur(planning[camera], (15, 15), 0).astype(np.float32)
        right = cv2.GaussianBlur(replay[camera], (15, 15), 0).astype(np.float32)
        difference = np.abs(left - right) / 255.0
        metrics[camera] = {
            "blurred_mean_abs": float(np.mean(difference)),
            "blurred_p95_abs": float(np.quantile(difference, 0.95)),
            "blurred_p99_abs": float(np.quantile(difference, 0.99)),
        }
    return metrics


def save_initial_observation(
    output_dir: Path, stem: str, images: dict[str, np.ndarray]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{stem}.npz", **images)
    composite = compose_three_camera(images)
    if not cv2.imwrite(
        str(output_dir / f"{stem}.png"), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError(f"failed to save {stem} initial observation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--pose-atol", type=float, default=1e-5)
    return parser.parse_args()


def load_runtime(args: argparse.Namespace) -> tuple[Any, dict[str, Any], Any]:
    root = args.robotwin_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "script"))
    sys.path.insert(0, str(root / "description" / "utils"))
    os.chdir(root)

    from envs import CONFIGS_PATH  # type: ignore
    from test_render import Sapien_TEST  # type: ignore

    Sapien_TEST()
    task_module = importlib.import_module(f"envs.{args.task}")
    task_env = getattr(task_module, args.task)()
    config_path = root / "task_config" / f"{args.task_config}.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "task_name": args.task,
            "task_config": args.task_config,
            "save_path": str(args.output_bundle.resolve()),
            "render_freq": 0,
        }
    )
    embodiment_types = yaml.safe_load(
        Path(CONFIGS_PATH, "_embodiment_config.yml").read_text(encoding="utf-8")
    )
    embodiment = config["embodiment"]
    if len(embodiment) != 1:
        raise ValueError("local strict-pair collector currently requires one dual-arm embodiment")
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    robot_config = yaml.safe_load(
        Path(robot_file, "config.yml").read_text(encoding="utf-8")
    )
    config.update(
        {
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "left_embodiment_config": robot_config,
            "right_embodiment_config": robot_config,
            "dual_arm_embodied": True,
            "embodiment_name": str(embodiment[0]),
        }
    )
    instruction_module = importlib.import_module("generate_episode_instructions")
    return task_env, config, instruction_module.generate_episode_descriptions


def close_safely(task_env: Any, *, clear_cache: bool = True) -> None:
    try:
        task_env.close_env(clear_cache=clear_cache)
    except Exception:
        pass
    viewer = getattr(task_env, "viewer", None)
    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    if args.seed < 0 or args.episode_index < 0:
        raise ValueError("seed and episode-index must be non-negative")
    args.output_bundle = args.output_bundle.resolve()
    args.output_bundle.mkdir(parents=True, exist_ok=True)
    task_env, base_config, generate_episode_descriptions = load_runtime(args)

    planning_config = dict(base_config)
    planning_config.update({"need_plan": True, "save_data": False})
    task_env.setup_demo(
        now_ep_num=args.episode_index, seed=args.seed, is_test=True, **planning_config
    )
    planning_observation = task_env.get_obs()
    planning_images = rgb_observations(planning_observation)
    planning_state = scene_state(task_env, planning_observation)
    try:
        episode_info = task_env.play_once()
        planning_success = bool(task_env.plan_success and task_env.check_success())
        if not planning_success:
            print(
                f"EXPERT_PLANNING_INFEASIBLE task={args.task} seed={args.seed}",
                flush=True,
            )
            raise SystemExit(20)
        task_env.save_traj_data(args.episode_index)
        trajectory = {
            "left_joint_path": deepcopy(task_env.left_joint_path),
            "right_joint_path": deepcopy(task_env.right_joint_path),
        }
    finally:
        close_safely(task_env)

    replay_config = dict(base_config)
    replay_config.update(
        {
            "need_plan": False,
            "save_data": True,
            "left_joint_path": trajectory["left_joint_path"],
            "right_joint_path": trajectory["right_joint_path"],
        }
    )
    task_env.setup_demo(
        now_ep_num=args.episode_index, seed=args.seed, is_test=True, **replay_config
    )
    replay_observation = task_env.get_obs()
    replay_images = rgb_observations(replay_observation)
    replay_state = scene_state(task_env, replay_observation)
    state_comparison = compare_scene_states(
        planning_state, replay_state, atol=args.pose_atol
    )
    if not state_comparison["exact"]:
        close_safely(task_env)
        print(
            f"STRICT_SCENE_MISMATCH task={args.task} seed={args.seed} "
            f"comparison={json.dumps(state_comparison, sort_keys=True)}",
            flush=True,
        )
        raise SystemExit(22)
    task_env.set_path_lst(replay_config)
    try:
        replay_info = task_env.play_once()
        replay_success = bool(task_env.plan_success and task_env.check_success())
        if not replay_success:
            print(
                f"EXPERT_REPLAY_INFEASIBLE task={args.task} seed={args.seed}",
                flush=True,
            )
            raise SystemExit(21)
    finally:
        close_safely(task_env)
    task_env.merge_pkl_to_hdf5_video()
    task_env.remove_data_cache()

    initial_dir = args.output_bundle / "initial_observations"
    prefix = f"episode{args.episode_index}"
    save_initial_observation(initial_dir, f"{prefix}_planning", planning_images)
    save_initial_observation(initial_dir, f"{prefix}_replay", replay_images)

    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    try:
        descriptions = generate_episode_descriptions(
            args.task, [episode_info["info"]], 100
        )[0]
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    instruction_dir = args.output_bundle / "instructions"
    instruction_dir.mkdir(parents=True, exist_ok=True)
    (instruction_dir / f"episode{args.episode_index}.json").write_text(
        json.dumps(descriptions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    scene_path = args.output_bundle / "scene_info.json"
    scene_db = json.loads(scene_path.read_text()) if scene_path.is_file() else {}
    scene_db[f"episode_{args.episode_index}"] = replay_info
    scene_path.write_text(
        json.dumps(scene_db, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata_dir = args.output_bundle / "pair_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "robotwin_local_expert_pair_episode_v1",
        "task": args.task,
        "task_config": args.task_config,
        "seed": args.seed,
        "episode_index": args.episode_index,
        "expert_planning_success": planning_success,
        "expert_replay_success": replay_success,
        "scene_state_comparison": state_comparison,
        "visual_metrics": visual_metrics(planning_images, replay_images),
        "planning_scene_state": planning_state,
        "replay_scene_state": replay_state,
        "expert_hdf5": str(
            (args.output_bundle / "data" / f"episode{args.episode_index}.hdf5").resolve()
        ),
        "expert_video": str(
            (args.output_bundle / "video" / f"episode{args.episode_index}.mp4").resolve()
        ),
    }
    metadata_path = metadata_dir / f"episode{args.episode_index}.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
