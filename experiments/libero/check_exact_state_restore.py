"""Check whether LIBERO supports deterministic exact-state counterfactual rollouts.

This is the hard precondition for the exact-state reward experiment: restoring a
saved MuJoCo state must reproduce both the anchor observation and the outcome of
an identical short action chunk.  The script only reads LIBERO and does not
modify its source tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)
from libero.libero import benchmark  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--num-states", type=int, default=3)
    parser.add_argument("--wait-steps", type=int, default=5)
    parser.add_argument("--exec-steps", type=int, default=8)
    parser.add_argument("--render-repeats", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=LIBERO_ENV_RESOLUTION)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("evaluate_results/exact_state_seed2042/state_restore_smoke.json"),
    )
    return parser.parse_args()


def _max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.shape != rhs.shape:
        return float("inf")
    return float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))))


def _image_diffs(lhs: dict[str, np.ndarray], rhs: dict[str, np.ndarray]) -> dict[str, float]:
    return {key: _max_abs_diff(lhs[key], rhs[key]) for key in sorted(lhs)}


def _image_metrics(
    lhs: dict[str, np.ndarray], rhs: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for key in sorted(lhs):
        left = np.asarray(lhs[key], dtype=np.float64)
        right = np.asarray(rhs[key], dtype=np.float64)
        delta = np.abs(left - right)
        mse = float(np.mean(np.square(left - right)))
        psnr = float("inf") if mse == 0.0 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
        metrics[key] = {
            "max_abs_diff": float(np.max(delta)),
            "mean_abs_diff": float(np.mean(delta)),
            "changed_fraction": float(np.mean(delta > 0.0)),
            "psnr_db": psnr,
        }
    return metrics


def _save_image_pair(
    output_dir: Path,
    state_index: int,
    pair_name: str,
    lhs: dict[str, np.ndarray],
    rhs: dict[str, np.ndarray],
) -> None:
    pair_dir = output_dir / f"state{state_index:02d}" / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)
    for camera_name in sorted(lhs):
        Image.fromarray(np.asarray(lhs[camera_name], dtype=np.uint8)).save(
            pair_dir / f"{camera_name}_first.png"
        )
        Image.fromarray(np.asarray(rhs[camera_name], dtype=np.uint8)).save(
            pair_dir / f"{camera_name}_second.png"
        )


def _fixed_action_chunk(exec_steps: int) -> np.ndarray:
    """Return a nontrivial, bounded chunk that exercises robot dynamics."""
    action = np.asarray([0.04, -0.025, 0.02, 0.015, -0.01, 0.02, -1.0], dtype=np.float64)
    return np.repeat(action[None], exec_steps, axis=0)


def _rollout_from_state(env: Any, state: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    obs = env.set_init_state(state.copy())
    rewards: list[float] = []
    dones: list[bool] = []
    for action in actions:
        obs, reward, done, _ = env.step(action.copy())
        rewards.append(float(reward))
        dones.append(bool(done))
    return {
        "state": env.get_sim_state().copy(),
        "images": get_libero_image(obs),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "dones": np.asarray(dones, dtype=np.bool_),
        "success": bool(env.check_success()),
    }


def _render_ensemble(
    env: Any, state: np.ndarray, render_repeats: int
) -> list[dict[str, np.ndarray]]:
    renders = []
    for _ in range(render_repeats):
        obs = env.set_init_state(state.copy())
        renders.append(get_libero_image(obs))
    return renders


def _save_render_ensembles(
    output_dir: Path,
    state_index: int,
    first_renders: list[dict[str, np.ndarray]],
    second_renders: list[dict[str, np.ndarray]],
) -> None:
    ensemble_dir = output_dir / f"state{state_index:02d}" / "replay_ensemble"
    for side, renders in (("first", first_renders), ("second", second_renders)):
        side_dir = ensemble_dir / side
        side_dir.mkdir(parents=True, exist_ok=True)
        for repeat_index, render in enumerate(renders):
            for camera_name, image in sorted(render.items()):
                Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
                    side_dir / f"repeat{repeat_index:02d}_{camera_name}.png"
                )


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

    suite_cls = benchmark.get_benchmark_dict()[args.task_suite]
    task_suite = suite_cls()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    if args.num_states > len(init_states):
        raise ValueError(
            f"Requested {args.num_states} states, but task only has {len(init_states)}."
        )

    env, task_description = get_libero_env(
        task,
        resolution=args.resolution,
        seed=args.seed,
    )
    actions = _fixed_action_chunk(args.exec_steps)
    records: list[dict[str, Any]] = []
    image_output_dir = args.output_json.parent / "state_restore_smoke_images"

    try:
        for state_index in range(args.num_states):
            env.reset()
            obs = env.set_init_state(init_states[state_index])
            for _ in range(args.wait_steps):
                obs, _, _, _ = env.step(get_libero_dummy_action())

            anchor_state = env.get_sim_state().copy()
            anchor_images = get_libero_image(obs)

            restored_obs = env.set_init_state(anchor_state.copy())
            restored_state = env.get_sim_state().copy()
            restored_images = get_libero_image(restored_obs)

            first = _rollout_from_state(env, anchor_state, actions)
            second = _rollout_from_state(env, anchor_state, actions)
            first_renders = _render_ensemble(env, first["state"], args.render_repeats)
            second_renders = _render_ensemble(env, second["state"], args.render_repeats)

            anchor_state_diff = _max_abs_diff(anchor_state, restored_state)
            anchor_image_diffs = _image_diffs(anchor_images, restored_images)
            anchor_image_metrics = _image_metrics(anchor_images, restored_images)
            replay_state_diff = _max_abs_diff(first["state"], second["state"])
            replay_image_diffs = _image_diffs(first["images"], second["images"])
            replay_image_metrics = _image_metrics(first["images"], second["images"])
            replay_reward_diff = _max_abs_diff(first["rewards"], second["rewards"])
            replay_done_equal = bool(np.array_equal(first["dones"], second["dones"]))
            replay_success_equal = first["success"] == second["success"]

            passed = bool(
                anchor_state_diff <= 1e-12
                and replay_state_diff <= 1e-10
                and max(
                    camera["mean_abs_diff"] for camera in replay_image_metrics.values()
                )
                <= 0.25
                and min(camera["psnr_db"] for camera in replay_image_metrics.values()) >= 45.0
                and replay_reward_diff == 0.0
                and replay_done_equal
                and replay_success_equal
            )
            record = {
                "state_index": state_index,
                "passed": passed,
                "anchor_state_max_abs_diff": anchor_state_diff,
                "anchor_image_max_abs_diffs": anchor_image_diffs,
                "anchor_image_metrics": anchor_image_metrics,
                "replay_state_max_abs_diff": replay_state_diff,
                "replay_image_max_abs_diffs": replay_image_diffs,
                "replay_image_metrics": replay_image_metrics,
                "replay_reward_max_abs_diff": replay_reward_diff,
                "replay_done_equal": replay_done_equal,
                "replay_success_equal": replay_success_equal,
            }
            records.append(record)
            _save_image_pair(
                image_output_dir,
                state_index,
                "anchor_restore",
                anchor_images,
                restored_images,
            )
            _save_image_pair(
                image_output_dir,
                state_index,
                "replay",
                first["images"],
                second["images"],
            )
            _save_render_ensembles(
                image_output_dir,
                state_index,
                first_renders,
                second_renders,
            )
            print(json.dumps(record, ensure_ascii=False))
    finally:
        env.close()

    result = {
        "schema_version": 1,
        "passed": all(record["passed"] for record in records),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "task_description": task_description,
        "seed": args.seed,
        "num_states": args.num_states,
        "wait_steps": args.wait_steps,
        "exec_steps": args.exec_steps,
        "render_repeats": args.render_repeats,
        "resolution": args.resolution,
        "action_chunk": actions.tolist(),
        "thresholds": {
            "anchor_state_max_abs_diff": 1e-12,
            "replay_state_max_abs_diff": 1e-10,
            "replay_image_mean_abs_diff": 0.25,
            "replay_image_min_psnr_db": 45.0,
            "replay_reward_max_abs_diff": 0.0,
        },
        "notes": {
            "anchor_images": (
                "Diagnostic only: the anchor observation is captured once and shared by all "
                "branches. A regenerated anchor render is not used by the experiment."
            ),
            "replay_images": (
                "Gate: repeated branch rollouts must have negligible rendering variation."
            ),
        },
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output_json}")
    if not result["passed"]:
        raise SystemExit("Exact-state restore smoke test failed; counterfactual collection is unsafe.")


if __name__ == "__main__":
    main()
