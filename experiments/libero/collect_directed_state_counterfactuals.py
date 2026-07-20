"""Collect matched toward/away LIBERO counterfactuals without model inference.

The collector reuses current observations and imagined goals from a validated
exact-state manifest.  It reconstructs the identical LIBERO model/controller
state, then executes policy, matched toward/away, and zero branches.  It never
trains or updates FastWAM and does not modify LIBERO source code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.collect_exact_state_counterfactuals import (  # noqa: E402
    _create_deterministic_env,
    _execute_actions_from_current_state,
    _render_endpoint_ensemble,
    _reset_controller_without_resetting_model,
)
from experiments.libero.imagination_reward_utils import (  # noqa: E402
    build_matched_direction_action_branches,
)
from experiments.libero.libero_utils import get_libero_dummy_action  # noqa: E402
from libero.libero import benchmark  # noqa: E402


BRANCH_NAMES = ("policy", "toward_bowl", "away_from_bowl", "zero")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-anchors", type=int, default=10)
    parser.add_argument("--translation-magnitude-cap", type=float, default=0.25)
    parser.add_argument("--eef-site", default="gripper0_grip_site")
    parser.add_argument("--target-body", default="akita_black_bowl_1_main")
    return parser.parse_args()


def _copy_camera_paths(
    source_paths: dict[str, str], output_dir: Path, prefix: str
) -> dict[str, str]:
    copied = {}
    for camera in ("agent", "wrist"):
        destination = output_dir / f"{prefix}_{camera}.png"
        shutil.copy2(source_paths[camera], destination)
        copied[camera] = str(destination.resolve())
    return copied


def _geometry(env: Any, *, eef_site: str, target_body: str) -> dict[str, Any]:
    sim = env.env.sim
    sim.forward()
    eef = np.asarray(sim.data.site_xpos[sim.model.site_name2id(eef_site)], dtype=np.float64)
    target = np.asarray(
        sim.data.body_xpos[sim.model.body_name2id(target_body)], dtype=np.float64
    )
    return {
        "eef_position": eef.tolist(),
        "target_position": target.tolist(),
        "eef_target_distance": float(np.linalg.norm(eef - target)),
    }


def _validate_source_manifest(source: dict[str, Any], num_anchors: int) -> None:
    if source.get("experiment") != "exact_state_counterfactual_reward_v2":
        raise ValueError("Source must be the validated exact-state Reward V2 manifest")
    if num_anchors <= 0 or num_anchors > len(source.get("anchors", [])):
        raise ValueError(
            f"num_anchors must be in [1, {len(source.get('anchors', []))}], got {num_anchors}"
        )
    if int(source.get("exec_steps", -1)) <= 0:
        raise ValueError("Source manifest has invalid exec_steps")


def main() -> None:
    args = _parse_args()
    start_time = time.time()
    source_path = args.source_manifest.resolve()
    source = json.loads(source_path.read_text())
    _validate_source_manifest(source, args.num_anchors)

    task_suite = benchmark.get_benchmark_dict()[source["task_suite"]]()
    task = task_suite.get_task(int(source["task_id"]))
    initial_states = task_suite.get_task_init_states(int(source["task_id"]))
    seed = int(source["seed"])
    wait_steps = int(source["wait_steps"])
    exec_steps = int(source["exec_steps"])
    render_repeats = int(source["render_repeats"])
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    env = _create_deterministic_env(task, seed=seed)
    anchor_records = []
    try:
        for source_anchor in source["anchors"][: args.num_anchors]:
            anchor_start = time.time()
            anchor_index = int(source_anchor["anchor_index"])
            anchor_dir = output_root / f"anchor{anchor_index:02d}"
            anchor_dir.mkdir(parents=True, exist_ok=True)
            anchor_state = np.load(source_anchor["anchor_state_path"])
            base_actions = np.load(source_anchor["base_actions_path"]).astype(np.float32)
            if base_actions.shape != (exec_steps, 7):
                raise ValueError(
                    f"Anchor {anchor_index} has unexpected base actions {base_actions.shape}"
                )

            _reset_controller_without_resetting_model(env)
            env.set_init_state(initial_states[anchor_index])
            for _ in range(wait_steps):
                env.step(get_libero_dummy_action())
            env.set_init_state(anchor_state.copy())
            reconstructed = env.get_sim_state().copy()
            reconstruction_error = float(np.max(np.abs(reconstructed - anchor_state)))
            if reconstruction_error > 1e-10:
                raise RuntimeError(
                    f"Anchor {anchor_index} reconstruction failed: {reconstruction_error}"
                )

            initial_geometry = _geometry(
                env, eef_site=args.eef_site, target_body=args.target_body
            )
            direction = np.asarray(initial_geometry["target_position"]) - np.asarray(
                initial_geometry["eef_position"]
            )
            direction /= np.linalg.norm(direction)
            branches = build_matched_direction_action_branches(
                base_actions,
                direction,
                translation_magnitude_cap=args.translation_magnitude_cap,
            )

            np.save(anchor_dir / "anchor_state.npy", anchor_state)
            np.save(anchor_dir / "base_actions.npy", base_actions)
            np.save(anchor_dir / "eef_to_target_unit_direction.npy", direction)
            current_paths = _copy_camera_paths(
                source_anchor["current_paths"], anchor_dir, "current"
            )
            goal_paths = _copy_camera_paths(
                source_anchor["predicted_goal_paths"], anchor_dir, "predicted_goal"
            )
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
                action_path = branch_dir / "actions.npy"
                final_state_path = branch_dir / "final_state.npy"
                np.save(action_path, actions)

                _reset_controller_without_resetting_model(env)
                env.set_init_state(initial_states[anchor_index])
                for _ in range(wait_steps):
                    env.step(get_libero_dummy_action())
                replayed_anchor = env.get_sim_state().copy()
                wait_replay_error = float(np.max(np.abs(replayed_anchor - anchor_state)))
                env.set_init_state(anchor_state.copy())
                reconstructed_anchor = env.get_sim_state().copy()
                reconstructed_error = float(
                    np.max(np.abs(reconstructed_anchor - anchor_state))
                )
                if reconstructed_error > 1e-10:
                    raise RuntimeError(
                        f"Branch {anchor_index}/{branch_name} reconstruction failed: "
                        f"{reconstructed_error}"
                    )
                branch_current_paths = _render_endpoint_ensemble(
                    env,
                    reconstructed_anchor,
                    repeats=render_repeats,
                    output_dir=branch_dir,
                    prefix="branch_current",
                )
                outcome = _execute_actions_from_current_state(env, actions)
                np.save(final_state_path, outcome["final_state"])
                final_geometry = _geometry(
                    env, eef_site=args.eef_site, target_body=args.target_body
                )
                distance_progress = (
                    initial_geometry["eef_target_distance"]
                    - final_geometry["eef_target_distance"]
                )
                actual_paths = _render_endpoint_ensemble(
                    env,
                    outcome["final_state"],
                    repeats=render_repeats,
                    output_dir=branch_dir,
                )
                branch_records.append(
                    {
                        "name": branch_name,
                        "noise_std": None,
                        "quality": float(distance_progress),
                        "success": outcome["success"],
                        "sim_rewards": outcome["rewards"],
                        "dones": outcome["dones"],
                        "reconstructed_anchor_state_max_abs_diff": reconstructed_error,
                        "wait_replay_state_max_abs_diff_before_exact_restore": (
                            wait_replay_error
                        ),
                        "current_paths": branch_current_paths,
                        "actions_path": str(action_path.resolve()),
                        "final_state_path": str(final_state_path.resolve()),
                        "actual_paths": actual_paths,
                        "geometry": {
                            "initial": initial_geometry,
                            "final": final_geometry,
                            "eef_target_distance_progress": float(distance_progress),
                        },
                    }
                )

            anchor_record = {
                "anchor_index": anchor_index,
                "source_anchor_index": anchor_index,
                "anchor_state_path": str((anchor_dir / "anchor_state.npy").resolve()),
                "base_actions_path": str((anchor_dir / "base_actions.npy").resolve()),
                "direction_path": str(
                    (anchor_dir / "eef_to_target_unit_direction.npy").resolve()
                ),
                "current_paths": current_paths,
                "reward_current_paths": reward_current_paths,
                "predicted_goal_paths": goal_paths,
                "initial_geometry": initial_geometry,
                "branches": branch_records,
                "duration_seconds": time.time() - anchor_start,
            }
            anchor_records.append(anchor_record)
            (anchor_dir / "metadata.json").write_text(
                json.dumps(anchor_record, indent=2) + "\n"
            )
            print(
                f"anchor {anchor_index + 1}/{args.num_anchors} complete "
                f"({anchor_record['duration_seconds']:.2f}s)"
            )
    finally:
        env.close()

    manifest = {
        "schema_version": 1,
        "experiment": "directed_exact_state_counterfactual_reward",
        "source_manifest": str(source_path),
        "task_suite": source["task_suite"],
        "task_id": source["task_id"],
        "task_description": source["task_description"],
        "seed": seed,
        "num_states": len(anchor_records),
        "wait_steps": wait_steps,
        "exec_steps": exec_steps,
        "render_repeats": render_repeats,
        "translation_magnitude_cap": float(args.translation_magnitude_cap),
        "eef_site": args.eef_site,
        "target_body": args.target_body,
        "branch_names": list(BRANCH_NAMES),
        "action_source": "source_policy_plus_matched_geometric_direction",
        "goal_source": "copied_source_infer_joint_last_aligned_frame",
        "branch_initialization": source["branch_initialization"],
        "checkpoint": source["checkpoint"],
        "dataset_stats_path": source["dataset_stats_path"],
        "total_model_inferences": 0,
        "total_short_rollouts": len(anchor_records) * len(BRANCH_NAMES),
        "total_simulator_action_steps": (
            len(anchor_records) * len(BRANCH_NAMES) * exec_steps
        ),
        "duration_seconds": time.time() - start_time,
        "anchors": anchor_records,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"collection complete in {manifest['duration_seconds']:.2f}s")


if __name__ == "__main__":
    main()
