#!/usr/bin/env python3
"""Render wrist RGB and record deployable proprio at collected branch states."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROPRIO_KEYS = (
    "robot0_joint_pos",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def group_pending_states(
    demos: np.ndarray,
    steps: np.ndarray,
    completed: np.ndarray,
    *,
    limit: int | None = None,
) -> dict[str, list[tuple[int, int]]]:
    pending = np.flatnonzero(~completed.astype(bool))
    if limit is not None:
        pending = pending[:limit]
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index in pending:
        grouped[str(demos[index])].append((int(steps[index]), int(index)))
    return {
        demo: sorted(entries)
        for demo, entries in sorted(grouped.items(), key=lambda item: item[0])
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _ensure_output(
    output: h5py.File,
    *,
    count: int,
    image_height: int,
    image_width: int,
    proprio_dim: int,
) -> h5py.Group:
    states = output.require_group("states")
    string_dtype = h5py.string_dtype("utf-8")
    definitions = {
        "source_demo": (string_dtype, (count,)),
        "source_split": (string_dtype, (count,)),
        "source_step": (np.int32, (count,)),
        "wrist_rgb": (np.uint8, (count, image_height, image_width, 3)),
        "proprio": (np.float32, (count, proprio_dim)),
        "state_linf": (np.float64, (count,)),
        "completed": (np.uint8, (count,)),
    }
    for name, (dtype, shape) in definitions.items():
        if name not in states:
            states.create_dataset(name, shape=shape, dtype=dtype, chunks=True)
        elif states[name].shape != shape:
            raise ValueError(f"Existing {name} shape {states[name].shape} != {shape}")
    return states


def collect(args: argparse.Namespace) -> dict[str, Any]:
    from robomimic.utils.env_utils import create_env_from_metadata
    from robomimic.utils.obs_utils import initialize_obs_utils_with_obs_specs

    initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )
    output_path = args.output.expanduser().resolve()
    summary_path = args.summary_json or output_path.with_suffix(".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.collection, "r") as collection:
        if not bool(collection.attrs.get("complete", False)):
            raise ValueError("Symmetric branch collection is incomplete")
        samples = collection["states"]
        count = int(collection.attrs["states_committed"])
        demos = samples["source_demo"].asstr()[...]
        splits = samples["source_split"].asstr()[...]
        steps = np.asarray(samples["source_step"], dtype=np.int32)
        expected_states = np.asarray(samples["branch_state"], dtype=np.float64)
        source_path = Path(str(collection.attrs["source_dataset"]))

        with h5py.File(source_path, "r") as source, h5py.File(output_path, "a") as output:
            first_demo = source["data"][demos[0]]
            proprio_dim = sum(first_demo["obs"][key].shape[1] for key in PROPRIO_KEYS)
            expected_attrs = {
                "format": "fastwam.robomimic_deployable_observations.v1",
                "source_collection": str(args.collection.resolve()),
                "source_dataset": str(source_path.resolve()),
                "camera_name": args.camera_name,
                "image_height": args.image_height,
                "image_width": args.image_width,
                "proprio_keys": json.dumps(PROPRIO_KEYS),
                "proprio_dim": proprio_dim,
                "states_total": count,
            }
            for name, expected in expected_attrs.items():
                if name in output.attrs and output.attrs[name] != expected:
                    raise ValueError(f"Output attribute {name} does not match requested run")
                output.attrs[name] = expected
            states = _ensure_output(
                output,
                count=count,
                image_height=args.image_height,
                image_width=args.image_width,
                proprio_dim=proprio_dim,
            )
            if not np.any(states["completed"]):
                states["source_demo"][:] = demos
                states["source_split"][:] = splits
                states["source_step"][:] = steps

            grouped = group_pending_states(
                demos,
                steps,
                np.asarray(states["completed"]),
                limit=args.max_states,
            )
            env_meta = json.loads(source["data"].attrs["env_args"])
            env_meta["env_kwargs"].pop("env_lang", None)
            env_meta["env_kwargs"]["reward_shaping"] = True
            env = create_env_from_metadata(
                env_meta=env_meta,
                render=False,
                render_offscreen=True,
                use_image_obs=False,
            )
            try:
                for demo_ordinal, (demo, entries) in enumerate(grouped.items(), start=1):
                    group = source["data"][demo]
                    obs = env.reset_to(
                        {
                            "model": str(group.attrs["model_file"]),
                            "states": np.asarray(group["states"][0]),
                        }
                    )
                    cursor = 0
                    actions = np.asarray(group["actions"])
                    for step, index in entries:
                        while cursor < step:
                            obs, _, _, _ = env.step(actions[cursor])
                            cursor += 1
                        actual_state = np.asarray(env.get_state()["states"])
                        linf = float(
                            np.max(np.abs(actual_state - expected_states[index]), initial=0.0)
                        )
                        if linf > args.state_tolerance:
                            raise RuntimeError(
                                f"State mismatch at {demo}:{step}: {linf} > {args.state_tolerance}"
                            )
                        image = env.render(
                            mode="rgb_array",
                            height=args.image_height,
                            width=args.image_width,
                            camera_name=args.camera_name,
                        )
                        proprio = np.concatenate(
                            [np.asarray(obs[key]).reshape(-1) for key in PROPRIO_KEYS]
                        ).astype(np.float32)
                        if image.shape != (args.image_height, args.image_width, 3):
                            raise RuntimeError(f"Unexpected rendered image shape: {image.shape}")
                        if not np.all(np.isfinite(proprio)):
                            raise RuntimeError("Non-finite proprio observation")
                        states["wrist_rgb"][index] = image
                        states["proprio"][index] = proprio
                        states["state_linf"][index] = linf
                        states["completed"][index] = 1
                    output.flush()
                    completed_count = int(np.count_nonzero(states["completed"]))
                    print(
                        json.dumps(
                            {
                                "demo": f"{demo_ordinal}/{len(grouped)}",
                                "source_demo": demo,
                                "states_completed": completed_count,
                                "states_total": count,
                            }
                        ),
                        flush=True,
                    )
            finally:
                close = getattr(getattr(env, "env", None), "close", None)
                if callable(close):
                    close()

            completed = np.asarray(states["completed"], dtype=bool)
            completed_count = int(np.count_nonzero(completed))
            report = {
                "output_path": str(output_path),
                "states_completed": completed_count,
                "states_total": count,
                "complete": completed_count == count,
                "camera_name": args.camera_name,
                "image_shape": [args.image_height, args.image_width, 3],
                "proprio_dim": proprio_dim,
                "max_state_linf": (
                    float(np.max(np.asarray(states["state_linf"])[completed], initial=0.0))
                    if completed_count
                    else 0.0
                ),
                "all_images_nonconstant": bool(
                    all(np.ptp(states["wrist_rgb"][index]) > 0 for index in np.flatnonzero(completed))
                ),
                "all_proprio_finite": bool(np.all(np.isfinite(states["proprio"][completed]))),
            }
            output.attrs["complete"] = report["complete"]
            output.flush()
            _write_json_atomic(summary_path, report)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--camera-name", default="robot0_eye_in_hand")
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--state-tolerance", type=float, default=1e-10)
    parser.add_argument("--max-states", type=int)
    args = parser.parse_args()
    if min(args.image_height, args.image_width) <= 0:
        parser.error("image dimensions must be positive")
    collect(args)


if __name__ == "__main__":
    main()
