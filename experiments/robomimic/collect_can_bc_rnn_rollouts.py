#!/usr/bin/env python3
"""Collect resumable online trajectories from a trained RoboMimic BC-RNN policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from experiments.robomimic.evaluate_can_bc_rnn_base import sha256


def episode_split(index: int, *, valid_every: int) -> str:
    if valid_every < 2:
        raise ValueError("valid_every must be at least 2")
    return "valid" if index % valid_every == valid_every - 1 else "train"


def branch_steps(
    action_count: int, *, warmup: int, horizon: int, stride: int
) -> np.ndarray:
    if min(horizon, stride) <= 0 or warmup < 0:
        raise ValueError("warmup must be nonnegative; horizon and stride must be positive")
    stop = action_count - horizon + 1
    if stop <= warmup:
        return np.empty(0, dtype=np.int32)
    return np.arange(warmup, stop, stride, dtype=np.int32)


def _set_or_validate_attr(group: h5py.Group, name: str, expected: Any) -> None:
    if name not in group.attrs:
        group.attrs[name] = expected
    elif group.attrs[name] != expected:
        raise ValueError(
            f"Existing collection attribute {name}={group.attrs[name]!r}, expected {expected!r}"
        )


def _replace_mask(output: h5py.File, split: str, names: list[str]) -> None:
    masks = output.require_group("mask")
    if split in masks:
        del masks[split]
    masks.create_dataset(
        split,
        data=np.asarray(names, dtype=h5py.string_dtype(encoding="utf-8")),
    )


def _write_trajectory(
    data: h5py.Group,
    *,
    demo_name: str,
    trajectory: dict[str, Any],
    stats: dict[str, float],
    seed: int,
    split: str,
) -> None:
    if demo_name in data:
        del data[demo_name]
    demo = data.create_group(demo_name)
    for key in ("actions", "states", "rewards", "dones"):
        demo.create_dataset(key, data=np.asarray(trajectory[key]))
    for observation_group in ("obs", "next_obs"):
        observations = demo.create_group(observation_group)
        for key, value in trajectory[observation_group].items():
            observations.create_dataset(key, data=np.asarray(value))
    initial_state = trajectory["initial_state_dict"]
    if "model" in initial_state:
        demo.attrs["model_file"] = initial_state["model"]
    demo.attrs["num_samples"] = len(trajectory["actions"])
    demo.attrs["episode_seed"] = seed
    demo.attrs["source_split"] = split
    demo.attrs["success"] = int(stats["Success_Rate"])


def _rebuild_masks(output: h5py.File, committed: int, *, valid_every: int) -> None:
    names = [f"demo_{index}" for index in range(committed)]
    for split in ("train", "valid"):
        _replace_mask(
            output,
            split,
            [
                name
                for index, name in enumerate(names)
                if episode_split(index, valid_every=valid_every) == split
            ],
        )


def write_state_index_atomic(
    path: Path,
    output: h5py.File,
    *,
    committed: int,
    valid_every: int,
    warmup: int,
    horizon: int,
    stride: int,
) -> int:
    demos: list[str] = []
    splits: list[str] = []
    steps: list[int] = []
    for index in range(committed):
        name = f"demo_{index}"
        count = len(output["data"][name]["actions"])
        selected = branch_steps(count, warmup=warmup, horizon=horizon, stride=stride)
        demos.extend([name] * len(selected))
        splits.extend([episode_split(index, valid_every=valid_every)] * len(selected))
        steps.extend(selected.tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_demo=np.asarray(demos, dtype="U32"),
            source_split=np.asarray(splits, dtype="U5"),
            source_step=np.asarray(steps, dtype=np.int32),
        )
    os.replace(temporary, path)
    return len(steps)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _summary(output: h5py.File, *, target: int, state_count: int) -> dict[str, Any]:
    committed = int(output.attrs["episodes_committed"])
    success = [int(output["data"][f"demo_{index}"].attrs["success"]) for index in range(committed)]
    train = len(output["mask"]["train"])
    valid = len(output["mask"]["valid"])
    return {
        "episodes_collected": committed,
        "target_episodes": target,
        "complete": committed >= target,
        "successes": int(sum(success)),
        "success_rate": float(np.mean(success)) if success else 0.0,
        "train_episodes": train,
        "valid_episodes": valid,
        "branchable_states": state_count,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.scripts.run_trained_agent import rollout

    checkpoint = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    state_index_path = args.state_index.expanduser().resolve()
    summary_path = args.summary_json or output_path.with_suffix(".summary.json")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = TorchUtils.get_torch_device(try_to_use_cuda=not args.cpu)
    policy, checkpoint_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint), device=device, verbose=True
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint_dict,
        render=False,
        render_offscreen=False,
        verbose=True,
    )
    try:
        with h5py.File(output_path, "a") as output:
            data = output.require_group("data")
            attributes = {
                "format": "fastwam.robomimic_bc_rnn_rollouts.v1",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "base_seed": args.seed,
                "target_episodes": args.episodes,
                "horizon": args.horizon,
                "valid_every": args.valid_every,
                "branch_warmup": args.branch_warmup,
                "branch_horizon": args.branch_horizon,
                "branch_stride": args.branch_stride,
            }
            for name, expected in attributes.items():
                _set_or_validate_attr(output, name, expected)
            output.attrs.setdefault("episodes_committed", 0)
            data.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
            committed = int(output.attrs["episodes_committed"])
            for name in list(data.keys()):
                if name.startswith("demo_") and int(name.rsplit("_", 1)[1]) >= committed:
                    del data[name]
            _rebuild_masks(output, committed, valid_every=args.valid_every)

            for index in range(committed, args.episodes):
                episode_seed = args.seed + index
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                stats, trajectory = rollout(
                    policy=policy,
                    env=env,
                    horizon=args.horizon,
                    return_obs=True,
                )
                split = episode_split(index, valid_every=args.valid_every)
                _write_trajectory(
                    data,
                    demo_name=f"demo_{index}",
                    trajectory=trajectory,
                    stats=stats,
                    seed=episode_seed,
                    split=split,
                )
                output.attrs.modify("episodes_committed", index + 1)
                _rebuild_masks(output, index + 1, valid_every=args.valid_every)
                output.flush()
                state_count = write_state_index_atomic(
                    state_index_path,
                    output,
                    committed=index + 1,
                    valid_every=args.valid_every,
                    warmup=args.branch_warmup,
                    horizon=args.branch_horizon,
                    stride=args.branch_stride,
                )
                report = _summary(output, target=args.episodes, state_count=state_count)
                report.update(
                    {
                        "output": str(output_path),
                        "state_index": str(state_index_path),
                        "device": str(device),
                    }
                )
                _write_json_atomic(summary_path, report)
                print(json.dumps(report, ensure_ascii=False), flush=True)
            final_committed = int(output.attrs["episodes_committed"])
            state_count = write_state_index_atomic(
                state_index_path,
                output,
                committed=final_committed,
                valid_every=args.valid_every,
                warmup=args.branch_warmup,
                horizon=args.branch_horizon,
                stride=args.branch_stride,
            )
            report = _summary(output, target=args.episodes, state_count=state_count)
            report.update(
                {
                    "output": str(output_path),
                    "state_index": str(state_index_path),
                    "device": str(device),
                }
            )
            output.attrs["complete"] = report["complete"]
            output.flush()
            _write_json_atomic(summary_path, report)
            return report
    finally:
        close = getattr(getattr(env, "env", env), "close", None)
        if callable(close):
            close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--valid-every", type=int, default=5)
    parser.add_argument("--branch-warmup", type=int, default=10)
    parser.add_argument("--branch-horizon", type=int, default=20)
    parser.add_argument("--branch-stride", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if min(args.episodes, args.horizon, args.branch_horizon, args.branch_stride) <= 0:
        parser.error("episode count, horizons, and stride must be positive")
    if args.branch_warmup < 0 or args.valid_every < 2:
        parser.error("branch warmup must be nonnegative and valid-every at least 2")
    report = collect(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
