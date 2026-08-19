#!/usr/bin/env python3
"""Collect exact-state counterfactual action branches from RoboMimic Can.

Each sample restores one saved intermediate simulator state, executes the
recorded action sequence, restores the *same* state, and executes a sequence
whose first few actions receive a bounded perturbation. The output is an
incrementally committed HDF5 file suitable for pairwise value learning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


NUMERIC_FIELDS: dict[str, tuple[Any, tuple[int, ...]]] = {
    "source_step": (np.int32, ()),
    "source_episode_success": (np.uint8, ()),
    "source_branch_contains_success": (np.uint8, ()),
    "noise_sigma": (np.float32, ()),
    "noise_seed": (np.int64, ()),
    "intervention_steps": (np.int16, ()),
    "horizon": (np.int16, ()),
    "base_reward_sum": (np.float32, ()),
    "candidate_reward_sum": (np.float32, ()),
    "base_score": (np.float32, ()),
    "candidate_score": (np.float32, ()),
    "delta_score": (np.float32, ()),
    "base_success": (np.uint8, ()),
    "candidate_success": (np.uint8, ()),
    "label": (np.int8, ()),
    "informative": (np.uint8, ()),
    "restore_linf": (np.float64, ()),
    "branch_initial_state_linf": (np.float64, ()),
    "stored_state_linf": (np.float64, ()),
    "branch_final_state_linf": (np.float64, ()),
}


def _decode_names(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _success_value(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value["task"])
    return bool(value)


def _select_source(
    *,
    rng: np.random.Generator,
    demo_names: list[str],
    action_lengths: dict[str, int],
    horizon: int,
    late_state_fraction: float,
) -> tuple[str, int]:
    name = demo_names[int(rng.integers(0, len(demo_names)))]
    max_step = action_lengths[name] - horizon
    if max_step < 0:
        raise ValueError(f"{name} has fewer than {horizon} actions")
    if rng.random() < late_state_fraction:
        low = max(0, max_step - 2 * horizon)
    else:
        low = 0
    step = int(rng.integers(low, max_step + 1))
    return name, step


def _make_candidate_actions(
    base_actions: np.ndarray,
    *,
    rng: np.random.Generator,
    sigma: float,
    intervention_steps: int,
    perturb_gripper: bool,
) -> np.ndarray:
    candidate = np.array(base_actions, dtype=np.float64, copy=True)
    span = min(intervention_steps, len(candidate))
    perturb_dimensions = candidate.shape[1] if perturb_gripper else candidate.shape[1] - 1
    noise = rng.normal(0.0, sigma, size=(span, perturb_dimensions))
    candidate[:span, :perturb_dimensions] += noise
    np.clip(candidate, -1.0, 1.0, out=candidate)
    return candidate.astype(base_actions.dtype, copy=False)


def _rollout_branch(
    env: Any,
    *,
    model: str,
    episode_initial_state: np.ndarray,
    prefix_actions: np.ndarray,
    branch_actions: np.ndarray,
) -> dict[str, Any]:
    """Rebuild controller history before executing the counterfactual branch."""

    env.reset_to({"model": model, "states": episode_initial_state})
    restored = np.asarray(env.get_state()["states"])
    restore_linf = float(np.max(np.abs(restored - episode_initial_state), initial=0.0))
    for action in prefix_actions:
        env.step(action)
    branch_initial_state = np.asarray(env.get_state()["states"])
    rewards: list[float] = []
    success = _success_value(env.is_success())
    for action in branch_actions:
        _, reward, _, info = env.step(action)
        rewards.append(float(reward))
        success = success or _success_value(info.get("is_success", env.is_success()))
    return {
        "restore_linf": restore_linf,
        "reward_sum": float(sum(rewards)),
        "success": success,
        "branch_initial_state": branch_initial_state,
        "final_state": np.asarray(env.get_state()["states"]),
    }


def _ensure_output_datasets(
    output: h5py.File,
    *,
    state_dim: int,
    action_dim: int,
) -> h5py.Group:
    samples = output.require_group("samples")
    committed = int(output.attrs.get("samples_committed", 0))
    string_dtype = h5py.string_dtype(encoding="utf-8")
    definitions: dict[str, tuple[Any, tuple[int, ...]]] = {
        **NUMERIC_FIELDS,
        "source_demo": (string_dtype, ()),
        "source_split": (string_dtype, ()),
        "initial_state": (np.float64, (state_dim,)),
        "base_action": (np.float32, (action_dim,)),
        "candidate_action": (np.float32, (action_dim,)),
        "residual_action": (np.float32, (action_dim,)),
        "base_final_state": (np.float64, (state_dim,)),
        "candidate_final_state": (np.float64, (state_dim,)),
    }
    for name, (dtype, tail_shape) in definitions.items():
        if name not in samples:
            samples.create_dataset(
                name,
                shape=(0, *tail_shape),
                maxshape=(None, *tail_shape),
                chunks=True,
                dtype=dtype,
            )
        elif len(samples[name]) != committed:
            samples[name].resize((committed, *samples[name].shape[1:]))
    return samples


def _append_sample(samples: h5py.Group, index: int, sample: dict[str, Any]) -> None:
    for name, value in sample.items():
        dataset = samples[name]
        dataset.resize((index + 1, *dataset.shape[1:]))
        dataset[index] = value


def _set_or_validate_attribute(output: h5py.File, name: str, expected: Any) -> None:
    """Prevent a resumed run from silently mixing incompatible settings."""

    if name not in output.attrs:
        output.attrs[name] = expected
        return
    actual = output.attrs[name]
    if isinstance(expected, float):
        matches = bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(f"Output attribute {name!r} is {actual!r}, expected {expected!r}")


def _summary(samples: h5py.Group, *, target_samples: int, margin: float) -> dict[str, Any]:
    count = len(samples["label"])
    labels = np.asarray(samples["label"])
    restore = np.asarray(samples["restore_linf"])
    branch_linf = np.asarray(samples["branch_final_state_linf"])
    base_success = np.asarray(samples["base_success"], dtype=bool)
    candidate_success = np.asarray(samples["candidate_success"], dtype=bool)
    expected_success = np.asarray(samples["source_branch_contains_success"], dtype=bool)
    reproduced_success = expected_success & base_success
    finite = all(
        bool(np.all(np.isfinite(np.asarray(samples[name]))))
        for name in ("base_score", "candidate_score", "delta_score", "restore_linf")
    )
    informative_count = int(np.count_nonzero(labels))
    return {
        "samples_collected": count,
        "target_samples": target_samples,
        "complete": count >= target_samples,
        "base_better": int(np.count_nonzero(labels < 0)),
        "candidate_better": int(np.count_nonzero(labels > 0)),
        "ties": int(np.count_nonzero(labels == 0)),
        "informative_count": informative_count,
        "informative_rate": float(informative_count / count) if count else 0.0,
        "score_margin": margin,
        "base_success_count": int(np.count_nonzero(base_success)),
        "candidate_success_count": int(np.count_nonzero(candidate_success)),
        "success_outcome_changed": int(np.count_nonzero(base_success != candidate_success)),
        "expected_success_count": int(np.count_nonzero(expected_success)),
        "reproduced_success_count": int(np.count_nonzero(reproduced_success)),
        "success_replay_rate": (
            float(np.count_nonzero(reproduced_success) / np.count_nonzero(expected_success))
            if np.count_nonzero(expected_success)
            else None
        ),
        "max_restore_linf": float(np.max(restore, initial=0.0)),
        "max_branch_initial_state_linf": float(
            np.max(np.asarray(samples["branch_initial_state_linf"]), initial=0.0)
        ),
        "max_stored_state_linf": float(
            np.max(np.asarray(samples["stored_state_linf"]), initial=0.0)
        ),
        "branch_diverged_count": int(np.count_nonzero(branch_linf > 1e-10)),
        "branch_diverged_rate": float(np.mean(branch_linf > 1e-10)) if count else 0.0,
        "all_finite": finite,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    from robomimic.utils.env_utils import create_env_from_metadata
    from robomimic.utils.obs_utils import initialize_obs_utils_with_obs_specs

    initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )

    source_path = args.dataset.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or output_path.with_suffix(".summary.json")

    with h5py.File(source_path, "r") as source:
        env_meta = json.loads(source["data"].attrs["env_args"])
        env_meta["env_kwargs"].pop("env_lang", None)
        # Dynamics stay unchanged; dense shaping makes short branches rankable.
        env_meta["env_kwargs"]["reward_shaping"] = True

        split_by_demo: dict[str, str] = {}
        for split in args.source_splits:
            if "mask" not in source or split not in source["mask"]:
                raise ValueError(f"Source dataset has no mask/{split}")
            for name in _decode_names(np.asarray(source["mask"][split])):
                previous = split_by_demo.setdefault(name, split)
                if previous != split:
                    raise ValueError(f"{name} occurs in both {previous} and {split}")
        action_lengths = {name: len(source["data"][name]["actions"]) for name in split_by_demo}
        eligible = sorted(
            (name for name, length in action_lengths.items() if length >= args.horizon),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if not eligible:
            raise ValueError("No demonstrations are long enough for the requested horizon")

        first_group = source["data"][eligible[0]]
        state_dim = int(first_group["states"].shape[1])
        action_dim = int(first_group["actions"].shape[1])
        env = create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        started = time.monotonic()
        try:
            with h5py.File(output_path, "a") as output:
                run_attributes = {
                    "format": "fastwam.robomimic_counterfactual.v2_prefix_replay",
                    "source_dataset": str(source_path),
                    "seed": args.seed,
                    "horizon": args.horizon,
                    "intervention_steps": args.intervention_steps,
                    "success_bonus": args.success_bonus,
                    "score_margin": args.score_margin,
                    "noise_sigmas": json.dumps(args.noise_sigmas),
                    "reward_shaping": True,
                }
                for name, expected in run_attributes.items():
                    _set_or_validate_attribute(output, name, expected)
                output.attrs.setdefault("samples_committed", 0)
                samples = _ensure_output_datasets(output, state_dim=state_dim, action_dim=action_dim)
                committed = int(output.attrs["samples_committed"])
                if committed > args.num_samples:
                    raise ValueError(
                        f"Output already has {committed} samples, more than target {args.num_samples}"
                    )

                for sample_index in range(committed, args.num_samples):
                    noise_seed = args.seed + sample_index
                    rng = np.random.default_rng(noise_seed)
                    demo_name, source_step = _select_source(
                        rng=rng,
                        demo_names=eligible,
                        action_lengths=action_lengths,
                        horizon=args.horizon,
                        late_state_fraction=args.late_state_fraction,
                    )
                    group = source["data"][demo_name]
                    episode_initial_state = np.asarray(group["states"][0])
                    stored_branch_state = np.asarray(group["states"][source_step])
                    prefix_actions = np.asarray(group["actions"][:source_step])
                    base_actions = np.asarray(
                        group["actions"][source_step : source_step + args.horizon]
                    )
                    sigma = float(args.noise_sigmas[int(rng.integers(0, len(args.noise_sigmas)))])
                    candidate_actions = _make_candidate_actions(
                        base_actions,
                        rng=rng,
                        sigma=sigma,
                        intervention_steps=args.intervention_steps,
                        perturb_gripper=args.perturb_gripper,
                    )
                    model = str(group.attrs["model_file"])
                    base = _rollout_branch(
                        env,
                        model=model,
                        episode_initial_state=episode_initial_state,
                        prefix_actions=prefix_actions,
                        branch_actions=base_actions,
                    )
                    candidate = _rollout_branch(
                        env,
                        model=model,
                        episode_initial_state=episode_initial_state,
                        prefix_actions=prefix_actions,
                        branch_actions=candidate_actions,
                    )
                    base_score = (
                        args.success_bonus * float(base["success"])
                        + base["reward_sum"] / len(base_actions)
                    )
                    candidate_score = (
                        args.success_bonus * float(candidate["success"])
                        + candidate["reward_sum"] / len(candidate_actions)
                    )
                    delta_score = candidate_score - base_score
                    if delta_score > args.score_margin:
                        label = 1
                    elif delta_score < -args.score_margin:
                        label = -1
                    else:
                        label = 0
                    sample = {
                        "source_demo": demo_name,
                        "source_split": split_by_demo[demo_name],
                        "source_step": source_step,
                        "source_episode_success": int(np.max(group["rewards"][()]) > 0),
                        "source_branch_contains_success": int(
                            np.max(group["rewards"][source_step : source_step + args.horizon]) > 0
                        ),
                        "noise_sigma": sigma,
                        "noise_seed": noise_seed,
                        "intervention_steps": min(args.intervention_steps, len(base_actions)),
                        "horizon": len(base_actions),
                        "initial_state": base["branch_initial_state"],
                        "base_action": base_actions[0],
                        "candidate_action": candidate_actions[0],
                        "residual_action": candidate_actions[0] - base_actions[0],
                        "base_reward_sum": base["reward_sum"],
                        "candidate_reward_sum": candidate["reward_sum"],
                        "base_score": base_score,
                        "candidate_score": candidate_score,
                        "delta_score": delta_score,
                        "base_success": int(base["success"]),
                        "candidate_success": int(candidate["success"]),
                        "label": label,
                        "informative": int(label != 0),
                        "restore_linf": max(base["restore_linf"], candidate["restore_linf"]),
                        "branch_initial_state_linf": float(
                            np.max(
                                np.abs(
                                    base["branch_initial_state"]
                                    - candidate["branch_initial_state"]
                                ),
                                initial=0.0,
                            )
                        ),
                        "stored_state_linf": float(
                            np.max(
                                np.abs(base["branch_initial_state"] - stored_branch_state),
                                initial=0.0,
                            )
                        ),
                        "base_final_state": base["final_state"],
                        "candidate_final_state": candidate["final_state"],
                        "branch_final_state_linf": float(
                            np.max(
                                np.abs(base["final_state"] - candidate["final_state"]),
                                initial=0.0,
                            )
                        ),
                    }
                    _append_sample(samples, sample_index, sample)
                    output.attrs.modify("samples_committed", sample_index + 1)

                    if (sample_index + 1) % args.flush_every == 0 or sample_index + 1 == args.num_samples:
                        output.flush()
                        summary = _summary(
                            samples,
                            target_samples=args.num_samples,
                            margin=args.score_margin,
                        )
                        elapsed = time.monotonic() - started
                        session_samples = sample_index + 1 - committed
                        rate = session_samples / elapsed if elapsed > 0 else 0.0
                        summary.update(
                            {
                                "output_path": str(output_path),
                                "elapsed_seconds_this_session": elapsed,
                                "samples_per_second_this_session": rate,
                                "eta_seconds": (
                                    (args.num_samples - sample_index - 1) / rate if rate > 0 else None
                                ),
                            }
                        )
                        _write_json_atomic(summary_path, summary)
                        print(json.dumps(summary, ensure_ascii=False), flush=True)

                output.attrs["complete"] = True
                output.flush()
                final_summary = _summary(
                    samples,
                    target_samples=args.num_samples,
                    margin=args.score_margin,
                )
                final_summary.update({"output_path": str(output_path), "summary_path": str(summary_path)})
                _write_json_atomic(summary_path, final_summary)
        finally:
            close = getattr(getattr(env, "env", None), "close", None)
            if callable(close):
                close()
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--intervention-steps", type=int, default=3)
    parser.add_argument("--noise-sigmas", type=float, nargs="+", default=[0.1, 0.2, 0.35])
    parser.add_argument("--source-splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--late-state-fraction", type=float, default=0.5)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--score-margin", type=float, default=1e-4)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--perturb-gripper", action="store_true")
    parser.add_argument("--require-smoke-quality", action="store_true")
    args = parser.parse_args()

    if args.num_samples <= 0 or args.horizon <= 0 or args.intervention_steps <= 0:
        parser.error("num-samples, horizon, and intervention-steps must be positive")
    if not 0.0 <= args.late_state_fraction <= 1.0:
        parser.error("late-state-fraction must be in [0, 1]")
    if not args.noise_sigmas or any(not math.isfinite(value) or value <= 0 for value in args.noise_sigmas):
        parser.error("noise-sigmas must contain positive finite values")

    summary = collect(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.require_smoke_quality:
        passed = (
            summary["complete"]
            and summary["all_finite"]
            and summary["max_restore_linf"] <= 1e-10
            and summary["max_branch_initial_state_linf"] <= 1e-10
            and summary["branch_diverged_rate"] >= 0.9
            and summary["informative_count"] > 0
            and (
                summary["expected_success_count"] == 0
                or summary["success_replay_rate"] >= 0.9
            )
        )
        if not passed:
            raise SystemExit("Counterfactual collection smoke-quality gate failed")


if __name__ == "__main__":
    main()
