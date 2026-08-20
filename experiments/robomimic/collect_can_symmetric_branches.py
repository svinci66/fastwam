#!/usr/bin/env python3
"""Collect multiple symmetric local action branches from each RoboMimic state."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from experiments.robomimic.collect_can_counterfactual_branches import _rollout_branch


def make_symmetric_directions(
    *,
    seed: int,
    direction_pairs: int,
    intervention_steps: int,
    action_dim: int,
    delta: float,
    perturb_gripper: bool = False,
) -> np.ndarray:
    perturb_dims = action_dim if perturb_gripper else action_dim - 1
    vector_dim = intervention_steps * perturb_dims
    if direction_pairs > vector_dim:
        raise ValueError("direction-pairs exceeds the local perturbation dimension")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(vector_dim, direction_pairs))
    orthogonal, _ = np.linalg.qr(matrix)
    directions = orthogonal.T.reshape(direction_pairs, intervention_steps, perturb_dims)
    directions /= np.maximum(np.max(np.abs(directions), axis=(1, 2), keepdims=True), 1e-12)
    directions *= delta
    symmetric = []
    for direction in directions:
        full = np.zeros((intervention_steps, action_dim), dtype=np.float32)
        full[:, :perturb_dims] = direction.astype(np.float32)
        symmetric.extend([full, -full])
    return np.stack(symmetric)


def _select_states(
    arrays: dict[str, np.ndarray],
    *,
    train_states: int,
    valid_states: int,
    seed: int,
) -> list[tuple[str, int, str]]:
    selected: list[tuple[str, int, str]] = []
    for offset, (split, count) in enumerate(
        (("train", train_states), ("valid", valid_states))
    ):
        indices = np.flatnonzero(arrays["source_split"] == split)
        if count > len(indices):
            raise ValueError(f"Requested {count} {split} states, only {len(indices)} available")
        rng = np.random.default_rng(seed + 100_000 * offset)
        chosen = indices[rng.permutation(len(indices))[:count]]
        selected.extend(
            (
                str(arrays["source_demo"][index]),
                int(arrays["source_step"][index]),
                split,
            )
            for index in chosen
        )
    return selected


def _set_or_validate(output: h5py.File, name: str, expected: Any) -> None:
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


def _ensure_datasets(
    output: h5py.File,
    *,
    candidates: int,
    intervention_steps: int,
    action_dim: int,
    state_dim: int,
) -> h5py.Group:
    samples = output.require_group("states")
    committed = int(output.attrs.get("states_committed", 0))
    string_dtype = h5py.string_dtype(encoding="utf-8")
    definitions: dict[str, tuple[Any, tuple[int, ...]]] = {
        "source_demo": (string_dtype, ()),
        "source_split": (string_dtype, ()),
        "source_step": (np.int32, ()),
        "direction_seed": (np.int64, ()),
        "branch_state": (np.float64, (state_dim,)),
        "base_action_chunk": (np.float32, (intervention_steps, action_dim)),
        "candidate_action_chunks": (
            np.float32,
            (candidates, intervention_steps, action_dim),
        ),
        "candidate_residual_chunks": (
            np.float32,
            (candidates, intervention_steps, action_dim),
        ),
        "base_score": (np.float32, ()),
        "candidate_scores": (np.float32, (candidates,)),
        "delta_scores": (np.float32, (candidates,)),
        "base_reward_sum": (np.float32, ()),
        "candidate_reward_sums": (np.float32, (candidates,)),
        "base_success": (np.uint8, ()),
        "candidate_successes": (np.uint8, (candidates,)),
        "base_final_state": (np.float64, (state_dim,)),
        "candidate_final_states": (np.float64, (candidates, state_dim)),
        "restore_linf": (np.float64, ()),
        "branch_initial_state_linf": (np.float64, ()),
        "best_candidate_index": (np.int16, ()),
        "best_delta_score": (np.float32, ()),
        "improving_candidate_count": (np.int16, ()),
        "candidate_diverged_count": (np.int16, ()),
    }
    for name, (dtype, tail) in definitions.items():
        if name not in samples:
            samples.create_dataset(
                name,
                shape=(0, *tail),
                maxshape=(None, *tail),
                chunks=True,
                dtype=dtype,
            )
        elif len(samples[name]) != committed:
            samples[name].resize((committed, *samples[name].shape[1:]))
    return samples


def _append(samples: h5py.Group, index: int, row: dict[str, Any]) -> None:
    for name, value in row.items():
        dataset = samples[name]
        dataset.resize((index + 1, *dataset.shape[1:]))
        dataset[index] = value


def _summary(
    samples: h5py.Group,
    *,
    target_states: int,
    candidates: int,
    score_margin: float,
) -> dict[str, Any]:
    count = len(samples["best_delta_score"])
    best_delta = np.asarray(samples["best_delta_score"])
    candidate_scores = np.asarray(samples["candidate_scores"])
    pair_difference = np.abs(candidate_scores[:, 0::2] - candidate_scores[:, 1::2])
    candidate_diverged = int(np.sum(np.asarray(samples["candidate_diverged_count"])))
    finite = all(
        bool(np.all(np.isfinite(np.asarray(samples[name]))))
        for name in (
            "base_score",
            "candidate_scores",
            "delta_scores",
            "restore_linf",
            "branch_initial_state_linf",
        )
    )
    total_candidates = count * candidates
    total_direction_pairs = count * (candidates // 2)
    return {
        "states_collected": count,
        "target_states": target_states,
        "complete": count >= target_states,
        "states_with_improving_candidate": int(np.count_nonzero(best_delta > score_margin)),
        "improving_state_rate": float(np.mean(best_delta > score_margin)) if count else 0.0,
        "best_delta_mean": float(np.mean(best_delta)) if count else 0.0,
        "best_delta_median": float(np.median(best_delta)) if count else 0.0,
        "decisive_symmetric_pairs": int(np.count_nonzero(pair_difference > score_margin)),
        "decisive_symmetric_pair_rate": (
            float(np.count_nonzero(pair_difference > score_margin) / total_direction_pairs)
            if total_direction_pairs
            else 0.0
        ),
        "candidate_diverged_count": candidate_diverged,
        "candidate_diverged_rate": (
            float(candidate_diverged / total_candidates) if total_candidates else 0.0
        ),
        "success_outcome_changed": int(
            np.count_nonzero(
                np.asarray(samples["candidate_successes"])
                != np.asarray(samples["base_success"])[:, None]
            )
        ),
        "max_restore_linf": float(np.max(np.asarray(samples["restore_linf"]), initial=0.0)),
        "max_branch_initial_state_linf": float(
            np.max(np.asarray(samples["branch_initial_state_linf"]), initial=0.0)
        ),
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
    with np.load(args.state_dataset, allow_pickle=False) as loaded:
        state_arrays = {key: loaded[key] for key in loaded.files}
    selected = _select_states(
        state_arrays,
        train_states=args.train_states,
        valid_states=args.valid_states,
        seed=args.seed,
    )
    target_states = len(selected)
    candidate_count = 2 * args.direction_pairs
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or output_path.with_suffix(".summary.json")

    with h5py.File(args.source_dataset, "r") as source:
        env_meta = json.loads(source["data"].attrs["env_args"])
        env_meta["env_kwargs"].pop("env_lang", None)
        env_meta["env_kwargs"]["reward_shaping"] = True
        first_group = source["data"][selected[0][0]]
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
                attributes = {
                    "format": "fastwam.robomimic_symmetric_branches.v1",
                    "source_dataset": str(args.source_dataset.resolve()),
                    "state_dataset": str(args.state_dataset.resolve()),
                    "seed": args.seed,
                    "horizon": args.horizon,
                    "intervention_steps": args.intervention_steps,
                    "direction_pairs": args.direction_pairs,
                    "delta": args.delta,
                    "success_bonus": args.success_bonus,
                    "score_margin": args.score_margin,
                    "train_states": args.train_states,
                    "valid_states": args.valid_states,
                }
                for name, expected in attributes.items():
                    _set_or_validate(output, name, expected)
                output.attrs.setdefault("states_committed", 0)
                samples = _ensure_datasets(
                    output,
                    candidates=candidate_count,
                    intervention_steps=args.intervention_steps,
                    action_dim=action_dim,
                    state_dim=state_dim,
                )
                committed = int(output.attrs["states_committed"])

                for state_index in range(committed, target_states):
                    demo_name, source_step, split = selected[state_index]
                    group = source["data"][demo_name]
                    if source_step + args.horizon > len(group["actions"]):
                        raise ValueError(f"{demo_name}:{source_step} lacks the requested horizon")
                    episode_initial_state = np.asarray(group["states"][0])
                    prefix_actions = np.asarray(group["actions"][:source_step])
                    base_actions = np.asarray(
                        group["actions"][source_step : source_step + args.horizon]
                    )
                    model = str(group.attrs["model_file"])
                    base = _rollout_branch(
                        env,
                        model=model,
                        episode_initial_state=episode_initial_state,
                        prefix_actions=prefix_actions,
                        branch_actions=base_actions,
                    )
                    base_score = (
                        args.success_bonus * float(base["success"])
                        + base["reward_sum"] / len(base_actions)
                    )

                    direction_seed = args.seed + state_index
                    residual_chunks = make_symmetric_directions(
                        seed=direction_seed,
                        direction_pairs=args.direction_pairs,
                        intervention_steps=args.intervention_steps,
                        action_dim=action_dim,
                        delta=args.delta,
                    )
                    candidate_chunks = np.clip(
                        base_actions[None, : args.intervention_steps] + residual_chunks,
                        -1.0,
                        1.0,
                    ).astype(base_actions.dtype)
                    # Store the realized residual after action clipping.
                    residual_chunks = candidate_chunks - base_actions[None, : args.intervention_steps]
                    candidate_scores = []
                    candidate_reward_sums = []
                    candidate_successes = []
                    candidate_final_states = []
                    restore_linf = base["restore_linf"]
                    branch_initial_linf = 0.0
                    candidate_diverged = 0
                    for candidate_chunk in candidate_chunks:
                        candidate_actions = np.array(base_actions, copy=True)
                        candidate_actions[: args.intervention_steps] = candidate_chunk
                        branch = _rollout_branch(
                            env,
                            model=model,
                            episode_initial_state=episode_initial_state,
                            prefix_actions=prefix_actions,
                            branch_actions=candidate_actions,
                        )
                        score = (
                            args.success_bonus * float(branch["success"])
                            + branch["reward_sum"] / len(candidate_actions)
                        )
                        candidate_scores.append(score)
                        candidate_reward_sums.append(branch["reward_sum"])
                        candidate_successes.append(int(branch["success"]))
                        candidate_final_states.append(branch["final_state"])
                        restore_linf = max(restore_linf, branch["restore_linf"])
                        branch_initial_linf = max(
                            branch_initial_linf,
                            float(
                                np.max(
                                    np.abs(
                                        base["branch_initial_state"]
                                        - branch["branch_initial_state"]
                                    ),
                                    initial=0.0,
                                )
                            ),
                        )
                        candidate_diverged += int(
                            np.max(
                                np.abs(base["final_state"] - branch["final_state"]),
                                initial=0.0,
                            )
                            > 1e-10
                        )
                    candidate_scores_array = np.asarray(candidate_scores, dtype=np.float32)
                    delta_scores = candidate_scores_array - base_score
                    best_index = int(np.argmax(delta_scores))
                    row = {
                        "source_demo": demo_name,
                        "source_split": split,
                        "source_step": source_step,
                        "direction_seed": direction_seed,
                        "branch_state": base["branch_initial_state"],
                        "base_action_chunk": base_actions[: args.intervention_steps],
                        "candidate_action_chunks": candidate_chunks,
                        "candidate_residual_chunks": residual_chunks,
                        "base_score": base_score,
                        "candidate_scores": candidate_scores_array,
                        "delta_scores": delta_scores,
                        "base_reward_sum": base["reward_sum"],
                        "candidate_reward_sums": np.asarray(
                            candidate_reward_sums, dtype=np.float32
                        ),
                        "base_success": int(base["success"]),
                        "candidate_successes": np.asarray(candidate_successes, dtype=np.uint8),
                        "base_final_state": base["final_state"],
                        "candidate_final_states": np.stack(candidate_final_states),
                        "restore_linf": restore_linf,
                        "branch_initial_state_linf": branch_initial_linf,
                        "best_candidate_index": best_index,
                        "best_delta_score": float(delta_scores[best_index]),
                        "improving_candidate_count": int(
                            np.count_nonzero(delta_scores > args.score_margin)
                        ),
                        "candidate_diverged_count": candidate_diverged,
                    }
                    _append(samples, state_index, row)
                    output.attrs.modify("states_committed", state_index + 1)

                    if (state_index + 1) % args.flush_every == 0 or state_index + 1 == target_states:
                        output.flush()
                        summary = _summary(
                            samples,
                            target_states=target_states,
                            candidates=candidate_count,
                            score_margin=args.score_margin,
                        )
                        elapsed = time.monotonic() - started
                        session_states = state_index + 1 - committed
                        rate = session_states / elapsed if elapsed > 0 else 0.0
                        summary.update(
                            {
                                "output_path": str(output_path),
                                "elapsed_seconds_this_session": elapsed,
                                "states_per_second_this_session": rate,
                                "eta_seconds": (
                                    (target_states - state_index - 1) / rate if rate > 0 else None
                                ),
                            }
                        )
                        _write_json_atomic(summary_path, summary)
                        print(json.dumps(summary, ensure_ascii=False), flush=True)

                output.attrs["complete"] = True
                output.flush()
                final_summary = _summary(
                    samples,
                    target_states=target_states,
                    candidates=candidate_count,
                    score_margin=args.score_margin,
                )
                final_summary.update(
                    {"output_path": str(output_path), "summary_path": str(summary_path)}
                )
                _write_json_atomic(summary_path, final_summary)
        finally:
            close = getattr(getattr(env, "env", None), "close", None)
            if callable(close):
                close()
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--state-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--train-states", type=int, required=True)
    parser.add_argument("--valid-states", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--intervention-steps", type=int, default=3)
    parser.add_argument("--direction-pairs", type=int, default=4)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--score-margin", type=float, default=1e-4)
    parser.add_argument("--flush-every", type=int, default=5)
    parser.add_argument("--require-smoke-quality", action="store_true")
    args = parser.parse_args()
    if min(args.train_states, args.valid_states, args.horizon, args.intervention_steps) <= 0:
        parser.error("state counts, horizon, and intervention-steps must be positive")
    if args.direction_pairs <= 0 or args.delta <= 0:
        parser.error("direction-pairs and delta must be positive")
    report = collect(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_smoke_quality:
        passed = (
            report["complete"]
            and report["all_finite"]
            and report["max_restore_linf"] <= 1e-10
            and report["max_branch_initial_state_linf"] <= 1e-10
            and report["candidate_diverged_rate"] >= 0.95
            and report["improving_state_rate"] >= 0.50
            and report["decisive_symmetric_pair_rate"] >= 0.50
        )
        if not passed:
            raise SystemExit("Symmetric multi-candidate smoke-quality gate failed")


if __name__ == "__main__":
    main()
