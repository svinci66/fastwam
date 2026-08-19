#!/usr/bin/env python3
"""Replay two RoboMimic Can trajectories from one exact simulator state.

This is an environment-level smoke test for same-state counterfactual data. It
does not train a policy. A passing run demonstrates that (1) state restoration
is numerically exact, (2) identical actions replay deterministically, and (3)
the paired good / bad actions produce different outcomes from the same state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _success_value(success: Any) -> bool:
    if isinstance(success, dict):
        return bool(success["task"])
    return bool(success)


def _replay(env: Any, *, model: str, initial_state: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    env.reset_to({"model": model, "states": initial_state})
    restored_state = np.asarray(env.get_state()["states"])
    restore_linf = float(np.max(np.abs(restored_state - initial_state), initial=0.0))
    rewards: list[float] = []
    success = _success_value(env.is_success())
    first_success_step: int | None = 0 if success else None
    for step, action in enumerate(actions, start=1):
        _, reward, _, info = env.step(action)
        rewards.append(float(reward))
        step_success = _success_value(info.get("is_success", env.is_success()))
        if step_success and first_success_step is None:
            first_success_step = step
        success = success or step_success
    return {
        "restore_linf": restore_linf,
        "success": success,
        "first_success_step": first_success_step,
        "reward_sum": float(sum(rewards)),
        "max_reward": float(max(rewards, default=0.0)),
        "steps": len(actions),
        "reward_trace": rewards,
        "final_state": np.asarray(env.get_state()["states"]),
    }


def validate_branching(
    dataset_path: str | Path,
    *,
    pair_index: int = 0,
    restore_tolerance: float = 1e-10,
    deterministic_tolerance: float = 1e-10,
) -> dict[str, Any]:
    # Imports stay local so the dataset-only audit does not require RoboMimic.
    from robomimic.utils.env_utils import create_env_from_metadata
    from robomimic.utils.obs_utils import initialize_obs_utils_with_obs_specs

    # Playback only consumes simulator state and reward. RoboMimic nevertheless
    # requires a modality table before its environment wrapper can return obs.
    initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )

    dataset_path = Path(dataset_path).expanduser().resolve()
    with h5py.File(dataset_path, "r") as dataset:
        env_meta = json.loads(dataset["data"].attrs["env_args"])
        env_meta["env_kwargs"].pop("env_lang", None)
        names = [f"demo_{2 * pair_index}", f"demo_{2 * pair_index + 1}"]
        if any(name not in dataset["data"] for name in names):
            raise ValueError(f"Pair {pair_index} ({names}) does not exist")
        trajectories = []
        for name in names:
            group = dataset["data"][name]
            trajectories.append(
                {
                    "name": name,
                    "actions": np.asarray(group["actions"]),
                    "initial_state": np.asarray(group["states"][0]),
                    "stored_success": bool(np.max(group["rewards"][()]) > 0),
                    "model": str(group.attrs["model_file"]),
                }
            )

    initial_state_linf = float(
        np.max(np.abs(trajectories[0]["initial_state"] - trajectories[1]["initial_state"]), initial=0.0)
    )
    if initial_state_linf > restore_tolerance:
        raise ValueError(f"Pair does not share an initial state: L_inf={initial_state_linf}")

    good = next((trajectory for trajectory in trajectories if trajectory["stored_success"]), None)
    bad = next((trajectory for trajectory in trajectories if not trajectory["stored_success"]), None)
    if good is None or bad is None:
        raise ValueError("Pair must contain exactly one successful and one failed trajectory")

    env = create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    try:
        shared_state = good["initial_state"]
        shared_model = good["model"]
        good_first = _replay(env, model=shared_model, initial_state=shared_state, actions=good["actions"])
        good_second = _replay(env, model=shared_model, initial_state=shared_state, actions=good["actions"])
        bad_branch = _replay(env, model=shared_model, initial_state=shared_state, actions=bad["actions"])
    finally:
        close = getattr(getattr(env, "env", None), "close", None)
        if callable(close):
            close()

    deterministic_final_linf = float(
        np.max(np.abs(good_first["final_state"] - good_second["final_state"]), initial=0.0)
    )
    branch_final_linf = float(
        np.max(np.abs(good_first["final_state"] - bad_branch["final_state"]), initial=0.0)
    )
    deterministic_rewards = np.array_equal(
        np.asarray(good_first["reward_trace"]), np.asarray(good_second["reward_trace"])
    )
    restore_exact = max(
        good_first["restore_linf"], good_second["restore_linf"], bad_branch["restore_linf"]
    ) <= restore_tolerance
    deterministic = deterministic_final_linf <= deterministic_tolerance and deterministic_rewards
    complementary_replay_outcomes = good_first["success"] and not bad_branch["success"]
    branch_diverged = branch_final_linf > deterministic_tolerance
    passed = restore_exact and deterministic and complementary_replay_outcomes and branch_diverged

    def public_rollout(rollout: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in rollout.items() if key not in {"final_state", "reward_trace"}}

    return {
        "dataset_path": str(dataset_path),
        "pair_index": pair_index,
        "good_demo": good["name"],
        "bad_demo": bad["name"],
        "shared_model_sha256": hashlib.sha256(shared_model.encode("utf-8")).hexdigest(),
        "paired_initial_state_linf": initial_state_linf,
        "good_replay": public_rollout(good_first),
        "good_repeat": public_rollout(good_second),
        "bad_branch": public_rollout(bad_branch),
        "deterministic_final_state_linf": deterministic_final_linf,
        "branch_final_state_linf": branch_final_linf,
        "restore_exact": restore_exact,
        "deterministic_replay": deterministic,
        "complementary_replay_outcomes": complementary_replay_outcomes,
        "branch_diverged": branch_diverged,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    report = validate_branching(args.dataset, pair_index=args.pair_index)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.allow_failure and not report["passed"]:
        raise SystemExit("State-branching validation failed")


if __name__ == "__main__":
    main()
