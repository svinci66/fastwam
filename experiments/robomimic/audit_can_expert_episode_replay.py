#!/usr/bin/env python3
"""Audit whether saved RoboMimic expert actions reproduce complete episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from experiments.robomimic.collect_can_counterfactual_branches import _success_value


def summarize_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stored = np.asarray([row["stored_success"] for row in rows], dtype=bool)
    replay = np.asarray([row["replay_success"] for row in rows], dtype=bool)
    return {
        "episodes": len(rows),
        "stored_successes": int(np.count_nonzero(stored)),
        "replayed_successes": int(np.count_nonzero(replay)),
        "success_replay_rate": (
            float(np.count_nonzero(stored & replay) / np.count_nonzero(stored))
            if np.any(stored)
            else None
        ),
        "failed_replay_demos": [
            row["source_demo"] for row in rows if row["stored_success"] and not row["replay_success"]
        ],
        "max_restore_linf": float(max(row["restore_linf"] for row in rows)),
        "max_stored_state_linf": float(max(row["max_stored_state_linf"] for row in rows)),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    from robomimic.utils.env_utils import create_env_from_metadata
    from robomimic.utils.obs_utils import initialize_obs_utils_with_obs_specs

    initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )
    with h5py.File(args.dataset, "r") as source:
        values = np.asarray(source["mask"][args.split])
        demos = [value.decode() if isinstance(value, bytes) else str(value) for value in values]
        demos.sort(key=lambda value: int(value.rsplit("_", 1)[1]))
        env_meta = json.loads(source["data"].attrs["env_args"])
        env_meta["env_kwargs"].pop("env_lang", None)
        env_meta["env_kwargs"]["reward_shaping"] = True
        env = create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        rows = []
        try:
            for ordinal, demo in enumerate(demos, start=1):
                group = source["data"][demo]
                states = np.asarray(group["states"])
                initial_state = states[0]
                env.reset_to({"model": str(group.attrs["model_file"]), "states": initial_state})
                restore_linf = float(
                    np.max(
                        np.abs(np.asarray(env.get_state()["states"]) - initial_state), initial=0.0
                    )
                )
                reward_sum = 0.0
                success = _success_value(env.is_success())
                max_stored_linf = 0.0
                first_state_mismatch_step = None
                for step, action in enumerate(np.asarray(group["actions"])):
                    _, reward, _, info = env.step(action)
                    reward_sum += float(reward)
                    success = success or _success_value(
                        info.get("is_success", env.is_success())
                    )
                    if step + 1 < len(states):
                        linf = float(
                            np.max(
                                np.abs(np.asarray(env.get_state()["states"]) - states[step + 1]),
                                initial=0.0,
                            )
                        )
                        max_stored_linf = max(max_stored_linf, linf)
                        if first_state_mismatch_step is None and linf > args.state_tolerance:
                            first_state_mismatch_step = step + 1
                stored_reward = np.asarray(group["rewards"])
                stored_done = np.asarray(group["dones"])
                stored_success = bool(np.any(stored_reward > 0.0) or np.any(stored_done > 0))
                row = {
                    "source_demo": demo,
                    "steps": len(group["actions"]),
                    "stored_success": int(stored_success),
                    "replay_success": int(success),
                    "stored_reward_sum": float(np.sum(stored_reward)),
                    "replay_reward_sum": reward_sum,
                    "restore_linf": restore_linf,
                    "max_stored_state_linf": max_stored_linf,
                    "first_state_mismatch_step": first_state_mismatch_step,
                }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "episode": f"{ordinal}/{len(demos)}",
                            "source_demo": demo,
                            "stored_success": int(stored_success),
                            "replay_success": int(success),
                            "max_stored_state_linf": max_stored_linf,
                        }
                    ),
                    flush=True,
                )
        finally:
            close = getattr(getattr(env, "env", None), "close", None)
            if callable(close):
                close()
    report = summarize_replay(rows)
    report.update(
        {
            "dataset": str(args.dataset.resolve()),
            "split": args.split,
            "state_tolerance": args.state_tolerance,
            "rows": rows,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--state-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
