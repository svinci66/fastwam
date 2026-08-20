#!/usr/bin/env python3
"""Execute a trained residual actor on held-out RoboMimic branch states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from experiments.robomimic.collect_can_counterfactual_branches import _rollout_branch
from experiments.robomimic.train_can_residual_actor import ResidualActor, _features


def summarize_branch_results(rows: list[dict[str, Any]], *, score_margin: float) -> dict[str, Any]:
    delta = np.asarray([row["delta_score"] for row in rows], dtype=np.float64)
    base_success = np.asarray([row["base_success"] for row in rows], dtype=bool)
    actor_success = np.asarray([row["actor_success"] for row in rows], dtype=bool)
    return {
        "states": len(rows),
        "delta_score_mean": float(np.mean(delta)),
        "delta_score_median": float(np.median(delta)),
        "delta_score_min": float(np.min(delta)),
        "delta_score_max": float(np.max(delta)),
        "improved_states": int(np.count_nonzero(delta > score_margin)),
        "tied_states": int(np.count_nonzero(np.abs(delta) <= score_margin)),
        "worsened_states": int(np.count_nonzero(delta < -score_margin)),
        "improvement_rate": float(np.mean(delta > score_margin)),
        "worsening_rate": float(np.mean(delta < -score_margin)),
        "base_successes": int(np.count_nonzero(base_success)),
        "actor_successes": int(np.count_nonzero(actor_success)),
        "success_gains": int(np.count_nonzero(~base_success & actor_success)),
        "success_losses": int(np.count_nonzero(base_success & ~actor_success)),
        "max_restore_linf": float(max(row["restore_linf"] for row in rows)),
        "max_branch_initial_state_linf": float(
            max(row["branch_initial_state_linf"] for row in rows)
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from robomimic.utils.env_utils import create_env_from_metadata
    from robomimic.utils.obs_utils import initialize_obs_utils_with_obs_specs

    initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(args.actor_checkpoint, map_location=device, weights_only=False)
    actor = ResidualActor(
        checkpoint["input_dim"],
        checkpoint["output_dim"],
        tuple(checkpoint["hidden_dims"]),
        checkpoint["residual_scale"],
        preserve_last_action_dim=checkpoint.get("preserve_last_action_dim", False),
        action_dim=checkpoint.get("action_dim"),
    ).to(device)
    actor.load_state_dict(checkpoint["model"])
    actor.eval()

    with h5py.File(args.collection, "r") as collection:
        if not bool(collection.attrs.get("complete", False)):
            raise ValueError("Collection must be complete")
        source_path = Path(str(collection.attrs["source_dataset"]))
        horizon = int(collection.attrs["horizon"])
        intervention_steps = int(collection.attrs["intervention_steps"])
        success_bonus = float(collection.attrs["success_bonus"])
        score_margin = float(collection.attrs["score_margin"])
        states = collection["states"]
        split = states["source_split"].asstr()[...]
        eligible = np.flatnonzero(split == args.split)
        if args.states > len(eligible):
            raise ValueError(f"Requested {args.states} states, only {len(eligible)} available")
        rng = np.random.default_rng(args.seed)
        selected = eligible[rng.permutation(len(eligible))[: args.states]]

        with h5py.File(source_path, "r") as source:
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
                for ordinal, index in enumerate(selected, start=1):
                    demo_value = states["source_demo"][index]
                    demo = demo_value.decode("utf-8") if isinstance(demo_value, bytes) else str(demo_value)
                    source_step = int(states["source_step"][index])
                    group = source["data"][demo]
                    base_action = np.asarray(states["base_action_chunk"][index], dtype=np.float32)
                    branch_state = np.asarray(states["branch_state"][index], dtype=np.float32)
                    feature = _features(
                        branch_state[None], base_action[None], checkpoint["normalization"]
                    )
                    with torch.no_grad():
                        residual = (
                            actor(torch.from_numpy(feature).to(device))
                            .cpu()
                            .numpy()
                            .reshape(intervention_steps, -1)
                        )
                    full_actions = np.asarray(
                        group["actions"][source_step : source_step + horizon]
                    ).copy()
                    full_actions[:intervention_steps] = np.clip(
                        base_action + residual, -1.0, 1.0
                    )
                    branch = _rollout_branch(
                        env,
                        model=str(group.attrs["model_file"]),
                        episode_initial_state=np.asarray(group["states"][0]),
                        prefix_actions=np.asarray(group["actions"][:source_step]),
                        branch_actions=full_actions,
                    )
                    actor_score = (
                        success_bonus * float(branch["success"])
                        + branch["reward_sum"] / horizon
                    )
                    base_score = float(states["base_score"][index])
                    initial_linf = float(
                        np.max(
                            np.abs(
                                np.asarray(branch["branch_initial_state"], dtype=np.float64)
                                - np.asarray(states["branch_state"][index], dtype=np.float64)
                            ),
                            initial=0.0,
                        )
                    )
                    row = {
                        "collection_index": int(index),
                        "source_demo": demo,
                        "source_step": source_step,
                        "base_score": base_score,
                        "actor_score": actor_score,
                        "delta_score": actor_score - base_score,
                        "base_success": int(states["base_success"][index]),
                        "actor_success": int(branch["success"]),
                        "residual_norm": float(np.linalg.norm(residual)),
                        "gripper_residual_max_abs": float(np.max(np.abs(residual[:, -1]))),
                        "restore_linf": float(branch["restore_linf"]),
                        "branch_initial_state_linf": initial_linf,
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "state": f"{ordinal}/{len(selected)}",
                                "delta_score": row["delta_score"],
                                "residual_norm": row["residual_norm"],
                            }
                        ),
                        flush=True,
                    )
            finally:
                close = getattr(getattr(env, "env", None), "close", None)
                if callable(close):
                    close()

    summary = summarize_branch_results(rows, score_margin=score_margin)
    summary.update(
        {
            "collection": str(args.collection.resolve()),
            "actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "split": args.split,
            "seed": args.seed,
            "score_margin": score_margin,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "training_method": checkpoint.get("training_method", "supervised"),
            "rows": rows,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.states <= 0:
        parser.error("states must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
