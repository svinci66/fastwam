#!/usr/bin/env python3
"""Audit each accepted online residual against a baseline-action terminal tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from experiments.robomimic.audit_can_residual_q_support import (
    _normalized_support_features,
    kth_neighbor_distance,
)
from experiments.robomimic.collect_can_counterfactual_branches import (
    _rollout_branch,
    _success_value,
)
from experiments.robomimic.evaluate_can_deployable_gated_branches import (
    _load_actor,
    _prepare_gate,
    _write_json_atomic,
    conservative_ensemble_advantage,
)
from experiments.robomimic.evaluate_can_deployable_online_episodes import (
    LiveVisionEncoder,
    PROPRIO_KEYS,
    _live_q_advantages,
    _load_q_models,
    load_vision_projection,
    padded_action_chunk,
    project_vision_feature,
)
from experiments.robomimic.train_can_residual_actor import _features


def summarize_terminal_tails(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["accepted"]]
    if not accepted:
        return {
            "decisions": len(rows),
            "accepted_interventions": 0,
            "terminal_success_gains": 0,
            "terminal_success_losses": 0,
            "terminal_success_preserved": 0,
            "terminal_failure_preserved": 0,
            "tail_reward_delta_mean": 0.0,
        }
    base_success = np.asarray([row["base_tail_success"] for row in accepted], dtype=bool)
    residual_success = np.asarray(
        [row["residual_tail_success"] for row in accepted], dtype=bool
    )
    reward_delta = np.asarray([row["tail_reward_delta"] for row in accepted], dtype=np.float64)
    return {
        "decisions": len(rows),
        "accepted_interventions": len(accepted),
        "terminal_success_gains": int(np.count_nonzero(~base_success & residual_success)),
        "terminal_success_losses": int(np.count_nonzero(base_success & ~residual_success)),
        "terminal_success_preserved": int(np.count_nonzero(base_success & residual_success)),
        "terminal_failure_preserved": int(np.count_nonzero(~base_success & ~residual_success)),
        "tail_reward_delta_mean": float(np.mean(reward_delta)),
        "tail_reward_delta_median": float(np.median(reward_delta)),
        "tail_reward_improved": int(np.count_nonzero(reward_delta > 1e-6)),
        "tail_reward_worsened": int(np.count_nonzero(reward_delta < -1e-6)),
        "max_branch_state_linf": float(
            max(row["branch_state_linf"] for row in accepted)
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
    actor, actor_checkpoint = _load_actor(args.actor_checkpoint, device)
    with np.load(args.actor_dataset, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    calibration = _prepare_gate(
        arrays,
        actor,
        actor_checkpoint,
        args.q_checkpoint,
        device=device,
        batch_size=args.batch_size,
        k=args.k,
        support_quantile=args.support_quantile,
        q_threshold_quantile=args.q_threshold_quantile,
        uncertainty_weight=args.uncertainty_weight,
        knn_chunk_size=args.knn_chunk_size,
    )
    ensemble = _load_q_models(args.q_checkpoint, device)
    metadata = actor_checkpoint["observation_metadata"]
    encoder = LiveVisionEncoder(Path(metadata["encoder_path"]), device)
    projection = load_vision_projection(metadata)
    normalization = actor_checkpoint["normalization"]
    target_shape = tuple(actor_checkpoint["target_shape"])
    chunk_steps = target_shape[0]
    camera_name = metadata["camera_name"]

    with h5py.File(args.dataset, "r") as source:
        if args.demo not in source["data"]:
            raise ValueError(f"Demo not found: {args.demo}")
        group = source["data"][args.demo]
        actions = np.asarray(group["actions"], dtype=np.float32)
        model_file = str(group.attrs["model_file"])
        initial_state = np.asarray(group["states"][0])
        env_meta = json.loads(source["data"].attrs["env_args"])
        env_meta["env_kwargs"].pop("env_lang", None)
        env_meta["env_kwargs"]["reward_shaping"] = True
        live_env = create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=True,
            use_image_obs=False,
        )
        branch_env = create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        rows: list[dict[str, Any]] = []
        executed_prefix: list[np.ndarray] = []
        accepted_before = 0
        cumulative_residual_norm = 0.0
        try:
            obs = live_env.reset_to({"model": model_file, "states": initial_state})
            online_success = _success_value(live_env.is_success())
            for start in range(0, len(actions), chunk_steps):
                base_chunk, available = padded_action_chunk(
                    actions, start, chunk_steps=chunk_steps
                )
                image = live_env.render(
                    mode="rgb_array",
                    height=args.image_size,
                    width=args.image_size,
                    camera_name=camera_name,
                )
                proprio = np.concatenate(
                    [np.asarray(obs[key]).reshape(-1) for key in PROPRIO_KEYS]
                ).astype(np.float32)
                vision = project_vision_feature(encoder(image), projection)
                state = np.concatenate([vision, proprio]).astype(np.float32)
                actor_feature = _features(
                    state[None], base_chunk[None], normalization
                )
                with torch.no_grad():
                    residual = (
                        actor(torch.from_numpy(actor_feature).to(device))
                        .cpu()
                        .numpy()
                        .reshape(target_shape)
                    )
                proposal = np.clip(base_chunk + residual, -1.0, 1.0)
                base_support = _normalized_support_features(
                    state[None], base_chunk[None], normalization
                )
                proposal_support = _normalized_support_features(
                    state[None], proposal[None], normalization
                )
                base_distance = float(
                    kth_neighbor_distance(
                        base_support,
                        calibration["support_references"],
                        k=args.k,
                        chunk_size=1,
                    )[0]
                )
                proposal_distance = float(
                    kth_neighbor_distance(
                        proposal_support,
                        calibration["support_references"],
                        k=args.k,
                        chunk_size=1,
                    )[0]
                )
                q_advantages = _live_q_advantages(
                    ensemble, state, base_chunk, proposal, device=device
                )
                conservative = float(
                    conservative_ensemble_advantage(
                        q_advantages[:, None],
                        uncertainty_weight=args.uncertainty_weight,
                    )[0]
                )
                in_support = (
                    base_distance <= calibration["support_threshold"]
                    and proposal_distance <= calibration["support_threshold"]
                )
                accepted = bool(
                    available == chunk_steps
                    and in_support
                    and conservative > calibration["q_threshold"]
                )
                row: dict[str, Any] = {
                    "start_step": start,
                    "progress": float(start / max(len(actions), 1)),
                    "accepted": accepted,
                    "accepted_interventions_before": accepted_before,
                    "cumulative_residual_norm_before": cumulative_residual_norm,
                    "in_support": in_support,
                    "conservative_q_advantage": conservative,
                    "base_support_distance": base_distance,
                    "proposal_support_distance": proposal_distance,
                    "residual_max_abs": float(np.max(np.abs(residual))),
                    "state": state.tolist(),
                    "base_action_chunk": base_chunk.tolist(),
                    "proposal_action_chunk": proposal.tolist(),
                    "residual_chunk": residual.tolist(),
                }
                if accepted:
                    prefix = np.asarray(executed_prefix, dtype=np.float32)
                    base_tail = actions[start:].copy()
                    residual_tail = base_tail.copy()
                    residual_tail[:available] = proposal[:available]
                    base_result = _rollout_branch(
                        branch_env,
                        model=model_file,
                        episode_initial_state=initial_state,
                        prefix_actions=prefix,
                        branch_actions=base_tail,
                    )
                    residual_result = _rollout_branch(
                        branch_env,
                        model=model_file,
                        episode_initial_state=initial_state,
                        prefix_actions=prefix,
                        branch_actions=residual_tail,
                    )
                    current_state = np.asarray(live_env.get_state()["states"])
                    branch_state_linf = max(
                        float(
                            np.max(
                                np.abs(result["branch_initial_state"] - current_state),
                                initial=0.0,
                            )
                        )
                        for result in (base_result, residual_result)
                    )
                    row.update(
                        {
                            "base_tail_success": int(base_result["success"]),
                            "residual_tail_success": int(residual_result["success"]),
                            "base_tail_reward": base_result["reward_sum"],
                            "residual_tail_reward": residual_result["reward_sum"],
                            "tail_reward_delta": (
                                residual_result["reward_sum"] - base_result["reward_sum"]
                            ),
                            "branch_state_linf": branch_state_linf,
                        }
                    )
                executed_chunk = proposal if accepted else base_chunk
                if accepted:
                    accepted_before += 1
                    cumulative_residual_norm += float(np.linalg.norm(residual))
                for action in executed_chunk[:available]:
                    obs, _, _, info = live_env.step(action)
                    executed_prefix.append(np.asarray(action, dtype=np.float32))
                    online_success = online_success or _success_value(
                        info.get("is_success", live_env.is_success())
                    )
                rows.append(row)
                _write_json_atomic(
                    args.output_json,
                    {
                        "complete": False,
                        "demo": args.demo,
                        "decisions_completed": len(rows),
                        "rows": rows,
                    },
                )
                print(
                    json.dumps(
                        {
                            "step": start,
                            "accepted": accepted,
                            "base_tail_success": row.get("base_tail_success"),
                            "residual_tail_success": row.get("residual_tail_success"),
                            "tail_reward_delta": row.get("tail_reward_delta"),
                        }
                    ),
                    flush=True,
                )
        finally:
            for env in (live_env, branch_env):
                close = getattr(getattr(env, "env", None), "close", None)
                if callable(close):
                    close()

    summary = summarize_terminal_tails(rows)
    summary.update(
        {
            "complete": True,
            "dataset": str(args.dataset.resolve()),
            "demo": args.demo,
            "episode_steps": len(actions),
            "actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "observation_metadata": metadata,
            "online_residual_success": int(online_success),
            "gate": {
                "support_threshold": calibration["support_threshold"],
                "q_advantage_threshold": calibration["q_threshold"],
                "k": args.k,
                "uncertainty_weight": args.uncertainty_weight,
            },
            "rows": rows,
        }
    )
    _write_json_atomic(args.output_json, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--demo", required=True)
    parser.add_argument("--actor-dataset", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--support-quantile", type=float, default=0.95)
    parser.add_argument("--q-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--knn-chunk-size", type=int, default=128)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
