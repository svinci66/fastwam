#!/usr/bin/env python3
"""Pair deployable residual branches against base actions with Q/OOD gating."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from experiments.robomimic.audit_can_residual_q_support import (
    _normalized_support_features,
    _q_advantage,
    kth_neighbor_distance,
)
from experiments.robomimic.collect_can_counterfactual_branches import _rollout_branch
from experiments.robomimic.evaluate_can_residual_branches import summarize_branch_results
from experiments.robomimic.train_can_residual_actor import ResidualActor, _features, _predict


def conservative_ensemble_advantage(
    advantages: np.ndarray, *, uncertainty_weight: float
) -> np.ndarray:
    if advantages.ndim != 2 or advantages.shape[0] == 0:
        raise ValueError("advantages must have shape [critics, states]")
    mean = advantages.mean(axis=0)
    if len(advantages) == 1:
        return mean
    return mean - uncertainty_weight * advantages.std(axis=0)


def source_key_lookup(demos: np.ndarray, steps: np.ndarray) -> dict[tuple[str, int], int]:
    lookup: dict[tuple[str, int], int] = {}
    for index, (demo, step) in enumerate(zip(demos.astype(str), steps)):
        key = (str(demo), int(step))
        if key in lookup:
            raise ValueError(f"Duplicate source key: {key}")
        lookup[key] = index
    return lookup


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _load_actor(
    checkpoint_path: Path, device: torch.device
) -> tuple[ResidualActor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    return actor, checkpoint


def _prepare_gate(
    arrays: dict[str, np.ndarray],
    actor: ResidualActor,
    actor_checkpoint: dict[str, Any],
    q_paths: list[Path],
    *,
    device: torch.device,
    batch_size: int,
    k: int,
    support_quantile: float,
    q_threshold_quantile: float,
    uncertainty_weight: float,
    knn_chunk_size: int,
) -> dict[str, Any]:
    train = arrays["source_split"] == "train"
    valid = arrays["source_split"] == "valid"
    improved = arrays["has_improvement"].astype(bool)
    actor_features = _features(
        arrays["state"], arrays["base_action_chunk"], actor_checkpoint["normalization"]
    )
    residual = _predict(
        actor,
        actor_features,
        device=device,
        batch_size=batch_size,
        output_shape=tuple(actor_checkpoint["target_shape"]),
    )
    proposal = np.clip(arrays["base_action_chunk"] + residual, -1.0, 1.0)

    normalization = actor_checkpoint["normalization"]
    train_base_support = _normalized_support_features(
        arrays["state"][train], arrays["base_action_chunk"][train], normalization
    )
    target_action = np.clip(
        arrays["base_action_chunk"][train] + arrays["target_residual_chunk"][train],
        -1.0,
        1.0,
    )
    train_target_support = _normalized_support_features(
        arrays["state"][train][improved[train]],
        target_action[improved[train]],
        normalization,
    )
    references = np.concatenate([train_base_support, train_target_support], axis=0)
    leave_one_out = kth_neighbor_distance(
        references,
        references,
        k=k,
        exclude_identical_index=True,
        chunk_size=knn_chunk_size,
    )
    support_threshold = float(np.quantile(leave_one_out, support_quantile))
    base_distance = np.full(len(arrays["state"]), np.inf, dtype=np.float64)
    proposal_distance = np.full(len(arrays["state"]), np.inf, dtype=np.float64)
    valid_base_support = _normalized_support_features(
        arrays["state"][valid], arrays["base_action_chunk"][valid], normalization
    )
    valid_proposal_support = _normalized_support_features(
        arrays["state"][valid], proposal[valid], normalization
    )
    base_distance[valid] = kth_neighbor_distance(
        valid_base_support, references, k=k, chunk_size=knn_chunk_size
    )
    proposal_distance[valid] = kth_neighbor_distance(
        valid_proposal_support, references, k=k, chunk_size=knn_chunk_size
    )
    in_support = (base_distance <= support_threshold) & (
        proposal_distance <= support_threshold
    )

    per_q = []
    observation_mode = actor_checkpoint.get("observation_metadata", {}).get(
        "observation_mode"
    )
    for path in q_paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint["state_mode"] != "full":
            raise ValueError(f"Deployable gate requires a full-observation Q: {path}")
        q_mode = checkpoint.get("observation_metadata", {}).get("observation_mode")
        if observation_mode is not None and q_mode != observation_mode:
            raise ValueError(f"Actor/Q observation mismatch: {observation_mode} != {q_mode}")
        per_q.append(
            _q_advantage(
                checkpoint,
                arrays["state"],
                arrays["base_action_chunk"],
                proposal,
                device=device,
                batch_size=batch_size,
            )
        )
    advantages = np.stack(per_q)
    conservative = conservative_ensemble_advantage(
        advantages, uncertainty_weight=uncertainty_weight
    )
    zero_target_train = train & ~improved
    if not np.any(zero_target_train):
        raise ValueError("Q threshold requires train zero-target states")
    q_threshold = float(np.quantile(conservative[zero_target_train], q_threshold_quantile))
    accepted = valid & in_support & (conservative > q_threshold)
    return {
        "residual": residual,
        "proposal": proposal,
        "advantages": advantages,
        "conservative_advantage": conservative,
        "q_threshold": q_threshold,
        "base_distance": base_distance,
        "proposal_distance": proposal_distance,
        "support_threshold": support_threshold,
        "in_support": in_support,
        "accepted": accepted,
        "support_reference_count": len(references),
        "support_references": references,
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
    if arrays["state"].shape[1] + arrays["base_action_chunk"].shape[1] * arrays[
        "base_action_chunk"
    ].shape[2] != actor_checkpoint["input_dim"]:
        raise ValueError("Actor dataset feature dimension does not match checkpoint")
    train_demos = set(arrays["source_demo"][arrays["source_split"] == "train"].tolist())
    valid_demos = set(arrays["source_demo"][arrays["source_split"] == "valid"].tolist())
    if train_demos & valid_demos:
        raise ValueError("Source trajectory leakage between train and validation")
    actor_lookup = source_key_lookup(arrays["source_demo"], arrays["source_step"])
    gate = _prepare_gate(
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
        eligible = np.flatnonzero(split == "valid")
        if args.states > len(eligible):
            raise ValueError(f"Requested {args.states} states, only {len(eligible)} available")
        selected = eligible[
            np.random.default_rng(args.selection_seed).permutation(len(eligible))[: args.states]
        ]

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
                for ordinal, collection_index in enumerate(selected, start=1):
                    demo = str(states["source_demo"].asstr()[collection_index])
                    source_step = int(states["source_step"][collection_index])
                    actor_index = actor_lookup[(demo, source_step)]
                    base_action = np.asarray(
                        states["base_action_chunk"][collection_index], dtype=np.float32
                    )
                    if not np.allclose(
                        base_action, arrays["base_action_chunk"][actor_index], rtol=0.0, atol=1e-7
                    ):
                        raise ValueError(f"Collection/actor action mismatch at {demo}:{source_step}")
                    proposed_residual = gate["residual"][actor_index]
                    accepted = bool(gate["accepted"][actor_index])
                    executed_residual = proposed_residual if accepted else np.zeros_like(proposed_residual)
                    group = source["data"][demo]
                    base_actions = np.asarray(
                        group["actions"][source_step : source_step + horizon]
                    ).copy()
                    rollout_kwargs = {
                        "model": str(group.attrs["model_file"]),
                        "episode_initial_state": np.asarray(group["states"][0]),
                        "prefix_actions": np.asarray(group["actions"][:source_step]),
                    }
                    # Re-run the baseline in this process. Comparing against the
                    # score saved by an older collection can turn tiny simulator
                    # replay drift into a false residual gain or regression.
                    base_branch = _rollout_branch(
                        env,
                        branch_actions=base_actions,
                        **rollout_kwargs,
                    )
                    base_score = (
                        success_bonus * float(base_branch["success"])
                        + base_branch["reward_sum"] / horizon
                    )
                    if accepted:
                        actor_actions = base_actions.copy()
                        actor_actions[:intervention_steps] = np.clip(
                            base_action + executed_residual, -1.0, 1.0
                        )
                        branch = _rollout_branch(
                            env,
                            branch_actions=actor_actions,
                            **rollout_kwargs,
                        )
                    else:
                        # A rejected intervention executes the fresh baseline by
                        # definition; do not introduce a redundant replay.
                        branch = base_branch
                    actor_score = (
                        success_bonus * float(branch["success"])
                        + branch["reward_sum"] / horizon
                    )
                    expected_branch_state = np.asarray(
                        states["branch_state"][collection_index], dtype=np.float64
                    )
                    initial_linf = max(
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(result["branch_initial_state"], dtype=np.float64)
                                    - expected_branch_state
                                ),
                                initial=0.0,
                            )
                        )
                        for result in (base_branch, branch)
                    )
                    recorded_base_score = float(states["base_score"][collection_index])
                    row = {
                        "collection_index": int(collection_index),
                        "actor_dataset_index": int(actor_index),
                        "source_demo": demo,
                        "source_step": source_step,
                        "gate_accepted": accepted,
                        "in_support": bool(gate["in_support"][actor_index]),
                        "base_support_distance": float(gate["base_distance"][actor_index]),
                        "proposal_support_distance": float(gate["proposal_distance"][actor_index]),
                        "conservative_q_advantage": float(
                            gate["conservative_advantage"][actor_index]
                        ),
                        "q_advantages": gate["advantages"][:, actor_index].tolist(),
                        "base_score": base_score,
                        "recorded_base_score": recorded_base_score,
                        "base_replay_score_drift": base_score - recorded_base_score,
                        "actor_score": actor_score,
                        "delta_score": actor_score - base_score,
                        "base_success": int(base_branch["success"]),
                        "recorded_base_success": int(states["base_success"][collection_index]),
                        "actor_success": int(branch["success"]),
                        "proposed_residual_norm": float(np.linalg.norm(proposed_residual)),
                        "residual_norm": float(np.linalg.norm(executed_residual)),
                        "executed_residual_max_abs": float(np.max(np.abs(executed_residual))),
                        "gripper_residual_max_abs": float(
                            np.max(np.abs(executed_residual[:, -1]))
                        ),
                        "restore_linf": float(
                            max(base_branch["restore_linf"], branch["restore_linf"])
                        ),
                        "branch_initial_state_linf": initial_linf,
                    }
                    rows.append(row)
                    if ordinal % args.flush_every == 0:
                        _write_json_atomic(
                            args.output_json,
                            {
                                "complete": False,
                                "states_completed": len(rows),
                                "states_requested": len(selected),
                                "actor_checkpoint": str(args.actor_checkpoint.resolve()),
                                "rows": rows,
                            },
                        )
                    print(
                        json.dumps(
                            {
                                "state": f"{ordinal}/{len(selected)}",
                                "accepted": accepted,
                                "delta_score": row["delta_score"],
                            }
                        ),
                        flush=True,
                    )
            finally:
                close = getattr(getattr(env, "env", None), "close", None)
                if callable(close):
                    close()

    summary = summarize_branch_results(rows, score_margin=score_margin)
    accepted = np.asarray([row["gate_accepted"] for row in rows], dtype=bool)
    base_replay_drift = np.asarray(
        [row["base_replay_score_drift"] for row in rows], dtype=np.float64
    )
    rejected_delta = np.asarray(
        [row["delta_score"] for row in rows if not row["gate_accepted"]],
        dtype=np.float64,
    )
    summary.update(
        {
            "complete": True,
            "collection": str(args.collection.resolve()),
            "actor_dataset": str(args.actor_dataset.resolve()),
            "actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "q_checkpoints": [str(path.resolve()) for path in args.q_checkpoint],
            "selection_seed": args.selection_seed,
            "score_margin": score_margin,
            "checkpoint_epoch": int(actor_checkpoint["epoch"]),
            "training_method": actor_checkpoint.get("training_method"),
            "observation_metadata": actor_checkpoint.get("observation_metadata", {}),
            "gate": {
                "k": args.k,
                "support_quantile": args.support_quantile,
                "support_threshold": gate["support_threshold"],
                "support_reference_count": gate["support_reference_count"],
                "q_threshold_quantile": args.q_threshold_quantile,
                "q_advantage_threshold": gate["q_threshold"],
                "uncertainty_weight": args.uncertainty_weight,
                "accepted_states": int(np.count_nonzero(accepted)),
                "intervention_rate": float(np.mean(accepted)),
            },
            "fresh_baseline_audit": {
                "mean_score_drift_from_recorded": float(np.mean(base_replay_drift)),
                "max_abs_score_drift_from_recorded": float(
                    np.max(np.abs(base_replay_drift), initial=0.0)
                ),
                "rejected_branch_max_abs_delta": float(
                    np.max(np.abs(rejected_delta), initial=0.0)
                ),
            },
            "constraints": {
                "max_executed_residual_abs": float(
                    max(row["executed_residual_max_abs"] for row in rows)
                ),
                "max_gripper_residual_abs": float(
                    max(row["gripper_residual_max_abs"] for row in rows)
                ),
                "checkpoint_residual_scale": float(actor_checkpoint["residual_scale"]),
            },
            "rows": rows,
        }
    )
    summary["pass_conditions"] = {
        "positive_mean_delta": summary["delta_score_mean"] > 0.0,
        "improvements_outnumber_regressions": (
            summary["improved_states"] > summary["worsened_states"]
        ),
        "no_success_losses": summary["success_losses"] == 0,
        "deterministic_restore": (
            summary["max_restore_linf"] <= 1e-10
            and summary["max_branch_initial_state_linf"] <= 1e-10
        ),
        "residual_is_bounded": (
            summary["constraints"]["max_executed_residual_abs"]
            <= float(actor_checkpoint["residual_scale"]) + 1e-7
        ),
        "gripper_is_preserved": summary["constraints"]["max_gripper_residual_abs"] == 0.0,
    }
    summary["passed"] = all(summary["pass_conditions"].values())
    _write_json_atomic(args.output_json, summary)
    print(
        json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--actor-dataset", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=20260820)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--support-quantile", type=float, default=0.95)
    parser.add_argument("--q-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--knn-chunk-size", type=int, default=128)
    parser.add_argument("--flush-every", type=int, default=5)
    args = parser.parse_args()
    if args.states <= 0 or args.flush_every <= 0:
        parser.error("states and flush-every must be positive")
    if not 0.0 < args.support_quantile < 1.0:
        parser.error("support-quantile must be between zero and one")
    if not 0.0 < args.q_threshold_quantile < 1.0:
        parser.error("q-threshold-quantile must be between zero and one")
    evaluate(args)


if __name__ == "__main__":
    main()
