#!/usr/bin/env python3
"""Run live wrist-SigLIP residual gating over complete RoboMimic expert episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

from experiments.robomimic.audit_can_residual_q_support import (
    _normalized_support_features,
    kth_neighbor_distance,
)
from experiments.robomimic.collect_can_counterfactual_branches import _success_value
from experiments.robomimic.evaluate_can_deployable_gated_branches import (
    _load_actor,
    _prepare_gate,
    _write_json_atomic,
    conservative_ensemble_advantage,
)
from experiments.robomimic.train_can_pairwise_q import PairwiseQ
from experiments.robomimic.train_can_q_guided_residual_actor import _q_features
from experiments.robomimic.train_can_residual_actor import _features


PROPRIO_KEYS = (
    "robot0_joint_pos",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def padded_action_chunk(
    actions: np.ndarray, start: int, *, chunk_steps: int
) -> tuple[np.ndarray, int]:
    available = min(chunk_steps, len(actions) - start)
    if available <= 0:
        raise ValueError("start must refer to an available action")
    chunk = np.asarray(actions[start : start + available], dtype=np.float32)
    if available < chunk_steps:
        padding = np.repeat(chunk[-1:], chunk_steps - available, axis=0)
        chunk = np.concatenate([chunk, padding], axis=0)
    return chunk, available


def summarize_online_episodes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reward_delta = np.asarray([row["reward_delta"] for row in rows], dtype=np.float64)
    baseline_success = np.asarray([row["baseline_success"] for row in rows], dtype=bool)
    residual_success = np.asarray([row["residual_success"] for row in rows], dtype=bool)
    decisions = sum(row["decisions"] for row in rows)
    interventions = sum(row["interventions"] for row in rows)
    return {
        "episodes": len(rows),
        "baseline_successes": int(np.count_nonzero(baseline_success)),
        "residual_successes": int(np.count_nonzero(residual_success)),
        "success_gains": int(np.count_nonzero(~baseline_success & residual_success)),
        "success_losses": int(np.count_nonzero(baseline_success & ~residual_success)),
        "baseline_success_rate": float(np.mean(baseline_success)),
        "residual_success_rate": float(np.mean(residual_success)),
        "reward_delta_mean": float(np.mean(reward_delta)),
        "reward_delta_median": float(np.median(reward_delta)),
        "reward_improved_episodes": int(np.count_nonzero(reward_delta > 1e-6)),
        "reward_tied_episodes": int(np.count_nonzero(np.abs(reward_delta) <= 1e-6)),
        "reward_worsened_episodes": int(np.count_nonzero(reward_delta < -1e-6)),
        "decisions": int(decisions),
        "interventions": int(interventions),
        "intervention_rate": float(interventions / decisions) if decisions else 0.0,
        "max_restore_linf": float(max(row["restore_linf"] for row in rows)),
        "max_gripper_residual_abs": float(
            max(row["max_gripper_residual_abs"] for row in rows)
        ),
        "max_residual_component_abs": float(
            max(row["max_residual_component_abs"] for row in rows)
        ),
    }


class LiveVisionEncoder:
    def __init__(self, path: Path, device: torch.device) -> None:
        from transformers import SiglipImageProcessor, SiglipVisionModel

        self.path = path.expanduser().resolve()
        self.device = device
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.processor = SiglipImageProcessor.from_pretrained(
            self.path, local_files_only=True
        )
        self.model = SiglipVisionModel.from_pretrained(
            self.path,
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(device).eval()

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        pixel_values = self.processor(
            images=[Image.fromarray(image).convert("RGB")], return_tensors="pt"
        )["pixel_values"].to(device=self.device, dtype=self.dtype)
        feature = self.model(pixel_values=pixel_values).pooler_output.float().cpu().numpy()
        if feature.shape[0] != 1 or not np.all(np.isfinite(feature)):
            raise RuntimeError("Invalid live SigLIP feature")
        return feature[0].astype(np.float32)


def _load_q_models(
    paths: list[Path], device: torch.device
) -> list[tuple[PairwiseQ, dict[str, Any]]]:
    loaded = []
    for path in paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = PairwiseQ(checkpoint["input_dim"], tuple(checkpoint["hidden_dims"])).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval().requires_grad_(False)
        loaded.append((model, checkpoint))
    return loaded


@torch.no_grad()
def _live_q_advantages(
    ensemble: list[tuple[PairwiseQ, dict[str, Any]]],
    state: np.ndarray,
    base_action: np.ndarray,
    proposal: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    state_tensor = torch.from_numpy(state[None]).to(device)
    base_tensor = torch.from_numpy(base_action[None]).to(device)
    proposal_tensor = torch.from_numpy(proposal[None]).to(device)
    advantages = []
    for model, checkpoint in ensemble:
        base_q = model(_q_features(state_tensor, base_tensor, checkpoint))
        proposal_q = model(_q_features(state_tensor, proposal_tensor, checkpoint))
        advantages.append(float((proposal_q - base_q).cpu()))
    return np.asarray(advantages, dtype=np.float32)


def _select_demo_names(mask: np.ndarray, *, episodes: int, seed: int) -> list[str]:
    names = [value.decode() if isinstance(value, bytes) else str(value) for value in mask]
    names.sort(key=lambda value: int(value.rsplit("_", 1)[1]))
    if episodes > len(names):
        raise ValueError(f"Requested {episodes} episodes, only {len(names)} are available")
    order = np.random.default_rng(seed).permutation(len(names))[:episodes]
    return [names[index] for index in order]


def _run_base_episode(env: Any, group: h5py.Group) -> dict[str, Any]:
    initial_state = np.asarray(group["states"][0])
    env.reset_to({"model": str(group.attrs["model_file"]), "states": initial_state})
    restore_linf = float(
        np.max(np.abs(np.asarray(env.get_state()["states"]) - initial_state), initial=0.0)
    )
    reward_sum = 0.0
    success = _success_value(env.is_success())
    for action in np.asarray(group["actions"]):
        _, reward, _, info = env.step(action)
        reward_sum += float(reward)
        success = success or _success_value(info.get("is_success", env.is_success()))
    return {"reward_sum": reward_sum, "success": success, "restore_linf": restore_linf}


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
    encoder_path = args.encoder_path or Path(
        actor_checkpoint["observation_metadata"]["encoder_path"]
    )
    encoder = LiveVisionEncoder(encoder_path, device)
    normalization = actor_checkpoint["normalization"]
    target_shape = tuple(actor_checkpoint["target_shape"])
    chunk_steps = target_shape[0]
    camera_name = actor_checkpoint["observation_metadata"]["camera_name"]

    output_path = args.output_json.expanduser().resolve()
    with h5py.File(args.dataset, "r") as source:
        demos = _select_demo_names(
            np.asarray(source["mask"][args.split]), episodes=args.episodes, seed=args.selection_seed
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
        rows = []
        try:
            for ordinal, demo in enumerate(demos, start=1):
                group = source["data"][demo]
                actions = np.asarray(group["actions"])
                baseline = _run_base_episode(env, group)

                initial_state = np.asarray(group["states"][0])
                obs = env.reset_to(
                    {"model": str(group.attrs["model_file"]), "states": initial_state}
                )
                residual_restore = float(
                    np.max(
                        np.abs(np.asarray(env.get_state()["states"]) - initial_state), initial=0.0
                    )
                )
                reward_sum = 0.0
                success = _success_value(env.is_success())
                decisions = 0
                interventions = 0
                max_residual = 0.0
                max_gripper = 0.0
                gate_rows = []
                for start in range(0, len(actions), chunk_steps):
                    base_chunk, available = padded_action_chunk(
                        actions, start, chunk_steps=chunk_steps
                    )
                    image = env.render(
                        mode="rgb_array",
                        height=args.image_size,
                        width=args.image_size,
                        camera_name=camera_name,
                    )
                    proprio = np.concatenate(
                        [np.asarray(obs[key]).reshape(-1) for key in PROPRIO_KEYS]
                    ).astype(np.float32)
                    state = np.concatenate([encoder(image), proprio]).astype(np.float32)
                    if state.shape != arrays["state"].shape[1:]:
                        raise RuntimeError(
                            f"Live observation shape {state.shape} != training {arrays['state'].shape[1:]}"
                        )
                    feature = _features(state[None], base_chunk[None], normalization)
                    with torch.no_grad():
                        residual = (
                            actor(torch.from_numpy(feature).to(device))
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
                            q_advantages[:, None], uncertainty_weight=args.uncertainty_weight
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
                    executed_chunk = proposal if accepted else base_chunk
                    decisions += 1
                    interventions += int(accepted)
                    executed_residual = residual if accepted else np.zeros_like(residual)
                    max_residual = max(max_residual, float(np.max(np.abs(executed_residual))))
                    max_gripper = max(
                        max_gripper, float(np.max(np.abs(executed_residual[:, -1])))
                    )
                    gate_rows.append(
                        {
                            "start_step": start,
                            "accepted": accepted,
                            "in_support": in_support,
                            "base_support_distance": base_distance,
                            "proposal_support_distance": proposal_distance,
                            "conservative_q_advantage": conservative,
                        }
                    )
                    for action in executed_chunk[:available]:
                        obs, reward, _, info = env.step(action)
                        reward_sum += float(reward)
                        success = success or _success_value(
                            info.get("is_success", env.is_success())
                        )

                row = {
                    "source_demo": demo,
                    "steps": len(actions),
                    "baseline_reward_sum": baseline["reward_sum"],
                    "residual_reward_sum": reward_sum,
                    "reward_delta": reward_sum - baseline["reward_sum"],
                    "baseline_success": int(baseline["success"]),
                    "residual_success": int(success),
                    "decisions": decisions,
                    "interventions": interventions,
                    "intervention_rate": interventions / decisions,
                    "restore_linf": max(baseline["restore_linf"], residual_restore),
                    "max_residual_component_abs": max_residual,
                    "max_gripper_residual_abs": max_gripper,
                    "gate_rows": gate_rows,
                }
                rows.append(row)
                _write_json_atomic(
                    output_path,
                    {
                        "complete": False,
                        "episodes_completed": len(rows),
                        "episodes_requested": len(demos),
                        "rows": rows,
                    },
                )
                print(
                    json.dumps(
                        {
                            "episode": f"{ordinal}/{len(demos)}",
                            "source_demo": demo,
                            "baseline_success": row["baseline_success"],
                            "residual_success": row["residual_success"],
                            "reward_delta": row["reward_delta"],
                            "interventions": interventions,
                        }
                    ),
                    flush=True,
                )
        finally:
            close = getattr(getattr(env, "env", None), "close", None)
            if callable(close):
                close()

    summary = summarize_online_episodes(rows)
    summary.update(
        {
            "complete": True,
            "dataset": str(args.dataset.resolve()),
            "split": args.split,
            "selection_seed": args.selection_seed,
            "actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "q_checkpoints": [str(path.resolve()) for path in args.q_checkpoint],
            "encoder_path": str(encoder.path),
            "gate": {
                "k": args.k,
                "support_quantile": args.support_quantile,
                "support_threshold": calibration["support_threshold"],
                "q_threshold_quantile": args.q_threshold_quantile,
                "q_advantage_threshold": calibration["q_threshold"],
                "uncertainty_weight": args.uncertainty_weight,
            },
            "rows": rows,
        }
    )
    summary["pass_conditions"] = {
        "baseline_replay_succeeds": summary["baseline_successes"] == len(rows),
        "no_success_losses": summary["success_losses"] == 0,
        "nonnegative_mean_reward_delta": summary["reward_delta_mean"] >= 0.0,
        "deterministic_restore": summary["max_restore_linf"] <= 1e-10,
        "bounded_residual": (
            summary["max_residual_component_abs"]
            <= float(actor_checkpoint["residual_scale"]) + 1e-7
        ),
        "gripper_preserved": summary["max_gripper_residual_abs"] == 0.0,
    }
    summary["passed"] = all(summary["pass_conditions"].values())
    _write_json_atomic(output_path, summary)
    print(
        json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--actor-dataset", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=20260820)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--support-quantile", type=float, default=0.95)
    parser.add_argument("--q-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--knn-chunk-size", type=int, default=128)
    args = parser.parse_args()
    if args.episodes <= 0 or args.image_size <= 0:
        parser.error("episodes and image-size must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
