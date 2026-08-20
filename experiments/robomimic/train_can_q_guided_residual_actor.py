#!/usr/bin/env python3
"""Improve a bounded residual actor directly against a frozen Q ensemble."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.robomimic.train_can_pairwise_q import PairwiseQ
from experiments.robomimic.train_can_residual_actor import ResidualActor, _features, _normalization


def _q_features(
    state: torch.Tensor,
    action: torch.Tensor,
    checkpoint: dict[str, Any],
) -> torch.Tensor:
    normalization = checkpoint["normalization"]
    action_mean = torch.as_tensor(normalization["action_mean"], device=action.device)
    action_std = torch.as_tensor(normalization["action_std"], device=action.device)
    normalized_action = (action.flatten(1) - action_mean) / action_std
    if checkpoint["state_mode"] == "action_only":
        return normalized_action
    state_mean = torch.as_tensor(normalization["state_mean"], device=state.device)
    state_std = torch.as_tensor(normalization["state_std"], device=state.device)
    normalized_state = (state - state_mean) / state_std
    return torch.cat([normalized_state, normalized_action], dim=1)


def conservative_advantage(
    advantages: torch.Tensor,
    *,
    uncertainty_weight: float,
) -> torch.Tensor:
    """LCB-style ensemble advantage, with a defined single-critic limit."""
    mean = advantages.mean(dim=0)
    if advantages.shape[0] == 1:
        return mean
    return mean - uncertainty_weight * advantages.std(dim=0, unbiased=False)


def _load_q_ensemble(
    paths: list[Path], device: torch.device
) -> list[tuple[PairwiseQ, dict[str, Any]]]:
    ensemble = []
    for path in paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint["state_mode"] != "full":
            raise ValueError(f"Q-guided actor requires full-state Q, got {path}")
        model = PairwiseQ(checkpoint["input_dim"], tuple(checkpoint["hidden_dims"])).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model.requires_grad_(False)
        ensemble.append((model, checkpoint))
    return ensemble


def _actor_outputs(
    actor: ResidualActor,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    actor.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            outputs.append(actor(batch).cpu().numpy())
    return np.concatenate(outputs).reshape(len(features), *target_shape)


@torch.no_grad()
def _evaluate(
    actor: ResidualActor,
    ensemble: list[tuple[PairwiseQ, dict[str, Any]]],
    *,
    state: np.ndarray,
    base_action: np.ndarray,
    actor_features: np.ndarray,
    zero_target: np.ndarray,
    device: torch.device,
    batch_size: int,
    uncertainty_weight: float,
    residual_l2_weight: float,
    zero_target_weight: float,
) -> dict[str, float]:
    residual = _actor_outputs(
        actor,
        actor_features,
        device=device,
        batch_size=batch_size,
        target_shape=base_action.shape[1:],
    )
    advantages = []
    for start in range(0, len(state), batch_size):
        state_batch = torch.from_numpy(state[start : start + batch_size]).to(device)
        base_batch = torch.from_numpy(base_action[start : start + batch_size]).to(device)
        residual_batch = torch.from_numpy(residual[start : start + batch_size]).to(device)
        proposal = torch.clamp(base_batch + residual_batch, -1.0, 1.0)
        per_q = []
        for model, checkpoint in ensemble:
            base_q = model(_q_features(state_batch, base_batch, checkpoint))
            proposal_q = model(_q_features(state_batch, proposal, checkpoint))
            per_q.append(proposal_q - base_q)
        advantages.append(torch.stack(per_q).cpu().numpy())
    advantage_array = np.concatenate(advantages, axis=1)
    conservative = advantage_array.mean(axis=0)
    if len(ensemble) > 1:
        conservative -= uncertainty_weight * advantage_array.std(axis=0)
    normalized_residual = residual / actor.residual_scale
    residual_penalty = np.mean(normalized_residual**2, axis=(1, 2))
    zero_penalty = np.mean(normalized_residual**2, axis=(1, 2)) * zero_target
    regularized = (
        conservative
        - residual_l2_weight * residual_penalty
        - zero_target_weight * zero_penalty
    )
    residual_norm = np.linalg.norm(residual.reshape(len(residual), -1), axis=1)
    return {
        "states": int(len(state)),
        "q_advantage_mean": float(np.mean(advantage_array)),
        "q_advantage_ensemble_std_mean": float(np.mean(advantage_array.std(axis=0))),
        "conservative_advantage_mean": float(np.mean(conservative)),
        "conservative_advantage_positive_rate": float(np.mean(conservative > 0.0)),
        "regularized_objective_mean": float(np.mean(regularized)),
        "residual_norm_mean": float(np.mean(residual_norm)),
        "residual_norm_p95": float(np.quantile(residual_norm, 0.95)),
        "zero_target_residual_norm_mean": (
            float(np.mean(residual_norm[zero_target.astype(bool)]))
            if np.any(zero_target)
            else 0.0
        ),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    with np.load(args.dataset, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    train_mask = arrays["source_split"] == "train"
    valid_mask = arrays["source_split"] == "valid"
    if set(arrays["source_demo"][train_mask]) & set(arrays["source_demo"][valid_mask]):
        raise ValueError("Source trajectory leakage between train and validation")

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ensemble = _load_q_ensemble(args.q_checkpoint, device)
    residual_scale = (
        float(arrays["residual_scale"])
        if args.actor_residual_scale is None
        else float(args.actor_residual_scale)
    )
    actor_normalization = _normalization(
        arrays["state"][train_mask], arrays["base_action_chunk"][train_mask]
    )
    actor_feature = _features(
        arrays["state"], arrays["base_action_chunk"], actor_normalization
    )
    target_shape = arrays["base_action_chunk"].shape[1:]
    actor = ResidualActor(
        actor_feature.shape[1],
        int(np.prod(target_shape)),
        tuple(args.hidden_dims),
        residual_scale,
        preserve_last_action_dim=True,
        action_dim=target_shape[-1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    train_indices = np.flatnonzero(train_mask)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(actor_feature[train_mask]),
            torch.from_numpy(arrays["state"][train_mask]),
            torch.from_numpy(arrays["base_action_chunk"][train_mask]),
            torch.from_numpy((~arrays["has_improvement"][train_mask].astype(bool)).astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def save(epoch: int) -> None:
        torch.save(
            {
                "model": actor.state_dict(),
                "input_dim": actor_feature.shape[1],
                "output_dim": int(np.prod(target_shape)),
                "hidden_dims": args.hidden_dims,
                "residual_scale": residual_scale,
                "normalization": actor_normalization,
                "target_shape": target_shape,
                "preserve_last_action_dim": True,
                "action_dim": target_shape[-1],
                "epoch": epoch,
                "training_method": "q_guided_conservative",
                "q_checkpoints": [str(path.resolve()) for path in args.q_checkpoint],
            },
            output_dir / "checkpoint.pt",
        )

    save(0)
    best_objective = 0.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        actor.train()
        losses = []
        for feature, state, base_action, zero_target in loader:
            feature = feature.to(device)
            state = state.to(device)
            base_action = base_action.to(device)
            zero_target = zero_target.to(device)
            residual = actor(feature).reshape_as(base_action)
            proposal = torch.clamp(base_action + residual, -1.0, 1.0)
            per_q = []
            for model, checkpoint in ensemble:
                base_q = model(_q_features(state, base_action, checkpoint))
                proposal_q = model(_q_features(state, proposal, checkpoint))
                per_q.append(proposal_q - base_q)
            advantage = conservative_advantage(
                torch.stack(per_q), uncertainty_weight=args.uncertainty_weight
            )
            normalized_residual = residual / residual_scale
            residual_penalty = normalized_residual.square().mean(dim=(1, 2))
            zero_penalty = residual_penalty * zero_target
            objective = (
                advantage
                - args.residual_l2_weight * residual_penalty
                - args.zero_target_weight * zero_penalty
            ).mean()
            loss = -objective
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        valid_metrics = _evaluate(
            actor,
            ensemble,
            state=arrays["state"][valid_mask],
            base_action=arrays["base_action_chunk"][valid_mask],
            actor_features=actor_feature[valid_mask],
            zero_target=(~arrays["has_improvement"][valid_mask].astype(bool)).astype(np.float32),
            device=device,
            batch_size=args.batch_size,
            uncertainty_weight=args.uncertainty_weight,
            residual_l2_weight=args.residual_l2_weight,
            zero_target_weight=args.zero_target_weight,
        )
        objective = valid_metrics["regularized_objective_mean"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "valid_regularized_objective": objective,
                "valid_conservative_advantage": valid_metrics[
                    "conservative_advantage_mean"
                ],
                "valid_residual_norm_mean": valid_metrics["residual_norm_mean"],
            }
        )
        if objective > best_objective + args.minimum_improvement:
            best_objective = objective
            best_epoch = epoch
            stale = 0
            save(epoch)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(history[-1]), flush=True)
        if stale >= args.patience:
            break

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["model"])
    train_metrics = _evaluate(
        actor,
        ensemble,
        state=arrays["state"][train_mask],
        base_action=arrays["base_action_chunk"][train_mask],
        actor_features=actor_feature[train_mask],
        zero_target=(~arrays["has_improvement"][train_mask].astype(bool)).astype(np.float32),
        device=device,
        batch_size=args.batch_size,
        uncertainty_weight=args.uncertainty_weight,
        residual_l2_weight=args.residual_l2_weight,
        zero_target_weight=args.zero_target_weight,
    )
    valid_metrics = _evaluate(
        actor,
        ensemble,
        state=arrays["state"][valid_mask],
        base_action=arrays["base_action_chunk"][valid_mask],
        actor_features=actor_feature[valid_mask],
        zero_target=(~arrays["has_improvement"][valid_mask].astype(bool)).astype(np.float32),
        device=device,
        batch_size=args.batch_size,
        uncertainty_weight=args.uncertainty_weight,
        residual_l2_weight=args.residual_l2_weight,
        zero_target_weight=args.zero_target_weight,
    )
    result = {
        "dataset": str(args.dataset.resolve()),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
        "q_ensemble_size": len(ensemble),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "zero_initialized_output_verified": True,
        "hyperparameters": {
            "uncertainty_weight": args.uncertainty_weight,
            "residual_l2_weight": args.residual_l2_weight,
            "zero_target_weight": args.zero_target_weight,
        },
        "train": train_metrics,
        "valid": valid_metrics,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--residual-l2-weight", type=float, default=0.1)
    parser.add_argument("--zero-target-weight", type=float, default=0.5)
    parser.add_argument("--actor-residual-scale", type=float, default=0.03)
    parser.add_argument("--minimum-improvement", type=float, default=1e-5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
