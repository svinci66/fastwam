"""Offline advantage-weighted regression for a small residual action head.

This module intentionally does not update FastWAM's flow policy.  It is the first
resource-feasible control experiment: FastWAM remains an immutable action prior and
the learner predicts only bounded residual actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .models import ResidualActor, ValueCritic
from .replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class AWRConfig:
    gamma: float = 0.99
    beta: float = 1.0
    max_advantage_weight: float = 20.0
    normalize_advantage_weights: bool = True
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 20
    max_grad_norm: float = 1.0
    use_goal_conditioning: bool = False
    seed: int = 42

    def validate(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.beta <= 0.0 or self.max_advantage_weight <= 0.0:
            raise ValueError("beta and max_advantage_weight must be positive")
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")


def build_context(
    observation_feature: torch.Tensor,
    proprio: torch.Tensor,
    goal_feature: torch.Tensor,
    *,
    use_goal_conditioning: bool,
) -> torch.Tensor:
    tensors = [observation_feature, proprio]
    if use_goal_conditioning:
        tensors.append(goal_feature)
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("all context tensors must have shape [B, D]")
    if len({tensor.shape[0] for tensor in tensors}) != 1:
        raise ValueError("all context tensors must share the same batch size")
    return torch.cat(tensors, dim=-1)


def advantage_weights(
    returns: torch.Tensor,
    values: torch.Tensor,
    *,
    beta: float,
    maximum: float,
    normalize: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    if returns.shape != values.shape:
        raise ValueError("returns and values must have identical shapes")
    if beta <= 0.0 or maximum <= 0.0:
        raise ValueError("beta and maximum must be positive")
    advantage = (returns - values).detach()
    weights = torch.exp(torch.clamp(advantage / beta, max=20.0))
    if normalize:
        weights = weights / weights.mean().clamp_min(eps)
    return weights.clamp(max=maximum)


def masked_action_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    effective_k: torch.Tensor,
) -> torch.Tensor:
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target must share shape [B, H, action_dim]")
    if effective_k.shape != (predicted.shape[0],):
        raise ValueError("effective_k must have shape [B]")
    horizon = predicted.shape[1]
    if torch.any(effective_k <= 0) or torch.any(effective_k > horizon):
        raise ValueError("effective_k values must be within the action horizon")
    mask = (
        torch.arange(horizon, device=predicted.device).unsqueeze(0)
        < effective_k.to(device=predicted.device).unsqueeze(1)
    ).to(predicted.dtype)
    per_step = torch.mean(torch.square(predicted - target), dim=-1)
    return torch.sum(per_step * mask, dim=1) / mask.sum(dim=1).clamp_min(1.0)


def compute_awr_losses(
    actor: ResidualActor,
    critic: ValueCritic,
    batch: Mapping[str, torch.Tensor],
    config: AWRConfig,
) -> dict[str, torch.Tensor]:
    context = build_context(
        batch["observation_feature"],
        batch["proprio"],
        batch["goal_feature"],
        use_goal_conditioning=config.use_goal_conditioning,
    )
    values = critic(context)
    returns = batch["return_to_go"]
    critic_loss = torch.mean(torch.square(values - returns))
    weights = advantage_weights(
        returns,
        values,
        beta=config.beta,
        maximum=config.max_advantage_weight,
        normalize=config.normalize_advantage_weights,
    )
    predicted_actions = actor(context, batch["baseline_actions"])
    action_error = masked_action_mse(
        predicted_actions,
        batch["executed_actions"],
        batch["effective_k"],
    )
    actor_loss = torch.mean(weights * action_error)
    return {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "mean_value": values.mean(),
        "mean_return": returns.mean(),
        "mean_advantage_weight": weights.mean(),
        "max_advantage_weight": weights.max(),
        "mean_action_mse": action_error.mean(),
    }


class ReplayTensorDataset(Dataset):
    def __init__(self, replay: ReplayBuffer, returns: np.ndarray):
        arrays = replay.arrays()
        returns = np.asarray(returns, dtype=np.float32)
        if returns.shape != (len(replay),):
            raise ValueError(f"returns must have shape {(len(replay),)}, got {returns.shape}")
        self.tensors = {
            "observation_feature": torch.from_numpy(arrays["observation_feature"]),
            "goal_feature": torch.from_numpy(arrays["goal_feature"]),
            "proprio": torch.from_numpy(arrays["proprio"]),
            "baseline_actions": torch.from_numpy(arrays["baseline_actions"]),
            "executed_actions": torch.from_numpy(arrays["executed_actions"]),
            "effective_k": torch.from_numpy(arrays["effective_k"]),
            "return_to_go": torch.from_numpy(returns),
        }

    def __len__(self) -> int:
        return int(self.tensors["return_to_go"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


def train_residual_awr(
    actor: ResidualActor,
    critic: ValueCritic,
    replay: ReplayBuffer,
    returns: np.ndarray,
    config: AWRConfig,
    *,
    device: torch.device | str,
) -> list[dict[str, float]]:
    """Train the small learner.  FastWAM is not accepted as an argument by design."""

    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(device)
    actor.to(device)
    critic.to(device)
    dataset = ReplayTensorDataset(replay, returns)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=config.actor_learning_rate, weight_decay=config.weight_decay
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=config.critic_learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        sums: dict[str, float] = {}
        batches = 0
        actor.train()
        critic.train()
        for cpu_batch in loader:
            batch = {key: value.to(device) for key, value in cpu_batch.items()}

            critic_optimizer.zero_grad(set_to_none=True)
            context = build_context(
                batch["observation_feature"],
                batch["proprio"],
                batch["goal_feature"],
                use_goal_conditioning=config.use_goal_conditioning,
            )
            critic_values = critic(context)
            critic_loss = torch.mean(torch.square(critic_values - batch["return_to_go"]))
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            critic_optimizer.step()

            actor_optimizer.zero_grad(set_to_none=True)
            losses = compute_awr_losses(actor, critic, batch, config)
            losses["actor_loss"].backward()
            nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            actor_optimizer.step()

            batches += 1
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
        history.append({"epoch": float(epoch), **{key: value / batches for key, value in sums.items()}})
    return history


def export_awr_config(config: AWRConfig) -> dict:
    config.validate()
    return asdict(config)
