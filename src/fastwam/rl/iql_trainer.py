"""Offline Implicit Q-Learning for FastWAM residual action chunks.

FastWAM remains frozen.  IQL learns two action-value critics, an expectile value
critic, and the same bounded deterministic residual actor used by the AWR
baseline.  This provides a stronger offline-RL test of the imagination reward
without evaluating out-of-distribution actions in the Bellman target.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .awr_trainer import (
    BalancedBatchSampler,
    TaskBalancedBatchSampler,
    build_context,
    masked_action_mse,
)
from .models import ActionValueCritic, ResidualActor, ValueCritic
from .replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class IQLConfig:
    gamma: float = 0.99
    expectile: float = 0.7
    advantage_temperature: float = 3.0
    max_advantage_weight: float = 100.0
    actor_learning_rate: float = 3e-4
    q_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    target_tau: float = 0.005
    batch_size: int = 64
    epochs: int = 20
    max_grad_norm: float = 1.0
    use_goal_conditioning: bool = False
    balance_tasks: bool = True
    bootstrap_timeouts: bool = False
    seed: int = 42

    def validate(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < self.expectile < 1.0:
            raise ValueError("expectile must be in (0, 1)")
        if self.advantage_temperature <= 0.0 or self.max_advantage_weight <= 0.0:
            raise ValueError("advantage temperature and maximum weight must be positive")
        if min(
            self.actor_learning_rate,
            self.q_learning_rate,
            self.value_learning_rate,
        ) <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 < self.target_tau <= 1.0:
            raise ValueError("target_tau must be in (0, 1]")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")


def expectile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    expectile: float,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("expectile prediction and target must have identical shapes")
    if not 0.0 < expectile < 1.0:
        raise ValueError("expectile must be in (0, 1)")
    difference = target.detach() - prediction
    weights = torch.where(difference >= 0.0, expectile, 1.0 - expectile)
    return torch.mean(weights * torch.square(difference))


def iql_advantage_weights(
    q_values: torch.Tensor,
    values: torch.Tensor,
    *,
    temperature: float,
    maximum: float,
) -> torch.Tensor:
    if q_values.shape != values.shape:
        raise ValueError("q_values and values must have identical shapes")
    if temperature <= 0.0 or maximum <= 0.0:
        raise ValueError("temperature and maximum must be positive")
    log_weights = temperature * (q_values - values).detach()
    return torch.exp(torch.clamp(log_weights, max=20.0)).clamp(max=maximum)


def _canonical_executed_actions(
    executed_actions: np.ndarray,
    baseline_actions: np.ndarray,
    effective_k: np.ndarray,
) -> np.ndarray:
    """Replace padded action suffixes with the baseline before Q conditioning."""

    result = np.asarray(executed_actions, dtype=np.float32).copy()
    baseline = np.asarray(baseline_actions, dtype=np.float32)
    horizon = result.shape[1]
    padding = np.arange(horizon)[None, :] >= np.asarray(effective_k)[:, None]
    result[padding] = baseline[padding]
    return result


class IQLReplayTensorDataset(Dataset):
    def __init__(
        self,
        replay: ReplayBuffer,
        rewards: np.ndarray,
        config: IQLConfig,
    ):
        arrays = replay.arrays()
        rewards = np.asarray(rewards, dtype=np.float32)
        if rewards.shape != (len(replay),) or np.any(~np.isfinite(rewards)):
            raise ValueError(f"rewards must be finite with shape {(len(replay),)}")

        next_baseline_actions = arrays["baseline_actions"].copy()
        episode_steps = {
            (transition.episode_id, transition.transition_index): index
            for index, transition in enumerate(replay.transitions)
        }
        for index, transition in enumerate(replay.transitions):
            next_index = episode_steps.get(
                (transition.episode_id, transition.transition_index + 1)
            )
            if next_index is not None:
                next_baseline_actions[index] = arrays["baseline_actions"][next_index]

        executed_actions = _canonical_executed_actions(
            arrays["executed_actions"],
            arrays["baseline_actions"],
            arrays["effective_k"],
        )
        terminated = arrays["terminated"].astype(np.bool_)
        truncated = arrays["truncated"].astype(np.bool_)
        bootstrap_mask = ~terminated
        if not config.bootstrap_timeouts:
            bootstrap_mask &= ~truncated

        self.tensors = {
            "observation_feature": torch.from_numpy(arrays["observation_feature"]),
            "next_observation_feature": torch.from_numpy(
                arrays["next_observation_feature"]
            ),
            "goal_feature": torch.from_numpy(arrays["goal_feature"]),
            "proprio": torch.from_numpy(arrays["proprio"]),
            "next_proprio": torch.from_numpy(arrays["next_proprio"]),
            "baseline_actions": torch.from_numpy(arrays["baseline_actions"]),
            "next_baseline_actions": torch.from_numpy(next_baseline_actions),
            "executed_actions": torch.from_numpy(executed_actions),
            "effective_k": torch.from_numpy(arrays["effective_k"]),
            "reward": torch.from_numpy(rewards),
            "bootstrap_mask": torch.from_numpy(bootstrap_mask.astype(np.float32)),
        }
        if "language_feature" in arrays:
            self.tensors["language_feature"] = torch.from_numpy(
                arrays["language_feature"]
            )
        task_keys = [(item.task_suite, item.task_id) for item in replay.transitions]
        unique_tasks = {key: index for index, key in enumerate(sorted(set(task_keys)))}
        self.task_group_ids = torch.tensor(
            [unique_tasks[key] for key in task_keys], dtype=torch.int64
        )

    def __len__(self) -> int:
        return int(self.tensors["reward"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


def compute_iql_losses(
    actor: ResidualActor,
    q_critics: tuple[ActionValueCritic, ActionValueCritic],
    target_q_critics: tuple[ActionValueCritic, ActionValueCritic],
    value_critic: ValueCritic,
    batch: Mapping[str, torch.Tensor],
    config: IQLConfig,
) -> dict[str, torch.Tensor]:
    context = build_context(
        batch["observation_feature"],
        batch["proprio"],
        batch["goal_feature"],
        use_goal_conditioning=config.use_goal_conditioning,
    )
    next_context = build_context(
        batch["next_observation_feature"],
        batch["next_proprio"],
        batch["goal_feature"],
        use_goal_conditioning=config.use_goal_conditioning,
    )
    language_feature = batch.get("language_feature")
    target_q_values = torch.minimum(
        *[
            critic(
                context,
                batch["baseline_actions"],
                batch["executed_actions"],
                language_feature,
            )
            for critic in target_q_critics
        ]
    ).detach()
    values = value_critic(
        context,
        baseline_actions=batch["baseline_actions"],
        language_feature=language_feature,
    )
    value_loss = expectile_loss(values, target_q_values, expectile=config.expectile)

    with torch.no_grad():
        next_values = value_critic(
            next_context,
            baseline_actions=batch["next_baseline_actions"],
            language_feature=language_feature,
        )
        discounts = torch.pow(
            torch.full_like(batch["reward"], config.gamma),
            batch["effective_k"].to(batch["reward"].dtype),
        )
        q_target = (
            batch["reward"]
            + discounts * batch["bootstrap_mask"] * next_values
        )
    q_values = [
        critic(
            context,
            batch["baseline_actions"],
            batch["executed_actions"],
            language_feature,
        )
        for critic in q_critics
    ]
    q_loss = sum(torch.mean(torch.square(value - q_target)) for value in q_values)

    actor_actions = actor(
        context,
        batch["baseline_actions"],
        language_feature=language_feature,
    )
    action_error = masked_action_mse(
        actor_actions,
        batch["executed_actions"],
        batch["effective_k"],
    )
    advantage_weight = iql_advantage_weights(
        target_q_values,
        values,
        temperature=config.advantage_temperature,
        maximum=config.max_advantage_weight,
    )
    actor_loss = torch.mean(advantage_weight * action_error)
    return {
        "actor_loss": actor_loss,
        "q_loss": q_loss,
        "value_loss": value_loss,
        "mean_q": torch.mean(torch.stack(q_values)),
        "mean_target_q": target_q_values.mean(),
        "mean_value": values.mean(),
        "mean_advantage_weight": advantage_weight.mean(),
        "max_advantage_weight": advantage_weight.max(),
        "mean_action_mse": action_error.mean(),
    }


@torch.no_grad()
def _soft_update(
    targets: tuple[ActionValueCritic, ActionValueCritic],
    sources: tuple[ActionValueCritic, ActionValueCritic],
    tau: float,
) -> None:
    for target, source in zip(targets, sources):
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.lerp_(source_parameter, tau)


def train_residual_iql(
    actor: ResidualActor,
    q_critics: tuple[ActionValueCritic, ActionValueCritic],
    value_critic: ValueCritic,
    replay: ReplayBuffer,
    rewards: np.ndarray,
    config: IQLConfig,
    *,
    device: torch.device | str,
) -> tuple[list[dict[str, float]], tuple[ActionValueCritic, ActionValueCritic]]:
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(device)
    actor.to(device)
    for critic in q_critics:
        critic.to(device)
    value_critic.to(device)
    target_q_critics = tuple(copy.deepcopy(critic).to(device).eval() for critic in q_critics)
    for critic in target_q_critics:
        critic.requires_grad_(False)

    dataset = IQLReplayTensorDataset(replay, rewards, config)
    generator = torch.Generator().manual_seed(config.seed)
    sampler_type = TaskBalancedBatchSampler if config.balance_tasks else BalancedBatchSampler
    sampler_input = dataset.task_group_ids if config.balance_tasks else len(dataset)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler_type(
            sampler_input,
            min(config.batch_size, len(dataset)),
            generator=generator,
        ),
    )
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=config.actor_learning_rate,
        weight_decay=config.weight_decay,
    )
    q_optimizer = torch.optim.AdamW(
        [parameter for critic in q_critics for parameter in critic.parameters()],
        lr=config.q_learning_rate,
        weight_decay=config.weight_decay,
    )
    value_optimizer = torch.optim.AdamW(
        value_critic.parameters(),
        lr=config.value_learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        sums: dict[str, float] = {}
        examples = 0
        actor.train()
        for critic in q_critics:
            critic.train()
        value_critic.train()
        for cpu_batch in loader:
            batch = {key: value.to(device) for key, value in cpu_batch.items()}

            value_optimizer.zero_grad(set_to_none=True)
            losses = compute_iql_losses(
                actor,
                q_critics,
                target_q_critics,
                value_critic,
                batch,
                config,
            )
            losses["value_loss"].backward()
            nn.utils.clip_grad_norm_(value_critic.parameters(), config.max_grad_norm)
            value_optimizer.step()

            q_optimizer.zero_grad(set_to_none=True)
            losses = compute_iql_losses(
                actor,
                q_critics,
                target_q_critics,
                value_critic,
                batch,
                config,
            )
            losses["q_loss"].backward()
            nn.utils.clip_grad_norm_(
                [parameter for critic in q_critics for parameter in critic.parameters()],
                config.max_grad_norm,
            )
            q_optimizer.step()

            actor_optimizer.zero_grad(set_to_none=True)
            losses = compute_iql_losses(
                actor,
                q_critics,
                target_q_critics,
                value_critic,
                batch,
                config,
            )
            losses["actor_loss"].backward()
            nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            actor_optimizer.step()
            _soft_update(target_q_critics, q_critics, config.target_tau)

            batch_examples = int(batch["reward"].shape[0])
            examples += batch_examples
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_examples
        history.append(
            {"epoch": float(epoch), **{key: value / examples for key, value in sums.items()}}
        )
    return history, target_q_critics


def export_iql_config(config: IQLConfig) -> dict:
    config.validate()
    return asdict(config)
