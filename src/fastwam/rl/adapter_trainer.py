"""AWR training utilities for a frozen residual actor plus a small adapter."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from .awr_trainer import (
    AWRConfig,
    ReplayTensorDataset,
    advantage_weights,
    build_context,
    masked_action_mse,
)
from .models import FrozenResidualAdapterActor, ValueCritic
from .replay_buffer import ReplayBuffer


class PairBehaviorBalancedBatchSampler(Sampler[list[int]]):
    """Uniformly sample task, pair, behavior, then a chunk within the episode."""

    def __init__(
        self,
        task_ids: Sequence[int],
        pair_ids: Sequence[str],
        behaviors: Sequence[str],
        maximum_batch_size: int,
        *,
        generator: torch.Generator,
    ) -> None:
        if not (len(task_ids) == len(pair_ids) == len(behaviors)) or not task_ids:
            raise ValueError("task, pair, and behavior labels must be non-empty and aligned")
        if maximum_batch_size <= 0:
            raise ValueError("maximum_batch_size must be positive")
        groups: dict[int, dict[str, dict[str, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for index, (task, pair, behavior) in enumerate(
            zip(task_ids, pair_ids, behaviors)
        ):
            behavior = str(behavior)
            if behavior not in {"expert", "policy"}:
                raise ValueError(
                    "pair-balanced adapter training requires expert/policy behavior, "
                    f"got {behavior!r}"
                )
            groups[int(task)][str(pair)][behavior].append(index)
        for task, pairs in groups.items():
            for pair, pair_groups in pairs.items():
                if set(pair_groups) != {"expert", "policy"}:
                    raise ValueError(
                        f"task={task} pair={pair!r} must contain expert and policy chunks"
                    )
        self.groups = {
            task: {
                pair: {
                    behavior: torch.tensor(indices, dtype=torch.int64)
                    for behavior, indices in pair_groups.items()
                }
                for pair, pair_groups in pairs.items()
            }
            for task, pairs in groups.items()
        }
        self.tasks = sorted(self.groups)
        self.pairs = {task: sorted(self.groups[task]) for task in self.tasks}
        self.dataset_size = len(task_ids)
        self.maximum_batch_size = int(maximum_batch_size)
        self.num_batches = int(np.ceil(self.dataset_size / self.maximum_batch_size))
        self.generator = generator

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        chunk_queues: dict[tuple[int, str, str], list[int]] = {}
        pair_queues: dict[int, list[str]] = {task: [] for task in self.tasks}
        behavior_counts: Counter[int] = Counter()

        def draw_pair(task: int) -> str:
            if not pair_queues[task]:
                order = torch.randperm(
                    len(self.pairs[task]), generator=self.generator
                ).tolist()
                pair_queues[task] = [self.pairs[task][index] for index in order]
            return pair_queues[task].pop()

        def draw_chunk(task: int, pair: str, behavior: str) -> int:
            key = (task, pair, behavior)
            if not chunk_queues.get(key):
                group = self.groups[task][pair][behavior]
                order = torch.randperm(len(group), generator=self.generator)
                chunk_queues[key] = group[order].tolist()
            return chunk_queues[key].pop()

        sizes = [
            len(batch)
            for batch in torch.tensor_split(
                torch.arange(self.dataset_size), self.num_batches
            )
        ]
        task_offset = int(
            torch.randint(len(self.tasks), (1,), generator=self.generator).item()
        )
        global_slot = 0
        for batch_size in sizes:
            batch = []
            for _ in range(batch_size):
                task = self.tasks[(task_offset + global_slot) % len(self.tasks)]
                pair = draw_pair(task)
                behavior = (
                    "expert" if behavior_counts[task] % 2 == 0 else "policy"
                )
                behavior_counts[task] += 1
                batch.append(draw_chunk(task, pair, behavior))
                global_slot += 1
            order = torch.randperm(len(batch), generator=self.generator).tolist()
            yield [batch[index] for index in order]


def replay_pair_labels(replay: ReplayBuffer) -> tuple[list[int], list[str], list[str]]:
    task_ids = [transition.task_id for transition in replay.transitions]
    pair_ids = []
    behaviors = []
    for transition in replay.transitions:
        suffix = f"-{transition.behavior_mode}"
        if transition.behavior_mode not in {"expert", "policy"} or not transition.episode_id.endswith(
            suffix
        ):
            raise ValueError(
                "adapter replay episode ids must end in expert/policy behavior: "
                f"{transition.episode_id!r}"
            )
        pair_ids.append(transition.episode_id[: -len(suffix)])
        behaviors.append(transition.behavior_mode)
    return task_ids, pair_ids, behaviors


def sampler_epoch_audit(
    sampler: PairBehaviorBalancedBatchSampler,
    *,
    task_ids: Sequence[int],
    pair_ids: Sequence[str],
    behaviors: Sequence[str],
) -> dict:
    indices = [index for batch in sampler for index in batch]
    if len(indices) != len(task_ids):
        raise RuntimeError("sampler epoch size differs from replay size")
    task_counts = Counter(int(task_ids[index]) for index in indices)
    behavior_counts = Counter(str(behaviors[index]) for index in indices)
    task_behavior_counts = Counter(
        f"task{int(task_ids[index])}/{behaviors[index]}" for index in indices
    )
    pair_counts = Counter(str(pair_ids[index]) for index in indices)
    return {
        "samples": len(indices),
        "task_counts": dict(sorted(task_counts.items())),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "task_behavior_counts": dict(sorted(task_behavior_counts.items())),
        "pair_count_min": min(pair_counts.values()),
        "pair_count_max": max(pair_counts.values()),
        "unique_pairs": len(pair_counts),
    }


def masked_temporal_smoothness(
    adapter_residual: torch.Tensor, effective_k: torch.Tensor
) -> torch.Tensor:
    if adapter_residual.ndim != 3:
        raise ValueError("adapter_residual must have shape [B,H,D]")
    if effective_k.shape != (adapter_residual.shape[0],):
        raise ValueError("effective_k must have shape [B]")
    if adapter_residual.shape[1] == 1:
        return torch.zeros(
            adapter_residual.shape[0],
            dtype=adapter_residual.dtype,
            device=adapter_residual.device,
        )
    difference = adapter_residual[:, 1:] - adapter_residual[:, :-1]
    valid = (
        torch.arange(adapter_residual.shape[1] - 1, device=adapter_residual.device)
        .unsqueeze(0)
        .lt((effective_k - 1).clamp_min(0).unsqueeze(1))
        .to(adapter_residual.dtype)
    )
    per_step = torch.mean(torch.square(difference), dim=-1)
    return torch.sum(per_step * valid, dim=1) / valid.sum(dim=1).clamp_min(1.0)


def train_residual_adapter_awr(
    actor: FrozenResidualAdapterActor,
    critic: ValueCritic,
    replay: ReplayBuffer,
    returns: np.ndarray,
    config: AWRConfig,
    *,
    anchor_weight: float,
    smoothness_weight: float,
    device: torch.device | str,
    epoch_end_callback: Callable[[int, FrozenResidualAdapterActor, ValueCritic], None]
    | None = None,
) -> tuple[list[dict[str, float]], dict]:
    config.validate()
    if anchor_weight < 0.0 or smoothness_weight < 0.0:
        raise ValueError("adapter loss weights must be non-negative")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(device)
    actor.to(device)
    actor.base_actor.eval().requires_grad_(False)
    critic.to(device)
    dataset = ReplayTensorDataset(replay, returns)
    task_ids, pair_ids, behaviors = replay_pair_labels(replay)
    audit_sampler = PairBehaviorBalancedBatchSampler(
        task_ids,
        pair_ids,
        behaviors,
        min(config.batch_size, len(dataset)),
        generator=torch.Generator().manual_seed(config.seed),
    )
    sampler_audit = sampler_epoch_audit(
        audit_sampler,
        task_ids=task_ids,
        pair_ids=pair_ids,
        behaviors=behaviors,
    )
    sampler = PairBehaviorBalancedBatchSampler(
        task_ids,
        pair_ids,
        behaviors,
        min(config.batch_size, len(dataset)),
        generator=torch.Generator().manual_seed(config.seed),
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    adapter_optimizer = torch.optim.AdamW(
        actor.adapter.parameters(),
        lr=config.actor_learning_rate,
        weight_decay=config.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.critic_learning_rate,
        weight_decay=config.weight_decay,
    )
    history = []
    for epoch in range(config.epochs):
        sums: dict[str, float] = {}
        examples = 0
        actor.adapter.train()
        actor.base_actor.eval()
        critic.train()
        for cpu_batch in loader:
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            context = build_context(
                batch["observation_feature"],
                batch["proprio"],
                batch["goal_feature"],
                use_goal_conditioning=config.use_goal_conditioning,
            )
            language = batch.get("language_feature")
            critic_optimizer.zero_grad(set_to_none=True)
            values = critic(
                context,
                baseline_actions=batch["baseline_actions"],
                language_feature=language,
            )
            critic_loss = torch.mean(torch.square(values - batch["return_to_go"]))
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            critic_optimizer.step()

            adapter_optimizer.zero_grad(set_to_none=True)
            values = critic(
                context,
                baseline_actions=batch["baseline_actions"],
                language_feature=language,
            )
            weights = advantage_weights(
                batch["return_to_go"],
                values,
                beta=config.beta,
                maximum=config.max_advantage_weight,
                normalize=config.normalize_advantage_weights,
            )
            base_actions, _, adapter_residual = actor.components(
                context,
                batch["baseline_actions"],
                language_feature=language,
            )
            corrected = base_actions + adapter_residual
            bounded = torch.maximum(
                torch.minimum(corrected, actor.base_actor.action_high),
                actor.base_actor.action_low,
            )
            predicted = torch.where(
                actor.adapter.adapter_scale == 0,
                base_actions,
                bounded,
            )
            action_error = masked_action_mse(
                predicted,
                batch["executed_actions"],
                batch["effective_k"],
            )
            anchor = torch.mean(torch.square(adapter_residual), dim=(1, 2))
            smoothness = masked_temporal_smoothness(
                adapter_residual, batch["effective_k"]
            )
            awr_loss = torch.mean(weights * action_error)
            actor_loss = (
                awr_loss
                + anchor_weight * anchor.mean()
                + smoothness_weight * smoothness.mean()
            )
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.adapter.parameters(), config.max_grad_norm)
            adapter_optimizer.step()

            metrics = {
                "actor_loss": actor_loss,
                "awr_loss": awr_loss,
                "critic_loss": critic_loss,
                "mean_value": values.mean(),
                "mean_return": batch["return_to_go"].mean(),
                "mean_advantage_weight": weights.mean(),
                "max_advantage_weight": weights.max(),
                "mean_action_mse": action_error.mean(),
                "mean_anchor_loss": anchor.mean(),
                "mean_smoothness_loss": smoothness.mean(),
                "adapter_rms": torch.sqrt(torch.mean(torch.square(adapter_residual))),
                "adapter_max_abs": torch.max(torch.abs(adapter_residual)),
            }
            batch_examples = int(batch["return_to_go"].shape[0])
            examples += batch_examples
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_examples
        history.append(
            {"epoch": float(epoch), **{key: value / examples for key, value in sums.items()}}
        )
        if epoch_end_callback is not None:
            epoch_end_callback(epoch + 1, actor, critic)
    return history, sampler_audit
