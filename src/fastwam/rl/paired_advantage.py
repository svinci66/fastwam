"""Supervised paired-outcome gate for conservative residual intervention.

The gate does not regress the absolute return used by IQL.  It learns whether a
candidate action chunk came from an episode that outperformed the clean
FastWAM episode with the same task and environment seed.  Equal controlled-
corruption outcomes are left unlabeled.  Executed residual ties can optionally
be labeled as non-improving negatives so the gate cannot learn only from the
rare successful residual rollouts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .models import ActionValueCritic
from .replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class PairedAdvantageTrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 30
    max_grad_norm: float = 1.0
    validation_seed_modulus: int = 5
    validation_seed_remainder: int = 4
    include_residual_equal_outcomes_as_negative: bool = False
    seed: int = 42

    def validate(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer settings")
        if self.batch_size <= 0 or self.epochs <= 0 or self.max_grad_norm <= 0.0:
            raise ValueError("batch_size, epochs, and max_grad_norm must be positive")
        if self.validation_seed_modulus <= 1:
            raise ValueError("validation_seed_modulus must be greater than one")
        if not 0 <= self.validation_seed_remainder < self.validation_seed_modulus:
            raise ValueError("validation_seed_remainder must be inside the modulus")


@dataclass(frozen=True)
class PairedAdvantageExamples:
    indices: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    episode_ids: tuple[str, ...]
    split: str

    def __len__(self) -> int:
        return int(self.indices.size)


def _canonical_actions(
    executed: np.ndarray, baseline: np.ndarray, effective_k: np.ndarray
) -> np.ndarray:
    result = np.asarray(executed, dtype=np.float32).copy()
    horizon = result.shape[1]
    padding = np.arange(horizon)[None, :] >= np.asarray(effective_k)[:, None]
    result[padding] = np.asarray(baseline, dtype=np.float32)[padding]
    return result


def build_paired_advantage_examples(
    replay: ReplayBuffer,
    config: PairedAdvantageTrainingConfig,
    *,
    split: str,
    behavior_modes: tuple[str, ...] | None = None,
) -> PairedAdvantageExamples:
    """Create trajectory-grouped labels without transition-level leakage."""

    config.validate()
    if split not in {"train", "validation", "all"}:
        raise ValueError("split must be 'train', 'validation', or 'all'")
    if behavior_modes is not None:
        behavior_modes = tuple(str(value) for value in behavior_modes)
        if not behavior_modes or any(not value for value in behavior_modes):
            raise ValueError("behavior_modes must contain at least one non-empty value")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, transition in enumerate(replay.transitions):
        grouped[transition.episode_id].append(index)

    episode_rows: dict[str, dict[str, Any]] = {}
    baselines: dict[tuple[str, int, int], str] = {}
    for episode_id, indices in grouped.items():
        indices.sort(key=lambda value: replay.transitions[value].transition_index)
        first = replay.transitions[indices[0]]
        row = {
            "indices": indices,
            "task_key": (first.task_suite, first.task_id, first.env_seed),
            "behavior": first.behavior_mode,
            "success": any(replay.transitions[index].success for index in indices),
            "env_seed": first.env_seed,
        }
        episode_rows[episode_id] = row
        if first.behavior_mode == "policy":
            key = row["task_key"]
            if key in baselines:
                raise ValueError(f"multiple clean policy episodes for pairing key {key}")
            baselines[key] = episode_id

    selected: list[tuple[str, list[int], int]] = []
    for episode_id, row in sorted(episode_rows.items()):
        if row["behavior"] == "policy":
            continue
        if behavior_modes is not None and row["behavior"] not in behavior_modes:
            continue
        baseline_id = baselines.get(row["task_key"])
        if baseline_id is None:
            continue
        baseline_success = bool(episode_rows[baseline_id]["success"])
        candidate_success = bool(row["success"])
        if baseline_success == candidate_success:
            if not (
                config.include_residual_equal_outcomes_as_negative
                and row["behavior"] == "residual"
            ):
                continue
            label = 0
        else:
            label = int(candidate_success and not baseline_success)
        is_validation = (
            int(row["env_seed"]) % config.validation_seed_modulus
            == config.validation_seed_remainder
        )
        if split != "all" and is_validation != (split == "validation"):
            continue
        selected.append(
            (episode_id, list(row["indices"]), label)
        )

    labels_present = {label for _, _, label in selected}
    if labels_present != {0, 1}:
        raise ValueError(
            f"paired {split} split must contain both labels, got {sorted(labels_present)}"
        )
    episodes_per_class = {
        label: sum(int(item_label == label) for _, _, item_label in selected)
        for label in (0, 1)
    }
    indices: list[int] = []
    labels: list[float] = []
    weights: list[float] = []
    episode_ids: list[str] = []
    for episode_id, episode_indices, label in selected:
        example_weight = 1.0 / (
            episodes_per_class[label] * len(episode_indices)
        )
        for index in episode_indices:
            indices.append(index)
            labels.append(float(label))
            weights.append(example_weight)
            episode_ids.append(episode_id)
    return PairedAdvantageExamples(
        indices=np.asarray(indices, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float64),
        episode_ids=tuple(episode_ids),
        split=split,
    )


class PairedAdvantageDataset(Dataset):
    def __init__(
        self,
        replay: ReplayBuffer,
        examples: PairedAdvantageExamples,
        *,
        context_override: np.ndarray | None = None,
        arrays_override: Mapping[str, np.ndarray] | None = None,
    ):
        arrays = replay.arrays() if arrays_override is None else arrays_override
        indices = examples.indices
        if context_override is None:
            context = np.concatenate(
                [arrays["observation_feature"], arrays["proprio"]], axis=1
            ).astype(np.float32)
        else:
            context = np.asarray(context_override, dtype=np.float32)
            if context.ndim != 2 or context.shape[0] != len(replay.transitions):
                raise ValueError(
                    "context_override must have shape "
                    f"[replay_size, context_dim], got {context.shape}"
                )
            if not np.all(np.isfinite(context)):
                raise ValueError("context_override must contain only finite values")
        actions = _canonical_actions(
            arrays["executed_actions"],
            arrays["baseline_actions"],
            arrays["effective_k"],
        )
        self.tensors = {
            "context": torch.from_numpy(context[indices]),
            "baseline_actions": torch.from_numpy(arrays["baseline_actions"][indices]),
            "candidate_actions": torch.from_numpy(actions[indices]),
            "label": torch.from_numpy(examples.labels),
        }
        if "language_feature" in arrays:
            self.tensors["language_feature"] = torch.from_numpy(
                arrays["language_feature"][indices]
            )

    def __len__(self) -> int:
        return int(self.tensors["label"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


def _logits(model: ActionValueCritic, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["context"],
        batch["baseline_actions"],
        batch["candidate_actions"],
        batch.get("language_feature"),
    )


def train_paired_advantage_ensemble(
    models: tuple[ActionValueCritic, ActionValueCritic],
    replay: ReplayBuffer,
    train_examples: PairedAdvantageExamples,
    config: PairedAdvantageTrainingConfig,
    *,
    device: torch.device | str,
    context_override: np.ndarray | None = None,
    arrays_override: Mapping[str, np.ndarray] | None = None,
) -> list[list[dict[str, float]]]:
    config.validate()
    device = torch.device(device)
    dataset = PairedAdvantageDataset(
        replay,
        train_examples,
        context_override=context_override,
        arrays_override=arrays_override,
    )
    histories: list[list[dict[str, float]]] = []
    for model_index, model in enumerate(models):
        model_seed = config.seed + 1009 * model_index
        torch.manual_seed(model_seed)
        generator = torch.Generator().manual_seed(model_seed)
        sampler = WeightedRandomSampler(
            torch.as_tensor(train_examples.weights, dtype=torch.double),
            num_samples=len(train_examples),
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(
            dataset,
            batch_size=min(config.batch_size, len(dataset)),
            sampler=sampler,
            generator=generator,
        )
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        history: list[dict[str, float]] = []
        for epoch in range(config.epochs):
            model.train()
            loss_sum = 0.0
            correct = 0
            examples = 0
            for cpu_batch in loader:
                batch = {key: value.to(device) for key, value in cpu_batch.items()}
                optimizer.zero_grad(set_to_none=True)
                logits = _logits(model, batch)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, batch["label"]
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                count = int(batch["label"].numel())
                examples += count
                loss_sum += float(loss.detach().cpu()) * count
                correct += int(
                    ((logits.detach() >= 0.0) == (batch["label"] >= 0.5)).sum()
                )
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": loss_sum / examples,
                    "accuracy": correct / examples,
                }
            )
        model.eval().requires_grad_(False)
        histories.append(history)
    return histories


@torch.no_grad()
def predict_paired_advantage(
    models: tuple[ActionValueCritic, ActionValueCritic],
    replay: ReplayBuffer,
    examples: PairedAdvantageExamples,
    *,
    device: torch.device | str,
    batch_size: int = 256,
    context_override: np.ndarray | None = None,
    arrays_override: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    dataset = PairedAdvantageDataset(
        replay,
        examples,
        context_override=context_override,
        arrays_override=arrays_override,
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    columns: list[list[np.ndarray]] = [[], []]
    device = torch.device(device)
    for cpu_batch in loader:
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        for index, model in enumerate(models):
            model.to(device).eval()
            columns[index].append(torch.sigmoid(_logits(model, batch)).cpu().numpy())
    return np.stack([np.concatenate(column) for column in columns], axis=1)


def build_temporal_context(
    replay: ReplayBuffer,
    *,
    history_length: int = 1,
    arrays_override: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build current state plus within-episode lagged state deltas.

    A history length of one exactly matches the existing paired gate context:
    concatenated frozen visual features and proprioception.  Longer histories
    append ``current - previous`` state deltas for preceding replans.  Missing
    history at the start of an episode is represented by zero deltas, and no
    information is ever borrowed across episode boundaries.
    """

    if history_length <= 0:
        raise ValueError("history_length must be positive")
    arrays = replay.arrays() if arrays_override is None else arrays_override
    current = np.concatenate(
        [arrays["observation_feature"], arrays["proprio"]], axis=1
    ).astype(np.float32)
    if history_length == 1:
        return current

    context = np.zeros(
        (current.shape[0], current.shape[1] * history_length), dtype=np.float32
    )
    context[:, : current.shape[1]] = current
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, transition in enumerate(replay.transitions):
        grouped[transition.episode_id].append(index)
    width = current.shape[1]
    for indices in grouped.values():
        indices.sort(key=lambda value: replay.transitions[value].transition_index)
        for position, index in enumerate(indices):
            for lag in range(1, history_length):
                if position < lag:
                    continue
                previous = indices[position - lag]
                start = lag * width
                context[index, start : start + width] = current[index] - current[previous]
    return context


def summarize_paired_predictions(
    examples: PairedAdvantageExamples,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    if probabilities.shape != (len(examples), 2):
        raise ValueError("probabilities must have shape [N,2]")
    conservative = probabilities.min(axis=1)
    negative = conservative[examples.labels == 0.0]
    positive = conservative[examples.labels == 1.0]
    if negative.size == 0 or positive.size == 0:
        raise ValueError("summary requires both labels")
    max_negative = max(0.5, float(np.max(negative)))
    if max_negative >= 1.0:
        raise ValueError(
            "cannot calibrate a strict threshold above a saturated negative"
        )
    threshold = float(
        np.nextafter(np.float32(max_negative), np.float32(1.0), dtype=np.float32)
    )
    approved = conservative >= threshold
    disagreement = np.abs(probabilities[:, 0] - probabilities[:, 1])
    episode_probabilities: dict[str, list[float]] = defaultdict(list)
    episode_labels: dict[str, float] = {}
    for episode_id, label, value in zip(
        examples.episode_ids, examples.labels, conservative
    ):
        episode_probabilities[episode_id].append(float(value))
        episode_labels[episode_id] = float(label)
    episode_rows = [
        {
            "episode_id": episode_id,
            "label": int(episode_labels[episode_id]),
            "mean_probability": float(np.mean(values)),
        }
        for episode_id, values in sorted(episode_probabilities.items())
    ]
    return {
        "split": examples.split,
        "transitions": len(examples),
        "episodes": len(episode_rows),
        "positive_episodes": sum(row["label"] for row in episode_rows),
        "negative_episodes": sum(1 - row["label"] for row in episode_rows),
        "recommended_threshold": threshold,
        "recommended_max_disagreement": float(
            np.quantile(disagreement[examples.labels == 1.0], 0.95)
        ),
        "transition_true_positive_rate": float(np.mean(approved[examples.labels == 1.0])),
        "transition_false_positive_rate": float(np.mean(approved[examples.labels == 0.0])),
        "transition_brier": float(np.mean(np.square(conservative - examples.labels))),
        "positive_probability_mean": float(np.mean(positive)),
        "negative_probability_mean": float(np.mean(negative)),
        "ensemble_disagreement_mean": float(np.mean(disagreement)),
        "episode_rows": episode_rows,
    }


def export_paired_advantage_config(config: PairedAdvantageTrainingConfig) -> dict:
    config.validate()
    return asdict(config)
