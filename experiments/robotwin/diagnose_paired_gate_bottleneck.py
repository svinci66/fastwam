#!/usr/bin/env python3
"""Diagnose data, capacity, and temporal-input limits of the residual gate.

This experiment deliberately uses only actual residual episodes paired with a
clean FastWAM episode under the same task and environment seed.  Expert and
synthetic-corruption episodes are excluded so that easy behavior-source cues do
not masquerade as residual-improvement generalization.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.rl.models import ActionValueCritic, ActionValueCriticConfig
from fastwam.rl.paired_advantage import (
    PairedAdvantageExamples,
    PairedAdvantageTrainingConfig,
    build_paired_advantage_examples,
    build_temporal_context,
    predict_paired_advantage,
    train_paired_advantage_ensemble,
)
from fastwam.rl.replay_buffer import ReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--overfit-epochs", type=int, default=200)
    return parser.parse_args()


def _episode_labels(examples: PairedAdvantageExamples) -> dict[str, int]:
    labels: dict[str, int] = {}
    for episode_id, label in zip(examples.episode_ids, examples.labels):
        integer = int(label >= 0.5)
        previous = labels.setdefault(episode_id, integer)
        if previous != integer:
            raise ValueError(f"episode {episode_id} has inconsistent labels")
    return labels


def _episode_indices(examples: PairedAdvantageExamples) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for offset, episode_id in enumerate(examples.episode_ids):
        grouped[episode_id].append(offset)
    return dict(grouped)


def subset_examples(
    examples: PairedAdvantageExamples,
    episode_ids: Iterable[str],
    *,
    split: str,
) -> PairedAdvantageExamples:
    """Select complete episodes and recompute class/episode-balanced weights."""

    selected_ids = set(episode_ids)
    if not selected_ids:
        raise ValueError("episode subset must not be empty")
    offsets = np.asarray(
        [index for index, value in enumerate(examples.episode_ids) if value in selected_ids],
        dtype=np.int64,
    )
    if offsets.size == 0:
        raise ValueError("episode subset has no matching transitions")
    episode_ids_array = np.asarray(examples.episode_ids, dtype=object)[offsets]
    labels = examples.labels[offsets]
    label_by_episode: dict[str, int] = {}
    transition_count: Counter[str] = Counter(str(value) for value in episode_ids_array)
    for episode_id, label in zip(episode_ids_array, labels):
        label_by_episode[str(episode_id)] = int(label >= 0.5)
    class_episode_counts = Counter(label_by_episode.values())
    if set(class_episode_counts) != {0, 1}:
        raise ValueError(
            f"episode subset must contain both labels, got {dict(class_episode_counts)}"
        )
    weights = np.asarray(
        [
            1.0
            / (
                class_episode_counts[int(label >= 0.5)]
                * transition_count[str(episode_id)]
            )
            for episode_id, label in zip(episode_ids_array, labels)
        ],
        dtype=np.float64,
    )
    return PairedAdvantageExamples(
        indices=examples.indices[offsets],
        labels=labels.copy(),
        weights=weights,
        episode_ids=tuple(str(value) for value in episode_ids_array),
        split=split,
    )


def stratified_folds(
    episode_labels: dict[str, int], *, folds: int, seed: int
) -> list[tuple[str, ...]]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    by_label: dict[int, list[str]] = defaultdict(list)
    for episode_id, label in episode_labels.items():
        by_label[label].append(episode_id)
    if set(by_label) != {0, 1}:
        raise ValueError("stratified folds require both labels")
    if any(len(values) < folds for values in by_label.values()):
        raise ValueError(
            f"each class needs at least {folds} episodes, got "
            f"{dict((key, len(value)) for key, value in by_label.items())}"
        )
    rng = random.Random(seed)
    result: list[list[str]] = [[] for _ in range(folds)]
    for label in (0, 1):
        values = sorted(by_label[label])
        rng.shuffle(values)
        for index, episode_id in enumerate(values):
            result[index % folds].append(episode_id)
    return [tuple(sorted(values)) for values in result]


def stratified_fraction(
    episode_ids: Sequence[str],
    episode_labels: dict[str, int],
    *,
    fraction: float,
    seed: int,
) -> tuple[str, ...]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1.0:
        return tuple(sorted(episode_ids))
    rng = random.Random(seed)
    selected: list[str] = []
    for label in (0, 1):
        values = sorted(value for value in episode_ids if episode_labels[value] == label)
        rng.shuffle(values)
        count = max(1, int(np.ceil(len(values) * fraction)))
        selected.extend(values[:count])
    return tuple(sorted(selected))


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional and aligned")
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("binary metrics require both labels")
    predictions = probabilities >= 0.5
    true_positive_rate = float(np.mean(predictions[labels == 1]))
    false_positive_rate = float(np.mean(predictions[labels == 0]))
    pairwise = positive[:, None] - negative[None, :]
    auroc = float(np.mean(pairwise > 0.0) + 0.5 * np.mean(pairwise == 0.0))
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": 0.5 * (true_positive_rate + 1.0 - false_positive_rate),
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "brier": float(np.mean(np.square(probabilities - labels))),
        "auroc": auroc,
        "positive_probability_mean": float(np.mean(positive)),
        "negative_probability_mean": float(np.mean(negative)),
    }


def summarize_episode_predictions(
    examples: PairedAdvantageExamples, probabilities: np.ndarray
) -> dict[str, Any]:
    conservative = np.asarray(probabilities, dtype=np.float64).min(axis=1)
    grouped: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, int] = {}
    for episode_id, label, probability in zip(
        examples.episode_ids, examples.labels, conservative
    ):
        grouped[episode_id].append(float(probability))
        labels[episode_id] = int(label >= 0.5)
    rows = [
        {
            "episode_id": episode_id,
            "label": labels[episode_id],
            "probability": float(np.mean(values)),
        }
        for episode_id, values in sorted(grouped.items())
    ]
    metrics = binary_metrics(
        np.asarray([row["label"] for row in rows]),
        np.asarray([row["probability"] for row in rows]),
    )
    metrics["episodes"] = len(rows)
    metrics["positive_episodes"] = sum(row["label"] for row in rows)
    metrics["negative_episodes"] = sum(1 - row["label"] for row in rows)
    metrics["episode_rows"] = rows
    return metrics


def _model_config(
    base: ActionValueCriticConfig,
    *,
    hidden_dims: tuple[int, ...],
    context_dim: int,
) -> ActionValueCriticConfig:
    values = asdict(base)
    values["hidden_dims"] = hidden_dims
    values["context_dim"] = context_dim
    return ActionValueCriticConfig(**values)


def fit_and_predict(
    *,
    replay: ReplayBuffer,
    train_examples: PairedAdvantageExamples,
    evaluation_examples: PairedAdvantageExamples,
    base_model_config: ActionValueCriticConfig,
    hidden_dims: tuple[int, ...],
    context: np.ndarray,
    arrays: dict[str, np.ndarray],
    epochs: int,
    seed: int,
    device: str,
    weight_decay: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_config = _model_config(
        base_model_config,
        hidden_dims=hidden_dims,
        context_dim=int(context.shape[1]),
    )
    models = (ActionValueCritic(model_config), ActionValueCritic(model_config))
    training_config = PairedAdvantageTrainingConfig(
        epochs=epochs,
        seed=seed,
        weight_decay=weight_decay,
        include_residual_equal_outcomes_as_negative=True,
    )
    histories = train_paired_advantage_ensemble(
        models,
        replay,
        train_examples,
        training_config,
        device=device,
        context_override=context,
        arrays_override=arrays,
    )
    train_probabilities = predict_paired_advantage(
        models,
        replay,
        train_examples,
        device=device,
        context_override=context,
        arrays_override=arrays,
    )
    evaluation_probabilities = predict_paired_advantage(
        models,
        replay,
        evaluation_examples,
        device=device,
        context_override=context,
        arrays_override=arrays,
    )
    training = {
        "parameters_each": sum(parameter.numel() for parameter in models[0].parameters()),
        "final_sampled_accuracy": [history[-1]["accuracy"] for history in histories],
        "final_sampled_loss": [history[-1]["loss"] for history in histories],
    }
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return train_probabilities, evaluation_probabilities, training


def cross_validate(
    *,
    replay: ReplayBuffer,
    all_examples: PairedAdvantageExamples,
    folds: Sequence[Sequence[str]],
    episode_labels: dict[str, int],
    base_model_config: ActionValueCriticConfig,
    hidden_dims: tuple[int, ...],
    context: np.ndarray,
    arrays: dict[str, np.ndarray],
    data_fraction: float,
    epochs: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    all_ids = set(episode_labels)
    out_of_fold_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, validation_ids_value in enumerate(folds):
        validation_ids = tuple(validation_ids_value)
        available_train_ids = tuple(sorted(all_ids - set(validation_ids)))
        train_ids = stratified_fraction(
            available_train_ids,
            episode_labels,
            fraction=data_fraction,
            seed=seed + 7919 * fold_index,
        )
        train_examples = subset_examples(
            all_examples, train_ids, split=f"train-fold-{fold_index}"
        )
        validation_examples = subset_examples(
            all_examples, validation_ids, split=f"validation-fold-{fold_index}"
        )
        train_probabilities, validation_probabilities, training = fit_and_predict(
            replay=replay,
            train_examples=train_examples,
            evaluation_examples=validation_examples,
            base_model_config=base_model_config,
            hidden_dims=hidden_dims,
            context=context,
            arrays=arrays,
            epochs=epochs,
            seed=seed + 104729 * fold_index,
            device=device,
        )
        train_summary = summarize_episode_predictions(train_examples, train_probabilities)
        validation_summary = summarize_episode_predictions(
            validation_examples, validation_probabilities
        )
        out_of_fold_rows.extend(validation_summary.pop("episode_rows"))
        train_summary.pop("episode_rows")
        fold_rows.append(
            {
                "fold": fold_index,
                "train_episode_ids": list(train_ids),
                "validation_episode_ids": list(validation_ids),
                "train": train_summary,
                "validation": validation_summary,
                "training": training,
            }
        )
    labels = np.asarray([row["label"] for row in out_of_fold_rows])
    probabilities = np.asarray([row["probability"] for row in out_of_fold_rows])
    aggregate = binary_metrics(labels, probabilities)
    aggregate["episodes"] = len(out_of_fold_rows)
    aggregate["episode_rows"] = sorted(
        out_of_fold_rows, key=lambda value: value["episode_id"]
    )
    return {
        "hidden_dims": list(hidden_dims),
        "context_dim": int(context.shape[1]),
        "data_fraction": data_fraction,
        "aggregate_out_of_fold": aggregate,
        "folds": fold_rows,
    }


def _dataset_summary(
    replay: ReplayBuffer, examples: PairedAdvantageExamples
) -> dict[str, Any]:
    labels = _episode_labels(examples)
    first_by_episode: dict[str, Any] = {}
    for index, episode_id in zip(examples.indices, examples.episode_ids):
        first_by_episode.setdefault(episode_id, replay.transitions[int(index)])
    task_rows: dict[str, Counter[str]] = defaultdict(Counter)
    for episode_id, label in labels.items():
        transition = first_by_episode[episode_id]
        task_rows[transition.task_description]["positive" if label else "negative"] += 1
    return {
        "transitions": len(examples),
        "episodes": len(labels),
        "positive_episodes": sum(labels.values()),
        "negative_episodes": sum(1 - value for value in labels.values()),
        "tasks": {key: dict(value) for key, value in sorted(task_rows.items())},
        "episode_labels": labels,
    }


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.folds < 2 or args.epochs <= 0 or args.overfit_epochs <= 0:
        raise ValueError("fold and epoch settings must be positive")
    replay = ReplayBuffer.load(args.replay_dir)
    arrays = replay.arrays()
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    q_values = dict(checkpoint["q_critic_config"])
    q_values["hidden_dims"] = tuple(q_values["hidden_dims"])
    base_model_config = ActionValueCriticConfig(**q_values)
    pair_config = PairedAdvantageTrainingConfig(
        include_residual_equal_outcomes_as_negative=True,
        seed=args.seed,
    )
    actual_examples = build_paired_advantage_examples(
        replay,
        pair_config,
        split="all",
        behavior_modes=("residual",),
    )
    episode_labels = _episode_labels(actual_examples)
    folds = stratified_folds(episode_labels, folds=args.folds, seed=args.seed)
    current_context = build_temporal_context(
        replay, history_length=1, arrays_override=arrays
    )
    current_hidden = tuple(int(value) for value in base_model_config.hidden_dims)

    identity = {
        "format": "fastwam_robotwin_paired_gate_bottleneck_diagnostic_v1",
        "replay_dir": str(args.replay_dir.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "seed": args.seed,
        "fold_count": args.folds,
        "epochs": args.epochs,
        "overfit_epochs": args.overfit_epochs,
    }
    if args.output_json.exists():
        payload = json.loads(args.output_json.read_text(encoding="utf-8"))
        mismatches = {
            key: (payload.get(key), value)
            for key, value in identity.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"existing output does not match this diagnostic: {mismatches}"
            )
        print(f"[diagnostic] resuming {args.output_json}", flush=True)
    else:
        payload = {
            **identity,
            "dataset": _dataset_summary(replay, actual_examples),
            "fold_episode_ids": [list(values) for values in folds],
            "capacity": [],
            "learning_curve": [],
            "temporal": [],
        }
        _write_payload(args.output_json, payload)

    if "overfit" not in payload:
        print("[diagnostic] overfit current model on all actual residual pairs", flush=True)
        overfit_train, _, overfit_training = fit_and_predict(
            replay=replay,
            train_examples=actual_examples,
            evaluation_examples=actual_examples,
            base_model_config=base_model_config,
            hidden_dims=current_hidden,
            context=current_context,
            arrays=arrays,
            epochs=args.overfit_epochs,
            seed=args.seed,
            device=args.device,
            weight_decay=0.0,
        )
        overfit = summarize_episode_predictions(actual_examples, overfit_train)
        overfit["training"] = overfit_training
        payload["overfit"] = overfit
        _write_payload(args.output_json, payload)
    else:
        overfit = payload["overfit"]
        print("[diagnostic] skip completed overfit section", flush=True)

    capacity_dims = (
        (128, 128),
        current_hidden,
        (512, 512, 512, 512),
        (1024, 1024, 1024, 1024),
    )
    capacity: list[dict[str, Any]] = list(payload.get("capacity", []))
    completed_capacity = {
        tuple(int(value) for value in row["hidden_dims"]) for row in capacity
    }
    for hidden_dims in capacity_dims:
        if hidden_dims in completed_capacity:
            print(
                f"[diagnostic] skip completed capacity hidden_dims={hidden_dims}",
                flush=True,
            )
            continue
        print(f"[diagnostic] capacity hidden_dims={hidden_dims}", flush=True)
        capacity.append(
            cross_validate(
                replay=replay,
                all_examples=actual_examples,
                folds=folds,
                episode_labels=episode_labels,
                base_model_config=base_model_config,
                hidden_dims=hidden_dims,
                context=current_context,
                arrays=arrays,
                data_fraction=1.0,
                epochs=args.epochs,
                seed=args.seed,
                device=args.device,
            )
        )
        payload["capacity"] = capacity
        _write_payload(args.output_json, payload)
    current_capacity = next(
        row for row in capacity if tuple(row["hidden_dims"]) == current_hidden
    )

    learning_curve: list[dict[str, Any]] = list(payload.get("learning_curve", []))
    completed_fractions = {float(row["data_fraction"]) for row in learning_curve}
    for fraction in (0.25, 0.5, 0.75):
        if fraction in completed_fractions:
            print(
                f"[diagnostic] skip completed learning-curve fraction={fraction:.2f}",
                flush=True,
            )
            continue
        print(f"[diagnostic] learning-curve fraction={fraction:.2f}", flush=True)
        learning_curve.append(
            cross_validate(
                replay=replay,
                all_examples=actual_examples,
                folds=folds,
                episode_labels=episode_labels,
                base_model_config=base_model_config,
                hidden_dims=current_hidden,
                context=current_context,
                arrays=arrays,
                data_fraction=fraction,
                epochs=args.epochs,
                seed=args.seed,
                device=args.device,
            )
        )
        payload["learning_curve"] = learning_curve
        _write_payload(args.output_json, payload)
    if not any(float(row["data_fraction"]) == 1.0 for row in learning_curve):
        learning_curve.append(current_capacity)
    learning_curve.sort(key=lambda row: float(row["data_fraction"]))
    payload["learning_curve"] = learning_curve
    _write_payload(args.output_json, payload)

    temporal: list[dict[str, Any]] = list(payload.get("temporal", []))
    if not any(int(row["history_length"]) == 1 for row in temporal):
        temporal_single = dict(current_capacity)
        temporal_single["history_length"] = 1
        temporal.append(temporal_single)
    if not any(int(row["history_length"]) == 3 for row in temporal):
        history_length = 3
        print(f"[diagnostic] temporal history_length={history_length}", flush=True)
        context = build_temporal_context(
            replay, history_length=history_length, arrays_override=arrays
        )
        result = cross_validate(
            replay=replay,
            all_examples=actual_examples,
            folds=folds,
            episode_labels=episode_labels,
            base_model_config=base_model_config,
            hidden_dims=current_hidden,
            context=context,
            arrays=arrays,
            data_fraction=1.0,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
        )
        result["history_length"] = history_length
        temporal.append(result)
        payload["temporal"] = temporal
        _write_payload(args.output_json, payload)
    else:
        print("[diagnostic] skip completed temporal history_length=3", flush=True)
    temporal.sort(key=lambda row: int(row["history_length"]))
    payload["temporal"] = temporal

    best_capacity = max(
        capacity, key=lambda row: row["aggregate_out_of_fold"]["balanced_accuracy"]
    )
    temporal_single, temporal_history = temporal
    diagnostic = {
        "current_model_can_fit_training_pairs": bool(
            overfit["balanced_accuracy"] >= 0.99 and overfit["brier"] <= 0.01
        ),
        "capacity_balanced_accuracy_gain_over_current": float(
            best_capacity["aggregate_out_of_fold"]["balanced_accuracy"]
            - current_capacity["aggregate_out_of_fold"]["balanced_accuracy"]
        ),
        "data_balanced_accuracy_gain_25_to_100": float(
            learning_curve[-1]["aggregate_out_of_fold"]["balanced_accuracy"]
            - learning_curve[0]["aggregate_out_of_fold"]["balanced_accuracy"]
        ),
        "temporal_balanced_accuracy_gain": float(
            temporal_history["aggregate_out_of_fold"]["balanced_accuracy"]
            - temporal_single["aggregate_out_of_fold"]["balanced_accuracy"]
        ),
        "warning": (
            "Only actual residual episodes are used, but labels remain episode-level; "
            "this diagnoses generalization and cannot assign causal credit to individual interventions."
        ),
    }
    payload["diagnostic"] = diagnostic
    _write_payload(args.output_json, payload)
    print(json.dumps(diagnostic, indent=2, sort_keys=True), flush=True)
    print(f"[diagnostic] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
