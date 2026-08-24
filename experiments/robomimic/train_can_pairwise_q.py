#!/usr/bin/env python3
"""Train a small shared Q(s, action_chunk) with a pairwise ranking loss."""

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


class PairwiseQ(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.SiLU(), nn.LayerNorm(hidden)])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _auc(target: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[target == 1]
    negative = scores[target == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def _metrics(target: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    prediction = logits >= 0.0
    positive = target == 1
    negative = ~positive
    positive_recall = float(np.mean(prediction[positive]))
    negative_recall = float(np.mean(~prediction[negative]))
    return {
        "samples": len(target),
        "candidate_better": int(np.count_nonzero(positive)),
        "base_better": int(np.count_nonzero(negative)),
        "accuracy": float(np.mean(prediction == positive)),
        "balanced_accuracy": 0.5 * (positive_recall + negative_recall),
        "candidate_better_recall": positive_recall,
        "base_better_recall": negative_recall,
        "auc": _auc(target.astype(np.int8), logits),
        "mean_logit": float(np.mean(logits)),
    }


def _prepare_features(
    state: np.ndarray,
    action: np.ndarray,
    *,
    state_mode: str,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> np.ndarray:
    normalized_action = (action.reshape(len(action), -1) - action_mean) / action_std
    if state_mode == "action_only":
        return normalized_action.astype(np.float32)
    normalized_state = (state - state_mean) / state_std
    return np.concatenate([normalized_state, normalized_action], axis=1).astype(np.float32)


def _initialize_full_from_action_only(
    model: PairwiseQ,
    action_checkpoint: dict[str, Any],
    *,
    state_dim: int,
    action_dim: int,
) -> PairwiseQ:
    """Initialize a full-state Q so that it exactly matches an action-only Q."""
    if action_checkpoint.get("state_mode") != "action_only":
        raise ValueError("Initialization checkpoint must be an action-only Q")
    if int(action_checkpoint["input_dim"]) != action_dim:
        raise ValueError(
            f"Action checkpoint input_dim={action_checkpoint['input_dim']} does not match {action_dim}"
        )
    hidden_dims = tuple(action_checkpoint["hidden_dims"])
    teacher = PairwiseQ(action_dim, hidden_dims)
    teacher.load_state_dict(action_checkpoint["model"])

    full_state = model.state_dict()
    action_state = teacher.state_dict()
    first_weight = "network.0.weight"
    for key, value in action_state.items():
        if key == first_weight:
            if full_state[key].shape[1] != state_dim + action_dim:
                raise ValueError("Unexpected full-state first-layer shape")
            full_state[key].zero_()
            full_state[key][:, state_dim:] = value
        else:
            if full_state[key].shape != value.shape:
                raise ValueError(f"Incompatible action checkpoint tensor: {key}")
            full_state[key].copy_(value)
    model.load_state_dict(full_state)
    return teacher


def _sample_weights(
    arrays: dict[str, np.ndarray],
    train_mask: np.ndarray,
    *,
    key: str | None,
    multiplier: float,
) -> np.ndarray:
    """Return per-row training weights for a binary event indicator."""
    if multiplier < 1.0:
        raise ValueError("Sample-weight multiplier must be at least 1")
    if key is None:
        return np.ones(int(np.count_nonzero(train_mask)), dtype=np.float32)
    if key not in arrays:
        raise KeyError(f"Sample-weight key is absent from dataset: {key}")
    values = np.asarray(arrays[key])
    if values.shape != train_mask.shape:
        raise ValueError(
            f"Sample-weight field {key!r} has shape {values.shape}; "
            f"expected {train_mask.shape}"
        )
    return np.where(values[train_mask] > 0, multiplier, 1.0).astype(np.float32)


@torch.no_grad()
def _predict(
    model: nn.Module,
    state: np.ndarray,
    base_action: np.ndarray,
    candidate_action: np.ndarray,
    *,
    state_mode: str,
    normalization: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    base_features = _prepare_features(
        state,
        base_action,
        state_mode=state_mode,
        **normalization,
    )
    candidate_features = _prepare_features(
        state,
        candidate_action,
        state_mode=state_mode,
        **normalization,
    )
    logits: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(state), batch_size):
        base = torch.from_numpy(base_features[start : start + batch_size]).to(device)
        candidate = torch.from_numpy(candidate_features[start : start + batch_size]).to(device)
        logits.append((model(candidate) - model(base)).cpu().numpy())
    return np.concatenate(logits)


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    with np.load(args.dataset, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    observation_metadata = {
        key: arrays[key].item()
        for key in (
            "observation_mode",
            "encoder_path",
            "camera_name",
            "proprio_keys",
            "vision_feature_dim",
            "vision_encoder_output_dim",
            "vision_projection_path",
            "proprio_dim",
        )
        if key in arrays
    }
    train_mask = arrays["source_split"] == "train"
    valid_mask = arrays["source_split"] == "valid"
    if np.any(np.isin(arrays["source_demo"][train_mask], arrays["source_demo"][valid_mask])):
        raise ValueError("Source trajectory leakage between train and validation")

    state_train = arrays["state"][train_mask]
    base_train = arrays["base_action_chunk"][train_mask]
    candidate_train = arrays["candidate_action_chunk"][train_mask]
    target_train = (arrays["label"][train_mask] > 0).astype(np.float32)
    action_train_flat = np.concatenate(
        [base_train.reshape(len(base_train), -1), candidate_train.reshape(len(candidate_train), -1)],
        axis=0,
    )
    normalization = {
        "state_mean": state_train.mean(axis=0),
        "state_std": np.maximum(state_train.std(axis=0), 1e-6),
        "action_mean": action_train_flat.mean(axis=0),
        "action_std": np.maximum(action_train_flat.std(axis=0), 1e-6),
    }
    base_features = _prepare_features(
        state_train,
        base_train,
        state_mode=args.state_mode,
        **normalization,
    )
    candidate_features = _prepare_features(
        state_train,
        candidate_train,
        state_mode=args.state_mode,
        **normalization,
    )
    sample_weights = _sample_weights(
        arrays,
        train_mask,
        key=args.sample_weight_key,
        multiplier=args.sample_weight_multiplier,
    )
    dataset = TensorDataset(
        torch.from_numpy(base_features),
        torch.from_numpy(candidate_features),
        torch.from_numpy(target_train),
        torch.from_numpy(sample_weights),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PairwiseQ(base_features.shape[1], tuple(args.hidden_dims)).to(device)
    action_teacher: PairwiseQ | None = None
    initialization_checkpoint: Path | None = args.initialize_action_checkpoint
    state_feature_dim = 0
    action_feature_dim = base_train.reshape(len(base_train), -1).shape[1]
    if initialization_checkpoint is not None:
        if args.state_mode != "full":
            raise ValueError("Action-prior initialization requires --state-mode full")
        initialization_checkpoint = initialization_checkpoint.expanduser().resolve()
        action_checkpoint = torch.load(
            initialization_checkpoint, map_location="cpu", weights_only=False
        )
        teacher_normalization = action_checkpoint["normalization"]
        for key in ("action_mean", "action_std"):
            if not np.allclose(teacher_normalization[key], normalization[key], atol=1e-6):
                raise ValueError(f"Action checkpoint uses incompatible {key}")
        state_feature_dim = state_train.shape[1]
        action_teacher = _initialize_full_from_action_only(
            model,
            action_checkpoint,
            state_dim=state_feature_dim,
            action_dim=action_feature_dim,
        ).to(device)
        action_teacher.eval()
        for parameter in action_teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    positive_count = float(np.count_nonzero(target_train))
    negative_count = float(len(target_train) - positive_count)
    class_weights = torch.tensor(
        [len(target_train) / (2.0 * negative_count), len(target_train) / (2.0 * positive_count)],
        dtype=torch.float32,
        device=device,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_balanced = -1.0
    best_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    state_valid = arrays["state"][valid_mask]
    base_valid = arrays["base_action_chunk"][valid_mask]
    candidate_valid = arrays["candidate_action_chunk"][valid_mask]
    target_valid = (arrays["label"][valid_mask] > 0).astype(np.int8)

    def save_checkpoint(epoch: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "input_dim": base_features.shape[1],
                "hidden_dims": args.hidden_dims,
                "state_mode": args.state_mode,
                "normalization": normalization,
                "observation_metadata": observation_metadata,
                "epoch": epoch,
                "initialization_checkpoint": (
                    str(initialization_checkpoint) if initialization_checkpoint else None
                ),
                "teacher_regularization": args.teacher_regularization,
                "sample_weight_key": args.sample_weight_key,
                "sample_weight_multiplier": args.sample_weight_multiplier,
            },
            output_dir / "checkpoint.pt",
        )

    if action_teacher is not None:
        initial_logits = _predict(
            model,
            state_valid,
            base_valid,
            candidate_valid,
            state_mode=args.state_mode,
            normalization=normalization,
            device=device,
            batch_size=args.batch_size,
        )
        initial_metrics = _metrics(target_valid, initial_logits)
        initial_loss = float(
            nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(initial_logits),
                torch.from_numpy(target_valid.astype(np.float32)),
            )
        )
        best_balanced = float(initial_metrics["balanced_accuracy"])
        best_loss = initial_loss
        best_epoch = 0
        save_checkpoint(0)
        history.append(
            {
                "epoch": 0,
                "train_loss": None,
                "valid_loss": initial_loss,
                "valid_balanced_accuracy": initial_metrics["balanced_accuracy"],
                "valid_auc": initial_metrics["auc"],
            }
        )
        print(json.dumps(history[-1]), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for base, candidate, target, sample_weight in loader:
            base = base.to(device)
            candidate = candidate.to(device)
            target = target.to(device)
            sample_weight = sample_weight.to(device)
            logits = model(candidate) - model(base)
            weights = class_weights[target.long()] * sample_weight
            elementwise_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss = torch.sum(elementwise_loss * weights) / torch.sum(weights)
            if action_teacher is not None and args.teacher_regularization > 0.0:
                with torch.no_grad():
                    teacher_logits = (
                        action_teacher(candidate[:, state_feature_dim:])
                        - action_teacher(base[:, state_feature_dim:])
                    )
                loss = loss + args.teacher_regularization * nn.functional.mse_loss(
                    logits, teacher_logits
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        valid_logits = _predict(
            model,
            state_valid,
            base_valid,
            candidate_valid,
            state_mode=args.state_mode,
            normalization=normalization,
            device=device,
            batch_size=args.batch_size,
        )
        valid_target_tensor = torch.from_numpy(target_valid.astype(np.float32))
        valid_loss = float(
            nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(valid_logits), valid_target_tensor
            )
        )
        valid_metrics = _metrics(target_valid, valid_logits)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "valid_loss": valid_loss,
                "valid_balanced_accuracy": valid_metrics["balanced_accuracy"],
                "valid_auc": valid_metrics["auc"],
            }
        )
        improved = (
            valid_metrics["balanced_accuracy"] > best_balanced + 1e-8
            or (
                abs(valid_metrics["balanced_accuracy"] - best_balanced) <= 1e-8
                and valid_loss < best_loss
            )
        )
        if improved:
            best_balanced = float(valid_metrics["balanced_accuracy"])
            best_loss = valid_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(epoch)
        else:
            epochs_without_improvement += 1
        if epoch % 10 == 0 or epoch == 1:
            print(json.dumps(history[-1]), flush=True)
        if epochs_without_improvement >= args.patience:
            break

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    train_logits = _predict(
        model,
        state_train,
        base_train,
        candidate_train,
        state_mode=args.state_mode,
        normalization=normalization,
        device=device,
        batch_size=args.batch_size,
    )
    valid_logits = _predict(
        model,
        state_valid,
        base_valid,
        candidate_valid,
        state_mode=args.state_mode,
        normalization=normalization,
        device=device,
        batch_size=args.batch_size,
    )
    train_target_int = target_train.astype(np.int8)
    majority_accuracy = float(max(np.mean(target_valid), 1.0 - np.mean(target_valid)))
    result = {
        "dataset": str(args.dataset.resolve()),
        "output_dir": str(output_dir),
        "state_mode": args.state_mode,
        "seed": args.seed,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "observation_metadata": observation_metadata,
        "initialization_checkpoint": (
            str(initialization_checkpoint) if initialization_checkpoint else None
        ),
        "teacher_regularization": args.teacher_regularization,
        "sample_weight_key": args.sample_weight_key,
        "sample_weight_multiplier": args.sample_weight_multiplier,
        "weighted_train_samples": int(np.count_nonzero(sample_weights > 1.0)),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "majority_baseline": {
            "accuracy": majority_accuracy,
            "balanced_accuracy": 0.5,
        },
        "train": _metrics(train_target_int, train_logits),
        "valid": _metrics(target_valid, valid_logits),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-mode", choices=("full", "action_only"), default="full")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--initialize-action-checkpoint", type=Path)
    parser.add_argument("--teacher-regularization", type=float, default=0.0)
    parser.add_argument("--sample-weight-key")
    parser.add_argument("--sample-weight-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
