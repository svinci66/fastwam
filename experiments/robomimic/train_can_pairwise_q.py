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
    dataset = TensorDataset(
        torch.from_numpy(base_features),
        torch.from_numpy(candidate_features),
        torch.from_numpy(target_train),
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

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for base, candidate, target in loader:
            base = base.to(device)
            candidate = candidate.to(device)
            target = target.to(device)
            logits = model(candidate) - model(base)
            weights = class_weights[target.long()]
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target, weight=weights
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "input_dim": base_features.shape[1],
                    "hidden_dims": args.hidden_dims,
                    "state_mode": args.state_mode,
                    "normalization": normalization,
                    "epoch": epoch,
                },
                output_dir / "checkpoint.pt",
            )
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
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
