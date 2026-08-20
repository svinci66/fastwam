#!/usr/bin/env python3
"""Train a zero-initialized bounded residual actor on aggregated targets."""

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


class ResidualActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        residual_scale: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.SiLU(), nn.LayerNorm(hidden)])
            previous = hidden
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(previous, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.residual_scale = float(residual_scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.residual_scale * torch.tanh(self.output(self.backbone(features)))


def _normalization(state: np.ndarray, base_action: np.ndarray) -> dict[str, np.ndarray]:
    action = base_action.reshape(len(base_action), -1)
    return {
        "state_mean": state.mean(axis=0),
        "state_std": np.maximum(state.std(axis=0), 1e-6),
        "action_mean": action.mean(axis=0),
        "action_std": np.maximum(action.std(axis=0), 1e-6),
    }


def _features(
    state: np.ndarray,
    base_action: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> np.ndarray:
    normalized_state = (state - normalization["state_mean"]) / normalization["state_std"]
    action = base_action.reshape(len(base_action), -1)
    normalized_action = (action - normalization["action_mean"]) / normalization["action_std"]
    return np.concatenate([normalized_state, normalized_action], axis=1).astype(np.float32)


@torch.no_grad()
def _predict(
    model: nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    output_shape: tuple[int, ...],
) -> np.ndarray:
    model.eval()
    predictions = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions).reshape(len(features), *output_shape)


def _actor_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    improved: np.ndarray,
) -> dict[str, Any]:
    flattened_target = target.reshape(len(target), -1)
    flattened_prediction = prediction.reshape(len(prediction), -1)
    target_norm = np.linalg.norm(flattened_target, axis=1)
    prediction_norm = np.linalg.norm(flattened_prediction, axis=1)
    positive = improved.astype(bool)
    negative = ~positive
    dot = np.sum(flattened_target[positive] * flattened_prediction[positive], axis=1)
    cosine = dot / np.maximum(
        np.linalg.norm(flattened_target[positive], axis=1)
        * np.linalg.norm(flattened_prediction[positive], axis=1),
        1e-8,
    )
    return {
        "samples": len(target),
        "improvement_targets": int(np.count_nonzero(positive)),
        "zero_targets": int(np.count_nonzero(negative)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "positive_mae": float(np.mean(np.abs(prediction[positive] - target[positive]))),
        "positive_cosine_mean": float(np.mean(cosine)),
        "positive_direction_alignment_rate": float(np.mean(dot > 0.0)),
        "positive_target_norm_mean": float(np.mean(target_norm[positive])),
        "positive_prediction_norm_mean": float(np.mean(prediction_norm[positive])),
        "zero_prediction_norm_mean": float(np.mean(prediction_norm[negative])),
        "zero_prediction_norm_p95": float(np.quantile(prediction_norm[negative], 0.95)),
        "zero_prediction_norm_max": float(np.max(prediction_norm[negative], initial=0.0)),
    }


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
    train_demos = set(arrays["source_demo"][train_mask].tolist())
    valid_demos = set(arrays["source_demo"][valid_mask].tolist())
    if train_demos & valid_demos:
        raise ValueError("Source trajectory leakage between train and validation")

    residual_scale = float(arrays["residual_scale"])
    normalization = _normalization(
        arrays["state"][train_mask], arrays["base_action_chunk"][train_mask]
    )
    train_features = _features(
        arrays["state"][train_mask],
        arrays["base_action_chunk"][train_mask],
        normalization,
    )
    valid_features = _features(
        arrays["state"][valid_mask],
        arrays["base_action_chunk"][valid_mask],
        normalization,
    )
    target_shape = arrays["target_residual_chunk"].shape[1:]
    train_target = arrays["target_residual_chunk"][train_mask].reshape(len(train_features), -1)
    valid_target = arrays["target_residual_chunk"][valid_mask]
    train_improved = arrays["has_improvement"][train_mask].astype(np.float32)
    valid_improved = arrays["has_improvement"][valid_mask].astype(bool)

    positive_count = float(np.count_nonzero(train_improved))
    negative_count = float(len(train_improved) - positive_count)
    sample_weights = np.where(
        train_improved > 0,
        len(train_improved) / (2.0 * positive_count),
        len(train_improved) / (2.0 * negative_count),
    ).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features),
            torch.from_numpy(train_target),
            torch.from_numpy(sample_weights),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = ResidualActor(
        input_dim=train_features.shape[1],
        output_dim=train_target.shape[1],
        hidden_dims=tuple(args.hidden_dims),
        residual_scale=residual_scale,
    ).to(device)
    initial_output = _predict(
        model,
        valid_features[: min(16, len(valid_features))],
        device=device,
        batch_size=args.batch_size,
        output_shape=target_shape,
    )
    if not np.array_equal(initial_output, np.zeros_like(initial_output)):
        raise RuntimeError("Residual actor output is not exactly zero at initialization")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for features, target, weight in loader:
            features = features.to(device)
            target = target.to(device)
            weight = weight.to(device)
            prediction = model(features)
            per_element = nn.functional.smooth_l1_loss(
                prediction, target, reduction="none", beta=args.huber_beta
            )
            loss = torch.mean(torch.mean(per_element, dim=1) * weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        valid_prediction = _predict(
            model,
            valid_features,
            device=device,
            batch_size=args.batch_size,
            output_shape=target_shape,
        )
        positive_valid = max(int(np.count_nonzero(valid_improved)), 1)
        negative_valid = max(len(valid_improved) - positive_valid, 1)
        valid_weights = np.where(
            valid_improved,
            len(valid_improved) / (2.0 * positive_valid),
            len(valid_improved) / (2.0 * negative_valid),
        )
        valid_per_sample = np.mean(np.abs(valid_prediction - valid_target), axis=(1, 2))
        valid_loss = float(np.mean(valid_per_sample * valid_weights))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "valid_balanced_mae": valid_loss,
            }
        )
        if valid_loss < best_loss - 1e-9:
            best_loss = valid_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "input_dim": train_features.shape[1],
                    "output_dim": train_target.shape[1],
                    "hidden_dims": args.hidden_dims,
                    "residual_scale": residual_scale,
                    "normalization": normalization,
                    "target_shape": target_shape,
                    "epoch": epoch,
                },
                output_dir / "checkpoint.pt",
            )
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(history[-1]), flush=True)
        if epochs_without_improvement >= args.patience:
            break

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    train_prediction = _predict(
        model,
        train_features,
        device=device,
        batch_size=args.batch_size,
        output_shape=target_shape,
    )
    valid_prediction = _predict(
        model,
        valid_features,
        device=device,
        batch_size=args.batch_size,
        output_shape=target_shape,
    )
    result = {
        "dataset": str(args.dataset.resolve()),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "device": str(device),
        "residual_scale": residual_scale,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "zero_initialized_output_verified": True,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train": _actor_metrics(
            arrays["target_residual_chunk"][train_mask],
            train_prediction,
            arrays["has_improvement"][train_mask].astype(bool),
        ),
        "valid": _actor_metrics(valid_target, valid_prediction, valid_improved),
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
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--huber-beta", type=float, default=0.02)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
