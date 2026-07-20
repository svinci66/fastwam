"""Small residual actor and value critic that leave FastWAM frozen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")
    if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
        raise ValueError("hidden_dims must contain positive widths")
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        width = int(width)
        layers.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.SiLU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class ResidualActorConfig:
    context_dim: int
    action_horizon: int = 8
    action_dim: int = 7
    hidden_dims: tuple[int, ...] = (512, 512)
    residual_scale: tuple[float, ...] = (0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.0)
    action_low: tuple[float, ...] = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    action_high: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    def validate(self) -> None:
        if self.context_dim <= 0 or self.action_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("context_dim, action_horizon, and action_dim must be positive")
        for name, values in (
            ("residual_scale", self.residual_scale),
            ("action_low", self.action_low),
            ("action_high", self.action_high),
        ):
            if len(values) != self.action_dim:
                raise ValueError(f"{name} must have action_dim={self.action_dim} entries")
        if any(value < 0.0 for value in self.residual_scale):
            raise ValueError("residual_scale must be non-negative")
        if any(low >= high for low, high in zip(self.action_low, self.action_high)):
            raise ValueError("each action_low value must be smaller than action_high")


class ResidualActor(nn.Module):
    """Predict a bounded correction to a frozen FastWAM action chunk."""

    def __init__(self, config: ResidualActorConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.network = _mlp(
            config.context_dim,
            config.hidden_dims,
            config.action_horizon * config.action_dim,
        )
        self.register_buffer(
            "residual_scale",
            torch.tensor(config.residual_scale, dtype=torch.float32).view(1, 1, -1),
        )
        self.register_buffer(
            "action_low", torch.tensor(config.action_low, dtype=torch.float32).view(1, 1, -1)
        )
        self.register_buffer(
            "action_high", torch.tensor(config.action_high, dtype=torch.float32).view(1, 1, -1)
        )

    def residual(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.config.context_dim:
            raise ValueError(
                f"context must have shape [B, {self.config.context_dim}], got {tuple(context.shape)}"
            )
        prediction = self.network(context).view(
            context.shape[0], self.config.action_horizon, self.config.action_dim
        )
        return torch.tanh(prediction) * self.residual_scale

    def forward(self, context: torch.Tensor, baseline_actions: torch.Tensor) -> torch.Tensor:
        expected = (context.shape[0], self.config.action_horizon, self.config.action_dim)
        if tuple(baseline_actions.shape) != expected:
            raise ValueError(
                f"baseline_actions must have shape {expected}, got {baseline_actions.shape}"
            )
        corrected = baseline_actions + self.residual(context)
        return torch.maximum(torch.minimum(corrected, self.action_high), self.action_low)

    def export_config(self) -> dict:
        return asdict(self.config)


@dataclass(frozen=True)
class ValueCriticConfig:
    context_dim: int
    hidden_dims: tuple[int, ...] = (512, 512)

    def validate(self) -> None:
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive values")


class ValueCritic(nn.Module):
    def __init__(self, config: ValueCriticConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.network = _mlp(config.context_dim, config.hidden_dims, 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.config.context_dim:
            raise ValueError(
                f"context must have shape [B, {self.config.context_dim}], got {tuple(context.shape)}"
            )
        return self.network(context).squeeze(-1)

    def export_config(self) -> dict:
        return asdict(self.config)
