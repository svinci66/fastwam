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


def _projector(input_dim: int, output_dim: int) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("projector dimensions must be positive")
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.SiLU(),
    )


def _validate_optional_projection(
    *, input_dim: int, output_dim: int, name: str
) -> None:
    if input_dim < 0 or output_dim < 0:
        raise ValueError(f"{name} dimensions must be non-negative")
    if (input_dim == 0) != (output_dim == 0):
        raise ValueError(
            f"{name}_feature_dim and {name}_embedding_dim must both be zero or positive"
        )


@dataclass(frozen=True)
class ResidualActorConfig:
    context_dim: int
    action_horizon: int = 8
    action_dim: int = 7
    hidden_dims: tuple[int, ...] = (512, 512)
    language_feature_dim: int = 0
    language_embedding_dim: int = 0
    baseline_action_embedding_dim: int = 0
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
        _validate_optional_projection(
            input_dim=self.language_feature_dim,
            output_dim=self.language_embedding_dim,
            name="language",
        )
        if self.baseline_action_embedding_dim < 0:
            raise ValueError("baseline_action_embedding_dim must be non-negative")


class ResidualActor(nn.Module):
    """Predict a bounded correction to a frozen FastWAM action chunk."""

    def __init__(self, config: ResidualActorConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.language_projector = (
            _projector(config.language_feature_dim, config.language_embedding_dim)
            if config.language_feature_dim > 0
            else None
        )
        self.baseline_action_projector = (
            _projector(
                config.action_horizon * config.action_dim,
                config.baseline_action_embedding_dim,
            )
            if config.baseline_action_embedding_dim > 0
            else None
        )
        self.network = _mlp(
            config.context_dim
            + config.language_embedding_dim
            + config.baseline_action_embedding_dim,
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

    def _conditioned_context(
        self,
        context: torch.Tensor,
        baseline_actions: torch.Tensor | None,
        language_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.config.context_dim:
            raise ValueError(
                f"context must have shape [B, {self.config.context_dim}], got {tuple(context.shape)}"
            )
        inputs = [context]
        if self.baseline_action_projector is not None:
            expected = (context.shape[0], self.config.action_horizon, self.config.action_dim)
            if baseline_actions is None or tuple(baseline_actions.shape) != expected:
                shape = None if baseline_actions is None else tuple(baseline_actions.shape)
                raise ValueError(f"baseline_actions must have shape {expected}, got {shape}")
            inputs.append(self.baseline_action_projector(baseline_actions.flatten(start_dim=1)))
        if self.language_projector is not None:
            expected = (context.shape[0], self.config.language_feature_dim)
            if language_feature is None or tuple(language_feature.shape) != expected:
                shape = None if language_feature is None else tuple(language_feature.shape)
                raise ValueError(f"language_feature must have shape {expected}, got {shape}")
            inputs.append(self.language_projector(language_feature))
        return torch.cat(inputs, dim=-1)

    def residual(
        self,
        context: torch.Tensor,
        baseline_actions: torch.Tensor | None = None,
        language_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditioned = self._conditioned_context(context, baseline_actions, language_feature)
        prediction = self.network(conditioned).view(
            context.shape[0], self.config.action_horizon, self.config.action_dim
        )
        return torch.tanh(prediction) * self.residual_scale

    def forward(
        self,
        context: torch.Tensor,
        baseline_actions: torch.Tensor,
        language_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (context.shape[0], self.config.action_horizon, self.config.action_dim)
        if tuple(baseline_actions.shape) != expected:
            raise ValueError(
                f"baseline_actions must have shape {expected}, got {baseline_actions.shape}"
            )
        corrected = baseline_actions + self.residual(
            context,
            baseline_actions=baseline_actions,
            language_feature=language_feature,
        )
        return torch.maximum(torch.minimum(corrected, self.action_high), self.action_low)

    def export_config(self) -> dict:
        return asdict(self.config)


@dataclass(frozen=True)
class ValueCriticConfig:
    context_dim: int
    hidden_dims: tuple[int, ...] = (512, 512)
    action_horizon: int = 0
    action_dim: int = 0
    language_feature_dim: int = 0
    language_embedding_dim: int = 0
    baseline_action_embedding_dim: int = 0

    def validate(self) -> None:
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive values")
        _validate_optional_projection(
            input_dim=self.language_feature_dim,
            output_dim=self.language_embedding_dim,
            name="language",
        )
        if self.baseline_action_embedding_dim < 0:
            raise ValueError("baseline_action_embedding_dim must be non-negative")
        if self.baseline_action_embedding_dim > 0 and (
            self.action_horizon <= 0 or self.action_dim <= 0
        ):
            raise ValueError(
                "action_horizon and action_dim must be positive when conditioning on baseline actions"
            )


class ValueCritic(nn.Module):
    def __init__(self, config: ValueCriticConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.language_projector = (
            _projector(config.language_feature_dim, config.language_embedding_dim)
            if config.language_feature_dim > 0
            else None
        )
        self.baseline_action_projector = (
            _projector(
                config.action_horizon * config.action_dim,
                config.baseline_action_embedding_dim,
            )
            if config.baseline_action_embedding_dim > 0
            else None
        )
        self.network = _mlp(
            config.context_dim
            + config.language_embedding_dim
            + config.baseline_action_embedding_dim,
            config.hidden_dims,
            1,
        )

    def forward(
        self,
        context: torch.Tensor,
        baseline_actions: torch.Tensor | None = None,
        language_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.config.context_dim:
            raise ValueError(
                f"context must have shape [B, {self.config.context_dim}], got {tuple(context.shape)}"
            )
        inputs = [context]
        if self.baseline_action_projector is not None:
            expected = (context.shape[0], self.config.action_horizon, self.config.action_dim)
            if baseline_actions is None or tuple(baseline_actions.shape) != expected:
                shape = None if baseline_actions is None else tuple(baseline_actions.shape)
                raise ValueError(f"baseline_actions must have shape {expected}, got {shape}")
            inputs.append(self.baseline_action_projector(baseline_actions.flatten(start_dim=1)))
        if self.language_projector is not None:
            expected = (context.shape[0], self.config.language_feature_dim)
            if language_feature is None or tuple(language_feature.shape) != expected:
                shape = None if language_feature is None else tuple(language_feature.shape)
                raise ValueError(f"language_feature must have shape {expected}, got {shape}")
            inputs.append(self.language_projector(language_feature))
        return self.network(torch.cat(inputs, dim=-1)).squeeze(-1)

    def export_config(self) -> dict:
        return asdict(self.config)


@dataclass(frozen=True)
class ActionValueCriticConfig:
    """Configuration for an action-conditioned Q critic.

    The critic sees both FastWAM's frozen baseline action chunk and the action
    chunk that was actually executed.  Keeping those inputs separate lets it
    estimate the value of a residual correction without asking the learner to
    rediscover the frozen action prior from pixels.
    """

    context_dim: int
    action_horizon: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (512, 512)
    language_feature_dim: int = 0
    language_embedding_dim: int = 0
    baseline_action_embedding_dim: int = 128
    action_embedding_dim: int = 128

    def validate(self) -> None:
        if self.context_dim <= 0 or self.action_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("context_dim, action_horizon, and action_dim must be positive")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive values")
        _validate_optional_projection(
            input_dim=self.language_feature_dim,
            output_dim=self.language_embedding_dim,
            name="language",
        )
        if self.baseline_action_embedding_dim <= 0 or self.action_embedding_dim <= 0:
            raise ValueError("baseline and action embedding dimensions must be positive")


class ActionValueCritic(nn.Module):
    """Estimate Q(s, a) for one bounded residual action chunk."""

    def __init__(self, config: ActionValueCriticConfig):
        super().__init__()
        config.validate()
        self.config = config
        flattened_action_dim = config.action_horizon * config.action_dim
        self.language_projector = (
            _projector(config.language_feature_dim, config.language_embedding_dim)
            if config.language_feature_dim > 0
            else None
        )
        self.baseline_action_projector = _projector(
            flattened_action_dim,
            config.baseline_action_embedding_dim,
        )
        self.action_projector = _projector(
            flattened_action_dim,
            config.action_embedding_dim,
        )
        self.network = _mlp(
            config.context_dim
            + config.language_embedding_dim
            + config.baseline_action_embedding_dim
            + config.action_embedding_dim,
            config.hidden_dims,
            1,
        )

    def forward(
        self,
        context: torch.Tensor,
        baseline_actions: torch.Tensor,
        actions: torch.Tensor,
        language_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.config.context_dim:
            raise ValueError(
                f"context must have shape [B, {self.config.context_dim}], "
                f"got {tuple(context.shape)}"
            )
        expected = (context.shape[0], self.config.action_horizon, self.config.action_dim)
        if tuple(baseline_actions.shape) != expected:
            raise ValueError(
                f"baseline_actions must have shape {expected}, "
                f"got {tuple(baseline_actions.shape)}"
            )
        if tuple(actions.shape) != expected:
            raise ValueError(f"actions must have shape {expected}, got {tuple(actions.shape)}")
        inputs = [
            context,
            self.baseline_action_projector(baseline_actions.flatten(start_dim=1)),
            self.action_projector(actions.flatten(start_dim=1)),
        ]
        if self.language_projector is not None:
            language_shape = (context.shape[0], self.config.language_feature_dim)
            if language_feature is None or tuple(language_feature.shape) != language_shape:
                shape = None if language_feature is None else tuple(language_feature.shape)
                raise ValueError(
                    f"language_feature must have shape {language_shape}, got {shape}"
                )
            inputs.append(self.language_projector(language_feature))
        return self.network(torch.cat(inputs, dim=-1)).squeeze(-1)

    def export_config(self) -> dict:
        return asdict(self.config)
