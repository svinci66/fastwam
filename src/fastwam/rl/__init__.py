"""Lightweight, server-portable reinforcement-learning components for FastWAM.

The package deliberately does not import LIBERO or instantiate FastWAM.  Rollout
collection and learner updates can therefore run in separate processes, or at
different times on a single GPU.
"""

from .models import ResidualActor, ResidualActorConfig, ValueCritic, ValueCriticConfig
from .online_policy import (
    OnlineResidualPolicy,
    ResidualPolicyOutput,
    combine_normalized_camera_features,
    load_residual_actor_checkpoint,
)
from .replay_buffer import REPLAY_SCHEMA_VERSION, ReplayBuffer, ReplayTransition
from .rewards import (
    CompositeRewardConfig,
    EpisodeShapingBudget,
    IMAGINATION_REWARD_TYPES,
    RewardBreakdown,
    compute_composite_reward,
    compute_imagination_reward,
    compute_imagination_progress,
)

__all__ = [
    "CompositeRewardConfig",
    "EpisodeShapingBudget",
    "IMAGINATION_REWARD_TYPES",
    "REPLAY_SCHEMA_VERSION",
    "ReplayBuffer",
    "ReplayTransition",
    "ResidualActor",
    "ResidualActorConfig",
    "OnlineResidualPolicy",
    "ResidualPolicyOutput",
    "RewardBreakdown",
    "ValueCritic",
    "ValueCriticConfig",
    "compute_composite_reward",
    "compute_imagination_reward",
    "compute_imagination_progress",
    "combine_normalized_camera_features",
    "load_residual_actor_checkpoint",
]
