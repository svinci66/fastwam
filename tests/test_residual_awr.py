import torch

from fastwam.rl.awr_trainer import (
    AWRConfig,
    advantage_weights,
    compute_awr_losses,
    masked_action_mse,
)
from fastwam.rl.models import ResidualActor, ResidualActorConfig, ValueCritic, ValueCriticConfig


def test_residual_actor_bounds_corrections_and_leaves_gripper_to_fastwam():
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=4,
            action_horizon=8,
            action_dim=7,
            hidden_dims=(16,),
            residual_scale=(0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.0),
        )
    )
    context = torch.randn(3, 4)
    baseline = torch.zeros(3, 8, 7)
    baseline[..., -1] = -1.0
    corrected = actor(context, baseline)
    residual = corrected - baseline
    assert torch.all(torch.abs(residual[..., :3]) <= 0.05 + 1e-6)
    assert torch.all(torch.abs(residual[..., 3:6]) <= 0.1 + 1e-6)
    assert torch.equal(corrected[..., -1], baseline[..., -1])


def test_action_mse_ignores_padding_after_effective_k():
    predicted = torch.zeros(2, 8, 7)
    target = torch.zeros_like(predicted)
    target[0, 4:] = 1000.0
    target[1, 7:] = 1000.0
    loss = masked_action_mse(predicted, target, torch.tensor([4, 7]))
    assert torch.equal(loss, torch.zeros(2))


def test_advantage_weights_are_detached_normalized_and_clipped():
    values = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    returns = torch.tensor([-10.0, 0.0, 10.0])
    weights = advantage_weights(
        returns,
        values,
        beta=1.0,
        maximum=2.0,
        normalize=True,
    )
    assert not weights.requires_grad
    assert float(weights.max()) <= 2.0
    assert weights[2] > weights[1] > weights[0]


def test_awr_loss_runs_without_updating_frozen_fastwam():
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=6,
            action_horizon=8,
            action_dim=7,
            hidden_dims=(16,),
        )
    )
    critic = ValueCritic(ValueCriticConfig(context_dim=6, hidden_dims=(16,)))
    batch = {
        "observation_feature": torch.randn(4, 2),
        "proprio": torch.randn(4, 4),
        "goal_feature": torch.randn(4, 2),
        "baseline_actions": torch.zeros(4, 8, 7),
        "executed_actions": torch.randn(4, 8, 7) * 0.01,
        "effective_k": torch.tensor([8, 8, 4, 6]),
        "return_to_go": torch.tensor([1.0, 0.5, -0.5, 0.0]),
    }
    losses = compute_awr_losses(actor, critic, batch, AWRConfig())
    assert losses["actor_loss"].ndim == 0
    assert losses["critic_loss"].ndim == 0
    assert torch.isfinite(losses["actor_loss"])
    assert torch.isfinite(losses["critic_loss"])
