import torch

from fastwam.rl.awr_trainer import (
    AWRConfig,
    BalancedBatchSampler,
    TaskBalancedBatchSampler,
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
            zero_init_output=False,
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


def test_residual_actor_zero_initialized_output_preserves_fastwam_actions():
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=4,
            action_horizon=2,
            action_dim=3,
            hidden_dims=(8,),
            residual_scale=(0.05, 0.05, 0.0),
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
            zero_init_output=True,
        )
    )
    context = torch.randn(3, 4)
    baseline = torch.rand(3, 2, 3) * 0.5

    residual = actor.residual(context)
    corrected = actor(context, baseline)

    torch.testing.assert_close(residual, torch.zeros_like(residual))
    torch.testing.assert_close(corrected, baseline)
    assert torch.count_nonzero(actor.network[-1].weight) == 0
    assert torch.count_nonzero(actor.network[-1].bias) == 0


def test_zero_initialized_output_layer_receives_a_training_gradient():
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=4,
            action_horizon=2,
            action_dim=3,
            hidden_dims=(8,),
            residual_scale=(0.05, 0.05, 0.05),
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
        )
    )

    actor.residual(torch.randn(3, 4)).sum().backward()

    assert actor.network[-1].weight.grad is not None
    assert torch.count_nonzero(actor.network[-1].weight.grad) > 0
    assert actor.network[-1].bias.grad is not None
    assert torch.count_nonzero(actor.network[-1].bias.grad) > 0


def test_residual_actor_can_disable_zero_output_initialization_for_ablation():
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=4,
            action_horizon=2,
            action_dim=3,
            hidden_dims=(8,),
            residual_scale=(0.05, 0.05, 0.0),
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
            zero_init_output=False,
        )
    )

    assert torch.count_nonzero(actor.network[-1].weight) > 0


def test_zero_scale_dimension_preserves_out_of_bounds_baseline_exactly():
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=2,
            action_horizon=1,
            action_dim=2,
            hidden_dims=(4,),
            residual_scale=(0.05, 0.0),
            action_low=(-1.0, 0.0),
            action_high=(1.0, 1.0),
        )
    )
    baseline = torch.tensor([[[0.0, -0.02]]])
    corrected = actor(torch.zeros(1, 2), baseline)
    torch.testing.assert_close(corrected[..., 1], baseline[..., 1])
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


def test_balanced_batch_sampler_avoids_tiny_final_batch():
    sampler = BalancedBatchSampler(
        72,
        64,
        generator=torch.Generator().manual_seed(42),
    )
    batches = list(sampler)
    assert [len(batch) for batch in batches] == [36, 36]
    assert sorted(index for batch in batches for index in batch) == list(range(72))


def test_actor_and_critic_use_language_and_baseline_conditioning():
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=4,
            action_horizon=2,
            action_dim=3,
            hidden_dims=(8,),
            language_feature_dim=5,
            language_embedding_dim=3,
            baseline_action_embedding_dim=4,
            residual_scale=(0.05, 0.1, 0.0),
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
        )
    )
    critic = ValueCritic(
        ValueCriticConfig(
            context_dim=4,
            hidden_dims=(8,),
            action_horizon=2,
            action_dim=3,
            language_feature_dim=5,
            language_embedding_dim=3,
            baseline_action_embedding_dim=4,
        )
    )
    context = torch.randn(2, 4)
    baseline = torch.randn(2, 2, 3)
    language = torch.randn(2, 5)
    assert actor(context, baseline, language).shape == baseline.shape
    assert critic(context, baseline, language).shape == (2,)


def test_awr_loss_passes_formal_conditioning_to_actor_and_critic():
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=6,
            action_horizon=2,
            action_dim=3,
            hidden_dims=(8,),
            language_feature_dim=5,
            language_embedding_dim=3,
            baseline_action_embedding_dim=4,
            residual_scale=(0.05, 0.1, 0.0),
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
        )
    )
    critic = ValueCritic(
        ValueCriticConfig(
            context_dim=6,
            hidden_dims=(8,),
            action_horizon=2,
            action_dim=3,
            language_feature_dim=5,
            language_embedding_dim=3,
            baseline_action_embedding_dim=4,
        )
    )
    batch = {
        "observation_feature": torch.randn(4, 2),
        "proprio": torch.randn(4, 4),
        "goal_feature": torch.randn(4, 2),
        "language_feature": torch.randn(4, 5),
        "baseline_actions": torch.zeros(4, 2, 3),
        "executed_actions": torch.randn(4, 2, 3) * 0.01,
        "effective_k": torch.tensor([2, 2, 1, 2]),
        "return_to_go": torch.tensor([1.0, 0.5, -0.5, 0.0]),
    }
    losses = compute_awr_losses(actor, critic, batch, AWRConfig())
    assert torch.isfinite(losses["actor_loss"])
    assert torch.isfinite(losses["critic_loss"])


def test_task_balanced_sampler_oversamples_small_task_without_changing_epoch_size():
    labels = torch.tensor([0] * 8 + [1] * 2)
    sampler = TaskBalancedBatchSampler(
        labels,
        5,
        generator=torch.Generator().manual_seed(42),
    )
    indices = [index for batch in sampler for index in batch]
    sampled_labels = labels[indices]
    assert len(indices) == len(labels)
    assert int((sampled_labels == 0).sum()) == 5
    assert int((sampled_labels == 1).sum()) == 5
