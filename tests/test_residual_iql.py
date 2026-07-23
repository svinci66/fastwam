import torch

from fastwam.rl.iql_trainer import (
    IQLConfig,
    compute_iql_losses,
    expectile_loss,
    iql_advantage_weights,
)
from fastwam.rl.models import (
    ActionValueCritic,
    ActionValueCriticConfig,
    ResidualActor,
    ResidualActorConfig,
    ValueCritic,
    ValueCriticConfig,
)


def _models():
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=6,
            action_horizon=2,
            action_dim=3,
            hidden_dims=(16,),
            language_feature_dim=5,
            language_embedding_dim=3,
            baseline_action_embedding_dim=4,
            residual_scale=(0.05, 0.1, 0.0),
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
        )
    )
    q_config = ActionValueCriticConfig(
        context_dim=6,
        action_horizon=2,
        action_dim=3,
        hidden_dims=(16,),
        language_feature_dim=5,
        language_embedding_dim=3,
        baseline_action_embedding_dim=4,
        action_embedding_dim=4,
    )
    q_critics = (ActionValueCritic(q_config), ActionValueCritic(q_config))
    target_q_critics = (ActionValueCritic(q_config), ActionValueCritic(q_config))
    for target, source in zip(target_q_critics, q_critics):
        target.load_state_dict(source.state_dict())
        target.requires_grad_(False)
    value = ValueCritic(
        ValueCriticConfig(
            context_dim=6,
            hidden_dims=(16,),
            action_horizon=2,
            action_dim=3,
            language_feature_dim=5,
            language_embedding_dim=3,
            baseline_action_embedding_dim=4,
        )
    )
    return actor, q_critics, target_q_critics, value


def test_expectile_weights_positive_and_negative_residuals_asymmetrically():
    prediction = torch.zeros(2)
    target = torch.tensor([1.0, -1.0])
    loss = expectile_loss(prediction, target, expectile=0.7)
    assert torch.isclose(loss, torch.tensor(0.5))


def test_iql_advantage_weights_are_detached_and_clipped():
    q_values = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    values = torch.zeros(3, requires_grad=True)
    weights = iql_advantage_weights(
        q_values,
        values,
        temperature=3.0,
        maximum=10.0,
    )
    assert not weights.requires_grad
    assert weights[2] == 10.0
    assert weights[2] > weights[1] > weights[0]


def test_action_value_critic_depends_on_executed_action_chunk():
    _, q_critics, _, _ = _models()
    context = torch.randn(3, 6)
    baseline = torch.zeros(3, 2, 3)
    language = torch.randn(3, 5)
    first = q_critics[0](context, baseline, baseline, language)
    second = q_critics[0](context, baseline, baseline + 0.1, language)
    assert first.shape == (3,)
    assert not torch.equal(first, second)


def test_iql_losses_are_finite_and_use_chunk_discount():
    actor, q_critics, target_q_critics, value = _models()
    batch = {
        "observation_feature": torch.randn(4, 2),
        "next_observation_feature": torch.randn(4, 2),
        "goal_feature": torch.randn(4, 2),
        "proprio": torch.randn(4, 4),
        "next_proprio": torch.randn(4, 4),
        "language_feature": torch.randn(4, 5),
        "baseline_actions": torch.zeros(4, 2, 3),
        "next_baseline_actions": torch.zeros(4, 2, 3),
        "executed_actions": torch.randn(4, 2, 3) * 0.01,
        "effective_k": torch.tensor([2, 2, 1, 2]),
        "reward": torch.tensor([1.0, 0.5, -0.5, 0.0]),
        "bootstrap_mask": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }
    losses = compute_iql_losses(
        actor,
        q_critics,
        target_q_critics,
        value,
        batch,
        IQLConfig(),
    )
    for name in ("actor_loss", "q_loss", "value_loss", "mean_action_mse"):
        assert losses[name].ndim == 0
        assert torch.isfinite(losses[name])
