import torch

from fastwam.rl.models import ResidualActor, ResidualActorConfig, ValueCritic, ValueCriticConfig
from scripts.train_libero_residual_awr import _state_dict_sha256, seed_training_process


def _construct_pair():
    actor = ResidualActor(
        ResidualActorConfig(context_dim=12, action_horizon=8, action_dim=7, hidden_dims=(16,))
    )
    critic = ValueCritic(ValueCriticConfig(context_dim=12, hidden_dims=(16,)))
    return actor, critic


def test_training_seed_controls_actor_and_critic_initialization():
    seed_training_process(42)
    first_actor, first_critic = _construct_pair()
    torch.rand(17)
    seed_training_process(42)
    second_actor, second_critic = _construct_pair()
    assert _state_dict_sha256(first_actor) == _state_dict_sha256(second_actor)
    assert _state_dict_sha256(first_critic) == _state_dict_sha256(second_critic)
