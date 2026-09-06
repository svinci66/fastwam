import torch

from fastwam.rl.models import (
    FrozenResidualAdapterActor,
    ResidualActor,
    ResidualActorConfig,
    ResidualAdapter,
    ResidualAdapterConfig,
    FrozenSpatialResidualAdapterActor,
    SpatialResidualAdapter,
    SpatialResidualAdapterConfig,
)
from fastwam.rl.online_policy import (
    RESIDUAL_ADAPTER_CHECKPOINT_FORMAT,
    RESIDUAL_SPATIAL_ADAPTER_CHECKPOINT_FORMAT,
    OnlineResidualPolicy,
    load_residual_actor_checkpoint,
)


def _models():
    actor_config = ResidualActorConfig(
        context_dim=6,
        action_horizon=4,
        action_dim=3,
        hidden_dims=(8,),
        language_feature_dim=5,
        language_embedding_dim=4,
        baseline_action_embedding_dim=4,
        residual_scale=(0.1, 0.1, 0.0),
        action_low=(-1.0, -1.0, -1.0),
        action_high=(1.0, 1.0, 1.0),
    )
    adapter_config = ResidualAdapterConfig(
        context_dim=6,
        action_horizon=4,
        action_dim=3,
        hidden_dims=(8,),
        language_feature_dim=5,
        language_embedding_dim=4,
        baseline_action_embedding_dim=4,
        ordinary_residual_embedding_dim=4,
        adapter_scale=(0.02, 0.02, 0.0),
    )
    base = ResidualActor(actor_config)
    with torch.no_grad():
        base.network[-1].weight.fill_(0.1)
    adapter = ResidualAdapter(adapter_config)
    return base, adapter, FrozenResidualAdapterActor(base, adapter)


def test_zero_initialized_adapter_is_bitwise_equal_to_base_actor():
    base, adapter, composed = _models()
    context = torch.randn(3, 6)
    baseline = torch.randn(3, 4, 3).clamp(-0.5, 0.5)
    language = torch.randn(3, 5)
    expected = base(context, baseline, language)
    actual = composed(context, baseline, language)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(
        adapter(context, baseline, expected - baseline, language)
    ) == 0


def test_adapter_is_bounded_and_never_changes_owned_gripper_dimension():
    base, adapter, composed = _models()
    with torch.no_grad():
        adapter.network[-1].bias.fill_(100.0)
    context = torch.randn(2, 6)
    baseline = torch.zeros(2, 4, 3)
    language = torch.randn(2, 5)
    base_actions, _, delta = composed.components(context, baseline, language)
    corrected = composed(context, baseline, language)
    assert torch.all(delta[..., :2].abs() <= 0.02)
    assert torch.count_nonzero(delta[..., 2]) == 0
    assert torch.equal(corrected[..., 2], base_actions[..., 2])


def test_composed_actor_freezes_base_parameters():
    base, _, composed = _models()
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert any(parameter.requires_grad for parameter in composed.adapter.parameters())


def test_adapter_checkpoint_loads_as_composed_actor(tmp_path):
    base, adapter, composed = _models()
    path = tmp_path / "adapter.pt"
    torch.save(
        {
            "format": RESIDUAL_ADAPTER_CHECKPOINT_FORMAT,
            "actor": base.state_dict(),
            "actor_config": base.export_config(),
            "adapter": adapter.state_dict(),
            "adapter_config": adapter.export_config(),
            "awr_config": {"use_goal_conditioning": False},
        },
        path,
    )
    loaded, payload = load_residual_actor_checkpoint(path, device="cpu")
    assert isinstance(loaded, FrozenResidualAdapterActor)
    assert payload["format"] == RESIDUAL_ADAPTER_CHECKPOINT_FORMAT
    context = torch.randn(2, 6)
    baseline = torch.randn(2, 4, 3).clamp(-0.5, 0.5)
    language = torch.randn(2, 5)
    assert torch.equal(
        loaded(context, baseline, language),
        composed(context, baseline, language),
    )


def test_zero_initialized_spatial_adapter_is_bitwise_equal_to_base_actor():
    base, _, _ = _models()
    adapter = SpatialResidualAdapter(
        SpatialResidualAdapterConfig(
            context_dim=6,
            action_horizon=4,
            action_dim=3,
            hidden_dims=(8,),
            language_feature_dim=5,
            language_embedding_dim=4,
            baseline_action_embedding_dim=4,
            ordinary_residual_embedding_dim=4,
            adapter_scale=(0.02, 0.02, 0.0),
            spatial_token_count=3,
            spatial_token_dim=7,
            spatial_embedding_dim=4,
        )
    )
    composed = FrozenSpatialResidualAdapterActor(base, adapter)
    context = torch.randn(2, 6)
    baseline = torch.randn(2, 4, 3).clamp(-0.5, 0.5)
    language = torch.randn(2, 5)
    spatial = torch.randn(2, 3, 7)
    assert torch.equal(
        composed(context, baseline, spatial, language),
        base(context, baseline, language),
    )


def test_spatial_adapter_checkpoint_requires_and_uses_external_spatial_feature(tmp_path):
    base, _, _ = _models()
    adapter = SpatialResidualAdapter(
        SpatialResidualAdapterConfig(
            context_dim=6,
            action_horizon=4,
            action_dim=3,
            hidden_dims=(8,),
            language_feature_dim=5,
            language_embedding_dim=4,
            baseline_action_embedding_dim=4,
            ordinary_residual_embedding_dim=4,
            adapter_scale=(0.02, 0.02, 0.0),
            spatial_token_count=3,
            spatial_token_dim=7,
            spatial_embedding_dim=4,
        )
    )
    checkpoint = tmp_path / "spatial.pt"
    torch.save(
        {
            "format": RESIDUAL_SPATIAL_ADAPTER_CHECKPOINT_FORMAT,
            "actor": base.state_dict(),
            "actor_config": base.export_config(),
            "adapter": adapter.state_dict(),
            "adapter_config": adapter.export_config(),
            "awr_config": {"use_goal_conditioning": False},
            "summary": {
                "feature_dim": 4,
                "proprio_dim": 2,
                "language_feature_dim": 5,
            },
            "replay_provenance": {
                "observation_encoder_version": "fastwam_video_expert_final_token_mean_l2_v1",
                "observation_feature_dim": 4,
                "fastwam_checkpoint_sha256": "a" * 64,
                "language_encoder_version": "fastwam_umt5_masked_mean_v1",
            },
        },
        checkpoint,
    )
    policy = OnlineResidualPolicy.from_checkpoint(
        checkpoint_path=checkpoint,
        encoder_path=None,
        device="cpu",
        encoder_dtype=torch.float32,
        encoder_version="fastwam_video_expert_final_token_mean_l2_v1",
        language_encoder_version="fastwam_umt5_masked_mean_v1",
        support_circuit_breaker_enabled=False,
    )
    assert policy.requires_external_observation_feature
    assert policy.requires_external_spatial_feature
    kwargs = {
        "observation_feature": torch.ones(4).numpy(),
        "proprio": torch.zeros(2).numpy(),
        "baseline_actions": torch.zeros(4, 3).numpy(),
        "language_feature": torch.zeros(5).numpy(),
    }
    import pytest

    with pytest.raises(ValueError, match="spatial_feature is required"):
        policy.correct_from_feature(**kwargs)
    output = policy.correct_from_feature(
        **kwargs, spatial_feature=torch.ones(3, 7).numpy()
    )
    assert output.corrected_actions.shape == (4, 3)
