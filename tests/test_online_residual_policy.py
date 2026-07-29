from pathlib import Path

import numpy as np
import pytest
import torch

from fastwam.rl.models import ResidualActor, ResidualActorConfig
from fastwam.rl.online_policy import (
    OnlineResidualPolicy,
    combine_normalized_camera_features,
    load_residual_actor_checkpoint,
)


class _FirstActionCritic(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, context, baseline_actions, actions, language_feature=None):
        del context, baseline_actions, language_feature
        return self.scale * actions[..., 0].mean(dim=1)


def test_combined_camera_features_match_agent_then_wrist_replay_order():
    feature = combine_normalized_camera_features(
        {
            "wrist": np.array([0.0, 2.0], dtype=np.float32),
            "agent": np.array([3.0, 0.0], dtype=np.float32),
        }
    )
    expected = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
    np.testing.assert_allclose(feature, expected, atol=1e-7)


def test_combined_camera_features_support_robotwin_three_camera_order():
    feature = combine_normalized_camera_features(
        {
            "right_wrist": np.array([0.0, 0.0, 4.0], dtype=np.float32),
            "head": np.array([2.0, 0.0, 0.0], dtype=np.float32),
            "left_wrist": np.array([0.0, 3.0, 0.0], dtype=np.float32),
        },
        camera_names=("head", "left_wrist", "right_wrist"),
    )
    expected = np.eye(3, dtype=np.float32).reshape(-1) / np.sqrt(3.0)
    np.testing.assert_allclose(feature, expected, atol=1e-7)


def test_residual_checkpoint_loader_rejects_goal_conditioning(tmp_path: Path):
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=4,
            action_horizon=2,
            action_dim=3,
            residual_scale=(0.05, 0.1, 0.0),
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
        )
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format": "fastwam_residual_awr_v1",
            "actor": actor.state_dict(),
            "actor_config": actor.export_config(),
            "awr_config": {"use_goal_conditioning": True},
        },
        path,
    )
    with pytest.raises(ValueError, match="use_goal_conditioning=false"):
        load_residual_actor_checkpoint(path, device="cpu")


def test_residual_checkpoint_loader_keeps_v1_backward_compatibility(tmp_path: Path):
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=4,
            action_horizon=2,
            action_dim=3,
            residual_scale=(0.05, 0.1, 0.0),
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
        )
    )
    legacy_config = actor.export_config()
    for key in (
        "language_feature_dim",
        "language_embedding_dim",
        "baseline_action_embedding_dim",
    ):
        legacy_config.pop(key)
    path = tmp_path / "legacy-checkpoint.pt"
    torch.save(
        {
            "format": "fastwam_residual_awr_v1",
            "actor": actor.state_dict(),
            "actor_config": legacy_config,
            "awr_config": {"use_goal_conditioning": False},
        },
        path,
    )
    loaded, payload = load_residual_actor_checkpoint(path, device="cpu")
    assert payload["format"] == "fastwam_residual_awr_v1"
    assert loaded.config.language_feature_dim == 0


def test_residual_checkpoint_loader_accepts_iql_actor(tmp_path: Path):
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=4,
            action_horizon=2,
            action_dim=3,
            residual_scale=(0.05, 0.1, 0.0),
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
        )
    )
    path = tmp_path / "iql-checkpoint.pt"
    torch.save(
        {
            "format": "fastwam_residual_iql_v1",
            "actor": actor.state_dict(),
            "actor_config": actor.export_config(),
            "iql_config": {"use_goal_conditioning": False},
        },
        path,
    )
    loaded, payload = load_residual_actor_checkpoint(path, device="cpu")
    assert payload["format"] == "fastwam_residual_iql_v1"
    assert loaded.config.context_dim == 4


def test_online_residual_policy_corrects_prefix_and_preserves_gripper():
    config = ResidualActorConfig(
        context_dim=4,
        action_horizon=2,
        action_dim=3,
        hidden_dims=(4,),
        residual_scale=(0.05, 0.1, 0.0),
        action_low=(-1.0, -1.0, -1.0),
        action_high=(1.0, 1.0, 1.0),
    )
    actor = ResidualActor(config)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.network[-1].bias.copy_(torch.tensor([1.0, -1.0, 2.0, 1.0, -1.0, 2.0]))
    policy = OnlineResidualPolicy(
        actor=actor,
        image_processor=None,
        vision_encoder=torch.nn.Identity(),
        device="cpu",
        encoder_dtype=torch.float32,
        checkpoint_path="checkpoint.pt",
        encoder_path="encoder",
        encoder_version="encoder-v1",
    )
    baseline = np.array(
        [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.25, -0.25, -1.0]],
        dtype=np.float32,
    )
    output = policy.correct_from_feature(
        observation_feature=np.array([1.0, 0.0], dtype=np.float32),
        proprio=np.array([0.0, 1.0], dtype=np.float32),
        baseline_actions=baseline,
    )
    assert output.corrected_actions.shape == baseline.shape
    np.testing.assert_array_equal(output.corrected_actions[2], baseline[2])
    np.testing.assert_array_equal(output.corrected_actions[:2, -1], baseline[:2, -1])
    assert np.all(output.residual_actions[:, 0] > 0.0)
    assert np.all(output.residual_actions[:, 1] < 0.0)
    np.testing.assert_array_equal(output.residual_actions[:, -1], 0.0)


def test_online_q_gate_applies_only_when_both_critics_prefer_candidate():
    config = ResidualActorConfig(
        context_dim=4,
        action_horizon=2,
        action_dim=3,
        hidden_dims=(4,),
        residual_scale=(0.05, 0.0, 0.0),
        action_low=(-1.0, -1.0, -1.0),
        action_high=(1.0, 1.0, 1.0),
    )
    actor = ResidualActor(config)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.network[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))

    kwargs = {
        "actor": actor,
        "image_processor": None,
        "vision_encoder": torch.nn.Identity(),
        "device": "cpu",
        "encoder_dtype": torch.float32,
        "checkpoint_path": "checkpoint.pt",
        "encoder_path": "encoder",
        "encoder_version": "encoder-v1",
        "q_gate_margin": 0.001,
        "q_gate_max_disagreement": 0.01,
    }
    inputs = {
        "observation_feature": np.ones(2, dtype=np.float32),
        "proprio": np.ones(2, dtype=np.float32),
        "baseline_actions": np.zeros((2, 3), dtype=np.float32),
    }
    accepted = OnlineResidualPolicy(
        **kwargs,
        q_critics=(_FirstActionCritic(1.0), _FirstActionCritic(0.9)),
    ).correct_from_feature(**inputs)
    assert accepted.gate_applied
    assert accepted.q_advantage_min > 0.001
    assert np.max(np.abs(accepted.residual_actions)) > 0.0

    rejected = OnlineResidualPolicy(
        **kwargs,
        q_critics=(_FirstActionCritic(1.0), _FirstActionCritic(-1.0)),
    ).correct_from_feature(**inputs)
    assert not rejected.gate_applied
    np.testing.assert_array_equal(rejected.corrected_actions, inputs["baseline_actions"])
    np.testing.assert_array_equal(rejected.residual_actions, 0.0)
    assert np.max(np.abs(rejected.candidate_residual_actions)) > 0.0


def test_online_language_conditioning_is_required_when_checkpoint_uses_it():
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=4,
            action_horizon=2,
            action_dim=3,
            hidden_dims=(4,),
            language_feature_dim=5,
            language_embedding_dim=3,
            baseline_action_embedding_dim=2,
            residual_scale=(0.05, 0.1, 0.0),
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
        )
    )
    policy = OnlineResidualPolicy(
        actor=actor,
        image_processor=None,
        vision_encoder=torch.nn.Identity(),
        device="cpu",
        encoder_dtype=torch.float32,
        checkpoint_path="checkpoint.pt",
        encoder_path="encoder",
        encoder_version="encoder-v1",
    )
    kwargs = {
        "observation_feature": np.ones(2, dtype=np.float32),
        "proprio": np.ones(2, dtype=np.float32),
        "baseline_actions": np.zeros((2, 3), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="language_feature is required"):
        policy.correct_from_feature(**kwargs)
    output = policy.correct_from_feature(
        **kwargs,
        language_feature=np.ones(5, dtype=np.float32),
    )
    assert output.corrected_actions.shape == (2, 3)
