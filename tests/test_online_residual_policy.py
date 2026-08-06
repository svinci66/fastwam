from pathlib import Path

import numpy as np
import pytest
import torch

from fastwam.rl.models import ResidualActor, ResidualActorConfig
from fastwam.rl.online_policy import (
    OnlineResidualPolicy,
    _encoder_dtype_for_device,
    combine_normalized_camera_features,
    load_residual_actor_checkpoint,
)
from fastwam.rl.support_gate import ResidualSupportIndex


class _FirstActionCritic(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, context, baseline_actions, actions, language_feature=None):
        del context, baseline_actions, language_feature
        return self.scale * actions[..., 0].mean(dim=1)


class _ConstantLogitGate(torch.nn.Module):
    def __init__(self, logit: float):
        super().__init__()
        self.logit = float(logit)

    def forward(self, context, baseline_actions, actions, language_feature=None):
        del baseline_actions, actions, language_feature
        return torch.full(
            (context.shape[0],), self.logit, device=context.device
        )


def test_cpu_residual_encoder_preserves_bfloat16_provenance():
    assert _encoder_dtype_for_device("cpu", torch.bfloat16) == torch.bfloat16
    assert _encoder_dtype_for_device("cpu", torch.float32) == torch.float32
    with pytest.raises(ValueError, match="not fp16"):
        _encoder_dtype_for_device("cpu", torch.float16)


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


def test_online_paired_advantage_gate_requires_conservative_probability():
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
        actor.network[-1].bias.copy_(
            torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        )
    inputs = {
        "observation_feature": np.ones(2, dtype=np.float32),
        "proprio": np.ones(2, dtype=np.float32),
        "baseline_actions": np.zeros((2, 3), dtype=np.float32),
    }
    kwargs = {
        "actor": actor,
        "image_processor": None,
        "vision_encoder": torch.nn.Identity(),
        "device": "cpu",
        "encoder_dtype": torch.float32,
        "checkpoint_path": "checkpoint.pt",
        "encoder_path": "encoder",
        "encoder_version": "encoder-v1",
        "paired_advantage_threshold": 0.8,
        "paired_advantage_max_disagreement": 0.1,
    }
    accepted = OnlineResidualPolicy(
        **kwargs,
        paired_advantage_gates=(
            _ConstantLogitGate(3.0),
            _ConstantLogitGate(2.8),
        ),
    ).correct_from_feature(**inputs)
    assert accepted.gate_applied
    assert accepted.paired_advantage_min_probability > 0.8
    assert accepted.paired_advantage_approved

    rejected = OnlineResidualPolicy(
        **kwargs,
        paired_advantage_gates=(
            _ConstantLogitGate(3.0),
            _ConstantLogitGate(0.0),
        ),
    ).correct_from_feature(**inputs)
    assert not rejected.gate_applied
    assert not rejected.paired_advantage_approved
    np.testing.assert_array_equal(rejected.corrected_actions, inputs["baseline_actions"])


def test_online_q_gate_uses_decaying_residual_risk_in_effective_margin():
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
        actor.network[-1].bias.copy_(
            torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
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
        q_critics=(_FirstActionCritic(1.0), _FirstActionCritic(0.9)),
        q_gate_risk_scale=1.0,
        q_gate_risk_decay=1.0,
    )
    inputs = {
        "observation_feature": np.ones(2, dtype=np.float32),
        "proprio": np.ones(2, dtype=np.float32),
        "baseline_actions": np.zeros((2, 3), dtype=np.float32),
    }

    first = policy.correct_from_feature(**inputs)
    second = policy.correct_from_feature(**inputs)
    assert first.gate_applied
    assert first.residual_risk_before == 0.0
    assert first.residual_risk_after == pytest.approx(first.candidate_residual_rms)
    assert not second.gate_applied
    assert second.gate_approved is False
    assert second.q_advantage_min < second.q_gate_effective_margin
    assert second.residual_risk_before == pytest.approx(first.residual_risk_after)
    assert second.residual_risk_after == pytest.approx(second.residual_risk_before)

    policy.reset()
    recovered = policy.correct_from_feature(**inputs)
    assert recovered.gate_applied
    assert recovered.residual_risk_before == 0.0


def test_online_outcome_confirmation_reanchors_after_failed_progress():
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
        actor.network[-1].bias.copy_(
            torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
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
        q_critics=(_FirstActionCritic(1.0), _FirstActionCritic(0.9)),
        outcome_confirmation_enabled=True,
        outcome_confirmation_min_progress=0.0,
        outcome_confirmation_reanchor_replans=1,
    )
    inputs = {
        "observation_feature": np.ones(2, dtype=np.float32),
        "proprio": np.ones(2, dtype=np.float32),
        "baseline_actions": np.zeros((2, 3), dtype=np.float32),
    }

    first = policy.correct_from_feature(**inputs)
    assert first.gate_applied
    assert first.outcome_confirmation_pending
    with pytest.raises(RuntimeError, match="record_intervention_outcome"):
        policy.correct_from_feature(**inputs)

    policy.record_intervention_outcome(-0.01)
    reanchored = policy.correct_from_feature(**inputs)
    assert reanchored.gate_approved
    assert not reanchored.gate_applied
    assert reanchored.outcome_blocked
    assert reanchored.outcome_reanchor_remaining == 0
    assert reanchored.last_outcome_confirmed is False

    retried = policy.correct_from_feature(**inputs)
    assert retried.gate_applied
    policy.record_intervention_outcome(0.01)
    continued = policy.correct_from_feature(**inputs)
    assert continued.gate_applied
    assert not continued.outcome_blocked
    assert continued.last_outcome_confirmed is True

    policy.reset()
    reset_output = policy.correct_from_feature(**inputs)
    assert reset_output.gate_applied
    assert reset_output.last_outcome_confirmed is None


def test_online_soft_scale_blends_residual_from_q_and_support_confidence():
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
        actor.network[-1].bias.copy_(
            torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        )
    expected_residual = float(0.05 * np.tanh(1.0))
    support = ResidualSupportIndex(
        observation_features=np.asarray([[1.0, 0.0]], dtype=np.float32),
        proprio=np.zeros((1, 2), dtype=np.float32),
        baseline_actions=np.zeros((1, 2, 3), dtype=np.float32),
        residual_actions=np.asarray(
            [[[expected_residual, 0.0, 0.0], [expected_residual, 0.0, 0.0]]],
            dtype=np.float32,
        ),
        state_local_radius=np.ones(1, dtype=np.float32),
        action_local_radius=np.ones(1, dtype=np.float32),
        task_ids=np.asarray([0]),
        task_names=("task",),
        language_prototypes=np.asarray([[1.0, 0.0]], dtype=np.float32),
        proprio_center=np.zeros(2, dtype=np.float32),
        proprio_scale=np.ones(2, dtype=np.float32),
        baseline_center=np.zeros((2, 3), dtype=np.float32),
        baseline_scale=np.ones((2, 3), dtype=np.float32),
        residual_scale=np.asarray([0.05, 0.0, 0.0], dtype=np.float32),
        state_threshold=1.0,
        action_threshold=1.0,
        state_increase_threshold=1.0,
        language_similarity_threshold=0.99,
        neighbors=1,
        score_neighbors=1,
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
        q_critics=(_FirstActionCritic(1.0), _FirstActionCritic(0.9)),
        support_index=support,
        soft_scale_enabled=True,
        soft_scale_q_full_advantage=0.1,
        soft_scale_support_full_margin=0.25,
    )
    output = policy.correct_from_feature(
        observation_feature=np.asarray([1.0, 0.0], dtype=np.float32),
        proprio=np.zeros(2, dtype=np.float32),
        baseline_actions=np.zeros((2, 3), dtype=np.float32),
        language_feature=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert output.gate_applied
    assert 0.0 < output.residual_scale_factor < 1.0
    assert output.residual_scale_factor == pytest.approx(
        output.q_scale_confidence
    )
    assert output.support_scale_confidence == pytest.approx(1.0)
    np.testing.assert_allclose(
        output.residual_actions,
        output.candidate_residual_actions * output.residual_scale_factor,
        atol=1e-7,
    )


def test_online_intervention_budget_counts_only_applied_residuals_and_resets():
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
        actor.network[-1].bias.copy_(
            torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
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
        max_interventions_per_episode=2,
    )
    inputs = {
        "observation_feature": np.ones(2, dtype=np.float32),
        "proprio": np.ones(2, dtype=np.float32),
        "baseline_actions": np.zeros((2, 3), dtype=np.float32),
    }
    denied = policy.correct_from_feature(**inputs, intervention_allowed=False)
    assert not denied.gate_applied
    assert denied.intervention_count == 0
    assert denied.intervention_budget_remaining == 2

    first = policy.correct_from_feature(**inputs)
    second = policy.correct_from_feature(**inputs)
    blocked = policy.correct_from_feature(**inputs)
    assert first.gate_applied and second.gate_applied
    assert second.intervention_count == 2
    assert second.intervention_budget_exhausted
    assert not blocked.gate_applied
    assert blocked.gate_approved
    assert not blocked.intervention_allowed
    assert blocked.intervention_count == 2

    policy.reset()
    recovered = policy.correct_from_feature(**inputs)
    assert recovered.gate_applied
    assert recovered.intervention_count == 1


def test_online_support_gate_shadow_and_episode_circuit_breaker():
    config = ResidualActorConfig(
        context_dim=4,
        action_horizon=2,
        action_dim=3,
        hidden_dims=(4,),
        language_feature_dim=2,
        language_embedding_dim=2,
        baseline_action_embedding_dim=2,
        residual_scale=(0.05, 0.0, 0.0),
        action_low=(-1.0, -1.0, -1.0),
        action_high=(1.0, 1.0, 1.0),
    )
    actor = ResidualActor(config)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.network[-1].bias.copy_(
            torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        )
    expected_residual = float(0.05 * np.tanh(1.0))
    support = ResidualSupportIndex(
        observation_features=np.asarray([[1.0, 0.0]], dtype=np.float32),
        proprio=np.asarray([[0.0, 0.0]], dtype=np.float32),
        baseline_actions=np.zeros((1, 2, 3), dtype=np.float32),
        residual_actions=np.asarray(
            [[[expected_residual, 0.0, 0.0], [expected_residual, 0.0, 0.0]]],
            dtype=np.float32,
        ),
        state_local_radius=np.ones(1, dtype=np.float32),
        action_local_radius=np.ones(1, dtype=np.float32),
        task_ids=np.asarray([0]),
        task_names=("task",),
        language_prototypes=np.asarray([[1.0, 0.0]], dtype=np.float32),
        proprio_center=np.zeros(2, dtype=np.float32),
        proprio_scale=np.ones(2, dtype=np.float32),
        baseline_center=np.zeros((2, 3), dtype=np.float32),
        baseline_scale=np.ones((2, 3), dtype=np.float32),
        residual_scale=np.asarray([0.05, 0.0, 0.0], dtype=np.float32),
        state_threshold=0.2,
        action_threshold=0.1,
        state_increase_threshold=0.05,
        language_similarity_threshold=0.99,
        neighbors=1,
        score_neighbors=1,
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
        language_encoder_version="language-v1",
        support_index=support,
        shadow_mode=True,
    )
    baseline = np.zeros((2, 3), dtype=np.float32)
    accepted = policy.correct_from_feature(
        observation_feature=np.asarray([1.0, 0.0], dtype=np.float32),
        proprio=np.zeros(2, dtype=np.float32),
        baseline_actions=baseline,
        language_feature=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    assert accepted.gate_approved
    assert not accepted.gate_applied
    assert accepted.shadow_mode
    np.testing.assert_array_equal(accepted.corrected_actions, baseline)

    out_of_support = policy.correct_from_feature(
        observation_feature=np.asarray([0.0, 1.0], dtype=np.float32),
        proprio=np.ones(2, dtype=np.float32),
        baseline_actions=baseline,
        language_feature=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    assert not out_of_support.gate_approved
    assert out_of_support.circuit_breaker_triggered
    assert out_of_support.circuit_breaker_active

    still_latched = policy.correct_from_feature(
        observation_feature=np.asarray([1.0, 0.0], dtype=np.float32),
        proprio=np.zeros(2, dtype=np.float32),
        baseline_actions=baseline,
        language_feature=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    assert still_latched.support_decision.in_support
    assert not still_latched.gate_approved
    policy.reset()
    recovered = policy.correct_from_feature(
        observation_feature=np.asarray([1.0, 0.0], dtype=np.float32),
        proprio=np.zeros(2, dtype=np.float32),
        baseline_actions=baseline,
        language_feature=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    assert recovered.gate_approved


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
