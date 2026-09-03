from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.robotwin.build_wan_vae_head_awr_replay import (
    collapse_record_feature,
    load_video_expert_record_features,
)
from fastwam.models.wan22.fastwam import (
    FASTWAM_VIDEO_EXPERT_FEATURE_VERSION,
    pool_video_expert_tokens,
)
from fastwam.rl.models import ResidualActor, ResidualActorConfig
from fastwam.rl.online_policy import OnlineResidualPolicy


def test_video_expert_token_pooling_is_parameter_free_and_l2_normalized():
    tokens = torch.tensor(
        [[[1.0, 0.0, 1.0], [3.0, 4.0, 1.0]]], dtype=torch.bfloat16
    )
    feature = pool_video_expert_tokens(tokens)
    expected = torch.tensor([[2.0, 2.0, 1.0]], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected, dim=-1, keepdim=True)
    assert feature.dtype == torch.float32
    torch.testing.assert_close(feature, expected)
    torch.testing.assert_close(
        torch.linalg.vector_norm(feature, dim=-1), torch.ones(1)
    )


def test_video_expert_token_pooling_rejects_invalid_features():
    with pytest.raises(ValueError, match=r"\[B, S, D\]"):
        pool_video_expert_tokens(torch.ones(2, 3))
    with pytest.raises(ValueError, match="finite and non-zero"):
        pool_video_expert_tokens(torch.zeros(1, 2, 3))


def test_native_checkpoint_loads_without_a_separate_vision_encoder(tmp_path: Path):
    actor = ResidualActor(
        ResidualActorConfig(
            context_dim=5,
            action_horizon=2,
            action_dim=3,
            hidden_dims=(4,),
            residual_scale=(0.05, 0.05, 0.0),
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
        )
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format": "fastwam_residual_awr_v2",
            "actor": actor.state_dict(),
            "actor_config": actor.export_config(),
            "awr_config": {"use_goal_conditioning": False},
            "summary": {"feature_dim": 3, "proprio_dim": 2},
            "replay_provenance": {
                "observation_encoder_version": FASTWAM_VIDEO_EXPERT_FEATURE_VERSION,
                "observation_feature_dim": 3,
                "fastwam_checkpoint_sha256": "a" * 64,
            },
        },
        checkpoint,
    )

    policy = OnlineResidualPolicy.from_checkpoint(
        checkpoint_path=checkpoint,
        encoder_path=None,
        device="cpu",
        encoder_dtype=torch.float32,
        encoder_version=None,
    )
    assert policy.requires_external_observation_feature
    assert policy.image_processor is None
    assert policy.vision_encoder is None
    with pytest.raises(RuntimeError, match="same FastWAM inference call"):
        policy.encode_observation({})

    output = policy.correct_from_feature(
        observation_feature=np.ones(3, dtype=np.float32),
        proprio=np.zeros(2, dtype=np.float32),
        baseline_actions=np.zeros((2, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(output.corrected_actions, np.zeros((2, 3)))


def test_native_replay_feature_loader_requires_versioned_captured_feature(
    tmp_path: Path,
):
    record_dir = tmp_path / "replan_0000"
    record_dir.mkdir()
    np.savez_compressed(
        record_dir / "rollout_arrays.npz",
        video_expert_feature=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )
    record = {
        "record_dir": str(record_dir),
        "rollout_arrays_file": "rollout_arrays.npz",
        "video_expert_feature_version": FASTWAM_VIDEO_EXPERT_FEATURE_VERSION,
        "video_expert_feature_dim": 3,
        "video_expert_checkpoint_sha256": "a" * 64,
    }
    encoded = load_video_expert_record_features([record])
    feature = collapse_record_feature(
        encoded[0]["current"], observation_source="fastwam_video_expert"
    )
    np.testing.assert_array_equal(feature, np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(
        encoded[0]["actual"]["native"], encoded[0]["current"]["native"]
    )

    bad = dict(record, video_expert_feature_version="siglip")
    with pytest.raises(ValueError, match="Missing or incompatible"):
        load_video_expert_record_features([bad])
