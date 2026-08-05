import json

import numpy as np
import pytest
import torch

from experiments.robotwin.build_residual_rl_replay import (
    combine_camera_features,
    discover_sourced_records,
    filter_records_by_trial_range,
    load_camera_normalization_manifest,
    pad_action_chunk,
    pad_environment_rewards,
    parse_env_seed_overrides,
    validate_episode_records,
)
from experiments.robotwin.analyze_imagination_rewards import (
    ROBOTWIN_CAMERA_IMAGE_SIZE,
    prepare_robotwin_camera_view,
    resolve_encoder_dtype,
)
from experiments.robotwin.imagination_reward_utils import ROBOTWIN_CAMERA_NAMES


def test_combine_camera_features_uses_all_three_views() -> None:
    features = {
        camera: np.asarray(
            [1.0 + index, 2.0 + index, 0.5 - index, -1.0], dtype=np.float32
        )
        for index, camera in enumerate(ROBOTWIN_CAMERA_NAMES)
    }
    combined = combine_camera_features(features)
    assert combined.shape == (12,)
    assert np.linalg.norm(combined) == pytest.approx(1.0)
    expected_blocks = [
        features[camera] / np.linalg.norm(features[camera])
        for camera in ROBOTWIN_CAMERA_NAMES
    ]
    expected = np.concatenate(expected_blocks)
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(combined, expected, rtol=1e-6, atol=1e-6)


def test_robotwin_replay_camera_preprocessing_matches_online_resize() -> None:
    view = np.zeros((128, 160, 3), dtype=np.uint8)
    prepared = prepare_robotwin_camera_view(view)
    assert prepared.size == (
        ROBOTWIN_CAMERA_IMAGE_SIZE,
        ROBOTWIN_CAMERA_IMAGE_SIZE,
    )


def test_replay_encoder_dtype_auto_matches_online_device_precision() -> None:
    assert resolve_encoder_dtype("auto", device="cpu") == torch.float32
    assert resolve_encoder_dtype("bf16", device="cpu") == torch.bfloat16
    with pytest.raises(ValueError, match="fp16 replay encoding"):
        resolve_encoder_dtype("fp16", device="cpu")


def test_pad_partial_robotwin_chunk_repeats_last_action_and_zeros_rewards() -> None:
    actions = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    padded = pad_action_chunk(actions, 4, name="actions")
    assert padded.tolist() == [[1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]
    rewards = pad_environment_rewards(np.asarray([0.0, 1.0]), 4)
    assert rewards.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_validate_robotwin_episode_requires_contiguous_final_boundary() -> None:
    records = [
        {
            "task_name": "task",
            "behavior": "policy",
            "trial_idx": 0,
            "replan_idx": 0,
            "terminated": False,
            "truncated": False,
        },
        {
            "task_name": "task",
            "behavior": "policy",
            "trial_idx": 0,
            "replan_idx": 1,
            "terminated": True,
            "truncated": False,
        },
    ]
    validate_episode_records(records)
    records[-1]["replan_idx"] = 2
    with pytest.raises(ValueError, match="non-contiguous"):
        validate_episode_records(records)


def test_camera_normalization_can_be_reused_from_manifest(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    cameras = {
        camera: {"center": float(index), "scale": float(index + 1)}
        for index, camera in enumerate(ROBOTWIN_CAMERA_NAMES)
    }
    manifest.write_text(json.dumps({"provenance": {"camera_normalization": {"cameras": cameras}}}))
    loaded = load_camera_normalization_manifest(manifest)
    assert loaded["cameras"] == cameras


def test_discover_sourced_records_keeps_input_episodes_separate(monkeypatch, tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    def fake_discover(paths):
        return [{"record_dir": str(paths[0])}]

    monkeypatch.setattr(
        "experiments.robotwin.build_residual_rl_replay.discover_records",
        fake_discover,
    )
    records = discover_sourced_records([first, second])
    assert len({record["source_id"] for record in records}) == 2


def test_discover_sourced_records_can_override_single_seed_captures(
    monkeypatch, tmp_path
) -> None:
    input_dir = tmp_path / "seed_capture"
    input_dir.mkdir()

    def fake_discover(paths):
        return [{"record_dir": str(paths[0]), "trial_idx": 0}]

    monkeypatch.setattr(
        "experiments.robotwin.build_residual_rl_replay.discover_records",
        fake_discover,
    )
    overrides = parse_env_seed_overrides([f"{input_dir}=4800001"])
    records = discover_sourced_records([input_dir], env_seed_overrides=overrides)
    assert records[0]["trial_idx"] == 4800001
    assert records[0]["raw_trial_idx"] == 0


def test_env_seed_override_does_not_leak_across_same_named_directories(
    monkeypatch, tmp_path
) -> None:
    first = tmp_path / "first" / "hanging_mug"
    second = tmp_path / "second" / "hanging_mug"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    def fake_discover(paths):
        return [{"record_dir": str(paths[0]), "trial_idx": 7}]

    monkeypatch.setattr(
        "experiments.robotwin.build_residual_rl_replay.discover_records",
        fake_discover,
    )
    overrides = parse_env_seed_overrides([f"{second}=4800002"])
    records = discover_sourced_records(
        [first, second], env_seed_overrides=overrides
    )
    assert records[0]["trial_idx"] == 7
    assert "raw_trial_idx" not in records[0]
    assert records[1]["trial_idx"] == 4800002
    assert records[1]["raw_trial_idx"] == 7


def test_parse_env_seed_overrides_rejects_malformed_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="INPUT_DIR=SEED"):
        parse_env_seed_overrides([str(tmp_path)])
    with pytest.raises(ValueError, match="must be an integer"):
        parse_env_seed_overrides([f"{tmp_path}=seed"])


def test_filter_records_by_trial_range_is_inclusive() -> None:
    records = [{"trial_idx": index} for index in range(7)]
    filtered = filter_records_by_trial_range(
        records, min_trial_index=2, max_trial_index=5
    )
    assert [record["trial_idx"] for record in filtered] == [2, 3, 4, 5]


def test_filter_records_by_trial_range_rejects_empty_or_reversed_range() -> None:
    records = [{"trial_idx": index} for index in range(3)]
    with pytest.raises(ValueError, match="must not exceed"):
        filter_records_by_trial_range(
            records, min_trial_index=3, max_trial_index=2
        )
    with pytest.raises(ValueError, match="removed every"):
        filter_records_by_trial_range(
            records, min_trial_index=10, max_trial_index=None
        )
