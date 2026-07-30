import json
from dataclasses import replace

from experiments.robotwin.merge_residual_replays import merge_replays
from fastwam.rl.replay_buffer import ReplayBuffer
from test_rl_replay_buffer import make_transition


def save_shard(path, transition, *, camera_center):
    ReplayBuffer([transition]).save(
        path,
        provenance={
            "reward_encoder_version": "siglip-test-v1",
            "camera_names": ["head", "left_wrist", "right_wrist"],
            "camera_weights": {
                "head": 1.0,
                "left_wrist": 1.0,
                "right_wrist": 1.0,
            },
            "camera_image_size": 224,
            "feature_fusion": "test-fusion-v1",
            "language_encoder_version": "test-language-v1",
            "language_pooling": "test-language-pooling-v1",
            "imagination_reward_type": transition.imagination_reward_type,
            "source_schema": "test-source-v1",
            "seed_fields": "test-seeds-v1",
            "camera_normalization": {
                "cameras": {
                    "head": {"center": camera_center, "scale": 1.0},
                    "left_wrist": {"center": 0.0, "scale": 1.0},
                    "right_wrist": {"center": 0.0, "scale": 1.0},
                }
            }
        },
    )


def test_merge_replays_canonicalizes_task_ids_and_namespaces_episodes(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    transition = make_transition("shared", 0, terminated=True)
    save_shard(first, replace(transition, task_id=4), camera_center=0.0)
    save_shard(second, replace(transition, task_id=0), camera_center=0.0)
    merged, provenance = merge_replays([first, second])
    assert len(merged) == 2
    assert {item.task_id for item in merged.transitions} == {0}
    assert len({item.episode_id for item in merged.transitions}) == 2
    assert provenance["input_replays"][0]["num_transitions"] == 1
    assert provenance["reward_encoder_version"] == "siglip-test-v1"
    assert provenance["camera_names"] == ["head", "left_wrist", "right_wrist"]
    assert provenance["camera_image_size"] == 224
    assert provenance["feature_fusion"] == "test-fusion-v1"
    assert provenance["language_encoder_version"] == "test-language-v1"


def test_merge_replays_rejects_normalization_mismatch(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    transition = make_transition("episode", 0, terminated=True)
    save_shard(first, transition, camera_center=0.0)
    save_shard(second, transition, camera_center=1.0)
    try:
        merge_replays([first, second])
    except ValueError as error:
        assert "camera normalizations" in str(error)
    else:
        raise AssertionError("Expected normalization mismatch to be rejected")
