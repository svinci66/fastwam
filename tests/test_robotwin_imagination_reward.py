from pathlib import Path

import numpy as np
from PIL import Image

from experiments.robotwin.analyze_imagination_rewards import discover_records
from experiments.robotwin.imagination_reward_utils import (
    apply_action_chunk_hold,
    apply_first_gripper_close_delay,
    apply_normalized_action_noise,
    save_aligned_transition,
    split_robotwin_camera_views,
    update_episode_success,
)


def test_split_robotwin_composite_preserves_three_views():
    frame = np.zeros((384, 320, 3), dtype=np.uint8)
    frame[:256] = 10
    frame[256:, :160] = 20
    frame[256:, 160:] = 30
    views = split_robotwin_camera_views(frame)
    assert views["head"].shape == (256, 320, 3)
    assert views["left_wrist"].shape == (128, 160, 3)
    assert views["right_wrist"].shape == (128, 160, 3)
    assert [int(views[name].mean()) for name in views] == [10, 20, 30]


def test_shared_noise_seed_scales_same_direction_and_excludes_grippers():
    baseline = np.zeros((4, 14), dtype=np.float32)
    mild, mild_epsilon = apply_normalized_action_noise(
        baseline, noise_std=0.1, rng=np.random.default_rng(7)
    )
    strong, strong_epsilon = apply_normalized_action_noise(
        baseline, noise_std=0.3, rng=np.random.default_rng(7)
    )
    np.testing.assert_array_equal(mild_epsilon, strong_epsilon)
    np.testing.assert_allclose(strong, 3.0 * mild, atol=1e-6)
    np.testing.assert_array_equal(mild[:, [6, 13]], 0.0)


def test_action_chunk_hold_freezes_only_arm_targets():
    current = np.arange(14, dtype=np.float32)
    baseline = np.stack([current + 1.0, current + 2.0])
    held, mask = apply_action_chunk_hold(
        baseline, current_action=current, hold_chunk=True
    )
    arm_indices = [index for index in range(14) if index not in (6, 13)]
    np.testing.assert_array_equal(
        held[:, arm_indices], np.tile(current[arm_indices], (2, 1))
    )
    np.testing.assert_array_equal(held[:, [6, 13]], baseline[:, [6, 13]])
    np.testing.assert_array_equal(mask[:, arm_indices], 1.0)
    np.testing.assert_array_equal(mask[:, [6, 13]], 0.0)


def test_gripper_close_delay_crosses_chunk_boundary_once():
    current = np.zeros(14, dtype=np.float32)
    current[[6, 13]] = 1.0
    first = np.tile(current, (3, 1))
    first[:, 6] = [0.8, 0.4, 0.0]
    delayed, mask, triggered, remaining = apply_first_gripper_close_delay(
        first,
        current_action=current,
        delay_steps=3,
        already_triggered=np.zeros(2, dtype=bool),
        remaining_steps=np.zeros(2, dtype=np.int64),
    )
    np.testing.assert_allclose(delayed[:, 6], [0.8, 0.8, 0.8])
    np.testing.assert_array_equal(mask[:, 6], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(triggered, [True, False])
    np.testing.assert_array_equal(remaining, [1, 0])

    second = np.tile(current, (2, 1))
    second[:, 6] = 0.0
    delayed, mask, triggered, remaining = apply_first_gripper_close_delay(
        second,
        current_action=delayed[-1],
        delay_steps=3,
        already_triggered=triggered,
        remaining_steps=remaining,
    )
    np.testing.assert_allclose(delayed[:, 6], [0.8, 0.0])
    np.testing.assert_array_equal(mask[:, 6], [1.0, 0.0])
    np.testing.assert_array_equal(triggered, [True, False])
    np.testing.assert_array_equal(remaining, [0, 0])


def test_saved_transition_can_backfill_episode_success(tmp_path: Path):
    frame = Image.fromarray(np.zeros((384, 320, 3), dtype=np.uint8))
    metadata_path = save_aligned_transition(
        tmp_path / "transition",
        current_frame=frame,
        predicted_goal_frame=frame,
        actual_frame=frame,
        metadata={"episode_success": False},
        rollout_arrays={"actions": np.zeros((2, 14), dtype=np.float32)},
    )
    update_episode_success([metadata_path], True)
    assert '"episode_success": true' in metadata_path.read_text(encoding="utf-8")


def test_reward_discovery_ignores_pairing_quarantine(tmp_path: Path):
    import json

    frame = Image.fromarray(np.zeros((384, 320, 3), dtype=np.uint8))
    for root in (tmp_path / "task", tmp_path / "pairing_quarantine" / "task"):
        root.mkdir(parents=True)
        (root / "metadata.json").write_text(
            json.dumps({"schema_version": "robotwin_imagination_transition_v1"}),
            encoding="utf-8",
        )
        for phase in ("current", "actual", "predicted_goal"):
            frame.save(root / f"{phase}.png")

    records = discover_records([tmp_path])

    assert len(records) == 1
    assert "pairing_quarantine" not in records[0]["record_dir"]
