from pathlib import Path

import numpy as np
from PIL import Image

from experiments.robotwin.imagination_reward_utils import (
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
