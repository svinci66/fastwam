import numpy as np

from experiments.libero.build_residual_rl_replay import build_replay
from experiments.libero.imagination_reward_utils import save_aligned_transition
from fastwam.rl.rewards import CompositeRewardConfig


def test_aligned_transition_saves_authoritative_rollout_arrays(tmp_path):
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    metadata_path = save_aligned_transition(
        tmp_path / "record",
        current_frame=frame,
        predicted_goal_frame=frame,
        actual_frame=frame,
        metadata={"alignment_valid": True},
        rollout_arrays={
            "baseline_actions": np.zeros((8, 7), dtype=np.float32),
            "executed_actions": np.ones((8, 7), dtype=np.float32),
        },
    )
    assert metadata_path.exists()
    with np.load(tmp_path / "record" / "rollout_arrays.npz", allow_pickle=False) as payload:
        assert payload["baseline_actions"].shape == (8, 7)
        assert np.all(payload["executed_actions"] == 1.0)


def test_replay_builder_uses_dual_camera_features_and_recorded_actions(tmp_path):
    arrays_path = tmp_path / "rollout_arrays.npz"
    baseline = np.zeros((8, 7), dtype=np.float32)
    executed = baseline.copy()
    executed[:, 0] = 0.01
    np.savez_compressed(
        arrays_path,
        proprio=np.zeros(8, dtype=np.float32),
        next_proprio=np.ones(8, dtype=np.float32),
        baseline_actions=baseline,
        planned_executed_actions=baseline,
        executed_actions=executed,
        environment_rewards=np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
    )
    record = {
        "record_dir": tmp_path,
        "arrays_path": arrays_path,
        "task_suite": "libero_goal",
        "task_id": 3,
        "task_description": "open the top drawer and put the bowl inside",
        "trial_idx": 0,
        "replan_idx": 0,
        "target_step": 8,
        "effective_k": 8,
        "alignment_valid": True,
        "transition_success": True,
        "terminated": True,
        "truncated": False,
        "env_seed": 42,
        "goal_seed": 43,
        "action_seed": 44,
        "action_mode": "noise",
        "action_noise_std": 0.15,
        "policy_version": "policy-sha",
        "predictor_version": "predictor-sha",
        "goal_frame_index": 2,
        "goal_tau": 8.0,
    }
    encoded = {
        "current": {
            "agent": np.array([1.0, 0.0], dtype=np.float32),
            "wrist": np.array([1.0, 0.0], dtype=np.float32),
        },
        "actual": {
            "agent": np.array([0.0, 1.0], dtype=np.float32),
            "wrist": np.array([0.0, 1.0], dtype=np.float32),
        },
        "predicted_goal": {
            "agent": np.array([0.0, 1.0], dtype=np.float32),
            "wrist": np.array([0.0, 1.0], dtype=np.float32),
        },
    }
    replay = build_replay(
        [record],
        [encoded],
        reward_encoder_version="siglip-sha",
        reward_config=CompositeRewardConfig(imitation_weight=0.0),
        camera_weights={"agent": 0.5, "wrist": 0.5},
        imitation_dimension_scales=None,
    )
    transition = replay.transitions[0]
    assert transition.reward.imagination_raw > 0.0
    assert transition.reward.environment_return == 1.0
    assert transition.reward.environment_component == 0.0
    np.testing.assert_array_equal(transition.executed_actions, executed)
    assert transition.observation_feature.shape == (4,)
