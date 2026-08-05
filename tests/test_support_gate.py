from pathlib import Path
from dataclasses import replace

import json
import numpy as np

from experiments.robotwin.build_residual_support_index import (
    _support_episode_indices,
    _support_task_name,
)
from fastwam.rl.replay_buffer import ReplayBuffer
from fastwam.rl.support_gate import ResidualSupportIndex, SUPPORT_INDEX_FORMAT
from test_rl_replay_buffer import make_transition


def _support_index() -> ResidualSupportIndex:
    return ResidualSupportIndex(
        observation_features=np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32
        ),
        proprio=np.asarray([[0.0], [0.1], [2.0]], dtype=np.float32),
        baseline_actions=np.zeros((3, 2, 2), dtype=np.float32),
        residual_actions=np.asarray(
            [
                [[0.02, 0.0], [0.02, 0.0]],
                [[0.03, 0.0], [0.03, 0.0]],
                [[-0.02, 0.0], [-0.02, 0.0]],
            ],
            dtype=np.float32,
        ),
        state_local_radius=np.ones(3, dtype=np.float32),
        action_local_radius=np.ones(3, dtype=np.float32),
        task_ids=np.asarray([0, 0, 1]),
        task_names=("task-a", "task-b"),
        language_prototypes=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        proprio_center=np.asarray([0.0], dtype=np.float32),
        proprio_scale=np.asarray([1.0], dtype=np.float32),
        baseline_center=np.zeros((2, 2), dtype=np.float32),
        baseline_scale=np.ones((2, 2), dtype=np.float32),
        residual_scale=np.asarray([0.05, 0.0], dtype=np.float32),
        state_threshold=0.2,
        action_threshold=0.3,
        state_increase_threshold=0.1,
        language_similarity_threshold=0.99,
        neighbors=2,
        score_neighbors=1,
    )


def test_support_builder_groups_instruction_variants_by_stable_task_id() -> None:
    first = replace(
        make_transition("episode-a", 0, terminated=True),
        task_suite="robotwin2.0",
        task_id=2,
        task_description="Hang the white mug on the rack.",
        behavior_mode="expert",
    )
    second = replace(
        make_transition("episode-b", 0, terminated=True),
        task_suite="robotwin2.0",
        task_id=2,
        task_description="Turn the beige mug and hang it on the medium rack.",
        behavior_mode="policy",
    )
    replay = ReplayBuffer([first, second])
    grouped = _support_episode_indices(replay)
    assert _support_task_name(first) == "robotwin2.0/task_0002"
    assert grouped == {"robotwin2.0/task_0002": ["episode-a", "episode-b"]}


def test_support_gate_checks_state_and_candidate_action():
    support = _support_index()
    common = {
        "baseline_actions": np.zeros((2, 2), dtype=np.float32),
        "language_feature": np.asarray([1.0, 0.0], dtype=np.float32),
    }
    accepted = support.evaluate(
        observation_feature=np.asarray([1.0, 0.0], dtype=np.float32),
        proprio=np.asarray([0.02], dtype=np.float32),
        candidate_residual_actions=np.asarray(
            [[0.02, 0.0], [0.02, 0.0]], dtype=np.float32
        ),
        **common,
    )
    assert accepted.task_name == "task-a"
    assert accepted.in_support

    unsupported_action = support.evaluate(
        observation_feature=np.asarray([1.0, 0.0], dtype=np.float32),
        proprio=np.asarray([0.02], dtype=np.float32),
        candidate_residual_actions=np.asarray(
            [[-0.05, 0.0], [-0.05, 0.0]], dtype=np.float32
        ),
        **common,
    )
    assert unsupported_action.state_in_support
    assert not unsupported_action.action_in_support

    unsupported_state = support.evaluate(
        observation_feature=np.asarray([0.0, 1.0], dtype=np.float32),
        proprio=np.asarray([3.0], dtype=np.float32),
        candidate_residual_actions=np.asarray(
            [[0.02, 0.0], [0.02, 0.0]], dtype=np.float32
        ),
        **common,
    )
    assert not unsupported_state.state_in_support


def test_support_index_round_trip(tmp_path: Path):
    support = _support_index()
    np.savez_compressed(
        tmp_path / "arrays.npz",
        observation_features=support.observation_features,
        proprio=support.proprio,
        baseline_actions=support.baseline_actions,
        residual_actions=support.residual_actions,
        state_local_radius=support.state_local_radius,
        action_local_radius=support.action_local_radius,
        task_ids=support.task_ids,
        language_prototypes=support.language_prototypes,
        language_prototype_task_ids=support.language_prototype_task_ids,
        proprio_center=support.proprio_center,
        proprio_scale=support.proprio_scale,
        baseline_center=support.baseline_center,
        baseline_scale=support.baseline_scale,
        residual_scale=support.residual_scale,
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "format": SUPPORT_INDEX_FORMAT,
                "task_names": list(support.task_names),
                "state_threshold": support.state_threshold,
                "action_threshold": support.action_threshold,
                "state_increase_threshold": support.state_increase_threshold,
                "language_similarity_threshold": support.language_similarity_threshold,
                "neighbors": support.neighbors,
                "score_neighbors": support.score_neighbors,
            }
        )
    )
    loaded = ResidualSupportIndex.load(tmp_path)
    assert loaded.task_names == support.task_names
    np.testing.assert_array_equal(
        loaded.language_prototype_task_ids,
        support.language_prototype_task_ids,
    )
    assert loaded.state_threshold == support.state_threshold


def test_support_gate_routes_multiple_language_prototypes_to_one_task() -> None:
    support = ResidualSupportIndex(
        observation_features=np.asarray([[1.0, 0.0]], dtype=np.float32),
        proprio=np.asarray([[0.0]], dtype=np.float32),
        baseline_actions=np.zeros((1, 2, 2), dtype=np.float32),
        residual_actions=np.zeros((1, 2, 2), dtype=np.float32),
        state_local_radius=np.ones(1, dtype=np.float32),
        action_local_radius=np.ones(1, dtype=np.float32),
        task_ids=np.asarray([0]),
        task_names=("robotwin2.0/task_0002",),
        language_prototypes=np.asarray(
            [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        ),
        language_prototype_task_ids=np.asarray([0, 0]),
        proprio_center=np.asarray([0.0], dtype=np.float32),
        proprio_scale=np.asarray([1.0], dtype=np.float32),
        baseline_center=np.zeros((2, 2), dtype=np.float32),
        baseline_scale=np.ones((2, 2), dtype=np.float32),
        residual_scale=np.asarray([0.05, 0.0], dtype=np.float32),
        state_threshold=1.0,
        action_threshold=1.0,
        state_increase_threshold=1.0,
        language_similarity_threshold=0.99,
        neighbors=1,
        score_neighbors=1,
    )
    decision = support.evaluate(
        observation_feature=np.asarray([1.0, 0.0], dtype=np.float32),
        proprio=np.asarray([0.0], dtype=np.float32),
        baseline_actions=np.zeros((2, 2), dtype=np.float32),
        candidate_residual_actions=np.zeros((2, 2), dtype=np.float32),
        language_feature=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    assert decision.task_name == "robotwin2.0/task_0002"
    assert decision.language_similarity == 1.0
