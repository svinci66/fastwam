import numpy as np
import pytest

from experiments.robomimic.collect_can_bc_rnn_rollouts import branch_steps, episode_split


def test_episode_split_keeps_trajectory_groups_disjoint():
    splits = [episode_split(index, valid_every=5) for index in range(10)]
    assert splits == [
        "train",
        "train",
        "train",
        "train",
        "valid",
        "train",
        "train",
        "train",
        "train",
        "valid",
    ]


def test_branch_steps_reserve_full_future_horizon():
    steps = branch_steps(100, warmup=10, horizon=20, stride=5)
    np.testing.assert_array_equal(steps[:3], np.asarray([10, 15, 20]))
    assert steps[-1] == 80
    assert np.all(steps + 20 <= 100)


def test_branch_steps_handles_short_episode_and_bad_arguments():
    assert len(branch_steps(20, warmup=10, horizon=20, stride=5)) == 0
    with pytest.raises(ValueError):
        branch_steps(100, warmup=-1, horizon=20, stride=5)
    with pytest.raises(ValueError):
        episode_split(0, valid_every=1)
