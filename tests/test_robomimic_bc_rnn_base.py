import pytest

from experiments.robomimic.train_can_bc_rnn_base import training_schedule


def test_smoke_schedule_is_bounded():
    schedule = training_schedule("smoke")
    assert schedule["epochs"] == 2
    assert schedule["epoch_steps"] == 3
    assert schedule["rollout_horizon"] == 10


def test_train_schedule_has_full_online_rollouts():
    schedule = training_schedule("train")
    assert schedule["epochs"] == 500
    assert schedule["rollout_episodes"] == 10
    assert schedule["rollout_horizon"] == 400
    assert schedule["rollout_warmstart"] > 0


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="Unsupported mode"):
        training_schedule("unknown")
