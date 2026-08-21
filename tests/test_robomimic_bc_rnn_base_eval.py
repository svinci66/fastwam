import pytest

from experiments.robomimic.evaluate_can_bc_rnn_base import summarize_rollouts


def test_summarize_rollouts():
    rows = [
        {"Success_Rate": 1.0, "Return": 1.0, "Horizon": 100.0},
        {"Success_Rate": 0.0, "Return": 0.0, "Horizon": 400.0},
    ]
    summary = summarize_rollouts(rows)
    assert summary["episodes"] == 2
    assert summary["successes"] == 1
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["horizon_mean"] == pytest.approx(250.0)


def test_summarize_rollouts_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one"):
        summarize_rollouts([])
