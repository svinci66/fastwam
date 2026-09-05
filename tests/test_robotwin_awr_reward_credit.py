import pytest

from experiments.robotwin.audit_awr_reward_credit import summarize_return_gap_rows


def test_return_gap_summary_reports_task_specific_shrinkage():
    result = summarize_return_gap_rows(
        [
            {"task": "a", "expert_failure_gap_change": 0.2},
            {"task": "a", "expert_failure_gap_change": -0.1},
            {"task": "b", "expert_failure_gap_change": 0.0},
        ]
    )
    assert result["pair_count"] == 3
    assert result["nonshrinking_count"] == 2
    assert result["nonshrinking_fraction"] == pytest.approx(2 / 3)
    assert result["per_task"]["a"]["nonshrinking_fraction"] == 0.5
    assert result["per_task"]["a"]["minimum_gap_change"] == -0.1
    assert result["per_task"]["b"]["strictly_improved_count"] == 0


def test_return_gap_summary_rejects_empty_rows():
    with pytest.raises(ValueError, match="no paired"):
        summarize_return_gap_rows([])
