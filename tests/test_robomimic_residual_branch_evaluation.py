from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.evaluate_can_residual_branches import summarize_branch_results


def test_branch_summary_counts_improvements_ties_and_regressions():
    rows = [
        {
            "delta_score": 0.2,
            "base_success": 0,
            "actor_success": 1,
            "restore_linf": 0.0,
            "branch_initial_state_linf": 0.0,
        },
        {
            "delta_score": 0.0,
            "base_success": 1,
            "actor_success": 1,
            "restore_linf": 0.0,
            "branch_initial_state_linf": 0.0,
        },
        {
            "delta_score": -0.1,
            "base_success": 1,
            "actor_success": 0,
            "restore_linf": 0.0,
            "branch_initial_state_linf": 0.0,
        },
    ]

    summary = summarize_branch_results(rows, score_margin=0.01)

    assert summary["improved_states"] == 1
    assert summary["tied_states"] == 1
    assert summary["worsened_states"] == 1
    assert summary["success_gains"] == 1
    assert summary["success_losses"] == 1
