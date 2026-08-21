from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.audit_can_expert_episode_replay import summarize_replay


def test_replay_summary_identifies_failed_successful_demonstrations():
    rows = [
        {
            "source_demo": "demo_0",
            "stored_success": 1,
            "replay_success": 1,
            "restore_linf": 0.0,
            "max_stored_state_linf": 0.1,
        },
        {
            "source_demo": "demo_1",
            "stored_success": 1,
            "replay_success": 0,
            "restore_linf": 0.0,
            "max_stored_state_linf": 0.2,
        },
    ]

    summary = summarize_replay(rows)

    assert summary["success_replay_rate"] == 0.5
    assert summary["failed_replay_demos"] == ["demo_1"]
