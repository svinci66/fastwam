from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.summarize_can_q_guided_validation import summarize


def test_q_guided_validation_summary_requires_consistent_actual_gain(tmp_path):
    for seed, mean, improvement, worsening in ((1, 0.1, 0.6, 0.1), (2, 0.05, 0.5, 0.2)):
        actor = tmp_path / f"actor_seed{seed}"
        actor.mkdir()
        (actor / "actual_valid100.json").write_text(
            json.dumps(
                {
                    "states": 100,
                    "delta_score_mean": mean,
                    "delta_score_median": mean,
                    "improvement_rate": improvement,
                    "worsening_rate": worsening,
                    "success_gains": 0,
                    "success_losses": 0,
                    "max_restore_linf": 0.0,
                    "max_branch_initial_state_linf": 0.0,
                }
            )
        )

    report = summarize(tmp_path)

    assert report["passed"] is True
    assert report["aggregate"]["delta_score_mean"]["min"] > 0.0
