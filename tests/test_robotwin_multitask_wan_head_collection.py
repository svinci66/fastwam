import json
import sys

from experiments.robotwin.audit_multitask_wan_head_collection import (
    EXPECTED_TASKS,
    main,
)


def test_balanced_four_task_collection_is_training_ready(tmp_path, monkeypatch):
    screen = {
        "per_task": {
            task: {
                "episodes": 10,
                "fastwam_successes": 5,
                "natural_failures": 5,
            }
            for task in EXPECTED_TASKS
        }
    }
    pairs = []
    for task in EXPECTED_TASKS:
        pairs.extend(
            {
                "task": task,
                "success_minus_failure": margin,
            }
            for margin in (0.5, 0.4, 0.3, 0.2, -0.1)
        )
    reward = {
        "schema_version": "robotwin_natural_failure_wan_vae_pair_reward_v1",
        "reward_cameras": ["head"],
        "pairs": pairs,
    }
    screen_path = tmp_path / "screen.json"
    reward_path = tmp_path / "reward.json"
    output_path = tmp_path / "audit.json"
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    reward_path.write_text(json.dumps(reward), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--screen-summary",
            str(screen_path),
            "--reward-json",
            str(reward_path),
            "--tasks",
            ",".join(EXPECTED_TASKS),
            "--episodes-per-task",
            "10",
            "--minimum-failures-per-task",
            "5",
            "--output-json",
            str(output_path),
            "--require-training-ready",
        ],
    )
    main()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["collection_complete"] is True
    assert result["reward_validation_pass"] is True
    assert result["training_ready"] is True
    assert result["macro_pairwise_accuracy"] == 0.8
