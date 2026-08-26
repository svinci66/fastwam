import json
from pathlib import Path

from experiments.robotwin.prepare_natural_failure_screen import select_seen_instruction
from experiments.robotwin.summarize_natural_failure_screen import parse_eval_log


def test_seen_instruction_selection_is_seed_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "episode0.json"
    path.write_text(json.dumps({"seen": ["third", "first", "second"]}))
    first = select_seen_instruction(path, 7)
    assert first in {"first", "second", "third"}
    assert first == select_seen_instruction(path, 7)


def test_parse_eval_log_pairs_accepted_seed_and_outcome(tmp_path: Path) -> None:
    path = tmp_path / "eval.log"
    path.write_text(
        "FASTWAM_ACCEPTED_ENV_SEED episode_id=0 seed=2\n"
        "\x1b[92mSuccess!\x1b[0m\n"
        "FASTWAM_ACCEPTED_ENV_SEED episode_id=1 seed=5\n"
        "\x1b[91mFail!\x1b[0m\n"
    )
    assert parse_eval_log(path) == [
        {"evaluation_episode_id": 0, "environment_seed": 2, "success": True},
        {"evaluation_episode_id": 1, "environment_seed": 5, "success": False},
    ]
