import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin import summarize_medium_task_screen


def test_medium_screen_organizes_failure_and_success_videos(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    artifact_dir = tmp_path / "artifacts"
    task_dir = run_dir / "hanging_mug"
    task_dir.mkdir(parents=True)
    (task_dir / "episode0.mp4").write_bytes(b"failure-video")
    (task_dir / "episode1.mp4").write_bytes(b"success-video")
    (run_dir / "eval_hanging_mug_20260825_000000.log").write_text(
        "FASTWAM_ACCEPTED_ENV_SEED episode_id=0 seed=4800001\n"
        "FASTWAM_EVAL_INSTRUCTION episode_id=0 seed=4800001 instruction='hang mug'\n"
        "Success rate: 0/1 => 0.0%\n"
        "FASTWAM_ACCEPTED_ENV_SEED episode_id=1 seed=4800002\n"
        "FASTWAM_EVAL_INSTRUCTION episode_id=1 seed=4800002 instruction='hang mug'\n"
        "Success rate: 1/2 => 50.0%\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_medium_task_screen.py",
            "--run-dir",
            str(run_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--tasks",
            "hanging_mug",
        ],
    )
    summarize_medium_task_screen.main()

    summary = json.loads((artifact_dir / "screen_summary.json").read_text())
    assert summary["rows"][0]["successes"] == 1
    assert (
        artifact_dir
        / "videos"
        / "failures"
        / "hanging_mug"
        / "baseline"
        / "seed4800001_episode0.mp4"
    ).read_bytes() == b"failure-video"
    review = list(csv.DictReader((artifact_dir / "failure_review.csv").open()))
    assert len(review) == 1
    assert review[0]["seed"] == "4800001"
