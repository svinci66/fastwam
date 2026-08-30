import json

import numpy as np

from experiments.robotwin.analyze_open_microwave_progress_diagnostic import analyze


def _write_episode(root, run_name, variant, *, ratios, success):
    record = (
        root
        / f"{run_name}_{variant}"
        / "open_microwave"
        / "imagination_transitions"
        / "open_microwave"
        / "residual"
        / "episode_0000"
        / "replan_0000"
    )
    record.mkdir(parents=True)
    progress = np.column_stack(
        [ratios, np.zeros(len(ratios)), np.ones(len(ratios)), ratios]
    ).astype(np.float32)
    np.savez_compressed(
        record / "rollout_arrays.npz",
        task_progress=progress,
        candidate_residual_actions=np.zeros((len(ratios) - 1, 14), dtype=np.float32),
    )
    (record / "metadata.json").write_text(
        json.dumps(
            {
                "environment_seed": 123,
                "episode_success": success,
                "rollout_arrays_file": "rollout_arrays.npz",
            }
        )
    )


def test_progress_diagnostic_reports_threshold_crossing_and_pair_delta(tmp_path):
    run_name = "diagnostic"
    _write_episode(
        tmp_path,
        run_name,
        "no_imagination",
        ratios=[0.0, 0.2, 0.55],
        success=False,
    )
    _write_episode(
        tmp_path,
        run_name,
        "imagination",
        ratios=[0.0, 0.3, 0.65],
        success=True,
    )

    result = analyze(tmp_path, run_name, ["no_imagination", "imagination"])

    pair = result["pairs"][0]
    assert pair["seed"] == 123
    assert pair["no_imagination_success"] is False
    assert pair["imagination_success"] is True
    assert np.isclose(pair["max_open_ratio_delta"], 0.1)
    candidate = result["variants"]["imagination"][0]
    assert candidate["threshold_crossing_action"] == 2
