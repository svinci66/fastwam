from experiments.robotwin.summarize_residual_iql_online_pair import parse_log


def test_parse_online_pair_log_extracts_success_and_residual_metrics(tmp_path):
    log = tmp_path / "eval_task.log"
    log.write_text(
        "\x1b[92mSuccess rate: \x1b[96m4/5\x1b[0m => \x1b[95m80.0%\x1b[0m\n"
        "[fastwam-residual] replan=0 rms=0.010000 max_abs=0.040000 "
        "gripper_max_abs=0.000000\n"
        "[fastwam-residual] replan=1 rms=0.020000 max_abs=0.050000 "
        "gripper_max_abs=0.000000\n",
        encoding="utf-8",
    )
    result = parse_log(log)
    assert result["successes"] == 4
    assert result["episodes"] == 5
    assert result["success_rate"] == 0.8
    assert result["num_residual_replans"] == 2
    assert result["residual_rms_mean"] == 0.015
    assert result["residual_max_abs"] == 0.05
    assert result["gripper_residual_max_abs"] == 0.0
