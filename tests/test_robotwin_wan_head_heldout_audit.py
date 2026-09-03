from experiments.robotwin.audit_wan_head_heldout_pair import audit


def payload(control, candidate, baseline=None, task="open_microwave"):
    def records(values):
        return [
            {"seed": 100 + index, "instruction": f"instruction-{index}", "success": value}
            for index, value in enumerate(values)
        ]

    rows = [
        {"variant": "no_imagination", "task": task, "status": "complete", "episode_records": records(control)},
        {"variant": "imagination", "task": task, "status": "complete", "episode_records": records(candidate)},
    ]
    if baseline is not None:
        rows.insert(
            0,
            {"variant": "baseline", "task": task, "status": "complete", "episode_records": records(baseline)},
        )
    return {
        "initial_state_audit": {task: {"exact_match": True}},
        "protocol_pairing_audit": {
            task: {"exact_seed_and_instruction_match": True}
        },
        "rows": rows,
    }


def test_heldout_confirmation_requires_positive_paired_net_gain():
    result = audit(payload([True] * 7 + [False] * 3, [True] * 9 + [False]))
    assert result["decision"] == "confirmed"
    assert result["paired_wins"] == 2
    assert result["paired_losses"] == 0
    assert result["strong_confirmation"] is True
    assert result["paired_exact_one_sided_p"] == 0.25
    assert result["statistical_confirmation"] is False


def test_heldout_tie_is_inconclusive_not_retuned():
    result = audit(payload([True] * 8 + [False] * 2, [True] * 8 + [False] * 2))
    assert result["decision"] == "inconclusive"


def test_expanded_heldout_requires_requested_pair_count_and_can_confirm_statistically():
    result = audit(
        payload([True] * 20 + [False] * 10, [True] * 27 + [False] * 3),
        expected_pairs=30,
    )
    assert result["paired_wins"] == 7
    assert result["paired_losses"] == 0
    assert result["paired_exact_one_sided_p"] == 0.0078125
    assert result["statistical_confirmation"] is True


def test_three_way_audit_reports_both_residual_comparisons_to_baseline():
    result = audit(
        payload(
            [True, True, False, False],
            [True, True, True, False],
            baseline=[True, False, False, False],
        ),
        expected_pairs=4,
    )
    assert result["baseline_successes"] == 1
    assert result["comparisons_to_baseline"]["no_imagination"] == {
        "reference_successes": 1,
        "contender_successes": 2,
        "paired_wins": 1,
        "paired_losses": 0,
        "both_success": 1,
        "both_failure": 2,
    }
    assert result["comparisons_to_baseline"]["imagination"] == {
        "reference_successes": 1,
        "contender_successes": 3,
        "paired_wins": 2,
        "paired_losses": 0,
        "both_success": 1,
        "both_failure": 1,
    }


def test_audit_supports_place_can_basket():
    result = audit(
        payload([False, True], [True, True], task="place_can_basket"),
        expected_pairs=2,
        task="place_can_basket",
    )
    assert result["task"] == "place_can_basket"
    assert result["candidate_successes"] == 2
    assert result["paired_wins"] == 1
