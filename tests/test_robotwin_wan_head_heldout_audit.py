from experiments.robotwin.audit_wan_head_heldout_pair import audit


def payload(control, candidate):
    def records(values):
        return [
            {"seed": 100 + index, "instruction": f"instruction-{index}", "success": value}
            for index, value in enumerate(values)
        ]

    return {
        "initial_state_audit": {"open_microwave": {"exact_match": True}},
        "protocol_pairing_audit": {
            "open_microwave": {"exact_seed_and_instruction_match": True}
        },
        "rows": [
            {"variant": "no_imagination", "task": "open_microwave", "status": "complete", "episode_records": records(control)},
            {"variant": "imagination", "task": "open_microwave", "status": "complete", "episode_records": records(candidate)},
        ],
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
