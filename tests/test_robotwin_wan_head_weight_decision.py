from experiments.robotwin.decide_wan_head_weight_candidate import decide


def payload(control, candidate):
    records = lambda values: [
        {"seed": index, "instruction": f"instruction-{index}", "success": value}
        for index, value in enumerate(values)
    ]
    return {
        "initial_state_audit": {"open_microwave": {"exact_match": True}},
        "protocol_pairing_audit": {
            "open_microwave": {"exact_seed_and_instruction_match": True}
        },
        "rows": [
            {
                "variant": "no_imagination",
                "task": "open_microwave",
                "status": "complete",
                "episode_records": records(control),
            },
            {
                "variant": "imagination",
                "task": "open_microwave",
                "status": "complete",
                "episode_records": records(candidate),
            },
        ],
    }


def test_clean_paired_gain_is_promoted():
    result = decide(
        payload([True, True, True, True, False], [True] * 5),
        weight=0.25,
        retry_on_tie=True,
    )
    assert result["decision"] == "promote_to_new_heldout"
    assert result["paired_wins"] == 1
    assert result["paired_losses"] == 0


def test_tie_retries_once_and_regression_redesigns():
    tied = decide(
        payload([True, True, True, True, False], [True, True, True, False, True]),
        weight=0.25,
        retry_on_tie=True,
    )
    assert tied["decision"] == "retry_lower_weight"
    final_tie = decide(
        payload([True, True, True, True, False], [True, True, True, False, True]),
        weight=0.1,
        retry_on_tie=False,
    )
    assert final_tie["decision"] == "redesign_reward"
