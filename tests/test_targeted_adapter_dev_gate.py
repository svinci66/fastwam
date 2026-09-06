from experiments.robotwin.audit_targeted_adapter_dev_gate import audit_dev_gate


def _summary(control: list[bool], treatment: list[bool]) -> dict:
    seeds = [4800282, 4800283, 4800286]
    return {
        "initial_state_audit": {"place_can_basket": {"exact_match": True}},
        "protocol_pairing_audit": {
            "place_can_basket": {"exact_seed_and_instruction_match": True}
        },
        "rows": [
            {
                "variant": variant,
                "episode_records": [
                    {"seed": seed, "success": success}
                    for seed, success in zip(seeds, outcomes)
                ],
            }
            for variant, outcomes in (
                ("no_imagination", control),
                ("imagination", treatment),
            )
        ],
    }


def test_dev_gate_requires_three_of_three_and_strict_control_improvement() -> None:
    report = audit_dev_gate(_summary([True, False, False], [True, True, True]))
    assert report["passed"]


def test_dev_gate_rejects_tie_even_when_both_are_safe() -> None:
    report = audit_dev_gate(_summary([True, True, True], [True, True, True]))
    assert not report["passed"]
    assert not report["checks"]["treatment_exceeds_same_data_control"]


def test_dev_gate_rejects_improvement_that_breaks_protected_state() -> None:
    report = audit_dev_gate(_summary([False, False, False], [True, True, False]))
    assert not report["passed"]
    assert not report["checks"]["treatment_preserves_all_three"]
