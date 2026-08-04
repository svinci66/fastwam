import pytest

from fastwam.rl.language_routing import resolve_residual_language_instruction


def test_residual_language_defaults_to_policy_instruction():
    assert (
        resolve_residual_language_instruction("Official unseen prompt.", None)
        == "Official unseen prompt."
    )


def test_residual_language_can_use_training_canonical_instruction():
    assert (
        resolve_residual_language_instruction(
            "Official unseen prompt.",
            "Canonical training prompt.",
        )
        == "Canonical training prompt."
    )


def test_residual_language_rejects_empty_policy_instruction():
    with pytest.raises(ValueError, match="policy_instruction must be non-empty"):
        resolve_residual_language_instruction("  ", "Canonical training prompt.")
