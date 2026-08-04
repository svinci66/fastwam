"""Language-routing helpers for residual policies."""

from __future__ import annotations


def resolve_residual_language_instruction(
    policy_instruction: str,
    configured_instruction: str | None,
) -> str:
    """Choose the instruction encoded for the residual actor, critics, and gate.

    The base policy always keeps ``policy_instruction``.  A configured instruction
    lets the residual branch reuse the canonical prompt seen during offline RL
    training without changing FastWAM's paper-protocol instruction.
    """

    policy_value = str(policy_instruction).strip()
    if not policy_value:
        raise ValueError("policy_instruction must be non-empty")
    if configured_instruction is None:
        return policy_value
    configured_value = str(configured_instruction).strip()
    return policy_value if not configured_value else configured_value
