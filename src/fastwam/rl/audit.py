"""Determinism-audit helpers shared by LIBERO evaluation code and tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np


def array_sha256(value: Any) -> str:
    """Hash an array together with its exact dtype and shape."""

    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def resolve_trial_indices(
    *,
    num_trials: int,
    trial_indices: Sequence[int] | None,
    available_states: int,
) -> list[int]:
    """Resolve default sequential trials or validate explicit state indices."""

    if num_trials <= 0:
        raise ValueError(f"num_trials must be positive, got {num_trials}")
    if available_states <= 0:
        raise ValueError(f"available_states must be positive, got {available_states}")
    if trial_indices is None:
        return list(range(num_trials))

    resolved = [int(index) for index in trial_indices]
    if not resolved:
        raise ValueError("trial_indices must be non-empty when provided")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"trial_indices must not contain duplicates, got {resolved}")
    invalid = [index for index in resolved if index < 0 or index >= available_states]
    if invalid:
        raise ValueError(
            "trial_indices are outside the available initial-state range "
            f"[0, {available_states - 1}]: {invalid}"
        )
    return resolved
