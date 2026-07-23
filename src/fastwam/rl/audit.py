"""Determinism-audit helpers shared by LIBERO evaluation code and tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np


def derive_episode_seed(
    *,
    base_seed: int,
    task_id: int,
    trial_index: int,
    stream: int = 0,
) -> int:
    """Derive an order-independent uint32 seed for one task/state/stream.

    Explicit task and trial terms prevent a resumed or reordered collection from
    changing the stochastic action sequence assigned to an initial state.  The
    stream term separates, for example, medium-noise and strong-noise behavior
    while keeping their frozen FastWAM policy seed identical.
    """

    values = {
        "base_seed": int(base_seed),
        "task_id": int(task_id),
        "trial_index": int(trial_index),
        "stream": int(stream),
    }
    if values["task_id"] < 0 or values["trial_index"] < 0 or values["stream"] < 0:
        raise ValueError(f"task_id, trial_index, and stream must be non-negative: {values}")
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="little")


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
