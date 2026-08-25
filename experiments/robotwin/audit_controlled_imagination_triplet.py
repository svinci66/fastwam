#!/usr/bin/env python3
"""Fail-closed audit for a clean/corrupted/corrected RoboTwin triplet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_records(root: Path) -> dict[int, tuple[dict, dict[str, np.ndarray]]]:
    records: dict[int, tuple[dict, dict[str, np.ndarray]]] = {}
    for metadata_path in sorted(root.glob("replan_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays_path = metadata_path.with_name("rollout_arrays.npz")
        with np.load(arrays_path) as payload:
            arrays = {key: payload[key].copy() for key in payload.files}
        records[int(metadata["replan_idx"])] = (metadata, arrays)
    if not records:
        raise FileNotFoundError(f"No aligned transitions found under {root}")
    return records


def _allclose(left: np.ndarray, right: np.ndarray, *, name: str) -> None:
    np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-6, err_msg=name)


def audit_triplet(
    *, clean_root: Path, corrupt_root: Path, correct_root: Path, intervention_replan: int
) -> dict:
    clean = _load_records(clean_root)
    corrupt = _load_records(corrupt_root)
    correct = _load_records(correct_root)
    common_replans = sorted(set(clean) & set(corrupt) & set(correct))
    if intervention_replan not in common_replans:
        raise AssertionError(
            f"Intervention replan {intervention_replan} is missing; common={common_replans}"
        )

    # Corrected executes exactly the clean policy, so their complete audit trace
    # must stay paired, not merely their initial observation.
    if set(clean) != set(correct):
        raise AssertionError(
            f"clean/corrected replan sets differ: {sorted(clean)} vs {sorted(correct)}"
        )
    for replan in sorted(clean):
        clean_meta, clean_arrays = clean[replan]
        correct_meta, correct_arrays = correct[replan]
        for key in (
            "initial_observation_sha256",
            "current_observation_sha256",
            "baseline_actions_sha256",
        ):
            if clean_meta[key] != correct_meta[key]:
                raise AssertionError(f"clean/corrected {key} differs at replan {replan}")
        _allclose(
            clean_arrays["executed_actions"],
            correct_arrays["executed_actions"],
            name=f"clean/corrected executed actions at replan {replan}",
        )

    clean_meta, clean_arrays = clean[intervention_replan]
    corrupt_meta, corrupt_arrays = corrupt[intervention_replan]
    correct_meta, correct_arrays = correct[intervention_replan]
    for key in (
        "initial_observation_sha256",
        "current_observation_sha256",
        "baseline_actions_sha256",
    ):
        values = {clean_meta[key], corrupt_meta[key], correct_meta[key]}
        if len(values) != 1:
            raise AssertionError(f"triplet {key} differs before intervention")

    corrupted = corrupt_arrays["controlled_corrupted_actions"]
    target = corrupt_arrays["controlled_target_residual_actions"]
    baseline = corrupt_arrays["baseline_actions"]
    _allclose(corrupted, corrupt_arrays["planned_actions"], name="corrupt planned")
    _allclose(corrupted, corrupt_arrays["executed_actions"], name="corrupt executed")
    _allclose(baseline, corrupted + target, name="inverse residual target")
    _allclose(
        corrupted,
        correct_arrays["controlled_corrupted_actions"],
        name="shared corrupted pseudo-baseline",
    )
    _allclose(
        target,
        correct_arrays["controlled_target_residual_actions"],
        name="shared inverse target",
    )
    _allclose(
        correct_arrays["baseline_actions"],
        correct_arrays["executed_actions"],
        name="corrected restores clean action",
    )

    normalized_delta = corrupt_arrays["normalized_noise_direction"]
    max_abs_delta = float(np.max(np.abs(normalized_delta)))
    if max_abs_delta > float(corrupt_meta["action_noise_std"]) + 1e-7:
        raise AssertionError(
            f"normalized corruption {max_abs_delta} exceeds registered bound"
        )
    np.testing.assert_array_equal(normalized_delta[:, [6, 13]], 0.0)
    if not np.any(np.abs(normalized_delta) > 0.0):
        raise AssertionError("controlled corruption is identically zero")

    return {
        "status": "pass",
        "intervention_replan": intervention_replan,
        "common_replans": common_replans,
        "clean_corrected_full_trace_replans": len(clean),
        "normalized_delta_max_abs": max_abs_delta,
        "environment_seed": clean_meta.get("environment_seed"),
        "instruction": clean_meta.get("task_description"),
        "episode_success": {
            "clean": bool(clean_meta["episode_success"]),
            "corrupted": bool(corrupt_meta["episode_success"]),
            "corrected": bool(correct_meta["episode_success"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--corrupt-root", type=Path, required=True)
    parser.add_argument("--correct-root", type=Path, required=True)
    parser.add_argument("--intervention-replan", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    summary = audit_triplet(
        clean_root=args.clean_root,
        corrupt_root=args.corrupt_root,
        correct_root=args.correct_root,
        intervention_replan=args.intervention_replan,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
