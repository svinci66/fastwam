#!/usr/bin/env python3
"""Fail-closed audit for frozen-plan RoboTwin trajectory triplets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_record(root: Path, replan: int) -> tuple[Path, dict, dict[str, np.ndarray]]:
    record_dir = root / f"replan_{replan:04d}"
    metadata_path = record_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays_path = record_dir / metadata["rollout_arrays_file"]
    with np.load(arrays_path) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    return record_dir, metadata, arrays


def _trajectory_hashes(record_dir: Path, metadata: dict, prefix: str) -> list[str]:
    files = metadata.get(f"{prefix}_trajectory_files")
    if not isinstance(files, list) or not files:
        raise AssertionError(f"missing {prefix} trajectory file list")
    paths = [record_dir / str(path) for path in files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {prefix} trajectory frames: {missing}")
    return [_sha256(path) for path in paths]


def _assert_equal_arrays(left: np.ndarray, right: np.ndarray, name: str) -> None:
    np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-6, err_msg=name)


def audit(
    *, clean_root: Path, corrupt_root: Path, correct_root: Path, replan: int
) -> dict:
    branches = {
        "clean": _load_record(clean_root, replan),
        "corrupt": _load_record(corrupt_root, replan),
        "correct": _load_record(correct_root, replan),
    }
    expected_offsets: list[int] | None = None
    predicted_hashes: dict[str, list[str]] = {}
    actual_hashes: dict[str, list[str]] = {}
    for name, (record_dir, metadata, _) in branches.items():
        if metadata.get("schema_version") != "robotwin_imagination_trajectory_v2":
            raise AssertionError(
                f"{name} uses unsupported schema {metadata.get('schema_version')!r}"
            )
        if metadata.get("trajectory_reference_policy") != "frozen_once_per_action_chunk":
            raise AssertionError(f"{name} did not register a frozen chunk reference")
        if metadata.get("trajectory_alignment_valid") is not True:
            raise AssertionError(f"{name} trajectory is incomplete or misaligned")
        branch_expected = [
            int(value) for value in metadata["trajectory_expected_action_offsets"]
        ]
        predicted_offsets = [
            int(value) for value in metadata["predicted_trajectory_action_offsets"]
        ]
        actual_offsets = [
            int(value) for value in metadata["actual_trajectory_action_offsets"]
        ]
        if predicted_offsets != branch_expected or actual_offsets != branch_expected:
            raise AssertionError(
                f"{name} trajectory offsets differ: expected={branch_expected}, "
                f"predicted={predicted_offsets}, actual={actual_offsets}"
            )
        if expected_offsets is None:
            expected_offsets = branch_expected
        elif branch_expected != expected_offsets:
            raise AssertionError("triplet uses different trajectory offsets")
        predicted_hashes[name] = _trajectory_hashes(record_dir, metadata, "predicted")
        actual_hashes[name] = _trajectory_hashes(record_dir, metadata, "actual")

    if not (
        predicted_hashes["clean"]
        == predicted_hashes["corrupt"]
        == predicted_hashes["correct"]
    ):
        raise AssertionError("triplet did not use the same frozen predicted trajectory")
    if actual_hashes["clean"] != actual_hashes["correct"]:
        raise AssertionError("clean/correct actual trajectories are not deterministic")

    clean_meta, clean_arrays = branches["clean"][1:]
    corrupt_meta, corrupt_arrays = branches["corrupt"][1:]
    correct_meta, correct_arrays = branches["correct"][1:]
    for key in (
        "initial_observation_sha256",
        "current_observation_sha256",
        "baseline_actions_sha256",
    ):
        if len({clean_meta[key], corrupt_meta[key], correct_meta[key]}) != 1:
            raise AssertionError(f"triplet differs before intervention: {key}")

    _assert_equal_arrays(
        clean_arrays["executed_actions"],
        correct_arrays["executed_actions"],
        "corrected action must reproduce clean action",
    )
    _assert_equal_arrays(
        corrupt_arrays["baseline_actions"],
        corrupt_arrays["controlled_corrupted_actions"]
        + corrupt_arrays["controlled_target_residual_actions"],
        "inverse residual must restore baseline",
    )
    _assert_equal_arrays(
        corrupt_arrays["controlled_corrupted_actions"],
        corrupt_arrays["executed_actions"],
        "corrupt branch must execute the controlled corruption",
    )
    if np.allclose(
        corrupt_arrays["executed_actions"], clean_arrays["executed_actions"], atol=1e-7
    ):
        raise AssertionError("controlled corruption did not change executed actions")

    return {
        "status": "pass",
        "schema_version": "robotwin_imagination_trajectory_v2",
        "intervention_replan": replan,
        "action_offsets": expected_offsets,
        "num_aligned_frames": len(expected_offsets or []),
        "same_frozen_prediction_across_triplet": True,
        "clean_correct_actual_trajectory_equal": True,
        "corrupt_actual_differs_from_clean": (
            actual_hashes["corrupt"] != actual_hashes["clean"]
        ),
        "episode_success": {
            "clean": bool(clean_meta["episode_success"]),
            "corrupt": bool(corrupt_meta["episode_success"]),
            "correct": bool(correct_meta["episode_success"]),
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
    result = audit(
        clean_root=args.clean_root,
        corrupt_root=args.corrupt_root,
        correct_root=args.correct_root,
        replan=args.intervention_replan,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
