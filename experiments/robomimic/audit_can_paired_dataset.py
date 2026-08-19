#!/usr/bin/env python3
"""Audit the structural assumptions of RoboMimic's Can Paired dataset.

The released dataset stores consecutive demonstrations as a pair. Each pair
should start from the same simulator state and contain one successful and one
unsuccessful trajectory. These properties make the dataset useful for testing
counterfactual value-ranking objectives without first collecting new robot
rollouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _demo_number(name: str) -> int:
    prefix, number = name.rsplit("_", 1)
    if prefix != "demo":
        raise ValueError(f"Unexpected demonstration name: {name}")
    return int(number)


def _decode_names(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def audit_paired_dataset(
    dataset_path: str | Path,
    *,
    initial_state_tolerance: float = 1e-10,
    success_reward_threshold: float = 0.0,
) -> dict[str, Any]:
    """Return a JSON-serializable report and fail closed on malformed data."""

    dataset_path = Path(dataset_path).expanduser().resolve()
    with h5py.File(dataset_path, "r") as dataset:
        if "data" not in dataset:
            raise ValueError("Dataset has no 'data' group")
        demo_names = sorted(dataset["data"].keys(), key=_demo_number)
        if not demo_names or len(demo_names) % 2:
            raise ValueError(f"Expected a positive even number of demos, got {len(demo_names)}")

        demos: list[dict[str, Any]] = []
        for name in demo_names:
            group = dataset["data"][name]
            rewards = np.asarray(group["rewards"])
            states = np.asarray(group["states"])
            actions = np.asarray(group["actions"])
            if len(states) == 0:
                raise ValueError(f"{name} has no simulator states")
            demos.append(
                {
                    "name": name,
                    "num_samples": int(group.attrs.get("num_samples", len(actions))),
                    "initial_state": states[0],
                    "success": bool(np.max(rewards, initial=-np.inf) > success_reward_threshold),
                    "max_reward": float(np.max(rewards, initial=-np.inf)),
                }
            )

        pair_reports: list[dict[str, Any]] = []
        valid_pair_count = 0
        max_initial_state_linf = 0.0
        for pair_index in range(len(demos) // 2):
            first, second = demos[2 * pair_index : 2 * pair_index + 2]
            if first["initial_state"].shape != second["initial_state"].shape:
                initial_linf = float("inf")
            else:
                initial_linf = float(
                    np.max(np.abs(first["initial_state"] - second["initial_state"]), initial=0.0)
                )
            same_initial_state = initial_linf <= initial_state_tolerance
            complementary_outcomes = first["success"] != second["success"]
            valid = same_initial_state and complementary_outcomes
            valid_pair_count += int(valid)
            max_initial_state_linf = max(max_initial_state_linf, initial_linf)
            pair_reports.append(
                {
                    "pair_index": pair_index,
                    "demos": [first["name"], second["name"]],
                    "successes": [first["success"], second["success"]],
                    "num_samples": [first["num_samples"], second["num_samples"]],
                    "initial_state_linf": initial_linf,
                    "same_initial_state": same_initial_state,
                    "complementary_outcomes": complementary_outcomes,
                    "valid": valid,
                }
            )

        masks: dict[str, list[str]] = {}
        if "mask" in dataset:
            for key in dataset["mask"]:
                masks[key] = _decode_names(np.asarray(dataset["mask"][key]))

        split_pair_integrity: dict[str, bool] = {}
        for split, names in masks.items():
            selected = set(names)
            split_pair_integrity[split] = all(
                ((first in selected) == (second in selected))
                for first, second in (pair["demos"] for pair in pair_reports)
            )

        success_count = sum(int(demo["success"]) for demo in demos)
        report = {
            "dataset_path": str(dataset_path),
            "dataset_bytes": dataset_path.stat().st_size,
            "demo_count": len(demos),
            "transition_count": int(sum(demo["num_samples"] for demo in demos)),
            "success_count": success_count,
            "failure_count": len(demos) - success_count,
            "pair_count": len(pair_reports),
            "valid_pair_count": valid_pair_count,
            "invalid_pair_count": len(pair_reports) - valid_pair_count,
            "max_initial_state_linf": max_initial_state_linf,
            "initial_state_tolerance": initial_state_tolerance,
            "split_sizes": {key: len(value) for key, value in masks.items()},
            "split_pair_integrity": split_pair_integrity,
            "all_pairs_valid": valid_pair_count == len(pair_reports),
            "pairs": pair_reports,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--initial-state-tolerance", type=float, default=1e-10)
    parser.add_argument("--allow-invalid", action="store_true")
    args = parser.parse_args()

    report = audit_paired_dataset(
        args.dataset,
        initial_state_tolerance=args.initial_state_tolerance,
    )
    summary = {key: value for key, value in report.items() if key != "pairs"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.allow_invalid and not report["all_pairs_valid"]:
        raise SystemExit("Paired dataset audit failed")


if __name__ == "__main__":
    main()
