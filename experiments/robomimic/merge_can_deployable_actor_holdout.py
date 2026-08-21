#!/usr/bin/env python3
"""Combine frozen training references with a trajectory-disjoint actor holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def merge(
    training_path: str | Path,
    holdout_path: str | Path,
    output_path: str | Path,
    *,
    training_demo_prefix: str = "frozen_train:",
) -> dict[str, Any]:
    training_path = Path(training_path).expanduser().resolve()
    holdout_path = Path(holdout_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    with np.load(training_path, allow_pickle=False) as loaded:
        training = {key: loaded[key] for key in loaded.files}
    with np.load(holdout_path, allow_pickle=False) as loaded:
        holdout = {key: loaded[key] for key in loaded.files}

    train_mask = training["source_split"] == "train"
    valid_mask = holdout["source_split"] == "valid"
    if not np.any(train_mask) or not np.any(valid_mask):
        raise ValueError("Both frozen training and new validation rows are required")
    if training.keys() != holdout.keys():
        raise ValueError("Training and holdout datasets have different fields")

    train_count = len(training["source_split"])
    holdout_count = len(holdout["source_split"])
    merged: dict[str, np.ndarray] = {}
    for key in training:
        train_value = training[key]
        holdout_value = holdout[key]
        if train_value.ndim == 0 and holdout_value.ndim == 0:
            if not np.array_equal(train_value, holdout_value):
                raise ValueError(f"Scalar metadata mismatch: {key}")
            merged[key] = train_value
        elif len(train_value) == train_count and len(holdout_value) == holdout_count:
            train_rows = train_value[train_mask]
            valid_rows = holdout_value[valid_mask]
            if key == "source_demo":
                train_rows = np.asarray(
                    [f"{training_demo_prefix}{value}" for value in train_rows.astype(str)]
                )
            merged[key] = np.concatenate([train_rows, valid_rows], axis=0)
        else:
            if not np.array_equal(train_value, holdout_value):
                raise ValueError(f"Static metadata mismatch: {key}")
            merged[key] = train_value

    train_demos = set(merged["source_demo"][merged["source_split"] == "train"].astype(str))
    valid_demos = set(merged["source_demo"][merged["source_split"] == "valid"].astype(str))
    if train_demos & valid_demos:
        raise ValueError("Merged dataset contains trajectory-name leakage")
    if not all(np.all(np.isfinite(merged[key])) for key in ("state", "base_action_chunk")):
        raise ValueError("Merged deployable features contain non-finite values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **merged)
    report = {
        "training_dataset": str(training_path),
        "holdout_dataset": str(holdout_path),
        "output": str(output_path),
        "train_rows": int(np.count_nonzero(merged["source_split"] == "train")),
        "valid_rows": int(np.count_nonzero(merged["source_split"] == "valid")),
        "state_dim": int(merged["state"].shape[1]),
        "training_demo_prefix": training_demo_prefix,
        "all_finite": True,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--holdout-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--training-demo-prefix", default="frozen_train:")
    args = parser.parse_args()
    report = merge(
        args.training_dataset,
        args.holdout_dataset,
        args.output,
        training_demo_prefix=args.training_demo_prefix,
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
