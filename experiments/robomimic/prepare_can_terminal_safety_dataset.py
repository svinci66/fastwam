#!/usr/bin/env python3
"""Convert terminal-tail audits into a history-aware pairwise safety dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def prepare(
    audit_paths: list[Path],
    source_dataset: Path,
    output_path: Path,
    *,
    success_bonus: float = 10.0,
    score_margin: float = 1e-6,
) -> dict[str, Any]:
    source_dataset = source_dataset.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    with h5py.File(source_dataset, "r") as source:
        split_lookup = {
            str(demo): split
            for split in source["mask"]
            for demo in source["mask"][split].asstr()[...]
        }

    rows: list[dict[str, Any]] = []
    observation_metadata: dict[str, Any] | None = None
    for path in audit_paths:
        payload = json.loads(path.read_text())
        if not payload.get("complete", False):
            raise ValueError(f"Incomplete terminal audit: {path}")
        demo = str(payload["demo"])
        current_metadata = payload.get("observation_metadata", {})
        if observation_metadata is None:
            observation_metadata = current_metadata
        elif current_metadata != observation_metadata:
            raise ValueError("Terminal audits use different observation metadata")
        if demo not in split_lookup:
            raise ValueError(f"Audit demo is absent from source masks: {demo}")
        for row in payload["rows"]:
            if not row["accepted"]:
                continue
            remaining_steps = max(1, int(payload.get("episode_steps", 0)) - row["start_step"])
            reward_delta = float(row["tail_reward_delta"])
            success_delta = int(row["residual_tail_success"]) - int(row["base_tail_success"])
            delta_score = success_bonus * success_delta + reward_delta / remaining_steps
            if abs(delta_score) <= score_margin:
                continue
            history = np.asarray(
                [
                    row["progress"],
                    row["accepted_interventions_before"],
                    row["cumulative_residual_norm_before"],
                ],
                dtype=np.float32,
            )
            state = np.asarray(row["state"], dtype=np.float32)
            rows.append(
                {
                    "state": np.concatenate([state, history]),
                    "base_action_chunk": np.asarray(row["base_action_chunk"], dtype=np.float32),
                    "candidate_action_chunk": np.asarray(
                        row["proposal_action_chunk"], dtype=np.float32
                    ),
                    "label": 1 if delta_score > 0 else -1,
                    "delta_score": delta_score,
                    "source_split": split_lookup[demo],
                    "source_demo": demo,
                    "source_step": int(row["start_step"]),
                    "comparison_type": "terminal_tail",
                    "terminal_outcome_changed": int(success_delta != 0),
                    "terminal_success_delta": success_delta,
                }
            )
    if not rows:
        raise ValueError("No decisive accepted terminal interventions were found")

    keys = rows[0].keys()
    arrays = {key: np.asarray([row[key] for row in rows]) for key in keys}
    arrays["state"] = arrays["state"].astype(np.float32)
    arrays["base_action_chunk"] = arrays["base_action_chunk"].astype(np.float32)
    arrays["candidate_action_chunk"] = arrays["candidate_action_chunk"].astype(np.float32)
    arrays["label"] = arrays["label"].astype(np.int8)
    arrays["delta_score"] = arrays["delta_score"].astype(np.float32)
    arrays["source_split"] = arrays["source_split"].astype("U5")
    arrays["source_demo"] = arrays["source_demo"].astype("U32")
    arrays["source_step"] = arrays["source_step"].astype(np.int32)
    arrays["comparison_type"] = arrays["comparison_type"].astype("U18")
    arrays["terminal_outcome_changed"] = arrays["terminal_outcome_changed"].astype(np.uint8)
    arrays["terminal_success_delta"] = arrays["terminal_success_delta"].astype(np.int8)
    arrays["observation_mode"] = np.asarray("siglip_wrist_proprio_history")
    arrays["history_dim"] = np.asarray(3, dtype=np.int32)
    for key, value in (observation_metadata or {}).items():
        if key == "observation_mode":
            continue
        arrays[key] = np.asarray(value)
    train_demos = set(arrays["source_demo"][arrays["source_split"] == "train"])
    valid_demos = set(arrays["source_demo"][arrays["source_split"] == "valid"])
    if train_demos & valid_demos:
        raise ValueError("Terminal safety dataset contains trajectory leakage")
    if not all(
        np.all(np.isfinite(arrays[key]))
        for key in ("state", "base_action_chunk", "candidate_action_chunk", "delta_score")
    ):
        raise ValueError("Terminal safety dataset contains non-finite values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    result = {
        "source_dataset": str(source_dataset),
        "audit_files": len(audit_paths),
        "output": str(output_path),
        "pairs": len(rows),
        "train_pairs": int(np.count_nonzero(arrays["source_split"] == "train")),
        "valid_pairs": int(np.count_nonzero(arrays["source_split"] == "valid")),
        "terminal_outcome_changed": int(np.count_nonzero(arrays["terminal_outcome_changed"])),
        "terminal_success_gains": int(np.count_nonzero(arrays["terminal_success_delta"] > 0)),
        "terminal_success_losses": int(np.count_nonzero(arrays["terminal_success_delta"] < 0)),
        "state_dim": int(arrays["state"].shape[1]),
        "all_finite": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--score-margin", type=float, default=1e-6)
    args = parser.parse_args()
    result = prepare(
        [path.expanduser().resolve() for path in args.audit],
        args.source_dataset,
        args.output,
        success_bonus=args.success_bonus,
        score_margin=args.score_margin,
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
