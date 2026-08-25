#!/usr/bin/env python3
"""Aggregate pre-registered Wan VAE trajectory reward validations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.input_json:
        payload = json.loads(path.read_text(encoding="utf-8"))
        branch = payload["branches"]
        clean = float(branch["clean"]["matched"]["equal_camera_mean"])
        corrupt = float(branch["corrupt"]["matched"]["equal_camera_mean"])
        correct = float(branch["correct"]["matched"]["equal_camera_mean"])
        rows.append(
            {
                "source": str(path),
                "replan": int(payload["replan"]),
                "clean_success": bool(branch["clean"]["episode_success"]),
                "corrupt_success": bool(branch["corrupt"]["episode_success"]),
                "correct_success": bool(branch["correct"]["episode_success"]),
                "clean_score": clean,
                "corrupt_score": corrupt,
                "correct_score": correct,
                "clean_minus_corrupt": clean - corrupt,
                "correct_minus_corrupt": correct - corrupt,
                "clean_matched_minus_shuffled": float(
                    branch["clean"]["matched_minus_shuffled"]
                ),
                "correct_matched_minus_shuffled": float(
                    branch["correct"]["matched_minus_shuffled"]
                ),
            }
        )

    discordant = [
        row
        for row in rows
        if row["clean_success"]
        and not row["corrupt_success"]
        and row["correct_success"]
    ]
    result = {
        "schema_version": "robotwin_frozen_plan_vae_reward_summary_v1",
        "num_validations": len(rows),
        "num_outcome_discordant": len(discordant),
        "rows": rows,
        "checks": {
            "all_clean_above_corrupt": all(
                row["clean_minus_corrupt"] > 0.0 for row in discordant
            ),
            "all_correct_above_corrupt": all(
                row["correct_minus_corrupt"] > 0.0 for row in discordant
            ),
            "all_clean_matched_above_shuffled": all(
                row["clean_matched_minus_shuffled"] > 0.0 for row in discordant
            ),
            "all_correct_matched_above_shuffled": all(
                row["correct_matched_minus_shuffled"] > 0.0 for row in discordant
            ),
        },
        "mean_clean_minus_corrupt": (
            None
            if not discordant
            else float(np.mean([row["clean_minus_corrupt"] for row in discordant]))
        ),
        "mean_correct_minus_corrupt": (
            None
            if not discordant
            else float(np.mean([row["correct_minus_corrupt"] for row in discordant]))
        ),
        "research_gate": (
            "insufficient_discordant_pairs"
            if len(discordant) < 8
            else "ready_for_registered_threshold_evaluation"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
