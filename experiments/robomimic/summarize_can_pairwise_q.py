#!/usr/bin/env python3
"""Aggregate matched full-state and action-only pairwise-Q runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def summarize(output_root: str | Path) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    full_runs = {
        int(path.parent.name.rsplit("seed", 1)[1]): json.loads(path.read_text())
        for path in output_root.glob("full_state_seed*/metrics.json")
    }
    action_runs = {
        int(path.parent.name.rsplit("seed", 1)[1]): json.loads(path.read_text())
        for path in output_root.glob("action_only_seed*/metrics.json")
    }
    seeds = sorted(set(full_runs) & set(action_runs))
    if not seeds:
        raise ValueError("No matched full-state and action-only seed runs found")

    rows = []
    for seed in seeds:
        full = full_runs[seed]
        action = action_runs[seed]
        rows.append(
            {
                "seed": seed,
                "full_balanced_accuracy": full["valid"]["balanced_accuracy"],
                "full_auc": full["valid"]["auc"],
                "full_accuracy": full["valid"]["accuracy"],
                "action_only_balanced_accuracy": action["valid"]["balanced_accuracy"],
                "action_only_auc": action["valid"]["auc"],
                "balanced_accuracy_gain": (
                    full["valid"]["balanced_accuracy"]
                    - action["valid"]["balanced_accuracy"]
                ),
                "auc_gain": full["valid"]["auc"] - action["valid"]["auc"],
            }
        )

    aggregate: dict[str, dict[str, float]] = {}
    for key in (
        "full_balanced_accuracy",
        "full_auc",
        "full_accuracy",
        "action_only_balanced_accuracy",
        "action_only_auc",
        "balanced_accuracy_gain",
        "auc_gain",
    ):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    passed = (
        aggregate["full_balanced_accuracy"]["min"] >= 0.60
        and aggregate["full_auc"]["min"] >= 0.60
        and aggregate["balanced_accuracy_gain"]["mean"] >= 0.02
        and aggregate["balanced_accuracy_gain"]["min"] > 0.0
    )
    return {
        "output_root": str(output_root),
        "seeds": seeds,
        "runs": rows,
        "aggregate": aggregate,
        "gate": {
            "requirements": {
                "minimum_full_balanced_accuracy_each_seed": 0.60,
                "minimum_full_auc_each_seed": 0.60,
                "minimum_mean_balanced_accuracy_gain_over_action_only": 0.02,
                "full_must_beat_action_only_on_every_seed": True,
            },
            "passed": passed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    report = summarize(args.output_root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if not report["gate"]["passed"]:
        raise SystemExit("Pairwise-Q multi-seed gate failed")


if __name__ == "__main__":
    main()
