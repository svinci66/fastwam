#!/usr/bin/env python3
"""Aggregate held-out simulator branch evaluations across actor seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def summarize(output_root: str | Path) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    runs = []
    for path in sorted(output_root.glob("actor_seed*/actual_valid100.json")):
        payload = json.loads(path.read_text())
        seed = int(path.parent.name.rsplit("seed", 1)[1])
        runs.append(
            {
                "seed": seed,
                "states": payload["states"],
                "delta_score_mean": payload["delta_score_mean"],
                "delta_score_median": payload["delta_score_median"],
                "improvement_rate": payload["improvement_rate"],
                "worsening_rate": payload["worsening_rate"],
                "success_gains": payload["success_gains"],
                "success_losses": payload["success_losses"],
                "max_restore_linf": payload["max_restore_linf"],
                "max_branch_initial_state_linf": payload[
                    "max_branch_initial_state_linf"
                ],
            }
        )
    if not runs:
        raise ValueError("No completed actual_valid100 evaluations found")

    aggregate = {}
    for key in ("delta_score_mean", "delta_score_median", "improvement_rate", "worsening_rate"):
        values = np.asarray([run[key] for run in runs], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    gate = {
        "positive_mean_delta_every_seed": aggregate["delta_score_mean"]["min"] > 0.0,
        "improvements_outnumber_regressions_every_seed": all(
            run["improvement_rate"] > run["worsening_rate"] for run in runs
        ),
        "no_success_losses": all(run["success_losses"] == 0 for run in runs),
        "restore_is_deterministic": all(
            run["max_restore_linf"] <= 1e-10
            and run["max_branch_initial_state_linf"] <= 1e-10
            for run in runs
        ),
    }
    return {
        "output_root": str(output_root),
        "seeds": [run["seed"] for run in runs],
        "runs": runs,
        "aggregate": aggregate,
        "gate": gate,
        "passed": all(gate.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.output_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
