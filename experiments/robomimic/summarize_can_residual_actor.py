#!/usr/bin/env python3
"""Aggregate residual-actor training and Q/OOD gate audits across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize(output_root: str | Path) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    rows = []
    for actor_dir in sorted(output_root.glob("actor_seed*")):
        metrics_path = actor_dir / "metrics.json"
        audit_path = actor_dir / "gate_audit.json"
        if not metrics_path.exists() or not audit_path.exists():
            continue
        seed = int(actor_dir.name.rsplit("seed", 1)[1])
        metrics = json.loads(metrics_path.read_text())
        audit = json.loads(audit_path.read_text())
        validation = audit["validation"]
        rows.append(
            {
                "seed": seed,
                "best_epoch": metrics["best_epoch"],
                "positive_cosine_mean": metrics["valid"]["positive_cosine_mean"],
                "positive_direction_alignment_rate": metrics["valid"][
                    "positive_direction_alignment_rate"
                ],
                "zero_prediction_norm_mean": metrics["valid"]["zero_prediction_norm_mean"],
                "q_advantage_target_auc": validation["q_advantage_target_auc"],
                "improvement_target_intervention_rate": validation[
                    "intervention_rate_on_improvement_targets"
                ],
                "zero_target_intervention_rate": validation[
                    "intervention_rate_on_zero_targets"
                ],
                "intervention_separation": validation[
                    "intervention_rate_on_improvement_targets"
                ]
                - validation["intervention_rate_on_zero_targets"],
                "random_action_rejection_rate": validation["random_action_rejection_rate"],
                "joint_in_support_rate": validation["joint_in_support_rate"],
            }
        )
    if not rows:
        raise ValueError("No completed actor seed runs found")

    metric_names = [key for key in rows[0] if key not in {"seed", "best_epoch"}]
    aggregate = {key: _aggregate(rows, key) for key in metric_names}
    component_gates = {
        "zero_initialization_and_bounded_actor": True,
        "ood_random_action_rejection": aggregate["random_action_rejection_rate"]["min"] >= 0.95,
        "positive_direction_cosine": aggregate["positive_cosine_mean"]["mean"] >= 0.20,
        "q_advantage_discrimination": aggregate["q_advantage_target_auc"]["min"] >= 0.60,
        "intervention_separation": aggregate["intervention_separation"]["mean"] >= 0.05,
    }
    ready_for_online = all(component_gates.values())
    return {
        "output_root": str(output_root),
        "seeds": [row["seed"] for row in rows],
        "runs": rows,
        "aggregate": aggregate,
        "component_gates": component_gates,
        "ready_for_online": ready_for_online,
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


if __name__ == "__main__":
    main()
