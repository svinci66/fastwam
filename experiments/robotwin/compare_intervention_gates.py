#!/usr/bin/env python3
"""Compare Q, OOD, and direct paired gates on causal intervention probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.rl.intervention_gate import (
    intervention_decision_metrics,
    load_intervention_gate_examples,
    predict_intervention_gate,
)
from fastwam.rl.online_policy import load_paired_advantage_gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-jsonl",
        type=Path,
        action="append",
        required=True,
        help="Scored intervention JSONL; repeatable.",
    )
    parser.add_argument("--paired-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--q-margin", type=float, default=0.003)
    parser.add_argument("--q-max-disagreement", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_intervention_gate_examples(args.pair_jsonl)
    payload = torch.load(
        args.paired_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    models = load_paired_advantage_gates(payload, device=args.device)
    probabilities = predict_intervention_gate(
        models, examples, device=args.device
    )
    paired_summary = payload.get("paired_advantage_summary", {})
    fit_summary = (
        paired_summary.get("diagnostic_fit", {})
        if isinstance(paired_summary, dict)
        else {}
    )
    threshold = fit_summary.get("recommended_threshold_for_smoke_only")
    if threshold is None:
        raise ValueError("checkpoint lacks its diagnostic smoke threshold")
    threshold = float(threshold)

    q_approved: list[bool] = []
    support_approved: list[bool] = []
    paired_approved: list[bool] = []
    rows: list[dict] = []
    for row, probability in zip(examples.rows, probabilities):
        q_min = row.get("shadow_q_advantage_min")
        q_disagreement = row.get("shadow_q_advantage_disagreement")
        support = row.get("shadow_support_in_distribution")
        if q_min is None or q_disagreement is None or support is None:
            raise ValueError(
                "comparison rows require shadow Q and OOD metadata; use scored_pairs.jsonl"
            )
        q_decision = (
            float(q_min) >= args.q_margin
            and float(q_disagreement) <= args.q_max_disagreement
        )
        support_decision = bool(support)
        paired_decision = float(np.min(probability)) >= threshold
        q_approved.append(q_decision)
        support_approved.append(support_decision)
        paired_approved.append(paired_decision)
        rows.append(
            {
                "pair_id": examples.pair_ids[len(rows)],
                "outcome_label": examples.outcome_labels[len(rows)],
                "q_approved": q_decision,
                "support_approved": support_decision,
                "q_ood_approved": q_decision and support_decision,
                "paired_probability_min": float(np.min(probability)),
                "paired_approved": paired_decision,
                "paired_ood_approved": paired_decision and support_decision,
            }
        )

    outcomes = examples.outcome_labels
    q_array = np.asarray(q_approved, dtype=bool)
    support_array = np.asarray(support_approved, dtype=bool)
    paired_array = np.asarray(paired_approved, dtype=bool)
    result = {
        "schema_version": "robotwin_intervention_gate_comparison_v1",
        "evaluation_scope": "diagnostic_training_overlap_not_held_out",
        "paired_checkpoint_deployment_ready": bool(
            payload.get("paired_advantage_deployment_ready", True)
        ),
        "q_margin": args.q_margin,
        "q_max_disagreement": args.q_max_disagreement,
        "paired_smoke_threshold": threshold,
        "metrics": {
            "q_only": intervention_decision_metrics(outcomes, q_array),
            "ood_only": intervention_decision_metrics(outcomes, support_array),
            "q_plus_ood": intervention_decision_metrics(
                outcomes, q_array & support_array
            ),
            "paired_only": intervention_decision_metrics(outcomes, paired_array),
            "paired_plus_ood": intervention_decision_metrics(
                outcomes, paired_array & support_array
            ),
        },
        "rows": rows,
        "warning": (
            "The direct paired gate was fitted on these pairs. Its metrics only "
            "verify pipeline capacity and must not be reported as held-out performance."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
