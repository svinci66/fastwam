#!/usr/bin/env python3
"""Fit a diagnostic gate on exact RoboTwin residual intervention pairs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.rl.intervention_gate import (
    audit_intervention_gate_coverage,
    discover_pair_jsonl,
    load_intervention_gate_examples,
    predict_intervention_gate,
    summarize_intervention_fit,
    train_intervention_gate_ensemble,
)
from fastwam.rl.models import ActionValueCritic, ActionValueCriticConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-root",
        type=Path,
        action="append",
        required=True,
        help="Directory searched recursively for accepted_pairs.jsonl; repeatable.",
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--non-improving-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pair_paths = discover_pair_jsonl(args.pair_root)
    examples = load_intervention_gate_examples(
        pair_paths, non_improving_weight=args.non_improving_weight
    )
    checkpoint = torch.load(
        args.base_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("format") != "fastwam_residual_iql_v1":
        raise ValueError("base checkpoint must be fastwam_residual_iql_v1")
    q_config = dict(checkpoint["q_critic_config"])
    q_config["hidden_dims"] = tuple(q_config["hidden_dims"])
    model_config = ActionValueCriticConfig(**q_config)
    expected_context = model_config.context_dim
    expected_actions = (model_config.action_horizon, model_config.action_dim)
    if examples.context.shape[1] != expected_context:
        raise ValueError(
            f"pair context dimension {examples.context.shape[1]} != {expected_context}"
        )
    if tuple(examples.baseline_actions.shape[1:]) != expected_actions:
        raise ValueError(
            "pair action shape "
            f"{examples.baseline_actions.shape[1:]} != {expected_actions}"
        )
    language_dim = (
        0 if examples.language_feature is None else examples.language_feature.shape[1]
    )
    if language_dim != model_config.language_feature_dim:
        raise ValueError(
            f"pair language dimension {language_dim} != {model_config.language_feature_dim}"
        )

    models = (ActionValueCritic(model_config), ActionValueCritic(model_config))
    histories = train_intervention_gate_ensemble(
        models,
        examples,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    probabilities = predict_intervention_gate(
        models, examples, device=args.device
    )
    coverage = audit_intervention_gate_coverage(examples)
    diagnostic_fit = summarize_intervention_fit(examples, probabilities)
    summary = {
        "schema_version": "robotwin_intervention_gate_diagnostic_v1",
        "data_coverage": coverage,
        "diagnostic_fit": diagnostic_fit,
        "pair_files": [str(path) for path in pair_paths],
        "warning": (
            "This checkpoint is a pipeline diagnostic, not a deployable gate. "
            "Threshold and fit metrics reuse the training pairs."
        ),
    }

    checkpoint["paired_advantage_gates"] = [model.state_dict() for model in models]
    checkpoint["paired_advantage_gate_config"] = asdict(model_config)
    checkpoint["paired_advantage_training_config"] = {
        "type": "exact_single_intervention_pairs_v1",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "non_improving_weight": args.non_improving_weight,
        "seed": args.seed,
    }
    checkpoint["paired_advantage_summary"] = summary
    checkpoint["paired_advantage_deployment_ready"] = False
    checkpoint["summary"] = dict(checkpoint["summary"])
    checkpoint["summary"]["intervention_gate_examples"] = len(examples)
    checkpoint["summary"]["intervention_gate_positive_examples"] = int(
        examples.labels.sum()
    )

    args.output_dir.mkdir(parents=True)
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")
    (args.output_dir / "history.json").write_text(
        json.dumps(histories, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
