#!/usr/bin/env python3
"""Train and attach a paired-outcome residual advantage gate to an IQL actor."""

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

from fastwam.rl.models import ActionValueCritic, ActionValueCriticConfig
from fastwam.rl.paired_advantage import (
    PairedAdvantageTrainingConfig,
    build_paired_advantage_examples,
    export_paired_advantage_config,
    predict_paired_advantage,
    summarize_paired_predictions,
    train_paired_advantage_ensemble,
)
from fastwam.rl.replay_buffer import ReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    replay = ReplayBuffer.load(args.replay_dir)
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "fastwam_residual_iql_v1":
        raise ValueError("base checkpoint must be fastwam_residual_iql_v1")
    config = PairedAdvantageTrainingConfig(epochs=args.epochs, seed=args.seed)
    train_examples = build_paired_advantage_examples(replay, config, split="train")
    validation_examples = build_paired_advantage_examples(
        replay, config, split="validation"
    )
    q_values = dict(checkpoint["q_critic_config"])
    q_values["hidden_dims"] = tuple(q_values["hidden_dims"])
    model_config = ActionValueCriticConfig(**q_values)
    models = (ActionValueCritic(model_config), ActionValueCritic(model_config))
    histories = train_paired_advantage_ensemble(
        models, replay, train_examples, config, device=args.device
    )
    train_probabilities = predict_paired_advantage(
        models, replay, train_examples, device=args.device
    )
    validation_probabilities = predict_paired_advantage(
        models, replay, validation_examples, device=args.device
    )
    summary = {
        "train": summarize_paired_predictions(train_examples, train_probabilities),
        "validation": summarize_paired_predictions(
            validation_examples, validation_probabilities
        ),
    }
    checkpoint["paired_advantage_gates"] = [
        model.state_dict() for model in models
    ]
    checkpoint["paired_advantage_gate_config"] = asdict(model_config)
    checkpoint["paired_advantage_training_config"] = export_paired_advantage_config(config)
    checkpoint["paired_advantage_summary"] = summary
    checkpoint["summary"] = dict(checkpoint["summary"])
    checkpoint["summary"]["paired_advantage_gate_parameters_each"] = sum(
        parameter.numel() for parameter in models[0].parameters()
    )
    checkpoint["summary"]["paired_advantage_train_transitions"] = len(train_examples)
    checkpoint["summary"]["paired_advantage_validation_transitions"] = len(
        validation_examples
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
