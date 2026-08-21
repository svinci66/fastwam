#!/usr/bin/env python3
"""Train a closed-loop RoboMimic BC-RNN base policy for the paired Can dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def training_schedule(mode: str) -> dict[str, int]:
    """Return the intentionally small smoke or first-stage training schedule."""
    if mode == "smoke":
        return {
            "epochs": 2,
            "epoch_steps": 3,
            "validation_steps": 3,
            "save_rate": 1,
            "rollout_rate": 1,
            "rollout_episodes": 2,
            "rollout_horizon": 10,
            "rollout_warmstart": 0,
        }
    if mode == "train":
        return {
            "epochs": 500,
            "epoch_steps": 100,
            "validation_steps": 20,
            "save_rate": 25,
            "rollout_rate": 25,
            "rollout_episodes": 10,
            "rollout_horizon": 400,
            "rollout_warmstart": 25,
        }
    raise ValueError(f"Unsupported mode: {mode}")


def build_config(args: argparse.Namespace) -> Any:
    # The official example is imported lazily so pure scheduling tests do not require
    # a complete RoboMimic training environment.
    from examples.train_bc_rnn import get_config

    schedule = training_schedule(args.mode)
    config = get_config(
        dataset_type="robosuite",
        dataset_path=str(args.dataset.resolve()),
        output_dir=str(args.output_dir.resolve()),
        debug=False,
    )
    with config.values_unlocked():
        config.experiment.name = f"can_paired_bc_rnn_{args.mode}_seed{args.seed}"
        config.experiment.logging.terminal_output_to_txt = False
        config.experiment.logging.log_tb = True
        config.experiment.render_video = False
        config.experiment.keep_all_videos = False
        config.experiment.epoch_every_n_steps = schedule["epoch_steps"]
        config.experiment.validation_epoch_every_n_steps = schedule["validation_steps"]
        config.experiment.save.every_n_epochs = schedule["save_rate"]
        # RoboMimic's low-noise GMM evaluation forces sigma to 1e-4, so its
        # validation NLL is not comparable to the training NLL. Select models by
        # periodic checkpoints and online rollout success instead.
        config.experiment.save.on_best_validation = False
        config.experiment.save.on_best_rollout_success_rate = True
        config.experiment.rollout.enabled = True
        config.experiment.rollout.rate = schedule["rollout_rate"]
        config.experiment.rollout.n = schedule["rollout_episodes"]
        config.experiment.rollout.horizon = schedule["rollout_horizon"]
        config.experiment.rollout.warmstart = schedule["rollout_warmstart"]
        config.train.num_epochs = args.epochs or schedule["epochs"]
        config.train.seed = args.seed
        config.train.cuda = not args.cpu
    return config


def train(args: argparse.Namespace) -> None:
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.scripts.train import train as robomimic_train

    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    if args.config_output is not None:
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        config.dump(str(args.config_output))
    if args.dry_run:
        print(config)
        return
    config.lock()
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    print(f"Starting {args.mode} BC-RNN training on {device}", flush=True)
    robomimic_train(config, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
