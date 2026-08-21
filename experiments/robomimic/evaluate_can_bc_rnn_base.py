#!/usr/bin/env python3
"""Evaluate a trained RoboMimic BC-RNN base policy with reproducible rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def summarize_rollouts(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one rollout is required")
    successes = np.asarray([row["Success_Rate"] for row in rows], dtype=np.float64)
    returns = np.asarray([row["Return"] for row in rows], dtype=np.float64)
    horizons = np.asarray([row["Horizon"] for row in rows], dtype=np.float64)
    return {
        "episodes": len(rows),
        "successes": int(np.count_nonzero(successes)),
        "success_rate": float(np.mean(successes)),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "horizon_mean": float(np.mean(horizons)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.scripts.run_trained_agent import rollout

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=not args.cpu)
    policy, checkpoint_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint), device=device, verbose=True
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint_dict,
        render=False,
        render_offscreen=False,
        verbose=True,
    )
    rows = []
    try:
        for episode in range(args.episodes):
            stats, _ = rollout(policy=policy, env=env, horizon=args.horizon)
            row = {key: float(value) for key, value in stats.items()}
            row["episode"] = episode
            rows.append(row)
            partial = {
                "complete": False,
                "episodes_completed": len(rows),
                "episodes_requested": args.episodes,
                "rows": rows,
            }
            write_json_atomic(args.output_json, partial)
            print(
                json.dumps(
                    {
                        "episode": f"{episode + 1}/{args.episodes}",
                        "success": int(row["Success_Rate"]),
                        "horizon": int(row["Horizon"]),
                    }
                ),
                flush=True,
            )
    finally:
        close = getattr(getattr(env, "env", env), "close", None)
        if callable(close):
            close()
    report = summarize_rollouts(rows)
    report.update(
        {
            "complete": True,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "seed": args.seed,
            "horizon": args.horizon,
            "device": str(device),
            "rows": rows,
        }
    )
    write_json_atomic(args.output_json, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0 or args.horizon <= 0:
        parser.error("episodes and horizon must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
