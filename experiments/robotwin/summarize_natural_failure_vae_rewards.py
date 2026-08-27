#!/usr/bin/env python3
"""Combine Wan VAE natural-failure reward results across collection runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.robotwin.validate_frozen_plan_vae_reward import CAMERA_NAMES


def summarize(inputs: list[Path], *, task: str | None = None) -> dict[str, Any]:
    pairs = []
    sources = []
    seen = set()
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "robotwin_natural_failure_wan_vae_pair_reward_v1":
            raise ValueError(f"Unsupported reward schema in {path}")
        sources.append(str(path.resolve()))
        for pair in payload["pairs"]:
            if task is not None and str(pair["task"]) != task:
                continue
            identity = (str(pair["task"]), int(pair["environment_seed"]))
            if identity in seen:
                raise ValueError(f"Duplicate task/seed pair: {identity}")
            seen.add(identity)
            row = dict(pair)
            row["source_result"] = str(path.resolve())
            pairs.append(row)
    if not pairs:
        raise ValueError("No reward pairs selected")

    margins = [float(pair["success_minus_failure"]) for pair in pairs]
    per_camera = {}
    for camera in CAMERA_NAMES:
        camera_margins = [
            float(
                pair["expert_success"]["camera_scores"][camera]
                - pair["fastwam_failure"]["camera_scores"][camera]
            )
            for pair in pairs
        ]
        per_camera[camera] = {
            "correctly_ranked_count": sum(margin > 0.0 for margin in camera_margins),
            "pairwise_accuracy": float(np.mean([margin > 0.0 for margin in camera_margins])),
            "mean_success_minus_failure": float(np.mean(camera_margins)),
            "margins": camera_margins,
        }
    return {
        "schema_version": "robotwin_natural_failure_wan_vae_reward_summary_v1",
        "task_filter": task,
        "source_results": sources,
        "pair_count": len(pairs),
        "correctly_ranked_count": sum(margin > 0.0 for margin in margins),
        "pairwise_accuracy": float(np.mean([margin > 0.0 for margin in margins])),
        "mean_success_minus_failure": float(np.mean(margins)),
        "median_success_minus_failure": float(np.median(margins)),
        "min_success_minus_failure": float(np.min(margins)),
        "max_success_minus_failure": float(np.max(margins)),
        "per_camera_pairwise": per_camera,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--task", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.inputs, task=(args.task.strip() or None))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
