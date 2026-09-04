#!/usr/bin/env python3
"""Merge compatible Wan-VAE pair-reward files without changing pair labels.

The merger is intentionally mechanical: it validates that every input uses the
same frozen trajectory-reward protocol, optionally restricts the union to an
explicit task allowlist, rejects duplicate task/seed pairs, and recomputes all
aggregate ranking diagnostics from the unchanged pair rows.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.robotwin.validate_frozen_plan_vae_reward import CAMERA_NAMES


EXPECTED_SCHEMA = "robotwin_natural_failure_wan_vae_pair_reward_v1"
PROTOCOL_FIELDS = (
    "schema_version",
    "feature_encoder",
    "trajectory_reference_policy",
    "time_offsets",
    "reward_cameras",
    "camera_weights",
    "latent_shape",
)


def _protocol(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in PROTOCOL_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"reward payload lacks protocol fields: {missing}")
    if payload["schema_version"] != EXPECTED_SCHEMA:
        raise ValueError(
            f"unsupported reward schema: {payload['schema_version']!r}"
        )
    return {field: copy.deepcopy(payload[field]) for field in PROTOCOL_FIELDS}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = np.asarray(
        [float(pair["success_minus_failure"]) for pair in rows], dtype=np.float64
    )
    if not rows:
        raise ValueError("no reward pairs selected")
    per_task: dict[str, dict[str, Any]] = {}
    for task in sorted({str(pair["task"]) for pair in rows}):
        task_margins = np.asarray(
            [
                float(pair["success_minus_failure"])
                for pair in rows
                if str(pair["task"]) == task
            ],
            dtype=np.float64,
        )
        per_task[task] = {
            "pair_count": int(task_margins.size),
            "correctly_ranked_count": int(np.sum(task_margins > 0.0)),
            "pairwise_accuracy": float(np.mean(task_margins > 0.0)),
            "mean_success_minus_failure": float(np.mean(task_margins)),
        }
    per_camera: dict[str, dict[str, Any]] = {}
    for camera in CAMERA_NAMES:
        camera_margins = np.asarray(
            [
                float(pair["expert_success"]["camera_scores"][camera])
                - float(pair["fastwam_failure"]["camera_scores"][camera])
                for pair in rows
            ],
            dtype=np.float64,
        )
        per_camera[camera] = {
            "correctly_ranked_count": int(np.sum(camera_margins > 0.0)),
            "pairwise_accuracy": float(np.mean(camera_margins > 0.0)),
            "mean_success_minus_failure": float(np.mean(camera_margins)),
        }
    return {
        "pair_count": len(rows),
        "correctly_ranked_count": int(np.sum(margins > 0.0)),
        "pairwise_accuracy": float(np.mean(margins > 0.0)),
        "mean_success_minus_failure": float(np.mean(margins)),
        "per_task": per_task,
        "per_camera_pairwise": per_camera,
    }


def merge_reward_payloads(
    inputs: list[tuple[Path, dict[str, Any]]], *, tasks: list[str]
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one reward payload is required")
    requested = [task.strip() for task in tasks if task.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError(f"duplicate task in allowlist: {requested}")
    allowlist = set(requested)

    reference = _protocol(inputs[0][1])
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    sources: list[dict[str, Any]] = []
    for path, payload in inputs:
        actual = _protocol(payload)
        if actual != reference:
            differences = {
                field: {"expected": reference[field], "actual": actual[field]}
                for field in PROTOCOL_FIELDS
                if actual[field] != reference[field]
            }
            raise ValueError(f"incompatible reward protocol in {path}: {differences}")
        selected_from_source = 0
        for source_pair in payload.get("pairs", []):
            task = str(source_pair["task"])
            if allowlist and task not in allowlist:
                continue
            pair = copy.deepcopy(source_pair)
            identity = (task, int(pair["environment_seed"]))
            if identity in seen:
                raise ValueError(f"duplicate task/environment-seed pair: {identity}")
            margin = float(pair["success_minus_failure"])
            expected_rank = bool(margin > 0.0)
            if bool(pair["correctly_ranked"]) != expected_rank:
                raise ValueError(f"inconsistent ranking label for {identity}")
            pair["source_reward_json"] = str(path.resolve())
            rows.append(pair)
            seen.add(identity)
            selected_from_source += 1
        sources.append(
            {
                "reward_json": str(path.resolve()),
                "source_pair_count": int(payload.get("pair_count", len(payload.get("pairs", [])))),
                "selected_pair_count": selected_from_source,
                "source_unique_encoded_frames": payload.get("unique_encoded_frames"),
            }
        )

    if allowlist:
        available = {str(pair["task"]) for pair in rows}
        missing = sorted(allowlist - available)
        if missing:
            raise ValueError(f"requested tasks are absent from merged inputs: {missing}")
    rows.sort(key=lambda pair: (str(pair["task"]), int(pair["environment_seed"])))
    # ``episode_id`` is local to a collection run and therefore may collide
    # after merging (for example, both old and new runs can contain episode 2).
    # Canonicalize it per task while preserving the source-local identifier.
    next_episode_id: dict[str, int] = {}
    for pair in rows:
        task = str(pair["task"])
        pair["source_episode_id"] = int(pair["episode_id"])
        pair["episode_id"] = next_episode_id.get(task, 0)
        next_episode_id[task] = int(pair["episode_id"]) + 1
    result = copy.deepcopy(reference)
    result.update(_summary(rows))
    result["pairs"] = rows
    result["selected_tasks"] = sorted({str(pair["task"]) for pair in rows})
    result["merge_provenance"] = {
        "selection_rule": "explicit_task_allowlist_then_unique_task_environment_seed",
        "episode_id_policy": "canonical_contiguous_per_task_preserve_source_episode_id",
        "pair_rewards_recomputed": False,
        "aggregate_statistics_recomputed": True,
        "sources": sources,
    }
    # The source encoders used independent caches, and a filtered source may
    # include cached frames from excluded tasks.  Preserve the source counts in
    # provenance instead of reporting a misleading merged cache cardinality.
    result["unique_encoded_frames"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in args.inputs
    ]
    result = merge_reward_payloads(
        inputs, tasks=[task.strip() for task in args.tasks.split(",") if task.strip()]
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in (
        "pair_count", "correctly_ranked_count", "pairwise_accuracy",
        "mean_success_minus_failure", "per_task", "selected_tasks",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
