"""Merge compatible residual replay shards with canonical task identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.rl.replay_buffer import ReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-replay", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_replays(input_replays: list[Path]) -> tuple[ReplayBuffer, dict]:
    if not input_replays:
        raise ValueError("At least one input replay is required")
    roots = [path.expanduser().resolve() for path in input_replays]
    manifests = [json.loads((root / "manifest.json").read_text()) for root in roots]
    compatibility_fields = (
        "schema_version",
        "target_k",
        "reward_encoder_version",
        "language_encoder_version",
        "imagination_reward_type",
    )
    for field in compatibility_fields:
        values = {json.dumps(manifest.get(field), sort_keys=True) for manifest in manifests}
        if len(values) != 1:
            raise ValueError(f"Replay manifests disagree on {field}: {sorted(values)}")
    camera_normalizations = {
        json.dumps(
            manifest.get("provenance", {}).get("camera_normalization"),
            sort_keys=True,
        )
        for manifest in manifests
    }
    if len(camera_normalizations) != 1:
        raise ValueError("Replay shards use different camera normalizations")

    shards = [ReplayBuffer.load(root) for root in roots]
    task_keys = sorted(
        {
            (transition.task_suite, transition.task_description)
            for shard in shards
            for transition in shard.transitions
        }
    )
    task_ids = {key: index for index, key in enumerate(task_keys)}
    merged = ReplayBuffer()
    task_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    for shard_index, (root, shard) in enumerate(zip(roots, shards)):
        namespace = f"shard{shard_index:03d}-{root.name}"
        for transition in shard.transitions:
            task_key = (transition.task_suite, transition.task_description)
            merged.append(
                replace(
                    transition,
                    episode_id=f"{namespace}/{transition.episode_id}",
                    task_id=task_ids[task_key],
                )
            )
            task_counts[transition.task_description] += 1
            behavior_counts[transition.behavior_mode] += 1
    provenance = {
        "merge_format": "robotwin_residual_replay_merge_v1",
        "camera_normalization": manifests[0]["provenance"]["camera_normalization"],
        "input_replays": [
            {
                "path": str(root),
                "manifest_sha256": _sha256(root / "manifest.json"),
                "num_transitions": len(shard),
            }
            for root, shard in zip(roots, shards)
        ],
        "canonical_tasks": [
            {"task_id": task_ids[key], "task_suite": key[0], "task_description": key[1]}
            for key in task_keys
        ],
        "task_transition_counts": dict(sorted(task_counts.items())),
        "behavior_transition_counts": dict(sorted(behavior_counts.items())),
    }
    return merged, provenance


def main() -> None:
    args = parse_args()
    merged, provenance = merge_replays(args.input_replay)
    output = merged.save(args.output_dir, provenance=provenance)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "num_transitions": len(merged),
                **provenance,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
