"""Build a strict seed manifest from pre-registered expert-feasibility records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--instructions-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    pool: dict,
    metadata_paths: list[Path],
    *,
    instructions_dir: Path,
    task: str,
    count: int,
    pool_path: Path,
) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    candidates = [int(value) for value in pool[task]["seeds"]]
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate pool contains duplicate seeds")
    if len(metadata_paths) != count:
        raise ValueError(f"expected {count} expert metadata files, found {len(metadata_paths)}")
    selected: list[int] = []
    selected_instructions: list[str] = []
    records = []
    for expected_episode, path in enumerate(metadata_paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        checks = {
            "schema": record.get("schema_version") == "robotwin_local_expert_pair_episode_v1",
            "task": record.get("task") == task,
            "task_config": record.get("task_config") == "demo_clean",
            "episode_index": int(record.get("episode_index", -1)) == expected_episode,
            "planning_success": bool(record.get("expert_planning_success")),
            "replay_success": bool(record.get("expert_replay_success")),
            "scene_exact": bool(record.get("scene_state_comparison", {}).get("exact")),
        }
        if not all(checks.values()):
            raise ValueError(f"expert feasibility audit failed for {path}: {checks}")
        seed = int(record["seed"])
        if seed not in candidates or seed in selected:
            raise ValueError(f"invalid or duplicate selected seed {seed}")
        instruction_path = instructions_dir / f"episode{expected_episode}.json"
        if not instruction_path.is_file():
            raise ValueError(f"missing official instruction record {instruction_path}")
        instruction_payload = json.loads(instruction_path.read_text(encoding="utf-8"))
        unseen = sorted(str(value) for value in instruction_payload.get("unseen", []))
        if not unseen:
            raise ValueError(f"no official unseen instructions in {instruction_path}")
        instruction = str(np.random.default_rng(seed).choice(unseen))
        selected.append(seed)
        selected_instructions.append(instruction)
        records.append(
            {
                "episode_index": expected_episode,
                "seed": seed,
                "metadata_sha256": sha256(path),
                "instruction_sha256": sha256(instruction_path),
            }
        )
    candidate_positions = [candidates.index(seed) for seed in selected]
    if candidate_positions != sorted(candidate_positions):
        raise ValueError("selected seeds do not preserve pre-registered candidate order")
    return {
        "_meta": {
            "schema_version": "robotwin_expert_feasible_heldout_manifest_v1",
            "description": (
                f"Fresh {task} held-out seeds selected only by exact expert "
                "feasibility."
            ),
            "task_config": "demo_clean",
            "selection_rule": "first expert-feasible candidates in pre-registered order",
            "expert_validation_mode": "prevalidated_exact_planning_and_replay",
            "instruction_selection": "sorted official unseen list with numpy default_rng(environment_seed)",
            "candidate_pool": str(pool_path.resolve()),
            "candidate_pool_sha256": sha256(pool_path),
            "expert_feasibility_records": records,
        },
        task: {"seeds": selected, "instructions": selected_instructions},
    }


def main() -> None:
    args = parse_args()
    paths = sorted(
        args.metadata_dir.glob("episode*.json"),
        key=lambda path: int(path.stem.removeprefix("episode")),
    )
    pool = json.loads(args.candidate_pool.read_text(encoding="utf-8"))
    manifest = build_manifest(
        pool,
        paths,
        instructions_dir=args.instructions_dir,
        task=args.task,
        count=args.count,
        pool_path=args.candidate_pool,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
