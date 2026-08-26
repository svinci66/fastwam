"""Prepare exact RoboTwin expert-source seeds for a natural FastWAM failure scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    return parser.parse_args()


def resolve_bundle_root(dataset_root: Path, task: str) -> Path:
    task_root = dataset_root / task
    matches = sorted(
        path.parent
        for path in task_root.rglob("seed.txt")
        if (path.parent / "data").is_dir()
        and (path.parent / "instructions").is_dir()
    )
    if len(matches) != 1:
        raise ValueError(
            f"{task}: expected one bundle containing seed.txt, data/, and "
            f"instructions/ below {task_root}, found {matches}"
        )
    return matches[0]


def select_seen_instruction(path: Path, environment_seed: int) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = sorted(str(value).strip() for value in payload.get("seen", []))
    candidates = [value for value in candidates if value]
    if not candidates:
        raise ValueError(f"no non-empty seen instructions in {path}")
    rng = np.random.default_rng(int(environment_seed))
    return str(rng.choice(candidates))


def build_cases(
    dataset_root: Path, tasks: list[str], episodes_per_task: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if episodes_per_task <= 0:
        raise ValueError("episodes-per-task must be positive")
    manifest: dict[str, object] = {
        "_meta": {
            "schema_version": "robotwin_natural_failure_manifest_v1",
            "instruction_split": "seen",
            "source": str(dataset_root.resolve()),
            "pairing_rule": "same_task_environment_seed_and_manifest_instruction",
        }
    }
    cases: list[dict[str, object]] = []
    for task in tasks:
        bundle = resolve_bundle_root(dataset_root, task)
        seeds = [int(value) for value in (bundle / "seed.txt").read_text().split()]
        if len(seeds) < episodes_per_task:
            raise ValueError(
                f"{task}: requested {episodes_per_task} episodes, found {len(seeds)} seeds"
            )
        task_seeds: list[int] = []
        task_instructions: list[str] = []
        task_episode_indices: list[int] = []
        for episode_index, environment_seed in enumerate(seeds[:episodes_per_task]):
            hdf5_path = bundle / "data" / f"episode{episode_index}.hdf5"
            instruction_path = bundle / "instructions" / f"episode{episode_index}.json"
            if not hdf5_path.is_file() or not instruction_path.is_file():
                raise FileNotFoundError(
                    f"{task} episode {episode_index}: missing {hdf5_path} or "
                    f"{instruction_path}"
                )
            instruction = select_seen_instruction(instruction_path, environment_seed)
            task_seeds.append(environment_seed)
            task_instructions.append(instruction)
            task_episode_indices.append(episode_index)
            cases.append(
                {
                    "task": task,
                    "episode_index": episode_index,
                    "environment_seed": environment_seed,
                    "instruction": instruction,
                    "expert_hdf5": str(hdf5_path.resolve()),
                    "instruction_source": str(instruction_path.resolve()),
                }
            )
        manifest[task] = {
            "seeds": task_seeds,
            "instructions": task_instructions,
            "source_episode_indices": task_episode_indices,
        }
    return manifest, cases


def main() -> None:
    args = parse_args()
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("tasks must be a non-empty comma-separated list without duplicates")
    manifest, cases = build_cases(args.dataset_root, tasks, args.episodes_per_task)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.cases_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.cases_jsonl.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "cases_jsonl": str(args.cases_jsonl.resolve()),
                "tasks": tasks,
                "case_count": len(cases),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
