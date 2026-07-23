"""Validate raw multi-task LIBERO transitions before replay construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--trial-indices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--expected-noise-stds",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Expected noise behavior scales. When omitted, preserve the legacy "
            "single 'noise' behavior audit."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _behavior_label(metadata: dict[str, Any], *, distinguish_noise: bool) -> str:
    mode = str(metadata["action_mode"])
    if mode != "noise" or not distinguish_noise:
        return mode
    return f"noise_{float(metadata.get('action_noise_std', 0.0)):.6f}"


def _load_records(collection_root: Path) -> list[tuple[Path, dict[str, Any], dict[str, np.ndarray]]]:
    records = []
    for metadata_path in sorted(collection_root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays_file = metadata.get("rollout_arrays_file")
        if not arrays_file:
            continue
        arrays_path = metadata_path.parent / str(arrays_file)
        if not arrays_path.is_file():
            raise FileNotFoundError(arrays_path)
        with np.load(arrays_path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        records.append((metadata_path, metadata, arrays))
    if not records:
        raise ValueError(f"no raw transition records found under {collection_root}")
    return records


def main() -> None:
    args = parse_args()
    expected_tasks = set(args.task_ids)
    expected_trials = set(args.trial_indices)
    expected_noise_stds = (
        None
        if args.expected_noise_stds is None
        else tuple(float(value) for value in args.expected_noise_stds)
    )
    if expected_noise_stds is not None and (
        any(value <= 0.0 for value in expected_noise_stds)
        or len(set(expected_noise_stds)) != len(expected_noise_stds)
    ):
        raise ValueError("expected noise stds must be distinct and positive")
    required_modes = (
        ("policy", "noise")
        if expected_noise_stds is None
        else (
            "policy",
            *(f"noise_{value:.6f}" for value in expected_noise_stds),
        )
    )
    records = _load_records(args.collection_root.resolve())

    per_task_mode_trials: dict[tuple[int, str], set[int]] = defaultdict(set)
    per_task_mode_count: Counter[tuple[int, str]] = Counter()
    episode_outcomes: dict[tuple[int, str, int], bool] = {}
    language_hashes: dict[int, set[str]] = defaultdict(set)
    language_versions: set[str] = set()
    residual_squares: list[np.ndarray] = []
    structural_errors: list[str] = []
    episode_seeds: dict[tuple[int, str, int], tuple[int, int]] = {}

    for metadata_path, metadata, arrays in records:
        task_id = int(metadata["task_id"])
        raw_mode = str(metadata["action_mode"])
        mode = _behavior_label(
            metadata, distinguish_noise=expected_noise_stds is not None
        )
        trial_idx = int(metadata["trial_idx"])
        if task_id not in expected_tasks:
            structural_errors.append(f"unexpected task {task_id} in {metadata_path}")
        if raw_mode not in {"policy", "noise"} or mode not in required_modes:
            structural_errors.append(f"unexpected behavior {mode!r} in {metadata_path}")
        per_task_mode_trials[task_id, mode].add(trial_idx)
        per_task_mode_count[task_id, mode] += 1
        episode_key = (task_id, mode, trial_idx)
        success = bool(metadata.get("episode_success", False))
        if episode_key in episode_outcomes and episode_outcomes[episode_key] != success:
            structural_errors.append(f"inconsistent episode success in {metadata_path}")
        episode_outcomes[episode_key] = success
        seed_pair = (
            int(metadata.get("policy_seed", -1)),
            int(metadata.get("action_seed", -1)),
        )
        if min(seed_pair) < 0:
            structural_errors.append(f"missing episode seeds in {metadata_path}")
        elif episode_key in episode_seeds and episode_seeds[episode_key] != seed_pair:
            structural_errors.append(f"inconsistent episode seeds in {metadata_path}")
        else:
            episode_seeds[episode_key] = seed_pair

        language = arrays.get("language_feature")
        if language is None or language.shape != (4096,) or not np.all(np.isfinite(language)):
            structural_errors.append(f"invalid UMT5 feature in {metadata_path}")
        else:
            language_hashes[task_id].add(_sha256(language))
        version = str(metadata.get("language_encoder_version", "")).strip()
        if not version:
            structural_errors.append(f"missing language encoder version in {metadata_path}")
        else:
            language_versions.add(version)

        baseline = arrays.get("baseline_actions")
        executed = arrays.get("executed_actions")
        effective_k = int(metadata.get("effective_k", 0))
        if (
            baseline is None
            or executed is None
            or baseline.shape != executed.shape
            or effective_k <= 0
            or effective_k > baseline.shape[0]
        ):
            structural_errors.append(f"invalid action arrays in {metadata_path}")
        else:
            residual_squares.append(np.square(executed[:effective_k] - baseline[:effective_k]))

    coverage = {}
    for task_id in sorted(expected_tasks):
        coverage[str(task_id)] = {}
        for mode in required_modes:
            trials = per_task_mode_trials[task_id, mode]
            missing = sorted(expected_trials - trials)
            coverage[str(task_id)][mode] = {
                "transition_count": per_task_mode_count[task_id, mode],
                "trial_indices": sorted(trials),
                "missing_trial_indices": missing,
            }
            if missing:
                structural_errors.append(
                    f"task {task_id} mode {mode} misses trial indices {missing}"
                )
        if len(language_hashes[task_id]) != 1:
            structural_errors.append(
                f"task {task_id} has {len(language_hashes[task_id])} distinct language features"
            )

    if len(language_versions) != 1:
        structural_errors.append(
            f"expected one language encoder version, got {sorted(language_versions)}"
        )
    distinct_task_features = len({next(iter(values)) for values in language_hashes.values() if values})
    if distinct_task_features != len(expected_tasks):
        structural_errors.append(
            "language features are not distinct across every requested task"
        )
    residual_rms = (
        float(np.sqrt(np.mean(np.concatenate([value.reshape(-1) for value in residual_squares]))))
        if residual_squares
        else 0.0
    )
    if residual_rms <= 1e-8:
        structural_errors.append("executed actions have no residual variation from baseline")

    if expected_noise_stds is not None:
        action_seed_owners: dict[int, tuple[int, str, int]] = {}
        for episode_key, (_, action_seed) in episode_seeds.items():
            owner = action_seed_owners.get(action_seed)
            if owner is not None and owner != episode_key:
                structural_errors.append(
                    f"action seed {action_seed} is shared by {owner} and {episode_key}"
                )
            action_seed_owners[action_seed] = episode_key
        for task_id in sorted(expected_tasks):
            for trial_idx in sorted(expected_trials):
                policy_seeds = {
                    episode_seeds[task_id, mode, trial_idx][0]
                    for mode in required_modes
                    if (task_id, mode, trial_idx) in episode_seeds
                }
                if len(policy_seeds) != 1:
                    structural_errors.append(
                        "paired behaviors do not share one FastWAM policy seed for "
                        f"task {task_id} trial {trial_idx}: {sorted(policy_seeds)}"
                    )

    successes = Counter(task_id for (task_id, _, _), success in episode_outcomes.items() if success)
    failures = Counter(task_id for (task_id, _, _), success in episode_outcomes.items() if not success)
    report = {
        "collection_root": str(args.collection_root.resolve()),
        "expected_task_ids": sorted(expected_tasks),
        "expected_trial_indices": sorted(expected_trials),
        "expected_behaviors": list(required_modes),
        "num_records": len(records),
        "coverage": coverage,
        "episode_success_counts": {str(task): successes[task] for task in sorted(expected_tasks)},
        "episode_failure_counts": {str(task): failures[task] for task in sorted(expected_tasks)},
        "language_encoder_versions": sorted(language_versions),
        "distinct_task_language_features": distinct_task_features,
        "executed_residual_rms": residual_rms,
        "structural_errors": structural_errors,
        "passed": not structural_errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if structural_errors:
        raise SystemExit("raw multi-task collection audit failed")


if __name__ == "__main__":
    main()
