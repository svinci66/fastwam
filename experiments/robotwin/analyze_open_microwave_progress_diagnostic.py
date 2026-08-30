"""Summarize door-joint progress for outcome-selected RoboTwin diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-base", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--variants", default="no_imagination,imagination")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _episode_records(root: Path) -> list[dict]:
    records: list[dict] = []
    for episode_dir in sorted(root.glob("episode_*")):
        metadata_paths = sorted(episode_dir.glob("replan_*/metadata.json"))
        if not metadata_paths:
            continue
        progress_parts: list[np.ndarray] = []
        residual_norms: list[np.ndarray] = []
        seed = None
        success = False
        for metadata_path in metadata_paths:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            seed = int(metadata["environment_seed"])
            success = success or bool(metadata.get("episode_success"))
            arrays_path = metadata_path.parent / metadata["rollout_arrays_file"]
            with np.load(arrays_path, allow_pickle=False) as arrays:
                if "task_progress" not in arrays:
                    raise ValueError(f"missing task_progress in {arrays_path}")
                progress = np.asarray(arrays["task_progress"], dtype=np.float64)
                if progress.ndim != 2 or progress.shape[1] != 4:
                    raise ValueError(f"invalid task_progress shape {progress.shape}")
                progress_parts.append(progress if not progress_parts else progress[1:])
                if "candidate_residual_actions" in arrays:
                    residual = np.asarray(
                        arrays["candidate_residual_actions"], dtype=np.float64
                    )
                    residual_norms.append(np.linalg.norm(residual, axis=1))
        if seed is None or not progress_parts:
            raise ValueError(f"incomplete episode diagnostic {episode_dir}")
        progress = np.concatenate(progress_parts, axis=0)
        ratio = progress[:, 3]
        delta = np.diff(ratio)
        crossed = np.flatnonzero(ratio >= 0.6)
        residual = (
            np.concatenate(residual_norms)
            if residual_norms
            else np.asarray([], dtype=np.float64)
        )
        records.append(
            {
                "episode_id": int(episode_dir.name.removeprefix("episode_")),
                "seed": seed,
                "environment_success": success,
                "success_from_progress": bool(crossed.size),
                "actions": int(ratio.size - 1),
                "replans": len(metadata_paths),
                "start_open_ratio": float(ratio[0]),
                "final_open_ratio": float(ratio[-1]),
                "max_open_ratio": float(np.max(ratio)),
                "threshold_crossing_action": (
                    None if not crossed.size else int(crossed[0])
                ),
                "net_open_progress": float(ratio[-1] - ratio[0]),
                "positive_open_progress": float(np.clip(delta, 0.0, None).sum()),
                "negative_open_progress": float(np.clip(delta, None, 0.0).sum()),
                "largest_positive_step": float(np.max(delta, initial=0.0)),
                "largest_negative_step": float(np.min(delta, initial=0.0)),
                "candidate_residual_l2_mean": (
                    None if not residual.size else float(np.mean(residual))
                ),
                "candidate_residual_l2_max": (
                    None if not residual.size else float(np.max(residual))
                ),
                "open_ratio_trace": ratio.tolist(),
            }
        )
    return records


def analyze(result_base: Path, run_name: str, variants: list[str]) -> dict:
    by_variant: dict[str, list[dict]] = {}
    for variant in variants:
        root = (
            result_base
            / f"{run_name}_{variant}"
            / "open_microwave"
            / "imagination_transitions"
            / "open_microwave"
            / "residual"
        )
        records = _episode_records(root)
        if not records:
            raise ValueError(f"no diagnostic transitions under {root}")
        by_variant[variant] = records
    keyed = {
        variant: {record["seed"]: record for record in records}
        for variant, records in by_variant.items()
    }
    seed_sets = [set(records) for records in keyed.values()]
    if not seed_sets or any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("diagnostic variants do not contain identical seeds")
    control = keyed["no_imagination"]
    candidate = keyed["imagination"]
    pairs = []
    for seed in sorted(seed_sets[0]):
        left = control[seed]
        right = candidate[seed]
        pairs.append(
            {
                "seed": seed,
                "no_imagination_success": left["environment_success"],
                "imagination_success": right["environment_success"],
                "no_imagination_max_open_ratio": left["max_open_ratio"],
                "imagination_max_open_ratio": right["max_open_ratio"],
                "max_open_ratio_delta": (
                    right["max_open_ratio"] - left["max_open_ratio"]
                ),
                "no_imagination_actions": left["actions"],
                "imagination_actions": right["actions"],
            }
        )
    return {
        "schema_version": "robotwin_open_microwave_progress_diagnostic_v1",
        "diagnostic_only": True,
        "warning": "Outcome-selected reruns must not be counted as held-out evaluation.",
        "success_threshold": 0.6,
        "variants": by_variant,
        "pairs": pairs,
    }


def main() -> None:
    args = parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    result = analyze(args.result_base, args.run_name, variants)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pairs": result["pairs"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
