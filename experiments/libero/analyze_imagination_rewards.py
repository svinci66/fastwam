"""Compute frozen-SigLIP imagination-progress rewards from saved LIBERO clips."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.imagination_reward_utils import compute_progress_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Directory containing saved metadata.json transition records. Repeatable.",
    )
    parser.add_argument("--encoder-path", required=True, help="Local frozen SigLIP checkpoint directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--clip-value", type=float, default=None)
    return parser.parse_args()


def discover_records(input_dirs: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        input_root = Path(input_dir).resolve()
        for metadata_path in sorted(input_root.rglob("metadata.json")):
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            if not bool(metadata.get("alignment_valid", False)):
                continue
            record_dir = metadata_path.parent
            record = dict(metadata)
            record["source_input_dir"] = str(input_root)
            record["record_dir"] = str(record_dir)
            record["current_path"] = str(record_dir / "current.png")
            record["goal_path"] = str(record_dir / "predicted_goal.png")
            record["actual_path"] = str(record_dir / "actual.png")
            records.append(record)
    return records


def encode_images(
    image_paths: list[str],
    *,
    encoder_path: str,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    from transformers import SiglipImageProcessor, SiglipVisionModel

    processor = SiglipImageProcessor.from_pretrained(encoder_path, local_files_only=True)
    model = SiglipVisionModel.from_pretrained(
        encoder_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    unique_paths = list(dict.fromkeys(image_paths))
    features: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(unique_paths), batch_size):
            batch_paths = unique_paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)
            pooled = torch.nn.functional.normalize(outputs.pooler_output.float(), dim=-1)
            for path, feature in zip(batch_paths, pooled.cpu().numpy()):
                features[path] = feature
    return features


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _episode_key(record: dict[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(record.get("source_input_dir", "")),
        str(record.get("task_suite", "")),
        int(record.get("task_id", -1)),
        int(record.get("trial_idx", -1)),
        str(record.get("action_mode", "unknown")),
    )


def select_wrong_goal_record(
    record: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Choose a distant goal from another episode of the same task."""
    current_episode = _episode_key(record)
    current_mode = str(record.get("action_mode", "unknown"))
    current_replan = int(record.get("replan_idx", 0))
    same_task = [
        candidate
        for candidate in records
        if _episode_key(candidate) != current_episode
        and str(candidate.get("task_suite", "")) == str(record.get("task_suite", ""))
        and int(candidate.get("task_id", -1)) == int(record.get("task_id", -1))
    ]
    if not same_task:
        return None

    same_mode = [
        candidate
        for candidate in same_task
        if str(candidate.get("action_mode", "unknown")) == current_mode
    ]
    candidates = same_mode or same_task
    temporally_distant = [
        candidate
        for candidate in candidates
        if abs(int(candidate.get("replan_idx", 0)) - current_replan) >= 2
    ]
    candidates = temporally_distant or candidates
    return max(
        candidates,
        key=lambda candidate: (
            abs(int(candidate.get("replan_idx", 0)) - current_replan),
            str(candidate.get("record_dir", "")),
        ),
    )


def build_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_episode_key(row)].append(row)

    episode_rows: list[dict[str, Any]] = []
    for key, episode_transitions in sorted(grouped.items()):
        progress_values = [float(row["imagination_progress"]) for row in episode_transitions]
        first = episode_transitions[0]
        episode_rows.append(
            {
                "source_input_dir": key[0],
                "task_suite": key[1],
                "task_id": key[2],
                "trial_idx": key[3],
                "action_mode": key[4],
                "success": bool(first.get("success", False)),
                "episode_policy_steps": int(first.get("episode_policy_steps", 0)),
                "num_valid_transitions": len(episode_transitions),
                "episode_imagination_return": float(np.sum(progress_values)),
                "episode_mean_imagination_progress": _mean(progress_values),
            }
        )
    return episode_rows


def summarize(
    rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_mode: dict[str, list[float]] = defaultdict(list)
    by_success: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("action_mode", "unknown"))].append(row["imagination_progress"])
        by_success[str(bool(row.get("success", False))).lower()].append(row["imagination_progress"])

    wrong_comparisons = [row["correct_beats_wrong"] for row in rows if "correct_beats_wrong" in row]
    closer_comparisons = [
        row["correct_goal_is_closer_after"]
        for row in rows
        if "correct_goal_is_closer_after" in row
    ]
    episode_return_by_mode: dict[str, list[float]] = defaultdict(list)
    episode_mean_by_mode: dict[str, list[float]] = defaultdict(list)
    episode_return_by_success: dict[str, list[float]] = defaultdict(list)
    episode_mean_by_success: dict[str, list[float]] = defaultdict(list)
    episode_success_by_mode: dict[str, list[float]] = defaultdict(list)
    episode_steps_by_mode: dict[str, list[float]] = defaultdict(list)
    paired_by_trial: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for episode in episode_rows:
        mode = str(episode["action_mode"])
        success = str(bool(episode["success"])).lower()
        episode_return_by_mode[mode].append(float(episode["episode_imagination_return"]))
        episode_mean_by_mode[mode].append(float(episode["episode_mean_imagination_progress"]))
        episode_return_by_success[success].append(float(episode["episode_imagination_return"]))
        episode_mean_by_success[success].append(float(episode["episode_mean_imagination_progress"]))
        episode_success_by_mode[mode].append(float(bool(episode["success"])))
        episode_steps_by_mode[mode].append(float(episode["episode_policy_steps"]))
        paired_key = (
            str(episode["task_suite"]),
            int(episode["task_id"]),
            int(episode["trial_idx"]),
        )
        paired_by_trial[paired_key][mode] = float(episode["episode_mean_imagination_progress"])

    fully_paired = [values for values in paired_by_trial.values() if {"policy", "noise", "zero"} <= values.keys()]
    policy_beats_noise = [values["policy"] > values["noise"] for values in fully_paired]
    noise_beats_zero = [values["noise"] > values["zero"] for values in fully_paired]
    full_order = [
        values["policy"] > values["noise"] > values["zero"] for values in fully_paired
    ]

    return {
        "num_transitions": len(rows),
        "num_episodes": len(episode_rows),
        "overall_mean_imagination_progress": _mean([row["imagination_progress"] for row in rows]),
        "mean_imagination_progress_by_action_mode": {
            key: _mean(values) for key, values in sorted(by_mode.items())
        },
        "mean_imagination_progress_by_episode_success": {
            key: _mean(values) for key, values in sorted(by_success.items())
        },
        "correct_goal_beats_wrong_fraction": _mean([float(value) for value in wrong_comparisons]),
        "correct_goal_is_closer_after_fraction": _mean(
            [float(value) for value in closer_comparisons]
        ),
        "num_wrong_goal_comparisons": len(wrong_comparisons),
        "wrong_goal_selection": (
            "same_task_different_episode_farthest_replan_prefer_same_action_mode"
        ),
        "mean_episode_imagination_return_by_action_mode": {
            key: _mean(values) for key, values in sorted(episode_return_by_mode.items())
        },
        "mean_episode_progress_per_transition_by_action_mode": {
            key: _mean(values) for key, values in sorted(episode_mean_by_mode.items())
        },
        "mean_episode_imagination_return_by_success": {
            key: _mean(values) for key, values in sorted(episode_return_by_success.items())
        },
        "mean_episode_progress_per_transition_by_success": {
            key: _mean(values) for key, values in sorted(episode_mean_by_success.items())
        },
        "episode_success_rate_by_action_mode": {
            key: _mean(values) for key, values in sorted(episode_success_by_mode.items())
        },
        "mean_episode_policy_steps_by_action_mode": {
            key: _mean(values) for key, values in sorted(episode_steps_by_mode.items())
        },
        "num_fully_paired_trials": len(fully_paired),
        "paired_policy_beats_noise_fraction": _mean(
            [float(value) for value in policy_beats_noise]
        ),
        "paired_noise_beats_zero_fraction": _mean([float(value) for value in noise_beats_zero]),
        "paired_policy_gt_noise_gt_zero_fraction": _mean(
            [float(value) for value in full_order]
        ),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    records = discover_records(args.input_dir)
    if not records:
        raise ValueError("No alignment-valid imagination transition records were found.")

    image_paths = [
        path
        for record in records
        for path in (record["current_path"], record["goal_path"], record["actual_path"])
    ]
    features = encode_images(
        image_paths,
        encoder_path=args.encoder_path,
        device=args.device,
        batch_size=args.batch_size,
    )

    rows: list[dict[str, Any]] = []
    for record in records:
        metrics = compute_progress_reward(
            features[record["current_path"]],
            features[record["actual_path"]],
            features[record["goal_path"]],
            clip_value=args.clip_value,
        )
        row = dict(record)
        row.update(metrics)

        wrong_record = select_wrong_goal_record(record, records)
        if wrong_record is not None:
            wrong_metrics = compute_progress_reward(
                features[record["current_path"]],
                features[record["actual_path"]],
                features[wrong_record["goal_path"]],
                clip_value=args.clip_value,
            )
            row["wrong_goal_record_dir"] = wrong_record["record_dir"]
            row["wrong_goal_imagination_progress"] = wrong_metrics["imagination_progress"]
            row["wrong_goal_distance_after"] = wrong_metrics["distance_after"]
            row["correct_beats_wrong"] = bool(
                row["imagination_progress"] > row["wrong_goal_imagination_progress"]
            )
            row["correct_goal_is_closer_after"] = bool(
                row["distance_after"] < row["wrong_goal_distance_after"]
            )
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "imagination_rewards.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    episode_rows = build_episode_rows(rows)
    with (output_dir / "episode_imagination_rewards.jsonl").open("w", encoding="utf-8") as stream:
        for episode in episode_rows:
            stream.write(json.dumps(episode, ensure_ascii=False) + "\n")

    summary = summarize(rows, episode_rows)
    summary["encoder_path"] = str(Path(args.encoder_path).resolve())
    summary["input_dirs"] = [str(Path(path).resolve()) for path in args.input_dir]
    with (output_dir / "imagination_reward_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
