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
        for metadata_path in sorted(Path(input_dir).rglob("metadata.json")):
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            if not bool(metadata.get("alignment_valid", False)):
                continue
            record_dir = metadata_path.parent
            record = dict(metadata)
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[float]] = defaultdict(list)
    by_success: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("action_mode", "unknown"))].append(row["imagination_progress"])
        by_success[str(bool(row.get("success", False))).lower()].append(row["imagination_progress"])

    wrong_comparisons = [row["correct_beats_wrong"] for row in rows if "correct_beats_wrong" in row]
    return {
        "num_transitions": len(rows),
        "overall_mean_imagination_progress": _mean([row["imagination_progress"] for row in rows]),
        "mean_imagination_progress_by_action_mode": {
            key: _mean(values) for key, values in sorted(by_mode.items())
        },
        "mean_imagination_progress_by_episode_success": {
            key: _mean(values) for key, values in sorted(by_success.items())
        },
        "correct_goal_beats_wrong_fraction": _mean([float(value) for value in wrong_comparisons]),
        "num_wrong_goal_comparisons": len(wrong_comparisons),
        "wrong_goal_selection": "next_alignment_valid_transition_cyclic",
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
    for index, record in enumerate(records):
        metrics = compute_progress_reward(
            features[record["current_path"]],
            features[record["actual_path"]],
            features[record["goal_path"]],
            clip_value=args.clip_value,
        )
        row = dict(record)
        row.update(metrics)

        if len(records) > 1:
            wrong_record = records[(index + 1) % len(records)]
            wrong_metrics = compute_progress_reward(
                features[record["current_path"]],
                features[record["actual_path"]],
                features[wrong_record["goal_path"]],
                clip_value=args.clip_value,
            )
            row["wrong_goal_record_dir"] = wrong_record["record_dir"]
            row["wrong_goal_imagination_progress"] = wrong_metrics["imagination_progress"]
            row["correct_beats_wrong"] = bool(
                row["imagination_progress"] > row["wrong_goal_imagination_progress"]
            )
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "imagination_rewards.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    summary["encoder_path"] = str(Path(args.encoder_path).resolve())
    summary["input_dirs"] = [str(Path(path).resolve()) for path in args.input_dir]
    with (output_dir / "imagination_reward_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
