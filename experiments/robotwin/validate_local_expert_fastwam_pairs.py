"""Validate local expert sources against actual FastWAM initial observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CAMERA_SLICES = {
    "head_camera": (slice(0, 256), slice(0, 320)),
    "left_camera": (slice(256, 384), slice(0, 160)),
    "right_camera": (slice(256, 384), slice(160, 320)),
}


def comparison_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError(f"image shape mismatch: {left.shape} != {right.shape}")
    left_blur = cv2.GaussianBlur(left, (15, 15), 0).astype(np.float32)
    right_blur = cv2.GaussianBlur(right, (15, 15), 0).astype(np.float32)
    difference = np.abs(left_blur - right_blur) / 255.0
    return {
        "blurred_mean_abs": float(np.mean(difference)),
        "blurred_p95_abs": float(np.quantile(difference, 0.95)),
        "blurred_p99_abs": float(np.quantile(difference, 0.99)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-mean-abs", type=float, default=0.06)
    parser.add_argument("--max-p95-abs", type=float, default=0.20)
    parser.add_argument("--require-valid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = [
        json.loads(line)
        for line in args.cases_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for case in cases:
        task = str(case["task"])
        episode = int(case["episode_index"])
        bundle = Path(case["expert_hdf5"]).parent.parent
        expert_path = (
            bundle / "initial_observations" / f"episode{episode}_planning.png"
        )
        fastwam_path = (
            args.run_dir
            / task
            / "imagination_transitions"
            / task
            / "policy"
            / f"episode_{episode:04d}"
            / "replan_0000"
            / "current.png"
        )
        expert = cv2.imread(str(expert_path), cv2.IMREAD_COLOR)
        fastwam = cv2.imread(str(fastwam_path), cv2.IMREAD_COLOR)
        if expert is None or fastwam is None:
            raise FileNotFoundError(
                f"missing initial image: expert={expert_path}, fastwam={fastwam_path}"
            )
        camera_metrics = {}
        valid = True
        for camera, slices in CAMERA_SLICES.items():
            values = comparison_metrics(expert[slices], fastwam[slices])
            values["valid"] = bool(
                values["blurred_mean_abs"] <= args.max_mean_abs
                and values["blurred_p95_abs"] <= args.max_p95_abs
            )
            camera_metrics[camera] = values
            valid = valid and bool(values["valid"])
        rows.append(
            {
                "task": task,
                "episode_index": episode,
                "environment_seed": int(case["environment_seed"]),
                "valid": valid,
                "expert_initial": str(expert_path.resolve()),
                "fastwam_initial": str(fastwam_path.resolve()),
                "camera_metrics": camera_metrics,
            }
        )
    report = {
        "schema_version": "robotwin_local_expert_fastwam_pair_audit_v1",
        "pair_count": len(rows),
        "valid_pair_count": sum(bool(row["valid"]) for row in rows),
        "all_valid": bool(rows) and all(bool(row["valid"]) for row in rows),
        "thresholds": {
            "max_mean_abs": args.max_mean_abs,
            "max_p95_abs": args.max_p95_abs,
        },
        "pairs": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_valid and not report["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
