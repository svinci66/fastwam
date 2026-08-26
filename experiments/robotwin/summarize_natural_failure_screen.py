"""Pair natural FastWAM failures with same-seed RoboTwin expert demonstrations."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ACCEPTED_RE = re.compile(r"FASTWAM_ACCEPTED_ENV_SEED episode_id=(\d+) seed=(\d+)")
OUTCOME_RE = re.compile(r"^(Success|Fail)!$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def parse_eval_log(path: Path) -> list[dict[str, Any]]:
    accepted: list[tuple[int, int]] = []
    outcomes: list[bool] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_RE.sub("", raw).strip()
        match = ACCEPTED_RE.search(line)
        if match:
            accepted.append((int(match.group(1)), int(match.group(2))))
            continue
        match = OUTCOME_RE.fullmatch(line)
        if match:
            outcomes.append(match.group(1) == "Success")
    if len(accepted) != len(outcomes):
        raise ValueError(
            f"{path}: accepted seeds={len(accepted)} but outcomes={len(outcomes)}"
        )
    if len({episode_id for episode_id, _ in accepted}) != len(accepted):
        raise ValueError(f"{path}: duplicate episode ids")
    return [
        {"evaluation_episode_id": episode_id, "environment_seed": seed, "success": success}
        for (episode_id, seed), success in zip(accepted, outcomes)
    ]


def decode_jpeg(value: Any) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(bytes(value), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode RoboTwin JPEG")
    return image


def compose_frame(handle: h5py.File, index: int) -> np.ndarray:
    head = cv2.resize(
        decode_jpeg(handle["observation/head_camera/rgb"][index]), (320, 256)
    )
    left = cv2.resize(
        decode_jpeg(handle["observation/left_camera/rgb"][index]), (160, 128)
    )
    right = cv2.resize(
        decode_jpeg(handle["observation/right_camera/rgb"][index]), (160, 128)
    )
    return np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)


def export_expert_video(hdf5_path: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".partial.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (320, 384)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {temporary}")
    try:
        with h5py.File(hdf5_path, "r") as handle:
            frame_count = int(handle["observation/head_camera/rgb"].shape[0])
            for index in range(frame_count):
                writer.write(compose_frame(handle, index))
    finally:
        writer.release()
    os.replace(temporary, output_path)


def latest_complete_log(run_dir: Path, task: str) -> Path:
    logs = sorted(run_dir.glob(f"eval_{task}_*.log"))
    for path in reversed(logs):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Success rate:" in text:
            return path
    raise FileNotFoundError(f"no complete evaluation log for {task} under {run_dir}")


def main() -> None:
    args = parse_args()
    cases = [
        json.loads(line)
        for line in args.cases_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    case_by_key = {
        (str(case["task"]), int(case["environment_seed"])): case for case in cases
    }
    rows: list[dict[str, Any]] = []
    for task in tasks:
        log_path = latest_complete_log(args.run_dir, task)
        for result in parse_eval_log(log_path):
            key = (task, int(result["environment_seed"]))
            if key not in case_by_key:
                raise KeyError(f"evaluation result has no expert source case: {key}")
            case = dict(case_by_key[key])
            episode_id = int(result["evaluation_episode_id"])
            policy_video = args.run_dir / task / f"episode{episode_id}.mp4"
            if not policy_video.is_file():
                raise FileNotFoundError(policy_video)
            row = {
                **case,
                **result,
                "eval_log": str(log_path.resolve()),
                "fastwam_video": str(policy_video.resolve()),
                "decision": "fastwam_success" if result["success"] else "natural_failure",
            }
            rows.append(row)

    failures = [row for row in rows if not bool(row["success"])]
    review_dir = args.artifact_dir / "manual_review"
    for candidate_index, row in enumerate(failures, start=1):
        name = (
            f"candidate_{candidate_index:02d}_{row['task']}_"
            f"episode{int(row['episode_index']):02d}_seed{int(row['environment_seed'])}"
        )
        candidate_dir = review_dir / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        expert_video = candidate_dir / "expert_success.mp4"
        if not expert_video.is_file():
            export_expert_video(Path(row["expert_hdf5"]), expert_video, args.fps)
        failure_link = candidate_dir / "fastwam_failure.mp4"
        failure_link.unlink(missing_ok=True)
        failure_link.symlink_to(Path(row["fastwam_video"]))
        metadata = {**row, "manual_review_required": True}
        (candidate_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        row["review_dir"] = str(candidate_dir.resolve())

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.artifact_dir / "status.jsonl"
    status_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    per_task = {
        task: {
            "episodes": sum(row["task"] == task for row in rows),
            "fastwam_successes": sum(
                row["task"] == task and bool(row["success"]) for row in rows
            ),
            "natural_failures": sum(
                row["task"] == task and not bool(row["success"]) for row in rows
            ),
        }
        for task in tasks
    }
    summary = {
        "schema_version": "robotwin_natural_failure_screen_v1",
        "source_case_count": len(cases),
        "evaluated_case_count": len(rows),
        "natural_failure_count": len(failures),
        "manual_annotation_scope": (
            "binary semantic validity plus approximate first-divergence replan; "
            "no framewise or action labeling"
        ),
        "per_task": per_task,
        "status_jsonl": str(status_path.resolve()),
        "manual_review_dir": str(review_dir.resolve()),
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
