"""Audit three-camera RoboTwin imagination rewards from aligned transitions."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.robotwin.imagination_reward_utils import (
    ROBOTWIN_CAMERA_NAMES,
    split_robotwin_camera_views,
)
from fastwam.rl.rewards import (
    GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE,
    compute_imagination_reward,
)


# Keep replay construction byte-for-byte aligned with
# ``OnlineResidualPolicy.encode_observation``.  The online path first resizes
# each split camera view to this square resolution and only then invokes the
# SigLIP processor.
ROBOTWIN_CAMERA_IMAGE_SIZE = 224


def resolve_encoder_dtype(value: str, *, device: str) -> torch.dtype:
    """Resolve the replay encoder precision used by online residual inference."""

    key = str(value).strip().lower()
    if key == "auto":
        key = "bf16" if torch.device(device).type == "cuda" else "fp32"
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if key not in mapping:
        raise ValueError(
            f"Unsupported encoder dtype {value!r}; expected auto, fp32, bf16, or fp16"
        )
    dtype = mapping[key]
    if torch.device(device).type == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 replay encoding is not supported on CPU")
    return dtype


def prepare_robotwin_camera_view(
    view: np.ndarray,
    *,
    image_size: int = ROBOTWIN_CAMERA_IMAGE_SIZE,
) -> Image.Image:
    """Apply the same per-camera resize used by online residual inference."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    array = np.asarray(view)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"camera view must be RGB [H,W,3], got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"camera view must use an integer dtype, got {array.dtype}")
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--encoder-dtype",
        default="auto",
        choices=("auto", "fp32", "bf16", "fp16"),
        help="SigLIP precision; auto matches online bf16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clip-value", type=float, default=0.1)
    parser.add_argument("--minimum-paired-trials", type=int, default=15)
    return parser.parse_args()


def behavior_label(record: dict[str, Any]) -> str:
    explicit = str(record.get("behavior_tag", "")).strip()
    if explicit:
        return explicit
    if str(record.get("action_mode", "policy")) == "policy":
        return "policy"
    return f"noise_{float(record.get('action_noise_std', 0.0)):.3f}"


def discover_records(input_dirs: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in input_dirs:
        for metadata_path in sorted(root.resolve().rglob("metadata.json")):
            if any("pairing_quarantine" in part for part in metadata_path.parts):
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema_version") != "robotwin_imagination_transition_v1":
                continue
            record = dict(metadata)
            record["record_dir"] = str(metadata_path.parent)
            record["behavior"] = behavior_label(record)
            for phase in ("current", "actual", "predicted_goal"):
                image_path = metadata_path.parent / f"{phase}.png"
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
                record[f"{phase}_path"] = str(image_path)
            records.append(record)
    if not records:
        raise ValueError("No RoboTwin imagination-transition records were found")
    return records


def encode_record_images(
    records: list[dict[str, Any]],
    *,
    encoder_path: Path,
    device: str,
    batch_size: int,
    camera_image_size: int = ROBOTWIN_CAMERA_IMAGE_SIZE,
    encoder_dtype: torch.dtype = torch.float32,
) -> list[dict[str, dict[str, np.ndarray]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if camera_image_size <= 0:
        raise ValueError("camera_image_size must be positive")
    from transformers import SiglipImageProcessor, SiglipVisionModel

    processor = SiglipImageProcessor.from_pretrained(encoder_path, local_files_only=True)
    model = SiglipVisionModel.from_pretrained(
        encoder_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=encoder_dtype,
    ).to(device).eval()
    images: list[Image.Image] = []
    keys: list[tuple[int, str, str]] = []
    for record_index, record in enumerate(records):
        for phase in ("current", "actual", "predicted_goal"):
            with Image.open(record[f"{phase}_path"]) as image:
                views = split_robotwin_camera_views(image.convert("RGB"))
            for camera in ROBOTWIN_CAMERA_NAMES:
                images.append(
                    prepare_robotwin_camera_view(
                        views[camera], image_size=camera_image_size
                    )
                )
                keys.append((record_index, phase, camera))

    encoded: list[dict[str, dict[str, np.ndarray]]] = [
        {phase: {} for phase in ("current", "actual", "predicted_goal")}
        for _ in records
    ]
    print(
        "[robotwin-replay-encoder] "
        f"records={len(records)} images={len(images)} batch_size={batch_size} "
        f"device={device} dtype={str(encoder_dtype).removeprefix('torch.')}",
        flush=True,
    )
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(images), batch_size)):
            batch = processor(images=images[start : start + batch_size], return_tensors="pt")
            output = model(
                pixel_values=batch["pixel_values"].to(
                    device=device, dtype=encoder_dtype
                )
            ).pooler_output
            output = torch.nn.functional.normalize(output.float(), dim=-1).cpu().numpy()
            for key, feature in zip(keys[start : start + batch_size], output):
                record_index, phase, camera = key
                encoded[record_index][phase][camera] = feature.astype(
                    np.float32, copy=False
                )
            completed = min(start + batch_size, len(images))
            if batch_index % 20 == 0 or completed == len(images):
                print(
                    "[robotwin-replay-encoder] "
                    f"encoded={completed}/{len(images)}",
                    flush=True,
                )
    return encoded


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    positions = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return float(np.interp(quantile, positions, values, left=values[0], right=values[-1]))


def fit_task_balanced_camera_normalization(
    records: list[dict[str, Any]],
    encoded: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, Any]:
    valid = [index for index, record in enumerate(records) if record["alignment_valid"]]
    if not valid:
        raise ValueError("At least one aligned transition is required")
    task_counts: dict[str, int] = defaultdict(int)
    for index in valid:
        task_counts[str(records[index]["task_name"])] += 1
    weights = np.asarray(
        [1.0 / task_counts[str(records[index]["task_name"])] for index in valid],
        dtype=np.float64,
    )
    scores: dict[str, list[float]] = {camera: [] for camera in ROBOTWIN_CAMERA_NAMES}
    for index in valid:
        features = encoded[index]
        raw = compute_imagination_reward(
            features["current"],
            features["actual"],
            features["predicted_goal"],
            reward_type="delta_alignment_v1",
            camera_weights={camera: 1.0 for camera in ROBOTWIN_CAMERA_NAMES},
            clip_value=1.0,
        )
        for camera in ROBOTWIN_CAMERA_NAMES:
            scores[camera].append(float(raw.per_camera[camera]["delta_alignment_reward"]))

    cameras: dict[str, dict[str, float]] = {}
    for camera in ROBOTWIN_CAMERA_NAMES:
        values = np.asarray(scores[camera], dtype=np.float64)
        q25 = _weighted_quantile(values, weights, 0.25)
        median = _weighted_quantile(values, weights, 0.5)
        q75 = _weighted_quantile(values, weights, 0.75)
        scale = q75 - q25
        if scale <= 1e-8:
            raise ValueError(f"Camera {camera} has degenerate global IQR={scale}")
        cameras[camera] = {
            "center": median,
            "scale": scale,
            "q25": q25,
            "q75": q75,
        }
    return {
        "type": "task_balanced_global_camera_median_iqr_tanh_v1",
        "num_tasks": len(task_counts),
        "num_valid_transitions": len(valid),
        "cameras": cameras,
    }


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def analyze(
    records: list[dict[str, Any]],
    encoded: list[dict[str, dict[str, np.ndarray]]],
    *,
    clip_value: float,
    minimum_paired_trials: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    normalization = fit_task_balanced_camera_normalization(records, encoded)
    camera_norm = {
        camera: {
            "center": settings["center"],
            "scale": settings["scale"],
        }
        for camera, settings in normalization["cameras"].items()
    }
    rows: list[dict[str, Any]] = []
    by_task_episode: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_task_episode[str(record["task_name"])].append(index)
    for index, (record, features) in enumerate(zip(records, encoded)):
        reward = compute_imagination_reward(
            features["current"],
            features["actual"],
            features["predicted_goal"],
            reward_type=GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE,
            camera_weights={camera: 1.0 for camera in ROBOTWIN_CAMERA_NAMES},
            camera_normalization=camera_norm,
            clip_value=clip_value,
            alignment_valid=bool(record["alignment_valid"]),
        )
        row = dict(record)
        row["imagination_reward"] = reward.clipped_progress
        row["per_camera"] = reward.per_camera

        wrong_candidates = [
            other
            for other in by_task_episode[str(record["task_name"])]
            if int(records[other]["trial_idx"]) != int(record["trial_idx"])
        ]
        if wrong_candidates:
            wrong_index = wrong_candidates[index % len(wrong_candidates)]
            wrong = compute_imagination_reward(
                features["current"],
                features["actual"],
                encoded[wrong_index]["predicted_goal"],
                reward_type=GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE,
                camera_weights={camera: 1.0 for camera in ROBOTWIN_CAMERA_NAMES},
                camera_normalization=camera_norm,
                clip_value=clip_value,
                alignment_valid=bool(record["alignment_valid"]),
            )
            row["shuffled_goal_reward"] = wrong.clipped_progress
            row["correct_beats_shuffled"] = reward.clipped_progress > wrong.clipped_progress
        rows.append(row)

    valid_rows = [row for row in rows if bool(row["alignment_valid"])]
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        grouped[(str(row["task_name"]), int(row["trial_idx"]), str(row["behavior"]))].append(row)
    episodes: list[dict[str, Any]] = []
    for (task, trial, behavior), values in sorted(grouped.items()):
        initial_hashes = {str(row["initial_observation_sha256"]) for row in values}
        if len(initial_hashes) != 1:
            raise ValueError(
                f"Episode {task}/{trial}/{behavior} contains multiple initial-state hashes"
            )
        episodes.append(
            {
                "task_name": task,
                "trial_idx": trial,
                "behavior": behavior,
                "success": bool(any(row["episode_success"] for row in values)),
                "initial_observation_sha256": next(iter(initial_hashes)),
                "num_transitions": len(values),
                "mean_imagination_reward": float(
                    np.mean([row["imagination_reward"] for row in values])
                ),
                "imagination_return": float(
                    np.sum([row["imagination_reward"] for row in values])
                ),
            }
        )

    noise_levels = sorted(
        {episode["behavior"] for episode in episodes if episode["behavior"].startswith("noise_")}
    )
    hold_levels = sorted(
        {episode["behavior"] for episode in episodes if episode["behavior"].startswith("hold_")},
        key=lambda value: float(value.removeprefix("hold_")),
    )
    gripper_delay_levels = sorted(
        {
            episode["behavior"]
            for episode in episodes
            if episode["behavior"].startswith("gripper_delay_")
        }
    )
    if len(noise_levels) >= 2:
        comparison_family = "noise"
        comparison_levels = noise_levels
    elif len(hold_levels) >= 2:
        comparison_family = "hold"
        comparison_levels = hold_levels
    else:
        comparison_family = None
        comparison_levels = []
    mild = comparison_levels[0] if comparison_levels else None
    strong = comparison_levels[-1] if len(comparison_levels) >= 2 else None
    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for episode in episodes:
        paired[(episode["task_name"], episode["trial_idx"])][episode["behavior"]] = episode
    complete = [
        item for item in paired.values()
        if mild is not None and strong is not None and {"policy", mild, strong} <= set(item)
    ]
    full_order = [
        item["policy"]["mean_imagination_reward"]
        > item[mild]["mean_imagination_reward"]
        > item[strong]["mean_imagination_reward"]
        for item in complete
    ]
    initial_state_matches = [
        len(
            {
                item["policy"]["initial_observation_sha256"],
                item[mild]["initial_observation_sha256"],
                item[strong]["initial_observation_sha256"],
            }
        )
        == 1
        for item in complete
    ]
    policy_gripper_pairs = [
        item
        for item in paired.values()
        if "policy" in item
        and any(level in item for level in gripper_delay_levels)
    ]
    policy_gt_gripper_delay = [
        item["policy"]["mean_imagination_reward"]
        > max(
            item[level]["mean_imagination_reward"]
            for level in gripper_delay_levels
            if level in item
        )
        for item in policy_gripper_pairs
    ]
    correct_shuffled = [
        bool(row["correct_beats_shuffled"])
        for row in valid_rows
        if "correct_beats_shuffled" in row
    ]
    record_integrity_fraction = float(
        np.mean(
            [
                bool(row["alignment_valid"])
                or bool(row["terminated"])
                or bool(row["truncated"])
                for row in rows
            ]
        )
    )
    saturation_fraction = float(
        np.mean(
            [
                abs(float(row["imagination_reward"])) >= 0.95 * clip_value
                for row in valid_rows
            ]
        )
    )
    success_rewards = [
        episode["mean_imagination_reward"] for episode in episodes if episode["success"]
    ]
    failure_rewards = [
        episode["mean_imagination_reward"] for episode in episodes if not episode["success"]
    ]
    ordering_fraction = _mean([float(value) for value in full_order])
    initial_state_match_fraction = _mean(
        [float(value) for value in initial_state_matches]
    )
    shuffled_fraction = _mean([float(value) for value in correct_shuffled])
    success_gt_failure = bool(
        success_rewards and failure_rewards and np.mean(success_rewards) > np.mean(failure_rewards)
    )
    sample_ready = len(complete) >= minimum_paired_trials
    behavior_summary: dict[str, dict[str, Any]] = {}
    for behavior in sorted({episode["behavior"] for episode in episodes}):
        values = [episode for episode in episodes if episode["behavior"] == behavior]
        behavior_summary[behavior] = {
            "num_episodes": len(values),
            "num_successes": int(sum(bool(value["success"]) for value in values)),
            "success_rate": float(np.mean([bool(value["success"]) for value in values])),
            "mean_imagination_reward": float(
                np.mean([value["mean_imagination_reward"] for value in values])
            ),
            "mean_imagination_return": float(
                np.mean([value["imagination_return"] for value in values])
            ),
        }
    gates = {
        "minimum_sample_ready": sample_ready,
        "paired_initial_state_match_eq_1": initial_state_match_fraction == 1.0,
        "temporal_record_integrity_ge_0_95": record_integrity_fraction >= 0.95,
        "policy_gt_mild_gt_strong_ge_0_70": (
            ordering_fraction is not None and ordering_fraction >= 0.70
        ),
        "correct_gt_shuffled_ge_0_70": (
            shuffled_fraction is not None and shuffled_fraction >= 0.70
        ),
        "saturation_lt_0_25": saturation_fraction < 0.25,
        "success_mean_gt_failure_mean": success_gt_failure,
    }
    summary = {
        "num_transitions": len(rows),
        "num_alignment_valid_transitions": len(valid_rows),
        "num_episodes": len(episodes),
        "num_complete_paired_trials": len(complete),
        "behavior_noise_levels": noise_levels,
        "behavior_hold_levels": hold_levels,
        "behavior_gripper_delay_levels": gripper_delay_levels,
        "behavior_comparison_family": comparison_family,
        "behavior_summary": behavior_summary,
        "temporal_record_integrity_fraction": record_integrity_fraction,
        "clip_saturation_fraction": saturation_fraction,
        "paired_policy_gt_mild_gt_strong_fraction": ordering_fraction,
        "paired_policy_gt_gripper_delay_fraction": _mean(
            [float(value) for value in policy_gt_gripper_delay]
        ),
        "paired_initial_state_match_fraction": initial_state_match_fraction,
        "correct_goal_beats_shuffled_fraction": shuffled_fraction,
        "mean_reward_success": _mean(success_rewards),
        "mean_reward_failure": _mean(failure_rewards),
        "camera_normalization": normalization,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
    }
    return rows, episodes, summary


def main() -> None:
    args = parse_args()
    encoder_dtype = resolve_encoder_dtype(args.encoder_dtype, device=args.device)
    records = discover_records(args.input_dir)
    encoded = encode_record_images(
        records,
        encoder_path=args.encoder_path,
        device=args.device,
        batch_size=args.batch_size,
        encoder_dtype=encoder_dtype,
    )
    rows, episodes, summary = analyze(
        records,
        encoded,
        clip_value=args.clip_value,
        minimum_paired_trials=args.minimum_paired_trials,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("transition_rewards.jsonl", rows),
        ("episode_rewards.jsonl", episodes),
    ):
        with (args.output_dir / filename).open("w", encoding="utf-8") as stream:
            for item in payload:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary["encoder_path"] = str(args.encoder_path.resolve())
    summary["encoder_dtype"] = str(encoder_dtype).removeprefix("torch.")
    (args.output_dir / "reward_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
