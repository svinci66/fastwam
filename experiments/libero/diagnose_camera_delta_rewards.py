"""Offline Reward V2 diagnostic for saved two-camera LIBERO transitions.

This script never launches LIBERO or updates model parameters.  It reuses aligned
``current / predicted_goal / actual`` triplets, separates the agent and wrist
cameras, and compares four reward candidates fixed before inspecting the results.
"""

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

from experiments.libero.analyze_imagination_rewards import (
    discover_records,
    select_wrong_goal_record,
)
from experiments.libero.imagination_reward_utils import (
    LIBERO_CAMERA_NAMES,
    compute_delta_alignment_reward,
    compute_progress_reward,
    split_horizontal_camera_views,
)


CANDIDATE_NAMES = (
    "concat_progress",
    "agent_delta_alignment",
    "dual_delta_alignment",
    "concat_progress_plus_dual_delta",
)
PHASE_NAMES = ("early", "middle", "late")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Directory containing saved metadata.json transition records. Repeatable.",
    )
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--direction-weight",
        type=float,
        default=0.01,
        help="Fixed coefficient in progress + direction; this script does not tune it.",
    )
    parser.add_argument("--diagnostic-seed", type=int, default=1042)
    parser.add_argument("--diagnostic-trials", type=int, nargs="+", default=[2, 3])
    return parser.parse_args()


def _encode_image_batch(processor: Any, model: Any, images: list[Image.Image], device: str) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt")
    outputs = model(pixel_values=inputs["pixel_values"].to(device))
    pooled = torch.nn.functional.normalize(outputs.pooler_output.float(), dim=-1)
    return pooled.cpu().numpy()


def encode_camera_features(
    image_paths: list[str],
    *,
    encoder_path: str,
    device: str,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Encode full, agent-view, and wrist-view images with one frozen encoder."""
    from transformers import SiglipImageProcessor, SiglipVisionModel

    processor = SiglipImageProcessor.from_pretrained(encoder_path, local_files_only=True)
    model = SiglipVisionModel.from_pretrained(
        encoder_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    # Preserve discovery order and reset batching per input dataset.  SigLIP's GPU
    # kernels have tiny batch-composition-dependent floating-point differences, so
    # this matches the original per-dataset Reward V1 analysis as closely as possible.
    paths = list(dict.fromkeys(image_paths))
    features: dict[str, dict[str, np.ndarray]] = {
        "concat": {},
        LIBERO_CAMERA_NAMES[0]: {},
        LIBERO_CAMERA_NAMES[1]: {},
    }
    with torch.inference_mode():
        for view_name in features:
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                images: list[Image.Image] = []
                for path in batch_paths:
                    with Image.open(path) as image:
                        rgb = image.convert("RGB")
                        if view_name == "concat":
                            selected = np.asarray(rgb, dtype=np.uint8)
                        else:
                            selected = split_horizontal_camera_views(rgb)[view_name]
                        images.append(Image.fromarray(selected))
                batch_features = _encode_image_batch(processor, model, images, device)
                for path, feature in zip(batch_paths, batch_features):
                    features[view_name][path] = feature
    return features


def save_feature_cache(
    cache_path: Path,
    image_paths: list[str],
    features: dict[str, dict[str, np.ndarray]],
) -> None:
    paths = sorted(set(image_paths))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        paths=np.asarray(paths),
        **{
            view: np.stack([features[view][path] for path in paths])
            for view in ("concat", *LIBERO_CAMERA_NAMES)
        },
    )


def load_feature_cache(
    cache_path: Path,
    image_paths: list[str],
) -> dict[str, dict[str, np.ndarray]] | None:
    if not cache_path.is_file():
        return None
    expected_paths = sorted(set(image_paths))
    with np.load(cache_path, allow_pickle=False) as cache:
        cached_paths = [str(path) for path in cache["paths"].tolist()]
        if cached_paths != expected_paths:
            return None
        return {
            view: {
                path: feature
                for path, feature in zip(cached_paths, np.asarray(cache[view]))
            }
            for view in ("concat", *LIBERO_CAMERA_NAMES)
        }


def _record_episode_key(record: dict[str, Any]) -> tuple[int, str, int, int, str]:
    return (
        int(record.get("policy_seed", -1)),
        str(record.get("task_suite", "")),
        int(record.get("task_id", -1)),
        int(record.get("trial_idx", -1)),
        str(record.get("action_mode", "unknown")),
    )


def assign_temporal_phases(records: list[dict[str, Any]]) -> dict[str, str]:
    """Assign early/middle/late by relative progress through each saved episode."""
    replans_by_episode: dict[tuple[int, str, int, int, str], list[int]] = defaultdict(list)
    for record in records:
        replans_by_episode[_record_episode_key(record)].append(int(record.get("replan_idx", 0)))

    phases: dict[str, str] = {}
    for record in records:
        replans = replans_by_episode[_record_episode_key(record)]
        current = int(record.get("replan_idx", 0))
        minimum, maximum = min(replans), max(replans)
        fraction = 0.0 if maximum == minimum else (current - minimum) / (maximum - minimum)
        phase_index = min(int(fraction * len(PHASE_NAMES)), len(PHASE_NAMES) - 1)
        phases[str(record["record_dir"])] = PHASE_NAMES[phase_index]
    return phases


def compute_candidate_metrics(
    features: dict[str, dict[str, np.ndarray]],
    *,
    current_path: str,
    actual_path: str,
    goal_path: str,
    direction_weight: float,
) -> dict[str, Any]:
    """Compute per-camera diagnostics and the four predeclared candidates."""
    view_metrics: dict[str, dict[str, float]] = {}
    for view in ("concat", *LIBERO_CAMERA_NAMES):
        progress = compute_progress_reward(
            features[view][current_path],
            features[view][actual_path],
            features[view][goal_path],
        )
        delta = compute_delta_alignment_reward(
            features[view][current_path],
            features[view][actual_path],
            features[view][goal_path],
        )
        view_metrics[view] = {**progress, **delta}

    dual_delta = float(
        np.mean(
            [view_metrics[view]["delta_alignment_reward"] for view in LIBERO_CAMERA_NAMES]
        )
    )
    concat_progress = float(view_metrics["concat"]["imagination_progress"])
    candidates = {
        "concat_progress": concat_progress,
        "agent_delta_alignment": float(view_metrics["agent"]["delta_alignment_reward"]),
        "dual_delta_alignment": dual_delta,
        "concat_progress_plus_dual_delta": float(
            concat_progress + direction_weight * dual_delta
        ),
    }
    return {"views": view_metrics, "candidates": candidates}


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def binary_auc(labels: list[bool], scores: list[float]) -> float | None:
    """Return tie-aware ROC AUC without adding a scikit-learn dependency."""
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return float(wins / (len(positives) * len(negatives)))


def build_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_record_episode_key(row)].append(row)

    result: list[dict[str, Any]] = []
    for key, transitions in sorted(grouped.items()):
        first = transitions[0]
        candidate_means = {
            candidate: _mean(
                [float(transition["candidate_rewards"][candidate]) for transition in transitions]
            )
            for candidate in CANDIDATE_NAMES
        }
        result.append(
            {
                "policy_seed": key[0],
                "task_suite": key[1],
                "task_id": key[2],
                "trial_idx": key[3],
                "action_mode": key[4],
                "success": bool(first.get("success", False)),
                "episode_policy_steps": int(first.get("episode_policy_steps", 0)),
                "num_valid_transitions": len(transitions),
                "mean_candidate_rewards": candidate_means,
            }
        )
    return result


def summarize_candidate(
    candidate: str,
    rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    transition_by_mode: dict[str, list[float]] = defaultdict(list)
    episode_by_mode: dict[str, list[float]] = defaultdict(list)
    episode_by_success: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        transition_by_mode[str(row["action_mode"])].append(
            float(row["candidate_rewards"][candidate])
        )
    for episode in episode_rows:
        score = float(episode["mean_candidate_rewards"][candidate])
        episode_by_mode[str(episode["action_mode"])].append(score)
        episode_by_success[str(bool(episode["success"])).lower()].append(score)

    paired: dict[tuple[int, str, int, int], dict[str, float]] = defaultdict(dict)
    paired_success: dict[tuple[int, str, int, int], dict[str, bool]] = defaultdict(dict)
    for episode in episode_rows:
        key = (
            int(episode["policy_seed"]),
            str(episode["task_suite"]),
            int(episode["task_id"]),
            int(episode["trial_idx"]),
        )
        mode = str(episode["action_mode"])
        paired[key][mode] = float(episode["mean_candidate_rewards"][candidate])
        paired_success[key][mode] = bool(episode["success"])

    fully_paired = [scores for scores in paired.values() if {"policy", "noise", "zero"} <= scores.keys()]
    success_failure_pairs: list[bool] = []
    for key, scores in paired.items():
        modes = sorted(scores)
        for left_index, left_mode in enumerate(modes):
            for right_mode in modes[left_index + 1 :]:
                left_success = paired_success[key][left_mode]
                right_success = paired_success[key][right_mode]
                if left_success == right_success:
                    continue
                successful_mode = left_mode if left_success else right_mode
                failed_mode = right_mode if left_success else left_mode
                success_failure_pairs.append(scores[successful_mode] > scores[failed_mode])

    wrong_comparisons = [
        bool(row["candidate_correct_beats_wrong"][candidate])
        for row in rows
        if "candidate_correct_beats_wrong" in row
    ]
    labels = [bool(episode["success"]) for episode in episode_rows]
    scores = [float(episode["mean_candidate_rewards"][candidate]) for episode in episode_rows]
    return {
        "mean_transition_reward_by_action_mode": {
            mode: _mean(values) for mode, values in sorted(transition_by_mode.items())
        },
        "mean_episode_reward_by_action_mode": {
            mode: _mean(values) for mode, values in sorted(episode_by_mode.items())
        },
        "mean_episode_reward_by_success": {
            success: _mean(values) for success, values in sorted(episode_by_success.items())
        },
        "episode_success_roc_auc": binary_auc(labels, scores),
        "correct_goal_beats_wrong_fraction": _mean(
            [float(value) for value in wrong_comparisons]
        ),
        "num_wrong_goal_comparisons": len(wrong_comparisons),
        "num_fully_paired_trials": len(fully_paired),
        "paired_policy_gt_noise_gt_zero_fraction": _mean(
            [float(scores["policy"] > scores["noise"] > scores["zero"]) for scores in fully_paired]
        ),
        "successful_mode_beats_failed_mode_fraction": _mean(
            [float(value) for value in success_failure_pairs]
        ),
        "num_success_failure_mode_pairs": len(success_failure_pairs),
    }


def build_phase_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["policy_seed"]), str(row["action_mode"]), str(row["phase"]))].append(row)
    result: list[dict[str, Any]] = []
    for (seed, mode, phase), phase_rows in sorted(grouped.items()):
        result.append(
            {
                "policy_seed": seed,
                "action_mode": mode,
                "phase": phase,
                "num_transitions": len(phase_rows),
                "mean_candidate_rewards": {
                    candidate: _mean(
                        [float(row["candidate_rewards"][candidate]) for row in phase_rows]
                    )
                    for candidate in CANDIDATE_NAMES
                },
            }
        )
    return result


def build_camera_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for view in ("concat", *LIBERO_CAMERA_NAMES):
            grouped[(int(row["policy_seed"]), str(row["action_mode"]), view)].append(row)
    result: list[dict[str, Any]] = []
    metric_names = (
        "imagination_progress",
        "actual_change_norm",
        "imagined_change_norm",
        "direction_alignment",
        "magnitude_ratio",
        "delta_alignment_reward",
    )
    for (seed, mode, view), view_rows in sorted(grouped.items()):
        result.append(
            {
                "policy_seed": seed,
                "action_mode": mode,
                "view": view,
                "num_transitions": len(view_rows),
                **{
                    f"mean_{metric}": _mean(
                        [float(row["view_metrics"][view][metric]) for row in view_rows]
                    )
                    for metric in metric_names
                },
            }
        )
    return result


def build_paired_trial_table(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for episode in episode_rows:
        grouped[(int(episode["policy_seed"]), int(episode["trial_idx"]))][
            str(episode["action_mode"])
        ] = episode
    result: list[dict[str, Any]] = []
    for (seed, trial), modes in sorted(grouped.items()):
        result.append(
            {
                "policy_seed": seed,
                "trial_idx": trial,
                "success_by_mode": {
                    mode: bool(episode["success"]) for mode, episode in sorted(modes.items())
                },
                "candidate_rewards_by_mode": {
                    candidate: {
                        mode: float(episode["mean_candidate_rewards"][candidate])
                        for mode, episode in sorted(modes.items())
                    }
                    for candidate in CANDIDATE_NAMES
                },
            }
        )
    return result


def plot_diagnostic_trials(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    seed: int,
    trials: list[int],
) -> list[str]:
    import matplotlib.pyplot as plt

    paths: list[str] = []
    colors = {"policy": "tab:blue", "noise": "tab:orange", "zero": "tab:green"}
    for trial in trials:
        selected = [
            row
            for row in rows
            if int(row["policy_seed"]) == seed and int(row["trial_idx"]) == trial
        ]
        if not selected:
            continue
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
        for axis, candidate in zip(axes.flat, CANDIDATE_NAMES):
            for mode in ("policy", "noise", "zero"):
                mode_rows = sorted(
                    [row for row in selected if row["action_mode"] == mode],
                    key=lambda row: int(row["replan_idx"]),
                )
                if not mode_rows:
                    continue
                x = [int(row["replan_idx"]) for row in mode_rows]
                y = [float(row["candidate_rewards"][candidate]) for row in mode_rows]
                axis.plot(x, y, marker=".", markersize=3, linewidth=1, color=colors[mode], label=mode)
            axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
            axis.set_title(candidate)
            axis.set_xlabel("replan index")
            axis.set_ylabel("transition reward")
            axis.grid(alpha=0.2)
        axes.flat[0].legend()
        figure.suptitle(f"LIBERO Reward V2 diagnostic: seed={seed}, trial={trial}")
        figure.tight_layout()
        path = output_dir / f"seed{seed}_trial{trial}_reward_curves.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.direction_weight < 0:
        raise ValueError("--direction-weight must be non-negative")

    records = discover_records(args.input_dir)
    if not records:
        raise ValueError("No alignment-valid imagination transition records were found.")
    phases = assign_temporal_phases(records)
    image_paths = [
        path
        for record in records
        for path in (record["current_path"], record["goal_path"], record["actual_path"])
    ]

    cache_path = Path(args.feature_cache) if args.feature_cache else None
    features = None
    if cache_path is not None and not args.rebuild_cache:
        features = load_feature_cache(cache_path, image_paths)
    if features is None:
        features = {"concat": {}, "agent": {}, "wrist": {}}
        for input_dir in args.input_dir:
            input_root = str(Path(input_dir).resolve())
            group_records = [
                record for record in records if record["source_input_dir"] == input_root
            ]
            group_paths = [
                path
                for record in group_records
                for path in (record["current_path"], record["goal_path"], record["actual_path"])
            ]
            group_features = encode_camera_features(
                group_paths,
                encoder_path=args.encoder_path,
                device=args.device,
                batch_size=args.batch_size,
            )
            for view in features:
                features[view].update(group_features[view])
        if cache_path is not None:
            save_feature_cache(cache_path, image_paths, features)

    rows: list[dict[str, Any]] = []
    records_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_seed[int(record.get("policy_seed", -1))].append(record)
    for record in records:
        metrics = compute_candidate_metrics(
            features,
            current_path=record["current_path"],
            actual_path=record["actual_path"],
            goal_path=record["goal_path"],
            direction_weight=args.direction_weight,
        )
        row = dict(record)
        row["phase"] = phases[str(record["record_dir"])]
        row["view_metrics"] = metrics["views"]
        row["candidate_rewards"] = metrics["candidates"]

        wrong_record = select_wrong_goal_record(
            record,
            records_by_seed[int(record.get("policy_seed", -1))],
        )
        if wrong_record is not None:
            wrong_metrics = compute_candidate_metrics(
                features,
                current_path=record["current_path"],
                actual_path=record["actual_path"],
                goal_path=wrong_record["goal_path"],
                direction_weight=args.direction_weight,
            )
            row["wrong_goal_record_dir"] = wrong_record["record_dir"]
            row["wrong_candidate_rewards"] = wrong_metrics["candidates"]
            row["candidate_correct_beats_wrong"] = {
                candidate: bool(
                    metrics["candidates"][candidate] > wrong_metrics["candidates"][candidate]
                )
                for candidate in CANDIDATE_NAMES
            }
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "reward_v2_transitions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    episode_rows = build_episode_rows(rows)
    with (output_dir / "reward_v2_episodes.jsonl").open("w", encoding="utf-8") as stream:
        for episode in episode_rows:
            stream.write(json.dumps(episode, ensure_ascii=False) + "\n")

    summary = {
        "status": "offline_diagnostic_only",
        "num_transitions": len(rows),
        "num_episodes": len(episode_rows),
        "camera_layout": "horizontal: agent | wrist; each view is 224x224",
        "direction_reward_formula": (
            "cos(actual-current, goal-current) * "
            "min(||actual-current|| / ||goal-current||, 1)"
        ),
        "direction_weight": args.direction_weight,
        "candidate_definitions": {
            "concat_progress": "d(current, goal) - d(actual, goal)",
            "agent_delta_alignment": "magnitude-weighted delta alignment on agent camera",
            "dual_delta_alignment": "equal mean of agent and wrist delta alignment",
            "concat_progress_plus_dual_delta": (
                f"concat_progress + {args.direction_weight} * dual_delta_alignment"
            ),
        },
        "candidate_summaries": {
            candidate: summarize_candidate(candidate, rows, episode_rows)
            for candidate in CANDIDATE_NAMES
        },
        "candidate_summaries_by_seed": {
            str(seed): {
                candidate: summarize_candidate(
                    candidate,
                    [row for row in rows if int(row["policy_seed"]) == seed],
                    [
                        episode
                        for episode in episode_rows
                        if int(episode["policy_seed"]) == seed
                    ],
                )
                for candidate in CANDIDATE_NAMES
            }
            for seed in sorted({int(row["policy_seed"]) for row in rows})
        },
        "phase_summary": build_phase_summary(rows),
        "camera_summary": build_camera_summary(rows),
        "paired_trial_table": build_paired_trial_table(episode_rows),
        "wrong_goal_selection": (
            "same_task_different_episode_farthest_replan_prefer_same_action_mode"
        ),
        "input_dirs": [str(Path(path).resolve()) for path in args.input_dir],
        "encoder_path": str(Path(args.encoder_path).resolve()),
    }
    summary["diagnostic_plot_paths"] = plot_diagnostic_trials(
        rows,
        output_dir,
        seed=args.diagnostic_seed,
        trials=args.diagnostic_trials,
    )
    with (output_dir / "reward_v2_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
