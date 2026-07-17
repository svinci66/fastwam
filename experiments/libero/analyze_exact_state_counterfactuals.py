"""Analyze exact-state LIBERO counterfactuals with frozen Reward V2 features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from experiments.libero.imagination_reward_utils import (
    compute_delta_alignment_reward,
    compute_progress_reward,
)


CAMERAS = ("agent", "wrist")
QUALITY_BRANCHES = ("policy", "noise_0.075", "noise_0.150", "noise_0.300")
EXPECTED_BRANCHES = (*QUALITY_BRANCHES, "zero")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-calibration-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2042)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    return parser.parse_args()


def _normalized_mean(features: list[np.ndarray]) -> np.ndarray:
    mean = np.mean(np.asarray(features, dtype=np.float32), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero feature mean")
    return mean / norm


def _rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation requires equal lists with at least two values")
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or samples <= 0:
        raise ValueError("Bootstrap requires non-empty values and positive samples")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    bootstrap_means = np.mean(array[indices], axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci95_lower": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_upper": float(np.quantile(bootstrap_means, 0.975)),
        "samples": samples,
    }


def _validate_collection(manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    exec_steps = int(manifest["exec_steps"])
    noise_stds = tuple(float(value) for value in manifest["noise_stds"])
    for anchor in manifest["anchors"]:
        base = np.load(anchor["base_actions_path"])
        epsilon = np.load(anchor["shared_noise_path"])
        branches = {branch["name"]: branch for branch in anchor["branches"]}
        if tuple(branches) != EXPECTED_BRANCHES:
            errors.append(
                f"anchor {anchor['anchor_index']}: branch order/names are {tuple(branches)}"
            )
        if base.shape != (exec_steps, 7) or epsilon.shape != (exec_steps, 6):
            errors.append(f"anchor {anchor['anchor_index']}: unexpected action/noise shape")
            continue
        for name, branch in branches.items():
            if float(branch.get("reconstructed_anchor_state_max_abs_diff", float("inf"))) > 1e-10:
                errors.append(
                    f"anchor {anchor['anchor_index']}/{name}: exact state reconstruction failed"
                )
            if len(branch.get("current_paths", [])) != int(manifest["render_repeats"]):
                errors.append(
                    f"anchor {anchor['anchor_index']}/{name}: missing branch-current renders"
                )
            actions = np.load(branch["actions_path"])
            if actions.shape != base.shape:
                errors.append(f"anchor {anchor['anchor_index']}/{name}: action shape mismatch")
                continue
            if name == "policy" and not np.array_equal(actions, base):
                errors.append(f"anchor {anchor['anchor_index']}: policy differs from base")
            elif name.startswith("noise_"):
                std = float(name.removeprefix("noise_"))
                expected = base.copy()
                expected[:, :-1] = np.clip(base[:, :-1] + std * epsilon, -1.0, 1.0)
                if not np.allclose(actions, expected, rtol=0.0, atol=1e-7):
                    errors.append(f"anchor {anchor['anchor_index']}/{name}: wrong shared noise")
                if not np.array_equal(actions[:, -1], base[:, -1]):
                    errors.append(f"anchor {anchor['anchor_index']}/{name}: gripper changed")
            elif name == "zero":
                if not (
                    np.array_equal(actions[:, :-1], np.zeros_like(actions[:, :-1]))
                    and np.array_equal(actions[:, -1], -np.ones(exec_steps))
                ):
                    errors.append(f"anchor {anchor['anchor_index']}: invalid zero action")
        if len(noise_stds) != 3:
            errors.append("expected exactly three noise levels")
    return {
        "passed": not errors,
        "errors": errors,
        "num_anchors": len(manifest["anchors"]),
        "num_branches": sum(len(anchor["branches"]) for anchor in manifest["anchors"]),
        "simulator_action_steps": sum(
            len(np.load(branch["actions_path"]))
            for anchor in manifest["anchors"]
            for branch in anchor["branches"]
        ),
    }


def _image_items(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    items: dict[str, dict[str, str]] = {}
    for anchor in manifest["anchors"]:
        anchor_index = int(anchor["anchor_index"])
        reward_current_paths = anchor.get("reward_current_paths")
        if reward_current_paths:
            for repeat_index, paths in enumerate(reward_current_paths):
                items[f"a{anchor_index:02d}/current{repeat_index:02d}"] = paths
        else:
            items[f"a{anchor_index:02d}/current"] = anchor["current_paths"]
        items[f"a{anchor_index:02d}/goal"] = anchor["predicted_goal_paths"]
        for branch in anchor["branches"]:
            for repeat_index, paths in enumerate(branch.get("current_paths", [])):
                items[
                    f"a{anchor_index:02d}/{branch['name']}/current{repeat_index:02d}"
                ] = paths
            for repeat_index, paths in enumerate(branch["actual_paths"]):
                items[
                    f"a{anchor_index:02d}/{branch['name']}/actual{repeat_index:02d}"
                ] = paths
    return items


def _encode_items(
    items: dict[str, dict[str, str]],
    *,
    encoder_path: str,
    device: str,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    from transformers import SiglipImageProcessor, SiglipVisionModel

    processor = SiglipImageProcessor.from_pretrained(encoder_path, local_files_only=True)
    model = SiglipVisionModel.from_pretrained(
        encoder_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    keys = list(items)
    output: dict[str, dict[str, np.ndarray]] = {
        view: {} for view in ("concat", *CAMERAS)
    }
    with torch.inference_mode():
        for view in output:
            for start in range(0, len(keys), batch_size):
                batch_keys = keys[start : start + batch_size]
                images = []
                for key in batch_keys:
                    camera_images = {
                        camera: np.asarray(
                            Image.open(items[key][camera]).convert("RGB"), dtype=np.uint8
                        )
                        for camera in CAMERAS
                    }
                    selected = (
                        camera_images[view]
                        if view in CAMERAS
                        else np.concatenate([camera_images[camera] for camera in CAMERAS], axis=1)
                    )
                    images.append(Image.fromarray(selected))
                inputs = processor(images=images, return_tensors="pt")
                encoded = model(pixel_values=inputs["pixel_values"].to(device)).pooler_output
                encoded = torch.nn.functional.normalize(encoded.float(), dim=-1).cpu().numpy()
                for key, feature in zip(batch_keys, encoded):
                    output[view][key] = feature
    return output


def _load_feature_cache(
    cache_path: Path, keys: list[str]
) -> dict[str, dict[str, np.ndarray]] | None:
    if not cache_path.is_file():
        return None
    with np.load(cache_path) as cache:
        cached_keys = [str(value) for value in cache["keys"].tolist()]
        if cached_keys != keys:
            return None
        return {
            view: {
                key: feature
                for key, feature in zip(keys, np.asarray(cache[view], dtype=np.float32))
            }
            for view in ("concat", *CAMERAS)
        }


def _load_frozen_camera_weight(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    calibration = json.loads(path.read_text())
    selection = calibration["selection"]
    return {
        "agent_weight": float(selection["selected_agent_weight"]),
        "wrist_weight": float(selection["selected_wrist_weight"]),
        "scales": {
            camera: float(calibration["camera_scales"][camera]) for camera in CAMERAS
        },
        "source": str(path.resolve()),
    }


def _build_reward_rows(
    manifest: dict[str, Any],
    features: dict[str, dict[str, np.ndarray]],
    frozen_weight: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    rows = []
    render_distances = []
    start_cosine_distances = []
    start_feature_l2 = []
    for anchor in manifest["anchors"]:
        anchor_index = int(anchor["anchor_index"])
        prefix = f"a{anchor_index:02d}"
        for branch in anchor["branches"]:
            camera_rewards = {}
            progress_rewards = {}
            view_metrics = {}
            for view in ("concat", *CAMERAS):
                branch_current_paths = branch.get("current_paths")
                reward_current_paths = anchor.get("reward_current_paths")
                if branch_current_paths:
                    current_features = [
                        features[view][
                            f"{prefix}/{branch['name']}/current{repeat_index:02d}"
                        ]
                        for repeat_index in range(len(branch_current_paths))
                    ]
                    current = _normalized_mean(current_features)
                    if reward_current_paths:
                        reference_current = _normalized_mean(
                            [
                                features[view][f"{prefix}/current{repeat_index:02d}"]
                                for repeat_index in range(len(reward_current_paths))
                            ]
                        )
                        start_cosine_distances.append(
                            float(1.0 - np.dot(current, reference_current))
                        )
                        start_feature_l2.append(
                            float(np.linalg.norm(current - reference_current))
                        )
                    render_distances.extend(
                        float(1.0 - np.dot(feature, current))
                        for feature in current_features
                    )
                elif reward_current_paths:
                    current_features = [
                        features[view][f"{prefix}/current{repeat_index:02d}"]
                        for repeat_index in range(len(reward_current_paths))
                    ]
                    current = _normalized_mean(current_features)
                    render_distances.extend(
                        float(1.0 - np.dot(feature, current))
                        for feature in current_features
                    )
                else:
                    current = features[view][f"{prefix}/current"]
                actual_features = [
                    features[view][f"{prefix}/{branch['name']}/actual{repeat_index:02d}"]
                    for repeat_index in range(len(branch["actual_paths"]))
                ]
                actual_mean = _normalized_mean(actual_features)
                render_distances.extend(
                    float(1.0 - np.dot(feature, actual_mean))
                    for feature in actual_features
                )
                goal = features[view][f"{prefix}/goal"]
                progress_details = compute_progress_reward(current, actual_mean, goal)
                progress_rewards[view] = progress_details["imagination_progress"]
                view_metrics[view] = {"progress": progress_details}
                if view in CAMERAS:
                    delta_details = compute_delta_alignment_reward(
                        current, actual_mean, goal
                    )
                    camera_rewards[view] = delta_details["delta_alignment_reward"]
                    view_metrics[view]["delta_alignment"] = delta_details

            reward_metrics = {
                "raw_dual": 0.5 * (camera_rewards["agent"] + camera_rewards["wrist"]),
                "agent": camera_rewards["agent"],
                "wrist": camera_rewards["wrist"],
                "concat_progress": progress_rewards["concat"],
                "dual_progress": 0.5
                * (progress_rewards["agent"] + progress_rewards["wrist"]),
            }
            if frozen_weight is not None:
                reward_metrics["frozen_camera_weight"] = sum(
                    frozen_weight[f"{camera}_weight"]
                    * camera_rewards[camera]
                    / frozen_weight["scales"][camera]
                    for camera in CAMERAS
                )
            rows.append(
                {
                    "anchor_index": anchor_index,
                    "branch": branch["name"],
                    "noise_std": branch["noise_std"],
                    "quality": branch["quality"],
                    "success": branch["success"],
                    "reward": reward_metrics,
                    "view_metrics": view_metrics,
                }
            )
    return (
        rows,
        {
            "mean_individual_to_ensemble_cosine_distance": float(np.mean(render_distances)),
            "max_individual_to_ensemble_cosine_distance": float(np.max(render_distances)),
        },
        {
            "mean_cosine_distance": (
                None if not start_cosine_distances else float(np.mean(start_cosine_distances))
            ),
            "max_cosine_distance": (
                None if not start_cosine_distances else float(np.max(start_cosine_distances))
            ),
            "max_feature_l2": (
                None if not start_feature_l2 else float(np.max(start_feature_l2))
            ),
            "thresholds": {
                "max_cosine_distance": 1e-4,
                "max_feature_l2": 0.015,
            },
            "passed": bool(
                start_cosine_distances
                and max(start_cosine_distances) <= 1e-4
                and max(start_feature_l2) <= 0.015
            ),
        },
    )


def summarize_metric(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_anchor: dict[int, dict[str, float]] = {}
    for row in rows:
        by_anchor.setdefault(int(row["anchor_index"]), {})[str(row["branch"])] = float(
            row["reward"][metric]
        )
    ordered = [by_anchor[index] for index in sorted(by_anchor)]
    qualities = [0.0, -0.075, -0.15, -0.30]
    spearman = [
        spearman_correlation(qualities, [anchor[name] for name in QUALITY_BRANCHES])
        for anchor in ordered
    ]
    policy_zero = [anchor["policy"] - anchor["zero"] for anchor in ordered]
    noise_zero = [anchor["noise_0.150"] - anchor["zero"] for anchor in ordered]
    policy_abs_mean = float(np.mean([abs(anchor["policy"]) for anchor in ordered]))
    zero_abs_mean = float(np.mean([abs(anchor["zero"]) for anchor in ordered]))
    return {
        "mean_reward_by_branch": {
            name: float(np.mean([anchor[name] for anchor in ordered]))
            for name in EXPECTED_BRANCHES
        },
        "per_anchor_spearman": spearman,
        "spearman_positive_count": int(sum(value > 0.0 for value in spearman)),
        "spearman_mean_bootstrap": bootstrap_mean_ci(
            spearman, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "policy_beats_zero_count": int(sum(value > 0.0 for value in policy_zero)),
        "policy_minus_zero_bootstrap": bootstrap_mean_ci(
            policy_zero, samples=bootstrap_samples, seed=bootstrap_seed + 1
        ),
        "noise_0.150_beats_zero_count": int(sum(value > 0.0 for value in noise_zero)),
        "noise_0.150_minus_zero_bootstrap": bootstrap_mean_ci(
            noise_zero, samples=bootstrap_samples, seed=bootstrap_seed + 2
        ),
        "zero_to_policy_abs_reward_ratio": (
            float("inf") if policy_abs_mean == 0.0 else zero_abs_mean / policy_abs_mean
        ),
        "high_noise_beats_policy_count": int(
            sum(anchor["noise_0.300"] > anchor["policy"] for anchor in ordered)
        ),
        "high_noise_beats_mid_noise_count": int(
            sum(anchor["noise_0.300"] > anchor["noise_0.150"] for anchor in ordered)
        ),
    }


def _primary_gates(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "spearman_positive_at_least_8_of_10": summary["spearman_positive_count"] >= 8,
        "spearman_bootstrap_lower_positive": summary["spearman_mean_bootstrap"][
            "ci95_lower"
        ]
        > 0.0,
        "policy_beats_zero_at_least_9_of_10": summary["policy_beats_zero_count"] >= 9,
        "policy_minus_zero_bootstrap_lower_positive": summary[
            "policy_minus_zero_bootstrap"
        ]["ci95_lower"]
        > 0.0,
        "noise_0.150_beats_zero_at_least_8_of_10": summary[
            "noise_0.150_beats_zero_count"
        ]
        >= 8,
        "noise_0.150_minus_zero_bootstrap_lower_positive": summary[
            "noise_0.150_minus_zero_bootstrap"
        ]["ci95_lower"]
        > 0.0,
        "zero_abs_reward_at_most_5pct_policy": summary[
            "zero_to_policy_abs_reward_ratio"
        ]
        <= 0.05,
    }


def _change_norm_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = {}
    for camera in CAMERAS:
        diagnostics[camera] = {
            branch: float(
                np.mean(
                    [
                        row["view_metrics"][camera]["delta_alignment"][
                            "actual_change_norm"
                        ]
                        for row in rows
                        if row["branch"] == branch
                    ]
                )
            )
            for branch in EXPECTED_BRANCHES
        }
        policy = diagnostics[camera]["policy"]
        diagnostics[camera]["zero_to_policy_change_norm_ratio"] = (
            float("inf") if policy == 0.0 else diagnostics[camera]["zero"] / policy
        )
    return diagnostics


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text())
    integrity = _validate_collection(manifest)
    if not integrity["passed"]:
        raise ValueError(f"Collection integrity failed: {integrity['errors']}")
    items = _image_items(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = list(items)
    feature_cache_path = args.output_dir / "features.npz"
    features = None if args.rebuild_feature_cache else _load_feature_cache(
        feature_cache_path, keys
    )
    if features is None:
        features = _encode_items(
            items,
            encoder_path=args.encoder_path,
            device=args.device,
            batch_size=args.batch_size,
        )
        np.savez_compressed(
            feature_cache_path,
            keys=np.asarray(keys),
            **{
                view: np.stack([features[view][key] for key in keys])
                for view in ("concat", *CAMERAS)
            },
        )

    frozen_weight = _load_frozen_camera_weight(args.camera_calibration_json)
    rows, render_stability, start_stability = _build_reward_rows(
        manifest, features, frozen_weight
    )
    if not start_stability["passed"]:
        raise ValueError(
            "Branch reconstructions do not reproduce the same anchor image features: "
            f"{start_stability}"
        )
    metric_names = list(rows[0]["reward"])
    metric_summaries = {
        metric: summarize_metric(
            rows,
            metric,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for metric in metric_names
    }
    gates = _primary_gates(metric_summaries["raw_dual"])
    passed = integrity["passed"] and start_stability["passed"] and all(gates.values())
    summary = {
        "schema_version": 1,
        "experiment": manifest["experiment"],
        "decision": (
            "supports_exact_state_reward_hypothesis"
            if passed
            else "does_not_yet_support_exact_state_reward_hypothesis"
        ),
        "passed": passed,
        "primary_metric": "raw_dual",
        "primary_gates": gates,
        "protocol": {
            "seed": manifest["seed"],
            "task_suite": manifest["task_suite"],
            "task_id": manifest["task_id"],
            "num_independent_anchors": manifest["num_states"],
            "exec_steps": manifest["exec_steps"],
            "render_repeats": manifest["render_repeats"],
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_ci": "paired-anchor percentile 95%",
            "quality_order": dict(zip(QUALITY_BRANCHES, [0.0, -0.075, -0.15, -0.30])),
        },
        "collection_integrity": integrity,
        "render_feature_stability": render_stability,
        "branch_anchor_feature_stability": start_stability,
        "actual_change_norm_diagnostic": _change_norm_diagnostics(rows),
        "frozen_camera_weight_diagnostic": frozen_weight,
        "metrics": metric_summaries,
    }
    with (args.output_dir / "reward_rows.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
