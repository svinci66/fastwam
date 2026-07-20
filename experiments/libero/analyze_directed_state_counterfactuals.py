"""Analyze matched toward/away exact-state LIBERO counterfactuals."""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from experiments.libero.analyze_exact_state_counterfactuals import (
    _build_reward_rows,
    _encode_items,
    _image_items,
    _load_feature_cache,
    _load_frozen_camera_weight,
    _normalized_mean,
    bootstrap_mean_ci,
    spearman_correlation,
)


BRANCH_NAMES = ("policy", "toward_bowl", "away_from_bowl", "zero")
FORMAL_NUM_ANCHORS = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--camera-calibration-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2042)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    return parser.parse_args()


def validate_directed_collection(manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    source_policy_state_errors: list[float] = []
    exec_steps = int(manifest["exec_steps"])
    cap = float(manifest["translation_magnitude_cap"])
    for anchor in manifest["anchors"]:
        anchor_index = int(anchor["anchor_index"])
        base = np.load(anchor["base_actions_path"])
        direction = np.load(anchor["direction_path"])
        branches = {branch["name"]: branch for branch in anchor["branches"]}
        if tuple(branches) != BRANCH_NAMES:
            errors.append(f"anchor {anchor_index}: unexpected branch order {tuple(branches)}")
            continue
        if base.shape != (exec_steps, 7) or direction.shape != (3,):
            errors.append(f"anchor {anchor_index}: unexpected base/direction shape")
            continue
        if not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-7):
            errors.append(f"anchor {anchor_index}: direction is not unit length")

        actions = {name: np.load(branches[name]["actions_path"]) for name in BRANCH_NAMES}
        for name, branch_actions in actions.items():
            if branch_actions.shape != base.shape:
                errors.append(f"anchor {anchor_index}/{name}: action shape mismatch")
            if float(
                branches[name].get("reconstructed_anchor_state_max_abs_diff", float("inf"))
            ) > 1e-10:
                errors.append(f"anchor {anchor_index}/{name}: exact restore failed")
            if len(branches[name].get("current_paths", [])) != int(
                manifest["render_repeats"]
            ):
                errors.append(f"anchor {anchor_index}/{name}: missing current renders")

        if not np.array_equal(actions["policy"], base):
            errors.append(f"anchor {anchor_index}: policy differs from source")
        toward = actions["toward_bowl"]
        away = actions["away_from_bowl"]
        if not np.allclose(toward[:, :3], -away[:, :3], rtol=0.0, atol=1e-7):
            errors.append(f"anchor {anchor_index}: directions are not exact opposites")
        if not np.allclose(
            np.linalg.norm(toward[:, :3], axis=1),
            np.linalg.norm(away[:, :3], axis=1),
            rtol=0.0,
            atol=1e-7,
        ):
            errors.append(f"anchor {anchor_index}: translation magnitudes differ")
        expected_magnitudes = np.minimum(np.linalg.norm(base[:, :3], axis=1), cap)
        if not np.allclose(
            np.linalg.norm(toward[:, :3], axis=1),
            expected_magnitudes,
            rtol=0.0,
            atol=1e-7,
        ):
            errors.append(f"anchor {anchor_index}: wrong capped translation magnitudes")
        for name in ("toward_bowl", "away_from_bowl"):
            if not np.array_equal(actions[name][:, 3:], base[:, 3:]):
                errors.append(f"anchor {anchor_index}/{name}: rotation or gripper changed")
        zero = actions["zero"]
        if not (
            np.array_equal(zero[:, :-1], np.zeros_like(zero[:, :-1]))
            and np.array_equal(zero[:, -1], -np.ones(exec_steps))
        ):
            errors.append(f"anchor {anchor_index}: invalid zero action")

        for name, branch in branches.items():
            geometry = branch["geometry"]
            expected_progress = (
                float(geometry["initial"]["eef_target_distance"])
                - float(geometry["final"]["eef_target_distance"])
            )
            if not np.isclose(
                expected_progress,
                float(geometry["eef_target_distance_progress"]),
                atol=1e-10,
            ):
                errors.append(f"anchor {anchor_index}/{name}: invalid geometry progress")

        source_policy_paths = anchor.get("source_policy_actual_paths", [])
        if len(source_policy_paths) != int(manifest["render_repeats"]):
            errors.append(f"anchor {anchor_index}: missing source policy renders")
        source_policy_state_path = anchor.get("source_policy_final_state_path")
        if not source_policy_state_path:
            errors.append(f"anchor {anchor_index}: missing source policy state")
        else:
            source_state = np.load(source_policy_state_path)
            new_state = np.load(branches["policy"]["final_state_path"])
            state_error = float(np.max(np.abs(source_state - new_state)))
            source_policy_state_errors.append(state_error)
            if state_error > 1e-7:
                errors.append(
                    f"anchor {anchor_index}: source policy state mismatch {state_error}"
                )

    return {
        "passed": not errors,
        "errors": errors,
        "num_anchors": len(manifest["anchors"]),
        "formal_num_anchors": FORMAL_NUM_ANCHORS,
        "formal_anchor_count_passed": len(manifest["anchors"]) == FORMAL_NUM_ANCHORS,
        "num_branches": sum(len(anchor["branches"]) for anchor in manifest["anchors"]),
        "simulator_action_steps": sum(
            len(np.load(branch["actions_path"]))
            for anchor in manifest["anchors"]
            for branch in anchor["branches"]
        ),
        "max_source_policy_final_state_abs_diff": (
            None
            if not source_policy_state_errors
            else max(source_policy_state_errors)
        ),
    }


def _directed_image_items(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    items = _image_items(manifest)
    for anchor in manifest["anchors"]:
        anchor_index = int(anchor["anchor_index"])
        for repeat_index, paths in enumerate(anchor["source_policy_actual_paths"]):
            items[f"a{anchor_index:02d}/source_policy/actual{repeat_index:02d}"] = paths
    return items


def _source_policy_feature_stability(
    manifest: dict[str, Any],
    features: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    cosine_distances = []
    feature_l2 = []
    for anchor in manifest["anchors"]:
        anchor_index = int(anchor["anchor_index"])
        prefix = f"a{anchor_index:02d}"
        new_count = len(next(
            branch["actual_paths"]
            for branch in anchor["branches"]
            if branch["name"] == "policy"
        ))
        source_count = len(anchor["source_policy_actual_paths"])
        for view in ("concat", "agent", "wrist"):
            new_mean = _normalized_mean(
                [
                    features[view][f"{prefix}/policy/actual{index:02d}"]
                    for index in range(new_count)
                ]
            )
            source_mean = _normalized_mean(
                [
                    features[view][f"{prefix}/source_policy/actual{index:02d}"]
                    for index in range(source_count)
                ]
            )
            cosine_distances.append(float(1.0 - np.dot(new_mean, source_mean)))
            feature_l2.append(float(np.linalg.norm(new_mean - source_mean)))
    max_cosine = max(cosine_distances)
    max_l2 = max(feature_l2)
    return {
        "mean_cosine_distance": float(np.mean(cosine_distances)),
        "max_cosine_distance": max_cosine,
        "max_feature_l2": max_l2,
        "thresholds": {
            "max_cosine_distance": 1e-4,
            "max_feature_l2": 0.015,
        },
        "passed": max_cosine <= 1e-4 and max_l2 <= 0.015,
    }


def _attach_geometry(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    geometry_by_key = {
        (int(anchor["anchor_index"]), branch["name"]): branch["geometry"]
        for anchor in manifest["anchors"]
        for branch in anchor["branches"]
    }
    for row in rows:
        geometry = geometry_by_key[(int(row["anchor_index"]), str(row["branch"]))]
        row["geometry"] = geometry
        row["quality"] = float(geometry["eef_target_distance_progress"])


def _load_hybrid_calibration(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    metrics = summary["metrics"]
    raw_scale = abs(float(metrics["raw_dual"]["mean_reward_by_branch"]["policy"]))
    progress_scale = abs(
        float(metrics["concat_progress"]["mean_reward_by_branch"]["policy"])
    )
    if raw_scale <= 0.0 or progress_scale <= 0.0:
        raise ValueError("Hybrid calibration scales must be positive")
    return {
        "raw_dual_scale": raw_scale,
        "concat_progress_scale": progress_scale,
        "weights": {"raw_dual": 0.5, "concat_progress": 0.5},
        "source": str(path.resolve()),
    }


def _attach_hybrid_reward(
    rows: list[dict[str, Any]], calibration: dict[str, Any]
) -> None:
    for row in rows:
        row["reward"]["normalized_equal_hybrid"] = 0.5 * (
            row["reward"]["raw_dual"] / calibration["raw_dual_scale"]
            + row["reward"]["concat_progress"]
            / calibration["concat_progress_scale"]
        )


def summarize_directed_metric(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    grouped: dict[int, dict[str, dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["anchor_index"]), {})[str(row["branch"])] = {
            "reward": float(row["reward"][metric]),
            "progress": float(row["geometry"]["eef_target_distance_progress"]),
        }
    anchors = [grouped[index] for index in sorted(grouped)]
    paired_differences = [
        anchor["toward_bowl"]["reward"] - anchor["away_from_bowl"]["reward"]
        for anchor in anchors
    ]
    spearman = [
        spearman_correlation(
            [anchor[name]["progress"] for name in BRANCH_NAMES],
            [anchor[name]["reward"] for name in BRANCH_NAMES],
        )
        for anchor in anchors
    ]
    policy_abs = float(
        np.mean([abs(anchor["policy"]["reward"]) for anchor in anchors])
    )
    zero_abs = float(np.mean([abs(anchor["zero"]["reward"]) for anchor in anchors]))
    positive_threshold = ceil(0.8 * len(anchors))
    return {
        "mean_reward_by_branch": {
            name: float(np.mean([anchor[name]["reward"] for anchor in anchors]))
            for name in BRANCH_NAMES
        },
        "toward_minus_away_per_anchor": paired_differences,
        "toward_beats_away_count": int(sum(value > 0.0 for value in paired_differences)),
        "required_positive_count": positive_threshold,
        "toward_minus_away_bootstrap": bootstrap_mean_ci(
            paired_differences, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "reward_geometry_spearman_per_anchor": spearman,
        "reward_geometry_spearman_positive_count": int(
            sum(value > 0.0 for value in spearman)
        ),
        "reward_geometry_spearman_bootstrap": bootstrap_mean_ci(
            spearman, samples=bootstrap_samples, seed=bootstrap_seed + 1
        ),
        "zero_to_policy_abs_reward_ratio": (
            float("inf") if policy_abs == 0.0 else zero_abs / policy_abs
        ),
    }


def summarize_geometry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[str, float]] = {}
    for row in rows:
        grouped.setdefault(int(row["anchor_index"]), {})[str(row["branch"])] = float(
            row["geometry"]["eef_target_distance_progress"]
        )
    anchors = [grouped[index] for index in sorted(grouped)]
    toward_away = [a["toward_bowl"] - a["away_from_bowl"] for a in anchors]
    return {
        "mean_eef_target_progress_by_branch_m": {
            name: float(np.mean([anchor[name] for anchor in anchors]))
            for name in BRANCH_NAMES
        },
        "toward_progress_positive_count": int(sum(a["toward_bowl"] > 0 for a in anchors)),
        "away_progress_negative_count": int(sum(a["away_from_bowl"] < 0 for a in anchors)),
        "toward_progress_exceeds_away_count": int(sum(value > 0 for value in toward_away)),
        "toward_minus_away_progress_bootstrap_m": bootstrap_mean_ci(
            toward_away, samples=10_000, seed=20_420
        ),
    }


def directed_primary_gates(
    metric_summary: dict[str, Any],
    geometry_summary: dict[str, Any],
    *,
    num_anchors: int,
) -> dict[str, bool]:
    required = ceil(0.8 * num_anchors)
    return {
        "formal_anchor_count_is_10": num_anchors == FORMAL_NUM_ANCHORS,
        "toward_geometry_positive_at_least_9_of_10": (
            geometry_summary["toward_progress_positive_count"] >= 9
        ),
        "away_geometry_negative_at_least_9_of_10": (
            geometry_summary["away_progress_negative_count"] >= 9
        ),
        "toward_geometry_beats_away_10_of_10": (
            geometry_summary["toward_progress_exceeds_away_count"] == num_anchors
        ),
        "reward_toward_beats_away_at_least_8_of_10": (
            metric_summary["toward_beats_away_count"] >= required
        ),
        "reward_toward_minus_away_bootstrap_lower_positive": (
            metric_summary["toward_minus_away_bootstrap"]["ci95_lower"] > 0.0
        ),
        "reward_geometry_spearman_positive_at_least_8_of_10": (
            metric_summary["reward_geometry_spearman_positive_count"] >= required
        ),
        "reward_geometry_spearman_bootstrap_lower_positive": (
            metric_summary["reward_geometry_spearman_bootstrap"]["ci95_lower"] > 0.0
        ),
        "zero_abs_reward_at_most_5pct_policy": (
            metric_summary["zero_to_policy_abs_reward_ratio"] <= 0.05
        ),
    }


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text())
    integrity = validate_directed_collection(manifest)
    if not integrity["passed"]:
        raise ValueError(f"Collection integrity failed: {integrity['errors']}")

    items = _directed_image_items(manifest)
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
                for view in ("concat", "agent", "wrist")
            },
        )

    frozen_weight = _load_frozen_camera_weight(args.camera_calibration_json)
    rows, render_stability, start_stability = _build_reward_rows(
        manifest, features, frozen_weight
    )
    if not start_stability["passed"]:
        raise ValueError(f"Branch anchor feature stability failed: {start_stability}")
    _attach_geometry(rows, manifest)
    source_policy_stability = _source_policy_feature_stability(manifest, features)
    if not source_policy_stability["passed"]:
        raise ValueError(
            "New policy endpoints do not reproduce source policy endpoint features: "
            f"{source_policy_stability}"
        )
    hybrid_calibration = _load_hybrid_calibration(args.calibration_summary)
    _attach_hybrid_reward(rows, hybrid_calibration)

    metric_names = list(rows[0]["reward"])
    metric_summaries = {
        metric: summarize_directed_metric(
            rows,
            metric,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for metric in metric_names
    }
    geometry_summary = summarize_geometry(rows)
    gates = directed_primary_gates(
        metric_summaries["raw_dual"],
        geometry_summary,
        num_anchors=len(manifest["anchors"]),
    )
    gates["source_policy_endpoint_feature_alignment"] = source_policy_stability["passed"]
    passed = (
        integrity["passed"]
        and start_stability["passed"]
        and source_policy_stability["passed"]
        and all(gates.values())
    )
    summary = {
        "schema_version": 1,
        "experiment": manifest["experiment"],
        "decision": (
            "supports_directed_imagination_reward_hypothesis"
            if passed
            else "does_not_yet_support_directed_imagination_reward_hypothesis"
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
            "translation_magnitude_cap": manifest["translation_magnitude_cap"],
            "branches": list(BRANCH_NAMES),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_ci": "paired-anchor percentile 95%",
            "no_model_inference": manifest["total_model_inferences"] == 0,
        },
        "collection_integrity": integrity,
        "render_feature_stability": render_stability,
        "branch_anchor_feature_stability": start_stability,
        "source_policy_endpoint_feature_stability": source_policy_stability,
        "geometry_control": geometry_summary,
        "hybrid_calibration": hybrid_calibration,
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
