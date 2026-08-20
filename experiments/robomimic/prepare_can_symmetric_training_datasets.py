#!/usr/bin/env python3
"""Convert symmetric RoboMimic branches into actor and pairwise-Q datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _text_array(values: np.ndarray, width: int) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=f"U{width}",
    )


def prepare_datasets(
    collection_path: str | Path,
    q_output_path: str | Path,
    actor_output_path: str | Path,
    *,
    residual_scale: float = 0.1,
    state_tolerance: float = 1e-10,
    actor_target_mode: str = "symmetric_gradient",
) -> dict[str, Any]:
    """Create independent Q-ranking and best-residual regression datasets.

    Q receives both candidate-vs-base comparisons and the harder direct
    comparisons between each symmetric +delta/-delta candidate pair. The actor
    receives exactly one target per state: the best candidate when its score
    clears the collector's margin, otherwise an exact zero residual.
    """
    collection_path = Path(collection_path).expanduser().resolve()
    q_output_path = Path(q_output_path).expanduser().resolve()
    actor_output_path = Path(actor_output_path).expanduser().resolve()
    if residual_scale <= 0:
        raise ValueError("residual_scale must be positive")
    if actor_target_mode not in {"best_candidate", "symmetric_gradient"}:
        raise ValueError(f"Unsupported actor_target_mode: {actor_target_mode}")

    with h5py.File(collection_path, "r") as source:
        if not bool(source.attrs.get("complete", False)):
            raise ValueError("Symmetric branch collection is not complete")
        if str(source.attrs.get("format", "")) != "fastwam.robomimic_symmetric_branches.v1":
            raise ValueError("Unsupported symmetric branch collection format")
        states = source["states"]
        count = int(source.attrs["states_committed"])
        if any(len(dataset) != count for dataset in states.values()):
            raise ValueError("Collection fields do not have a common committed length")

        state = np.asarray(states["branch_state"], dtype=np.float32)
        base = np.asarray(states["base_action_chunk"], dtype=np.float32)
        candidates = np.asarray(states["candidate_action_chunks"], dtype=np.float32)
        residuals = np.asarray(states["candidate_residual_chunks"], dtype=np.float32)
        candidate_scores = np.asarray(states["candidate_scores"], dtype=np.float32)
        delta_scores = np.asarray(states["delta_scores"], dtype=np.float32)
        split = _text_array(np.asarray(states["source_split"]), 5)
        demo = _text_array(np.asarray(states["source_demo"]), 32)
        step = np.asarray(states["source_step"], dtype=np.int32)
        score_margin = float(source.attrs["score_margin"])

        if candidates.shape[1] % 2:
            raise ValueError("Candidate count must contain complete symmetric pairs")
        finite_arrays = (state, base, candidates, residuals, candidate_scores, delta_scores)
        if not all(np.all(np.isfinite(array)) for array in finite_arrays):
            raise ValueError("Collection contains non-finite training values")
        if float(np.max(np.asarray(states["restore_linf"]), initial=0.0)) > state_tolerance:
            raise ValueError("Environment restore error exceeds tolerance")
        if (
            float(np.max(np.asarray(states["branch_initial_state_linf"]), initial=0.0))
            > state_tolerance
        ):
            raise ValueError("Branch initial-state mismatch exceeds tolerance")

    q_rows: dict[str, list[Any]] = {
        key: []
        for key in (
            "state",
            "base_action_chunk",
            "candidate_action_chunk",
            "label",
            "delta_score",
            "source_split",
            "source_demo",
            "source_step",
            "comparison_type",
        )
    }

    def append_q(
        index: int,
        first_action: np.ndarray,
        second_action: np.ndarray,
        delta: float,
        comparison_type: str,
    ) -> None:
        if abs(delta) <= score_margin:
            return
        q_rows["state"].append(state[index])
        q_rows["base_action_chunk"].append(first_action)
        q_rows["candidate_action_chunk"].append(second_action)
        q_rows["label"].append(1 if delta > 0 else -1)
        q_rows["delta_score"].append(delta)
        q_rows["source_split"].append(split[index])
        q_rows["source_demo"].append(demo[index])
        q_rows["source_step"].append(step[index])
        q_rows["comparison_type"].append(comparison_type)

    for index in range(count):
        for candidate_index in range(candidates.shape[1]):
            append_q(
                index,
                base[index],
                candidates[index, candidate_index],
                float(delta_scores[index, candidate_index]),
                "candidate_vs_base",
            )
        for pair_index in range(0, candidates.shape[1], 2):
            append_q(
                index,
                candidates[index, pair_index],
                candidates[index, pair_index + 1],
                float(candidate_scores[index, pair_index + 1] - candidate_scores[index, pair_index]),
                "symmetric_pair",
            )

    q_arrays = {
        "state": np.stack(q_rows["state"]).astype(np.float32),
        "base_action_chunk": np.stack(q_rows["base_action_chunk"]).astype(np.float32),
        "candidate_action_chunk": np.stack(q_rows["candidate_action_chunk"]).astype(np.float32),
        "label": np.asarray(q_rows["label"], dtype=np.int8),
        "delta_score": np.asarray(q_rows["delta_score"], dtype=np.float32),
        "source_split": np.asarray(q_rows["source_split"], dtype="U5"),
        "source_demo": np.asarray(q_rows["source_demo"], dtype="U32"),
        "source_step": np.asarray(q_rows["source_step"], dtype=np.int32),
        "comparison_type": np.asarray(q_rows["comparison_type"], dtype="U18"),
    }

    best_index = np.argmax(delta_scores, axis=1)
    row_index = np.arange(count)
    best_delta = delta_scores[row_index, best_index]
    has_improvement = best_delta > score_margin
    if actor_target_mode == "best_candidate":
        raw_target = residuals[row_index, best_index]
    else:
        positive_direction = 0.5 * (residuals[:, 0::2] - residuals[:, 1::2])
        symmetric_score_difference = candidate_scores[:, 0::2] - candidate_scores[:, 1::2]
        flat_direction = positive_direction.reshape(count, positive_direction.shape[1], -1)
        squared_norm = np.sum(flat_direction * flat_direction, axis=2)
        # Central finite differences projected onto each sampled direction.
        # The common factor of two is irrelevant because the combined vector is
        # normalized below. Dividing by direction energy prevents directions
        # with a larger L2 norm from dominating merely because of their scale.
        coefficient = symmetric_score_difference / np.maximum(squared_norm, 1e-12)
        gradient = np.sum(coefficient[:, :, None] * flat_direction, axis=1)
        max_abs = np.max(np.abs(gradient), axis=1)
        normalized = gradient / np.maximum(max_abs[:, None], 1e-12)
        raw_target = (residual_scale * normalized).reshape(residuals.shape[0], *residuals.shape[2:])
    target = np.where(has_improvement[:, None, None], raw_target, 0.0)
    clipped_target = np.clip(target, -residual_scale, residual_scale).astype(np.float32)
    target_was_clipped = np.any(
        np.abs(target) > residual_scale + 1e-8, axis=(1, 2)
    ).astype(np.uint8)
    actor_arrays = {
        "state": state,
        "base_action_chunk": base,
        "target_residual_chunk": clipped_target,
        "has_improvement": has_improvement.astype(np.uint8),
        "best_delta_score": np.where(has_improvement, best_delta, 0.0).astype(np.float32),
        "target_was_clipped": target_was_clipped,
        "source_split": split,
        "source_demo": demo,
        "source_step": step,
        "candidate_count": np.full(count, candidates.shape[1], dtype=np.int16),
        "residual_scale": np.asarray(residual_scale, dtype=np.float32),
    }

    train_demos = set(demo[split == "train"].tolist())
    valid_demos = set(demo[split == "valid"].tolist())
    overlap = sorted(train_demos & valid_demos)
    if overlap:
        raise ValueError("Source trajectory leakage between train and validation")

    q_output_path.parent.mkdir(parents=True, exist_ok=True)
    actor_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(q_output_path, **q_arrays)
    np.savez_compressed(actor_output_path, **actor_arrays)

    q_splits: dict[str, Any] = {}
    actor_splits: dict[str, Any] = {}
    for split_name in ("train", "valid"):
        q_mask = q_arrays["source_split"] == split_name
        actor_mask = actor_arrays["source_split"] == split_name
        q_splits[split_name] = {
            "pairs": int(np.count_nonzero(q_mask)),
            "candidate_better": int(np.count_nonzero(q_arrays["label"][q_mask] > 0)),
            "first_action_better": int(np.count_nonzero(q_arrays["label"][q_mask] < 0)),
            "candidate_vs_base": int(
                np.count_nonzero(q_mask & (q_arrays["comparison_type"] == "candidate_vs_base"))
            ),
            "symmetric_pair": int(
                np.count_nonzero(q_mask & (q_arrays["comparison_type"] == "symmetric_pair"))
            ),
        }
        actor_splits[split_name] = {
            "states": int(np.count_nonzero(actor_mask)),
            "improvement_targets": int(
                np.count_nonzero(actor_arrays["has_improvement"][actor_mask])
            ),
            "zero_targets": int(
                np.count_nonzero(~actor_arrays["has_improvement"][actor_mask].astype(bool))
            ),
            "clipped_targets": int(
                np.count_nonzero(actor_arrays["target_was_clipped"][actor_mask])
            ),
        }

    return {
        "collection": str(collection_path),
        "q_output": str(q_output_path),
        "actor_output": str(actor_output_path),
        "score_margin": score_margin,
        "residual_scale": residual_scale,
        "actor_target_mode": actor_target_mode,
        "states": count,
        "candidates_per_state": int(candidates.shape[1]),
        "q_pairs": len(q_arrays["label"]),
        "q_splits": q_splits,
        "actor_splits": actor_splits,
        "source_demo_overlap": overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--q-output", type=Path, required=True)
    parser.add_argument("--actor-output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument(
        "--actor-target-mode",
        choices=("best_candidate", "symmetric_gradient"),
        default="symmetric_gradient",
    )
    args = parser.parse_args()
    report = prepare_datasets(
        args.collection,
        args.q_output,
        args.actor_output,
        residual_scale=args.residual_scale,
        actor_target_mode=args.actor_target_mode,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
