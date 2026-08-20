#!/usr/bin/env python3
"""Aggregate counterfactual candidates into one residual target per state."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def prepare_residual_dataset(
    pairwise_dataset: str | Path,
    output_path: str | Path,
    *,
    residual_scale: float = 0.1,
    equality_tolerance: float = 1e-6,
) -> dict[str, Any]:
    pairwise_dataset = Path(pairwise_dataset).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(pairwise_dataset, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}

    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (demo, step) in enumerate(zip(arrays["source_demo"], arrays["source_step"])):
        groups[(str(demo), int(step))].append(index)

    output: dict[str, list[Any]] = defaultdict(list)
    mixed_outcome_states = 0
    for (demo, step), indices in sorted(groups.items()):
        reference = indices[0]
        state = arrays["state"][reference]
        base = arrays["base_action_chunk"][reference]
        split = str(arrays["source_split"][reference])
        for index in indices[1:]:
            if str(arrays["source_split"][index]) != split:
                raise ValueError(f"State {(demo, step)} crosses train/valid splits")
            if not np.allclose(arrays["state"][index], state, rtol=0.0, atol=equality_tolerance):
                raise ValueError(f"State mismatch within {(demo, step)}")
            if not np.allclose(
                arrays["base_action_chunk"][index], base, rtol=0.0, atol=equality_tolerance
            ):
                raise ValueError(f"Base action mismatch within {(demo, step)}")

        positive = [index for index in indices if int(arrays["label"][index]) > 0]
        negative = [index for index in indices if int(arrays["label"][index]) < 0]
        mixed_outcome_states += int(bool(positive) and bool(negative))
        if positive:
            best = max(positive, key=lambda index: float(arrays["delta_score"][index]))
            raw_residual = arrays["candidate_action_chunk"][best] - base
            target_residual = np.clip(raw_residual, -residual_scale, residual_scale)
            best_delta = float(arrays["delta_score"][best])
            has_improvement = 1
            target_was_clipped = int(np.any(np.abs(raw_residual) > residual_scale + 1e-8))
        else:
            target_residual = np.zeros_like(base)
            best_delta = 0.0
            has_improvement = 0
            target_was_clipped = 0

        output["state"].append(state.astype(np.float32))
        output["base_action_chunk"].append(base.astype(np.float32))
        output["target_residual_chunk"].append(target_residual.astype(np.float32))
        output["has_improvement"].append(has_improvement)
        output["best_delta_score"].append(best_delta)
        output["target_was_clipped"].append(target_was_clipped)
        output["source_split"].append(split)
        output["source_demo"].append(demo)
        output["source_step"].append(step)
        output["candidate_count"].append(len(indices))

    prepared = {
        "state": np.stack(output["state"]),
        "base_action_chunk": np.stack(output["base_action_chunk"]),
        "target_residual_chunk": np.stack(output["target_residual_chunk"]),
        "has_improvement": np.asarray(output["has_improvement"], dtype=np.uint8),
        "best_delta_score": np.asarray(output["best_delta_score"], dtype=np.float32),
        "target_was_clipped": np.asarray(output["target_was_clipped"], dtype=np.uint8),
        "source_split": np.asarray(output["source_split"], dtype="U5"),
        "source_demo": np.asarray(output["source_demo"], dtype="U16"),
        "source_step": np.asarray(output["source_step"], dtype=np.int32),
        "candidate_count": np.asarray(output["candidate_count"], dtype=np.int16),
        "residual_scale": np.asarray(residual_scale, dtype=np.float32),
    }
    np.savez_compressed(output_path, **prepared)

    splits: dict[str, Any] = {}
    for split in ("train", "valid"):
        mask = prepared["source_split"] == split
        improved = prepared["has_improvement"][mask].astype(bool)
        splits[split] = {
            "states": int(np.count_nonzero(mask)),
            "source_demos": len(set(prepared["source_demo"][mask].tolist())),
            "improvement_targets": int(np.count_nonzero(improved)),
            "zero_targets": int(np.count_nonzero(~improved)),
            "clipped_improvement_targets": int(
                np.count_nonzero(prepared["target_was_clipped"][mask])
            ),
        }
    return {
        "pairwise_dataset": str(pairwise_dataset),
        "output_path": str(output_path),
        "raw_pair_count": len(arrays["label"]),
        "unique_state_count": len(prepared["state"]),
        "mixed_candidate_outcome_states": mixed_outcome_states,
        "residual_scale": residual_scale,
        "state_shape": list(prepared["state"].shape),
        "action_chunk_shape": list(prepared["base_action_chunk"].shape),
        "splits": splits,
        "source_demo_overlap": sorted(
            set(prepared["source_demo"][prepared["source_split"] == "train"].tolist())
            & set(prepared["source_demo"][prepared["source_split"] == "valid"].tolist())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    args = parser.parse_args()
    if args.residual_scale <= 0:
        parser.error("residual-scale must be positive")

    report = prepare_residual_dataset(
        args.pairwise_dataset,
        args.output,
        residual_scale=args.residual_scale,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if report["source_demo_overlap"]:
        raise SystemExit("Train/valid source trajectories overlap")


if __name__ == "__main__":
    main()
