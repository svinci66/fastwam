"""Select one late-stage open-microwave chunk from a baseline rollout.

The selected chunk is only an anchor.  Counterfactual branches must replay the
episode and pass exact pre-intervention observation/action audits before their
outcomes are compared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "robotwin_open_microwave_chunk_anchor_v1"


def discover_chunks(transition_root: Path) -> list[dict]:
    chunks: list[dict] = []
    for metadata_path in sorted(transition_root.rglob("replan_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("task_name") != "open_microwave":
            continue
        arrays_path = metadata_path.parent / str(metadata["rollout_arrays_file"])
        with np.load(arrays_path, allow_pickle=False) as arrays:
            progress = np.asarray(arrays["task_progress"], dtype=np.float64)
        if progress.ndim != 2 or progress.shape[0] < 2 or progress.shape[1] != 4:
            raise ValueError(f"invalid task_progress shape in {arrays_path}: {progress.shape}")
        ratio = progress[:, 3]
        chunks.append(
            {
                "record_dir": str(metadata_path.parent.resolve()),
                "environment_seed": int(metadata["environment_seed"]),
                "trial_idx": int(metadata["trial_idx"]),
                "replan_idx": int(metadata["replan_idx"]),
                "start_open_ratio": float(ratio[0]),
                "end_open_ratio": float(ratio[-1]),
                "max_open_ratio": float(np.max(ratio)),
                "open_ratio_delta": float(ratio[-1] - ratio[0]),
                "executed_actions": int(ratio.size - 1),
                "current_observation_sha256": metadata.get(
                    "current_observation_sha256"
                ),
                "baseline_actions_sha256": metadata.get("baseline_actions_sha256"),
            }
        )
    if not chunks:
        raise ValueError(f"no open_microwave chunks found in {transition_root}")
    seeds = {chunk["environment_seed"] for chunk in chunks}
    trials = {chunk["trial_idx"] for chunk in chunks}
    if len(seeds) != 1 or len(trials) != 1:
        raise ValueError(
            "anchor selection requires exactly one episode, got "
            f"seeds={sorted(seeds)} trials={sorted(trials)}"
        )
    return chunks


def select_anchor(chunks: list[dict], target_ratio: float) -> dict:
    candidates = [
        chunk
        for chunk in chunks
        if chunk["start_open_ratio"] < 0.6 and chunk["executed_actions"] > 0
    ]
    if not candidates:
        raise ValueError("baseline rollout has no pre-success chunk")
    selected = min(
        candidates,
        key=lambda chunk: (
            abs(chunk["start_open_ratio"] - target_ratio),
            chunk["replan_idx"],
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": "closest_pre_success_start_ratio",
        "target_open_ratio": float(target_ratio),
        "candidate_count": len(candidates),
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-root", type=Path, required=True)
    parser.add_argument("--target-open-ratio", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-replan", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.target_open_ratio < 0.6:
        raise ValueError("target-open-ratio must be in [0, 0.6)")
    result = select_anchor(
        discover_chunks(args.transition_root.expanduser().resolve()),
        args.target_open_ratio,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.output_replan is not None:
        args.output_replan.parent.mkdir(parents=True, exist_ok=True)
        args.output_replan.write_text(
            f"{result['selected']['replan_idx']}\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
