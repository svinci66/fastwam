#!/usr/bin/env python3
"""Clean counterfactual branches and reconstruct action chunks for pairwise Q."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from experiments.robomimic.collect_can_counterfactual_branches import (
    _decode_names,
    _make_candidate_actions,
    _select_source,
)


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def prepare_dataset(
    collection_path: str | Path,
    output_path: str | Path,
    *,
    late_state_fraction: float = 0.5,
    state_tolerance: float = 1e-10,
) -> dict[str, Any]:
    collection_path = Path(collection_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(collection_path, "r") as collection:
        if not bool(collection.attrs.get("complete", False)):
            raise ValueError("Counterfactual collection is not complete")
        source_path = Path(_text(collection.attrs["source_dataset"]))
        horizon = int(collection.attrs["horizon"])
        default_intervention_steps = int(collection.attrs["intervention_steps"])
        noise_sigmas = json.loads(_text(collection.attrs["noise_sigmas"]))
        samples = collection["samples"]
        count = int(collection.attrs["samples_committed"])
        if any(len(dataset) != count for dataset in samples.values()):
            raise ValueError("Collection fields do not have a common committed length")

        with h5py.File(source_path, "r") as source:
            split_by_demo: dict[str, str] = {}
            for split in ("train", "valid"):
                for name in _decode_names(np.asarray(source["mask"][split])):
                    split_by_demo[name] = split
            action_lengths = {name: len(source["data"][name]["actions"]) for name in split_by_demo}
            eligible = sorted(
                (name for name, length in action_lengths.items() if length >= horizon),
                key=lambda name: int(name.rsplit("_", 1)[1]),
            )

            clean: dict[str, list[Any]] = {
                "state": [],
                "base_action_chunk": [],
                "candidate_action_chunk": [],
                "label": [],
                "delta_score": [],
                "source_split": [],
                "source_demo": [],
                "source_step": [],
                "base_success": [],
                "candidate_success": [],
                "source_branch_contains_success": [],
            }
            rejection_counts = {
                "tie": 0,
                "nonfinite": 0,
                "restore_error": 0,
                "branch_start_mismatch": 0,
                "reconstruction_mismatch": 0,
            }

            for index in range(count):
                label = int(samples["label"][index])
                if label == 0:
                    rejection_counts["tie"] += 1
                    continue
                numeric_values = np.concatenate(
                    [
                        np.asarray(samples["initial_state"][index]).reshape(-1),
                        np.asarray(samples["base_action"][index]).reshape(-1),
                        np.asarray(samples["candidate_action"][index]).reshape(-1),
                        np.asarray([samples["delta_score"][index]]),
                    ]
                )
                if not np.all(np.isfinite(numeric_values)):
                    rejection_counts["nonfinite"] += 1
                    continue
                if float(samples["restore_linf"][index]) > state_tolerance:
                    rejection_counts["restore_error"] += 1
                    continue
                if float(samples["branch_initial_state_linf"][index]) > state_tolerance:
                    rejection_counts["branch_start_mismatch"] += 1
                    continue

                demo_name = _text(samples["source_demo"][index])
                source_step = int(samples["source_step"][index])
                intervention_steps = int(samples["intervention_steps"][index])
                noise_seed = int(samples["noise_seed"][index])
                sigma = float(samples["noise_sigma"][index])
                group = source["data"][demo_name]
                base_actions = np.asarray(group["actions"][source_step : source_step + horizon])

                # Replay the collector's RNG calls to reconstruct all perturbed
                # actions, not merely the first action saved in the HDF5 row.
                rng = np.random.default_rng(noise_seed)
                selected_demo, selected_step = _select_source(
                    rng=rng,
                    demo_names=eligible,
                    action_lengths=action_lengths,
                    horizon=horizon,
                    late_state_fraction=late_state_fraction,
                )
                selected_sigma = float(noise_sigmas[int(rng.integers(0, len(noise_sigmas)))])
                candidate_actions = _make_candidate_actions(
                    base_actions,
                    rng=rng,
                    sigma=selected_sigma,
                    intervention_steps=intervention_steps,
                    perturb_gripper=False,
                )
                reconstruction_matches = (
                    selected_demo == demo_name
                    and selected_step == source_step
                    and np.isclose(selected_sigma, sigma)
                    and np.allclose(
                        candidate_actions[0],
                        np.asarray(samples["candidate_action"][index]),
                        rtol=0.0,
                        atol=1e-7,
                    )
                )
                if not reconstruction_matches:
                    rejection_counts["reconstruction_mismatch"] += 1
                    continue

                chunk_steps = min(default_intervention_steps, intervention_steps, len(base_actions))
                clean["state"].append(np.asarray(samples["initial_state"][index], dtype=np.float32))
                clean["base_action_chunk"].append(base_actions[:chunk_steps].astype(np.float32))
                clean["candidate_action_chunk"].append(candidate_actions[:chunk_steps].astype(np.float32))
                clean["label"].append(label)
                clean["delta_score"].append(float(samples["delta_score"][index]))
                clean["source_split"].append(_text(samples["source_split"][index]))
                clean["source_demo"].append(demo_name)
                clean["source_step"].append(source_step)
                clean["base_success"].append(int(samples["base_success"][index]))
                clean["candidate_success"].append(int(samples["candidate_success"][index]))
                clean["source_branch_contains_success"].append(
                    int(samples["source_branch_contains_success"][index])
                )

    arrays = {
        "state": np.stack(clean["state"]),
        "base_action_chunk": np.stack(clean["base_action_chunk"]),
        "candidate_action_chunk": np.stack(clean["candidate_action_chunk"]),
        "label": np.asarray(clean["label"], dtype=np.int8),
        "delta_score": np.asarray(clean["delta_score"], dtype=np.float32),
        "source_split": np.asarray(clean["source_split"], dtype="U5"),
        "source_demo": np.asarray(clean["source_demo"], dtype="U16"),
        "source_step": np.asarray(clean["source_step"], dtype=np.int32),
        "base_success": np.asarray(clean["base_success"], dtype=np.uint8),
        "candidate_success": np.asarray(clean["candidate_success"], dtype=np.uint8),
        "source_branch_contains_success": np.asarray(
            clean["source_branch_contains_success"], dtype=np.uint8
        ),
    }
    np.savez_compressed(output_path, **arrays)

    split_reports: dict[str, Any] = {}
    for split in ("train", "valid"):
        mask = arrays["source_split"] == split
        labels = arrays["label"][mask]
        split_reports[split] = {
            "samples": int(np.count_nonzero(mask)),
            "source_demos": len(set(arrays["source_demo"][mask].tolist())),
            "base_better": int(np.count_nonzero(labels < 0)),
            "candidate_better": int(np.count_nonzero(labels > 0)),
        }
    report = {
        "collection_path": str(collection_path),
        "source_dataset": str(source_path),
        "output_path": str(output_path),
        "input_samples": count,
        "accepted_samples": len(arrays["label"]),
        "rejected_samples": count - len(arrays["label"]),
        "rejection_counts": rejection_counts,
        "state_shape": list(arrays["state"].shape),
        "action_chunk_shape": list(arrays["base_action_chunk"].shape),
        "splits": split_reports,
        "source_demo_overlap": sorted(
            set(arrays["source_demo"][arrays["source_split"] == "train"].tolist())
            & set(arrays["source_demo"][arrays["source_split"] == "valid"].tolist())
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--late-state-fraction", type=float, default=0.5)
    args = parser.parse_args()

    report = prepare_dataset(
        args.collection,
        args.output,
        late_state_fraction=args.late_state_fraction,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if report["rejection_counts"]["reconstruction_mismatch"]:
        raise SystemExit("Candidate action-chunk reconstruction failed")
    if report["source_demo_overlap"]:
        raise SystemExit("Train/valid source trajectories overlap")


if __name__ == "__main__":
    main()
