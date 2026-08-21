#!/usr/bin/env python3
"""Replace privileged simulator state with wrist SigLIP plus proprio features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def prepare(
    q_source_path: str | Path,
    actor_source_path: str | Path,
    observation_path: str | Path,
    q_output_path: str | Path,
    actor_output_path: str | Path,
    *,
    vision_pca_dim: int | None = None,
    projection_output_path: str | Path | None = None,
    projection_input_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = [Path(path).expanduser().resolve() for path in (
        q_source_path, actor_source_path, observation_path, q_output_path, actor_output_path
    )]
    q_source_path, actor_source_path, observation_path, q_output_path, actor_output_path = paths
    with np.load(q_source_path, allow_pickle=False) as loaded:
        q = {key: loaded[key] for key in loaded.files}
    with np.load(actor_source_path, allow_pickle=False) as loaded:
        actor = {key: loaded[key] for key in loaded.files}
    with np.load(observation_path, allow_pickle=False) as loaded:
        observation = {key: loaded[key] for key in loaded.files}

    lookup = {
        (str(demo), int(step)): index
        for index, (demo, step) in enumerate(
            zip(observation["source_demo"], observation["source_step"])
        )
    }
    raw_vision = observation["vision_feature"].astype(np.float32)
    explained_variance_ratio = None
    projection_path = None
    if projection_input_path is not None:
        if vision_pca_dim is not None or projection_output_path is not None:
            raise ValueError(
                "projection_input_path cannot be combined with PCA fitting arguments"
            )
        projection_path = Path(projection_input_path).expanduser().resolve()
        with np.load(projection_path, allow_pickle=False) as loaded:
            projection = {key: loaded[key] for key in loaded.files}
        mean = np.asarray(projection["mean"], dtype=np.float32)
        components = np.asarray(projection["components"], dtype=np.float32)
        input_dim = int(projection["input_dim"])
        output_dim = int(projection["output_dim"])
        if str(projection.get("fitted_split", "")) != "train":
            raise ValueError("Loaded PCA projection was not fitted on the training split")
        if input_dim != raw_vision.shape[1] or mean.shape != (input_dim,):
            raise ValueError("Loaded PCA projection input shape is incompatible")
        if components.shape != (output_dim, input_dim):
            raise ValueError("Loaded PCA projection component shape is incompatible")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(components)):
            raise ValueError("Loaded PCA projection contains non-finite values")
        vision_pca_dim = output_dim
        vision = ((raw_vision - mean) @ components.T).astype(np.float32)
    elif vision_pca_dim is not None:
        train = observation["source_split"] == "train"
        if vision_pca_dim <= 0 or vision_pca_dim > min(np.count_nonzero(train), raw_vision.shape[1]):
            raise ValueError("vision_pca_dim is incompatible with train states or feature dim")
        mean = raw_vision[train].mean(axis=0, dtype=np.float64)
        centered_train = raw_vision[train].astype(np.float64) - mean
        _, singular_values, right = np.linalg.svd(centered_train, full_matrices=False)
        components = right[:vision_pca_dim]
        vision = ((raw_vision.astype(np.float64) - mean) @ components.T).astype(np.float32)
        total_variance = float(np.sum(singular_values**2))
        explained_variance_ratio = (
            float(np.sum(singular_values[:vision_pca_dim] ** 2) / total_variance)
            if total_variance > 0
            else 0.0
        )
        if projection_output_path is None:
            raise ValueError("projection_output_path is required with vision_pca_dim")
        projection_path = Path(projection_output_path).expanduser().resolve()
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            projection_path,
            mean=mean.astype(np.float32),
            components=components.astype(np.float32),
            input_dim=np.asarray(raw_vision.shape[1], dtype=np.int32),
            output_dim=np.asarray(vision_pca_dim, dtype=np.int32),
            fitted_split=np.asarray("train"),
        )
    else:
        vision = raw_vision
    deployable = np.concatenate([vision, observation["proprio"]], axis=1).astype(np.float32)

    def replace_state(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        indices = np.asarray(
            [lookup[(str(demo), int(step))] for demo, step in zip(arrays["source_demo"], arrays["source_step"])],
            dtype=np.int64,
        )
        result = dict(arrays)
        result["state"] = deployable[indices]
        result.update(
            observation_mode=np.asarray("siglip_wrist_proprio"),
            encoder_path=observation["encoder_path"],
            camera_name=observation["camera_name"],
            proprio_keys=observation["proprio_keys"],
            vision_feature_dim=np.asarray(vision.shape[1], dtype=np.int32),
            vision_encoder_output_dim=np.asarray(raw_vision.shape[1], dtype=np.int32),
            proprio_dim=np.asarray(observation["proprio"].shape[1], dtype=np.int32),
        )
        if projection_path is not None:
            result["vision_projection_path"] = np.asarray(str(projection_path))
        return result

    q_output = replace_state(q)
    actor_output = replace_state(actor)
    q_output_path.parent.mkdir(parents=True, exist_ok=True)
    actor_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(q_output_path, **q_output)
    np.savez_compressed(actor_output_path, **actor_output)
    report = {
        "q_source": str(q_source_path),
        "actor_source": str(actor_source_path),
        "observations": str(observation_path),
        "q_output": str(q_output_path),
        "actor_output": str(actor_output_path),
        "observation_mode": "siglip_wrist_proprio",
        "vision_encoder_output_dim": int(raw_vision.shape[1]),
        "vision_feature_dim": int(vision.shape[1]),
        "proprio_dim": int(observation["proprio"].shape[1]),
        "combined_state_dim": int(deployable.shape[1]),
        "vision_pca_dim": vision_pca_dim,
        "vision_pca_train_explained_variance_ratio": explained_variance_ratio,
        "vision_projection_path": str(projection_path) if projection_path is not None else None,
        "q_rows": len(q_output["state"]),
        "actor_rows": len(actor_output["state"]),
        "all_finite": bool(
            np.all(np.isfinite(q_output["state"]))
            and np.all(np.isfinite(actor_output["state"]))
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-source", type=Path, required=True)
    parser.add_argument("--actor-source", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--q-output", type=Path, required=True)
    parser.add_argument("--actor-output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--vision-pca-dim", type=int)
    parser.add_argument("--projection-output", type=Path)
    parser.add_argument("--projection-input", type=Path)
    args = parser.parse_args()
    report = prepare(
        args.q_source,
        args.actor_source,
        args.observations,
        args.q_output,
        args.actor_output,
        vision_pca_dim=args.vision_pca_dim,
        projection_output_path=args.projection_output,
        projection_input_path=args.projection_input,
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
