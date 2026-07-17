"""Measure EGL replay noise in the frozen visual feature space used by Reward V2."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.libero.analyze_imagination_rewards import encode_images


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-ensemble-cosine-distance", type=float, default=1e-4)
    parser.add_argument("--max-ensemble-feature-l2", type=float, default=0.015)
    return parser.parse_args()


def _pair_key(path: Path) -> tuple[str, str, str]:
    relative = path.relative_to(path.parents[2])
    state_name = relative.parts[0]
    pair_name = relative.parts[1]
    suffix = "_first.png" if path.name.endswith("_first.png") else "_second.png"
    camera = path.name.removesuffix(suffix)
    return state_name, pair_name, camera


def main() -> None:
    args = _parse_args()
    image_paths = sorted(args.input_dir.rglob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found below {args.input_dir}")

    grouped: dict[tuple[str, str, str], dict[str, Path]] = defaultdict(dict)
    pair_paths = [
        path
        for path in image_paths
        if path.name.endswith("_first.png") or path.name.endswith("_second.png")
    ]
    for path in pair_paths:
        key = _pair_key(path)
        side = "first" if path.name.endswith("_first.png") else "second"
        grouped[key][side] = path.resolve()
    incomplete = [key for key, paths in grouped.items() if set(paths) != {"first", "second"}]
    if incomplete:
        raise ValueError(f"Incomplete render pairs: {incomplete}")

    resolved_paths = [str(path.resolve()) for path in image_paths]
    features = encode_images(
        resolved_paths,
        encoder_path=args.encoder_path,
        device=args.device,
        batch_size=args.batch_size,
    )

    rows = []
    for (state_name, pair_name, camera), paths in sorted(grouped.items()):
        first = features[str(paths["first"])]
        second = features[str(paths["second"])]
        cosine_distance = float(1.0 - np.dot(first, second))
        feature_l2 = float(np.linalg.norm(first - second))
        rows.append(
            {
                "state": state_name,
                "pair": pair_name,
                "camera": camera,
                "cosine_distance": cosine_distance,
                "feature_l2": feature_l2,
            }
        )

    ensemble_groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for path in image_paths:
        if "replay_ensemble" not in path.parts:
            continue
        relative = path.relative_to(args.input_dir)
        state_name, pair_name, side = relative.parts[:3]
        camera = path.stem.split("_", maxsplit=1)[1]
        ensemble_groups[(state_name, side, camera)].append(path.resolve())

    ensemble_states = sorted({key[0] for key in ensemble_groups})
    for state_name in ensemble_states:
        cameras = sorted(
            {key[2] for key in ensemble_groups if key[0] == state_name}
        )
        for camera in cameras:
            side_features = {}
            for side in ("first", "second"):
                paths = sorted(ensemble_groups[(state_name, side, camera)])
                if not paths:
                    raise ValueError(
                        f"Missing replay ensemble for {state_name}/{side}/{camera}"
                    )
                mean_feature = np.mean(
                    [features[str(path)] for path in paths], axis=0
                )
                side_features[side] = mean_feature / np.linalg.norm(mean_feature)
            first = side_features["first"]
            second = side_features["second"]
            rows.append(
                {
                    "state": state_name,
                    "pair": "replay_ensemble",
                    "camera": camera,
                    "render_repeats": len(
                        ensemble_groups[(state_name, "first", camera)]
                    ),
                    "cosine_distance": float(1.0 - np.dot(first, second)),
                    "feature_l2": float(np.linalg.norm(first - second)),
                }
            )

    replay_rows = [row for row in rows if row["pair"] == "replay_ensemble"]
    if not replay_rows:
        raise ValueError("No replay render ensembles found")
    max_cosine_distance = max(row["cosine_distance"] for row in replay_rows)
    max_feature_l2 = max(row["feature_l2"] for row in replay_rows)
    passed = bool(
        max_cosine_distance <= args.max_ensemble_cosine_distance
        and max_feature_l2 <= args.max_ensemble_feature_l2
    )
    result = {
        "schema_version": 1,
        "passed": passed,
        "encoder_path": str(Path(args.encoder_path).resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "thresholds": {
            "max_ensemble_cosine_distance": args.max_ensemble_cosine_distance,
            "max_ensemble_feature_l2": args.max_ensemble_feature_l2,
        },
        "ensemble_max_cosine_distance": max_cosine_distance,
        "ensemble_max_feature_l2": max_feature_l2,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("Replay rendering drift is too large for Reward V2.")


if __name__ == "__main__":
    main()
