"""Score exported single-intervention pairs with one global camera normalization."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.robotwin.analyze_imagination_rewards import (
    encode_record_images,
    fit_task_balanced_camera_normalization,
    resolve_encoder_dtype,
)
from experiments.robotwin.build_single_intervention_pairs import _outcome_label
from experiments.robotwin.imagination_reward_utils import ROBOTWIN_CAMERA_NAMES
from fastwam.rl.rewards import (
    GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE,
    compute_imagination_reward,
)


def discover_pair_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.expanduser().resolve().rglob("accepted_pairs.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["pair_file"] = str(path.resolve())
            rows.append(row)
    if not rows:
        raise ValueError(f"No accepted_pairs.jsonl rows found in {root}")
    return rows


def _transition_record(record_dir: str) -> dict[str, Any]:
    root = Path(record_dir).expanduser().resolve()
    metadata_path = root / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = dict(payload)
    payload["record_dir"] = str(root)
    for phase in ("current", "actual", "predicted_goal"):
        path = root / f"{phase}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload[f"{phase}_path"] = str(path)
    return payload


def _candidate_stratum(metadata: dict[str, Any], *, q_margin: float) -> str:
    if metadata.get("residual_support_in_distribution") is False:
        return "ood_rejected"
    q_min = metadata.get("residual_q_advantage_min")
    if q_min is not None and float(q_min) < q_margin:
        return "q_rejected"
    if bool(metadata.get("residual_gate_approved", False)):
        return "approved"
    return "other_rejected"


def score_pairs(
    pairs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    encoded: list[dict[str, dict[str, np.ndarray]]],
    *,
    camera_normalization: dict[str, Any],
    clip_value: float,
    local_progress_threshold: float,
    q_margin: float,
) -> list[dict[str, Any]]:
    if len(records) != len(encoded):
        raise ValueError("record and feature counts differ")
    encoded_by_dir = {
        str(Path(record["record_dir"]).resolve()): features
        for record, features in zip(records, encoded)
    }
    metadata_by_dir = {
        str(Path(record["record_dir"]).resolve()): record for record in records
    }
    camera_norm = {
        camera: {
            "center": float(settings["center"]),
            "scale": float(settings["scale"]),
        }
        for camera, settings in camera_normalization["cameras"].items()
    }
    scored: list[dict[str, Any]] = []
    for pair in pairs:
        baseline_dir = str(Path(pair["baseline_record_dir"]).resolve())
        residual_dir = str(Path(pair["residual_record_dir"]).resolve())
        baseline_features = encoded_by_dir[baseline_dir]
        residual_features = encoded_by_dir[residual_dir]
        # Both branches passed exact pre-intervention matching. Use the shadow
        # baseline's current observation and imagined goal as the common frame
        # of reference, so only the realized next observation differs.
        baseline_reward = compute_imagination_reward(
            baseline_features["current"],
            baseline_features["actual"],
            baseline_features["predicted_goal"],
            reward_type=GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE,
            camera_weights={camera: 1.0 for camera in ROBOTWIN_CAMERA_NAMES},
            camera_normalization=camera_norm,
            clip_value=clip_value,
            alignment_valid=True,
        )
        residual_reward = compute_imagination_reward(
            baseline_features["current"],
            residual_features["actual"],
            baseline_features["predicted_goal"],
            reward_type=GLOBAL_CAMERA_NORMALIZED_REWARD_TYPE,
            camera_weights={camera: 1.0 for camera in ROBOTWIN_CAMERA_NAMES},
            camera_normalization=camera_norm,
            clip_value=clip_value,
            alignment_valid=True,
        )
        delta = float(residual_reward.clipped_progress - baseline_reward.clipped_progress)
        shadow_metadata = metadata_by_dir[baseline_dir]
        row = dict(pair)
        row.update(
            {
                "candidate_stratum": _candidate_stratum(
                    shadow_metadata, q_margin=q_margin
                ),
                "shadow_q_advantage_min": shadow_metadata.get(
                    "residual_q_advantage_min"
                ),
                "shadow_q_advantage_disagreement": shadow_metadata.get(
                    "residual_q_advantage_disagreement"
                ),
                "shadow_support_in_distribution": shadow_metadata.get(
                    "residual_support_in_distribution"
                ),
                "shadow_support_state_score": shadow_metadata.get(
                    "residual_support_state_score"
                ),
                "shadow_support_action_score": shadow_metadata.get(
                    "residual_support_action_score"
                ),
                "baseline_imagination_progress": float(
                    baseline_reward.clipped_progress
                ),
                "residual_imagination_progress": float(
                    residual_reward.clipped_progress
                ),
                "local_progress_delta": delta,
                "per_camera_local_progress_delta": {
                    camera: float(
                        residual_reward.per_camera[camera][
                            "normalized_delta_alignment"
                        ]
                        - baseline_reward.per_camera[camera][
                            "normalized_delta_alignment"
                        ]
                    )
                    for camera in ROBOTWIN_CAMERA_NAMES
                },
            }
        )
        row["label"] = _outcome_label(
            baseline_success=bool(row["baseline_episode_success"]),
            residual_success=bool(row["residual_episode_success"]),
            local_progress_delta=delta,
            local_progress_threshold=local_progress_threshold,
        )
        scored.append(row)
    return scored


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_stratum[str(row["candidate_stratum"])].append(row)
    return {
        "schema_version": "robotwin_single_intervention_statistics_v1",
        "pair_count": len(scored),
        "label_counts": dict(Counter(str(row["label"]) for row in scored)),
        "baseline_success_count": int(
            sum(bool(row["baseline_episode_success"]) for row in scored)
        ),
        "residual_success_count": int(
            sum(bool(row["residual_episode_success"]) for row in scored)
        ),
        "mean_local_progress_delta": float(
            np.mean([float(row["local_progress_delta"]) for row in scored])
        ),
        "local_progress_positive_count": int(
            sum(float(row["local_progress_delta"]) > 0.0 for row in scored)
        ),
        "strata": {
            stratum: {
                "count": len(rows),
                "labels": dict(Counter(str(row["label"]) for row in rows)),
                "baseline_successes": int(
                    sum(bool(row["baseline_episode_success"]) for row in rows)
                ),
                "residual_successes": int(
                    sum(bool(row["residual_episode_success"]) for row in rows)
                ),
                "mean_local_progress_delta": float(
                    np.mean([float(row["local_progress_delta"]) for row in rows])
                ),
            }
            for stratum, rows in sorted(by_stratum.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoder-dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--clip-value", type=float, default=0.1)
    parser.add_argument("--local-progress-threshold", type=float, default=0.01)
    parser.add_argument("--q-margin", type=float, default=0.003)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pairs = discover_pair_rows(args.pair_root)
    records_by_dir: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        for key in ("baseline_record_dir", "residual_record_dir"):
            record = _transition_record(str(pair[key]))
            records_by_dir[str(Path(record["record_dir"]).resolve())] = record
    records = list(records_by_dir.values())
    dtype = resolve_encoder_dtype(args.encoder_dtype, device=args.device)
    encoded = encode_record_images(
        records,
        encoder_path=args.encoder_path,
        device=args.device,
        batch_size=args.batch_size,
        encoder_dtype=dtype,
    )
    normalization = fit_task_balanced_camera_normalization(records, encoded)
    scored = score_pairs(
        pairs,
        records,
        encoded,
        camera_normalization=normalization,
        clip_value=args.clip_value,
        local_progress_threshold=args.local_progress_threshold,
        q_margin=args.q_margin,
    )
    summary = summarize(scored)
    summary["camera_normalization"] = normalization
    summary["encoder_path"] = str(args.encoder_path.expanduser().resolve())
    summary["encoder_dtype"] = str(dtype).removeprefix("torch.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scored_pairs.jsonl").open("w", encoding="utf-8") as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "statistics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
