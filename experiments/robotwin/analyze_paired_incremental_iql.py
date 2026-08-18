#!/usr/bin/env python3
"""Compare old/new IQL actor and dual-Q scores at paired interventions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.rl.online_policy import load_iql_q_critics, load_residual_actor_checkpoint
from fastwam.rl.replay_buffer import ReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--rescue-replay", type=Path, required=True)
    parser.add_argument("--hard-negative-replay", type=Path, required=True)
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--q-margin", type=float, default=0.005)
    parser.add_argument("--max-q-disagreement", type=float, default=0.02)
    return parser.parse_args()


def _task_rollout_root(record_dir: Path, task_name: str) -> Path:
    for parent in record_dir.expanduser().resolve().parents:
        if parent.name == task_name and (parent / "imagination_transitions").is_dir():
            return parent
    raise ValueError(f"cannot resolve rollout root for {record_dir}")


def _source_transition_indices(
    selection_bucket: dict[str, Any], replay: ReplayBuffer
) -> dict[tuple[str, int], int]:
    source_ids = {
        (str(Path(source["path"]).resolve()), int(source["environment_seed"])):
        f"source{index:03d}-"
        for index, source in enumerate(selection_bucket["sources"])
    }
    lookup: dict[tuple[str, int], int] = {}
    for pair in selection_bucket["pairs"]:
        source_root = _task_rollout_root(
            Path(pair["residual_record_dir"]), str(pair["task_name"])
        )
        source_prefix = source_ids[(str(source_root), int(pair["environment_seed"]))]
        replan = int(pair["intervention_replan_idx"])
        matches = [
            index
            for index, transition in enumerate(replay.transitions)
            if source_prefix in transition.episode_id
            and transition.transition_index == replan
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one replay transition for {source_prefix} replan {replan}, "
                f"got {len(matches)}"
            )
        lookup[(str(Path(pair["residual_record_dir"]).resolve()), replan)] = matches[0]
    return lookup


def _checkpoint_rows(
    checkpoint: Path,
    *,
    bucket_name: str,
    bucket: dict[str, Any],
    replay: ReplayBuffer,
    lookup: dict[tuple[str, int], int],
    device: torch.device,
    q_margin: float,
    max_q_disagreement: float,
) -> list[dict[str, Any]]:
    actor, payload = load_residual_actor_checkpoint(checkpoint, device=device)
    critics = load_iql_q_critics(payload, device=device, source="target")
    arrays = replay.arrays()
    rows: list[dict[str, Any]] = []
    for pair in bucket["pairs"]:
        replan = int(pair["intervention_replan_idx"])
        index = lookup[(str(Path(pair["residual_record_dir"]).resolve()), replan)]
        context = torch.from_numpy(
            np.concatenate(
                [arrays["observation_feature"][index], arrays["proprio"][index]]
            )[None]
        ).to(device)
        baseline = torch.from_numpy(arrays["baseline_actions"][index : index + 1]).to(device)
        language = (
            None
            if "language_feature" not in arrays
            else torch.from_numpy(arrays["language_feature"][index : index + 1]).to(device)
        )
        with torch.inference_mode():
            candidate = actor(context, baseline, language_feature=language)
            advantages = []
            for critic in critics:
                baseline_q = critic(context, baseline, baseline, language)
                candidate_q = critic(context, baseline, candidate, language)
                advantages.append(float((candidate_q - baseline_q).item()))
        candidate_np = candidate.float().cpu().numpy()[0]
        baseline_np = arrays["baseline_actions"][index]
        executed_np = arrays["executed_actions"][index]
        effective_k = int(arrays["effective_k"][index])
        residual = candidate_np[:effective_k] - baseline_np[:effective_k]
        captured_residual = executed_np[:effective_k] - baseline_np[:effective_k]
        q_min = min(advantages)
        disagreement = abs(advantages[0] - advantages[1])
        rows.append(
            {
                "bucket": bucket_name,
                "label": str(pair["label"]),
                "task_name": str(pair["task_name"]),
                "environment_seed": int(pair["environment_seed"]),
                "intervention_replan_idx": replan,
                "q_advantages": advantages,
                "q_advantage_min": q_min,
                "q_disagreement": disagreement,
                "q_gate_approved": bool(
                    q_min >= q_margin and disagreement <= max_q_disagreement
                ),
                "candidate_residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
                "captured_residual_rms": float(
                    np.sqrt(np.mean(np.square(captured_residual)))
                ),
                "candidate_to_captured_action_mse": float(
                    np.mean(np.square(candidate_np[:effective_k] - executed_np[:effective_k]))
                ),
            }
        )
    return rows


def _auc(positives: list[float], negatives: list[float]) -> float:
    comparisons = [
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    ]
    return float(np.mean(comparisons))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    summary = {}
    for label, values in sorted(by_label.items()):
        summary[label] = {
            "count": len(values),
            "q_advantage_min_mean": float(
                np.mean([value["q_advantage_min"] for value in values])
            ),
            "q_advantage_min_median": float(
                np.median([value["q_advantage_min"] for value in values])
            ),
            "q_gate_approval_rate": float(
                np.mean([value["q_gate_approved"] for value in values])
            ),
            "candidate_residual_rms_mean": float(
                np.mean([value["candidate_residual_rms"] for value in values])
            ),
            "candidate_to_captured_action_mse_mean": float(
                np.mean([value["candidate_to_captured_action_mse"] for value in values])
            ),
        }
    positive = [row["q_advantage_min"] for row in rows if row["label"] == "rescue"]
    negative = [row["q_advantage_min"] for row in rows if row["label"] != "rescue"]
    return {
        "by_label": summary,
        "rescue_vs_non_rescue_q_auc": _auc(positive, negative),
    }


def main() -> None:
    args = parse_args()
    if args.q_margin < 0.0 or args.max_q_disagreement < 0.0:
        raise ValueError("Q gate thresholds must be non-negative")
    device = torch.device(args.device)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    bucket_replays = {
        "rescue": ReplayBuffer.load(args.rescue_replay),
        "hard_negative": ReplayBuffer.load(args.hard_negative_replay),
    }
    lookups = {
        name: _source_transition_indices(selection["buckets"][name], replay)
        for name, replay in bucket_replays.items()
    }
    result: dict[str, Any] = {
        "format": "robotwin_paired_incremental_iql_audit_v1",
        "q_margin": args.q_margin,
        "max_q_disagreement": args.max_q_disagreement,
        "checkpoints": {},
    }
    for name, checkpoint in (
        ("old", args.old_checkpoint),
        ("new", args.new_checkpoint),
    ):
        rows = []
        for bucket_name, replay in bucket_replays.items():
            rows.extend(
                _checkpoint_rows(
                    checkpoint,
                    bucket_name=bucket_name,
                    bucket=selection["buckets"][bucket_name],
                    replay=replay,
                    lookup=lookups[bucket_name],
                    device=device,
                    q_margin=args.q_margin,
                    max_q_disagreement=args.max_q_disagreement,
                )
            )
        result["checkpoints"][name] = {
            "path": str(checkpoint.resolve()),
            "summary": _summarize(rows),
            "rows": rows,
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                name: checkpoint["summary"]
                for name, checkpoint in result["checkpoints"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
