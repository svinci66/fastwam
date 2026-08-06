"""Select diverse residual candidates from a Q+OOD shadow rollout."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def discover_shadow_candidates(root: Path, *, q_margin: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in sorted(root.expanduser().resolve().rglob("metadata.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != "robotwin_imagination_transition_v1":
            continue
        if str(row.get("action_mode")) != "residual" or not bool(
            row.get("residual_shadow_mode", False)
        ):
            continue
        if not bool(row.get("alignment_valid", False)):
            continue
        candidate_rms = float(row.get("residual_candidate_rms", 0.0))
        if not np.isfinite(candidate_rms) or candidate_rms <= 0.0:
            continue
        q_min = row.get("residual_q_advantage_min")
        support = row.get("residual_support_in_distribution")
        if support is False:
            stratum = "ood_rejected"
        elif q_min is not None and float(q_min) < q_margin:
            stratum = "q_rejected"
        elif bool(row.get("residual_gate_approved", False)):
            stratum = "approved"
        else:
            stratum = "other_rejected"
        candidates.append(
            {
                "task_name": str(row["task_name"]),
                "environment_seed": int(row["environment_seed"]),
                "trial_idx": int(row["trial_idx"]),
                "replan_idx": int(row["replan_idx"]),
                "stratum": stratum,
                "candidate_rms": candidate_rms,
                "q_advantage_min": None if q_min is None else float(q_min),
                "support_state_score": row.get("residual_support_state_score"),
                "support_action_score": row.get("residual_support_action_score"),
                "record_dir": str(path.parent.resolve()),
            }
        )
    if not candidates:
        raise ValueError(f"No eligible shadow candidates found in {root}")
    return candidates


def _stratum_priority(row: dict[str, Any]) -> tuple[float, int]:
    stratum = str(row["stratum"])
    if stratum == "approved":
        return (-float(row["candidate_rms"]), int(row["replan_idx"]))
    if stratum == "q_rejected":
        q_min = row["q_advantage_min"]
        return (
            float("inf") if q_min is None else float(q_min),
            int(row["replan_idx"]),
        )
    if stratum == "ood_rejected":
        state = float(row.get("support_state_score") or 0.0)
        action = float(row.get("support_action_score") or 0.0)
        return (-max(state, action), int(row["replan_idx"]))
    return (-float(row["candidate_rms"]), int(row["replan_idx"]))


def select_candidates(
    candidates: list[dict[str, Any]], *, max_per_episode: int
) -> list[dict[str, Any]]:
    if max_per_episode <= 0:
        raise ValueError("max_per_episode must be positive")
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[
            (
                str(row["task_name"]),
                int(row["environment_seed"]),
                int(row["trial_idx"]),
            )
        ].append(row)

    selected: list[dict[str, Any]] = []
    stratum_order = ("approved", "q_rejected", "ood_rejected", "other_rejected")
    for episode_key, rows in sorted(grouped.items()):
        episode_selected: list[dict[str, Any]] = []
        for stratum in stratum_order:
            values = sorted(
                (row for row in rows if row["stratum"] == stratum),
                key=_stratum_priority,
            )
            if values and len(episode_selected) < max_per_episode:
                episode_selected.append(values[0])
        remaining = [row for row in rows if row not in episode_selected]
        while remaining and len(episode_selected) < max_per_episode:
            # Fill unused slots with the replan farthest from those already
            # selected, avoiding a batch concentrated in one task phase.
            def diversity_key(row: dict[str, Any]) -> tuple[int, float, int]:
                distance = min(
                    abs(int(row["replan_idx"]) - int(other["replan_idx"]))
                    for other in episode_selected
                )
                return (
                    distance,
                    float(row["candidate_rms"]),
                    -int(row["replan_idx"]),
                )

            chosen = max(remaining, key=diversity_key)
            episode_selected.append(chosen)
            remaining.remove(chosen)
        selected.extend(sorted(episode_selected, key=lambda row: int(row["replan_idx"])))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--q-margin", type=float, default=0.003)
    parser.add_argument("--max-per-episode", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    args = parser.parse_args()

    candidates = discover_shadow_candidates(args.input_dir, q_margin=args.q_margin)
    selected = select_candidates(candidates, max_per_episode=args.max_per_episode)
    payload = {
        "schema_version": "robotwin_single_intervention_candidate_plan_v1",
        "input_dir": str(args.input_dir.expanduser().resolve()),
        "q_margin": args.q_margin,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "candidate_strata": dict(Counter(row["stratum"] for row in candidates)),
        "selected_strata": dict(Counter(row["stratum"] for row in selected)),
        "selected": selected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with args.output_tsv.open("w", encoding="utf-8") as stream:
        stream.write("task_name\tenvironment_seed\ttrial_idx\treplan_idx\tstratum\n")
        for row in selected:
            stream.write(
                f"{row['task_name']}\t{row['environment_seed']}\t"
                f"{row['trial_idx']}\t{row['replan_idx']}\t{row['stratum']}\n"
            )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
