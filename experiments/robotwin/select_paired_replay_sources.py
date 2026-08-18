#!/usr/bin/env python3
"""Select deduplicated full-rollout sources from accepted RoboTwin pairs.

The selector keeps causal rescue evidence separate from hard negatives so that
the small rescue set can be sampled more often without duplicating every
historical tie.  A replay builder must consume the returned task roots (rather
than the individual intervention directories) because IQL requires complete,
contiguous episodes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TIE_SOURCE_MARKERS = (
    "robotwin_rescue_screen_20260817_approved_pairs",
    "robotwin_single_hold_collection_20260818",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tie-source-marker",
        action="append",
        default=[],
        help=(
            "Keep terminal_tie_unscored rows only when the accepted-pairs path "
            "contains this marker. Repeat to add sources."
        ),
    )
    return parser.parse_args()


def _task_rollout_root(record_dir: Path, task_name: str) -> Path:
    resolved = record_dir.expanduser().resolve()
    for parent in resolved.parents:
        if parent.name == task_name and (parent / "imagination_transitions").is_dir():
            return parent
    raise ValueError(
        f"cannot find full task rollout root above record {resolved} for {task_name!r}"
    )


def _pair_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["task_name"]),
        int(row["environment_seed"]),
        int(row["intervention_replan_idx"]),
        str(row.get("current_observation_sha256", "")),
        str(Path(row["baseline_record_dir"]).expanduser().resolve()),
        str(Path(row["residual_record_dir"]).expanduser().resolve()),
    )


def _source_identity(source: dict[str, Any]) -> tuple[str, int]:
    return (str(source["path"]), int(source["environment_seed"]))


def select_sources(
    pair_root: Path,
    *,
    tie_source_markers: tuple[str, ...],
) -> dict[str, Any]:
    pair_files = sorted(pair_root.expanduser().resolve().rglob("accepted_pairs.jsonl"))
    selected_rows: dict[str, list[dict[str, Any]]] = {
        "rescue": [],
        "hard_negative": [],
    }
    seen_pairs: set[tuple[Any, ...]] = set()
    scanned_counts: Counter[str] = Counter()
    duplicate_rows = 0

    for pair_file in pair_files:
        with pair_file.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "accepted":
                    continue
                label = str(row.get("label"))
                scanned_counts[label] += 1
                identity = _pair_identity(row)
                if identity in seen_pairs:
                    duplicate_rows += 1
                    continue
                seen_pairs.add(identity)
                bucket = None
                if label == "rescue":
                    bucket = "rescue"
                elif label == "regression":
                    bucket = "hard_negative"
                elif label == "terminal_tie_unscored" and any(
                    marker in str(pair_file) for marker in tie_source_markers
                ):
                    bucket = "hard_negative"
                if bucket is None:
                    continue
                selected_rows[bucket].append(
                    {
                        **row,
                        "accepted_pairs_path": str(pair_file),
                        "accepted_pairs_line": line_number,
                    }
                )

    payload: dict[str, Any] = {
        "format": "robotwin_paired_replay_source_selection_v1",
        "pair_root": str(pair_root.expanduser().resolve()),
        "tie_source_markers": list(tie_source_markers),
        "scanned_label_counts": dict(sorted(scanned_counts.items())),
        "duplicate_pair_rows_removed": duplicate_rows,
        "buckets": {},
    }
    for bucket, rows in selected_rows.items():
        sources: list[dict[str, Any]] = []
        seen_sources: set[tuple[str, int]] = set()
        for row in rows:
            seed = int(row["environment_seed"])
            for role in ("baseline", "residual"):
                root = _task_rollout_root(
                    Path(row[f"{role}_record_dir"]), str(row["task_name"])
                )
                source = {
                    "path": str(root),
                    "environment_seed": seed,
                    "task_name": str(row["task_name"]),
                    "role": role,
                }
                identity = _source_identity(source)
                if identity not in seen_sources:
                    seen_sources.add(identity)
                    sources.append(source)
        group_counts = Counter(
            f"{row['task_name']}:{int(row['environment_seed'])}" for row in rows
        )
        payload["buckets"][bucket] = {
            "pair_count": len(rows),
            "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
            "task_seed_group_count": len(group_counts),
            "task_seed_pair_counts": dict(sorted(group_counts.items())),
            "source_count": len(sources),
            "sources": sources,
            "pairs": rows,
        }
    return payload


def main() -> None:
    args = parse_args()
    markers = tuple(args.tie_source_marker) or DEFAULT_TIE_SOURCE_MARKERS
    payload = select_sources(args.pair_root, tie_source_markers=markers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compact = {
        "output": str(args.output.resolve()),
        "scanned_label_counts": payload["scanned_label_counts"],
        "duplicate_pair_rows_removed": payload["duplicate_pair_rows_removed"],
        "buckets": {
            name: {
                key: value
                for key, value in bucket.items()
                if key in {"pair_count", "label_counts", "task_seed_group_count", "source_count"}
            }
            for name, bucket in payload["buckets"].items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
