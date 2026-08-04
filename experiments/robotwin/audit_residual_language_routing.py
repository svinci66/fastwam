"""Audit official-prompt OOD scores and canonical residual-language routing."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.rl.replay_buffer import ReplayBuffer


INSTRUCTION_RE = re.compile(
    r"FASTWAM_EVAL_INSTRUCTION episode_id=(\d+) seed=(\d+) instruction=(.+)$"
)
SIMILARITY_RE = re.compile(r"support_language_similarity=([0-9.eE+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--support-index", type=Path, required=True)
    parser.add_argument("--instruction-manifest", type=Path, required=True)
    parser.add_argument("--online-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    if np.any(~np.isfinite(rows)) or np.any(norms <= 0.0):
        raise ValueError("language features must be finite and non-zero")
    return rows / norms


def _official_episode_scores(path: Path) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        instruction_match = INSTRUCTION_RE.search(line)
        if instruction_match:
            current = {
                "episode_id": int(instruction_match.group(1)),
                "seed": int(instruction_match.group(2)),
                "instruction": ast.literal_eval(instruction_match.group(3)),
                "similarities": [],
            }
            episodes.append(current)
            continue
        similarity_match = SIMILARITY_RE.search(line)
        if current is not None and similarity_match:
            current["similarities"].append(float(similarity_match.group(1)))
    for episode in episodes:
        values = np.asarray(episode.pop("similarities"), dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"missing support similarity in {path}: {episode}")
        episode["observed_similarity"] = float(np.mean(values))
        episode["num_replans"] = int(values.size)
    return episodes


def main() -> None:
    args = parse_args()
    instruction_manifest = json.loads(
        args.instruction_manifest.read_text(encoding="utf-8")
    )
    support_metadata = json.loads(
        (args.support_index / "metadata.json").read_text(encoding="utf-8")
    )
    with np.load(args.support_index / "arrays.npz") as support_arrays:
        prototypes = _normalize(support_arrays["language_prototypes"])
    replay = ReplayBuffer.load(args.replay_dir)
    replay_manifest = json.loads(
        (args.replay_dir / "manifest.json").read_text(encoding="utf-8")
    )
    replay_arrays = replay.arrays()
    replay_language = replay_arrays.get("language_feature")
    if replay_language is None:
        raise ValueError("replay has no language_feature")
    summary = json.loads(args.online_summary.read_text(encoding="utf-8"))
    residual_rows = {
        row["task"]: row
        for row in summary["rows"]
        if row["variant"] == "imagination" and row["status"] == "complete"
    }
    task_names = list(support_metadata["task_names"])
    threshold = float(support_metadata["language_similarity_threshold"])
    manifest_encoder = instruction_manifest["language_encoder_version"]
    replay_encoder = replay_manifest.get("language_encoder_version")
    tasks: dict[str, dict[str, object]] = {}
    all_official_scores: list[float] = []
    all_routed_scores: list[float] = []
    for task_key, canonical_instruction in instruction_manifest["tasks"].items():
        if canonical_instruction not in task_names:
            raise ValueError(
                f"canonical instruction for {task_key} is absent from support index"
            )
        if task_key not in residual_rows:
            raise ValueError(f"online summary has no residual row for {task_key}")
        task_id = task_names.index(canonical_instruction)
        replay_indices = [
            index
            for index, transition in enumerate(replay.transitions)
            if transition.task_description == canonical_instruction
        ]
        if not replay_indices:
            raise ValueError(f"replay has no rows for {canonical_instruction!r}")
        normalized_rows = _normalize(replay_language[replay_indices])
        canonical_feature = _normalize(np.mean(normalized_rows, axis=0, keepdims=True))[0]
        similarities = prototypes @ canonical_feature
        routed_task_id = int(np.argmax(similarities))
        routed_similarity = float(similarities[routed_task_id])
        official_episodes = _official_episode_scores(
            Path(residual_rows[task_key]["log"])
        )
        official_scores = [
            float(episode["observed_similarity"]) for episode in official_episodes
        ]
        all_official_scores.extend(official_scores)
        all_routed_scores.append(routed_similarity)
        tasks[task_key] = {
            "canonical_instruction": canonical_instruction,
            "num_replay_transitions": len(replay_indices),
            "replay_to_prototype_similarity_min": float(
                np.min(normalized_rows @ prototypes[task_id])
            ),
            "canonical_route_similarity": routed_similarity,
            "canonical_route_resolved_task": task_names[routed_task_id],
            "canonical_route_accepted": routed_similarity >= threshold,
            "official_similarity_min": float(np.min(official_scores)),
            "official_similarity_mean": float(np.mean(official_scores)),
            "official_similarity_max": float(np.max(official_scores)),
            "official_accepted_episodes": int(
                np.sum(np.asarray(official_scores) >= threshold)
            ),
            "official_episodes": official_episodes,
        }
    checks = {
        "encoder_matches_replay": manifest_encoder == replay_encoder,
        "support_task_set_matches_manifest": set(task_names)
        == set(instruction_manifest["tasks"].values()),
        "all_official_prompts_rejected": bool(
            np.all(np.asarray(all_official_scores) < threshold)
        ),
        "all_canonical_routes_accepted": bool(
            np.all(np.asarray(all_routed_scores) >= threshold)
        ),
        "all_canonical_routes_resolve_correct_task": all(
            value["canonical_route_resolved_task"]
            == value["canonical_instruction"]
            for value in tasks.values()
        ),
    }
    payload = {
        "format": "fastwam_robotwin_residual_language_routing_audit_v1",
        "language_encoder_version": replay_encoder,
        "language_similarity_threshold": threshold,
        "official_prompt_summary": {
            "episodes": len(all_official_scores),
            "accepted": int(
                np.sum(np.asarray(all_official_scores) >= threshold)
            ),
            "minimum": float(np.min(all_official_scores)),
            "mean": float(np.mean(all_official_scores)),
            "maximum": float(np.max(all_official_scores)),
        },
        "canonical_route_summary": {
            "tasks": len(all_routed_scores),
            "accepted": int(np.sum(np.asarray(all_routed_scores) >= threshold)),
            "minimum": float(np.min(all_routed_scores)),
            "mean": float(np.mean(all_routed_scores)),
            "maximum": float(np.max(all_routed_scores)),
        },
        "checks": checks,
        "tasks": tasks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"language routing audit failed: {checks}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
