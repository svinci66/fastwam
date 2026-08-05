"""Build an episode-disjoint kNN support gate for a residual-IQL checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.rl.online_policy import load_residual_actor_checkpoint
from fastwam.rl.replay_buffer import ReplayBuffer
from fastwam.rl.support_gate import (
    SUPPORT_INDEX_FORMAT,
    ResidualSupportIndex,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--score-neighbors", type=int, default=3)
    parser.add_argument("--language-similarity-threshold", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_successes(replay: ReplayBuffer) -> dict[str, bool]:
    success: dict[str, bool] = defaultdict(bool)
    for transition in replay.transitions:
        success[transition.episode_id] |= bool(transition.success)
    return dict(success)


def _support_task_name(transition) -> str:
    """Return a stable task identity independent of instruction wording."""

    return f"{transition.task_suite}/task_{int(transition.task_id):04d}"


def _support_episode_indices(
    replay: ReplayBuffer,
) -> dict[str, list[str]]:
    successes = _episode_successes(replay)
    by_task: dict[str, set[str]] = defaultdict(set)
    for transition in replay.transitions:
        if (
            transition.behavior_mode in {"expert", "policy"}
            and successes[transition.episode_id]
        ):
            by_task[_support_task_name(transition)].add(transition.episode_id)
    return {task: sorted(episodes) for task, episodes in by_task.items()}


def _split_episodes(
    episodes_by_task: dict[str, list[str]],
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    reference: set[str] = set()
    calibration: set[str] = set()
    for task, episodes in sorted(episodes_by_task.items()):
        if len(episodes) < 2:
            raise ValueError(
                f"task {task!r} needs at least two successful expert/policy episodes, "
                f"found {len(episodes)}"
            )
        ordered = sorted(
            episodes,
            key=lambda episode: hashlib.sha256(
                f"{seed}:{task}:{episode}".encode("utf-8")
            ).digest(),
        )
        calibration_count = max(
            1, min(len(ordered) - 1, math.ceil(len(ordered) * calibration_fraction))
        )
        calibration.update(ordered[:calibration_count])
        reference.update(ordered[calibration_count:])
    return reference, calibration


def _robust_center_scale(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(value, axis=0).astype(np.float32)
    q25 = np.quantile(value, 0.25, axis=0)
    q75 = np.quantile(value, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-3).astype(np.float32)
    return center, scale


def _task_balanced_quantile(
    values: np.ndarray,
    labels: Iterable[str],
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = np.asarray(list(labels), dtype=str)
    if values.size == 0 or values.shape != labels.shape:
        raise ValueError("balanced quantile values and labels must be non-empty and aligned")
    counts = Counter(labels.tolist())
    weights = np.asarray([1.0 / counts[label] for label in labels], dtype=np.float64)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cutoff = quantile * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, cutoff, side="left")), values.size - 1)
    return float(values[order[index]])


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must be in (0, 1)")
    if not 0.5 <= args.quantile < 1.0:
        raise ValueError("quantile must be in [0.5, 1)")
    if args.neighbors <= 0:
        raise ValueError("neighbors must be positive")
    if args.score_neighbors <= 0 or args.score_neighbors > args.neighbors:
        raise ValueError("score-neighbors must be in [1, neighbors]")

    replay = ReplayBuffer.load(args.replay_dir)
    arrays = replay.arrays()
    if "language_feature" not in arrays:
        raise ValueError("support index requires language-conditioned replay data")
    actor, payload = load_residual_actor_checkpoint(args.checkpoint, device=args.device)
    episodes_by_task = _support_episode_indices(replay)
    reference_episodes, calibration_episodes = _split_episodes(
        episodes_by_task,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    reference_indices = np.asarray(
        [
            index
            for index, transition in enumerate(replay.transitions)
            if (
                transition.episode_id in reference_episodes
                and transition.behavior_mode in {"expert", "policy"}
            )
        ],
        dtype=np.int64,
    )
    calibration_indices = np.asarray(
        [
            index
            for index, transition in enumerate(replay.transitions)
            if (
                transition.episode_id in calibration_episodes
                and transition.behavior_mode in {"expert", "policy"}
            )
        ],
        dtype=np.int64,
    )
    if reference_indices.size == 0 or calibration_indices.size == 0:
        raise ValueError("episode split produced an empty reference or calibration set")

    task_names = tuple(sorted(episodes_by_task))
    task_to_id = {task: index for index, task in enumerate(task_names)}
    reference_task_ids = np.asarray(
        [task_to_id[_support_task_name(replay.transitions[index])] for index in reference_indices],
        dtype=np.int64,
    )
    language_prototypes = []
    for task in task_names:
        task_rows = [
            index
            for index in reference_indices
            if _support_task_name(replay.transitions[index]) == task
        ]
        prototype = np.mean(arrays["language_feature"][task_rows], axis=0)
        prototype /= np.linalg.norm(prototype)
        language_prototypes.append(prototype.astype(np.float32))
    language_prototypes_array = np.stack(language_prototypes)

    proprio_center, proprio_scale = _robust_center_scale(
        arrays["proprio"][reference_indices]
    )
    baseline_center, baseline_scale = _robust_center_scale(
        arrays["baseline_actions"][reference_indices]
    )
    residual_scale = np.asarray(actor.config.residual_scale, dtype=np.float32)
    reference_residuals = (
        arrays["executed_actions"][reference_indices]
        - arrays["baseline_actions"][reference_indices]
    )
    reference_episode_ids = np.asarray(
        [replay.transitions[index].episode_id for index in reference_indices]
    )
    support_for_radii = ResidualSupportIndex(
        observation_features=arrays["observation_feature"][reference_indices],
        proprio=arrays["proprio"][reference_indices],
        baseline_actions=arrays["baseline_actions"][reference_indices],
        residual_actions=reference_residuals,
        state_local_radius=np.ones(reference_indices.size, dtype=np.float32),
        action_local_radius=np.ones(reference_indices.size, dtype=np.float32),
        task_ids=reference_task_ids,
        task_names=task_names,
        language_prototypes=language_prototypes_array,
        proprio_center=proprio_center,
        proprio_scale=proprio_scale,
        baseline_center=baseline_center,
        baseline_scale=baseline_scale,
        residual_scale=residual_scale,
        state_threshold=1e9,
        action_threshold=1e9,
        state_increase_threshold=1e9,
        language_similarity_threshold=args.language_similarity_threshold,
        neighbors=args.neighbors,
        score_neighbors=args.score_neighbors,
    )
    enabled = residual_scale > 0.0
    reference_residual_z = (
        reference_residuals[..., enabled] / residual_scale[enabled]
    )
    state_local_radius: list[float] = []
    action_local_radius: list[float] = []
    for local_index, replay_index in enumerate(reference_indices):
        transition = replay.transitions[replay_index]
        candidates, distances = support_for_radii.state_distances(
            observation_feature=arrays["observation_feature"][replay_index],
            proprio=arrays["proprio"][replay_index],
            baseline_actions=arrays["baseline_actions"][replay_index],
            task_id=task_to_id[_support_task_name(transition)],
        )
        different_episode = (
            reference_episode_ids[candidates] != transition.episode_id
        )
        candidates = candidates[different_episode]
        distances = distances[different_episode]
        if candidates.size == 0:
            raise ValueError(
                f"no episode-disjoint reference neighbor for {transition.episode_id}"
            )
        order = np.argsort(distances)[: min(args.neighbors, candidates.size)]
        neighbors = candidates[order]
        state_local_radius.append(
            float(
                np.median(
                    distances[order[: min(args.score_neighbors, order.size)]]
                )
            )
        )
        residual_distances = np.sqrt(
            np.mean(
                np.square(
                    reference_residual_z[neighbors]
                    - reference_residual_z[local_index]
                ),
                axis=(1, 2),
            )
        )
        action_local_radius.append(
            float(
                np.median(
                    np.sort(residual_distances)[
                        : min(args.score_neighbors, residual_distances.size)
                    ]
                )
            )
        )
    state_local_radius_array = np.asarray(state_local_radius, dtype=np.float32)
    action_local_radius_array = np.asarray(action_local_radius, dtype=np.float32)
    for radii in (state_local_radius_array, action_local_radius_array):
        positive = radii[radii > 1e-6]
        floor = 1e-3 if positive.size == 0 else max(
            1e-3, float(np.quantile(positive, 0.1)) * 0.1
        )
        np.maximum(radii, floor, out=radii)
    support = ResidualSupportIndex(
        observation_features=arrays["observation_feature"][reference_indices],
        proprio=arrays["proprio"][reference_indices],
        baseline_actions=arrays["baseline_actions"][reference_indices],
        residual_actions=reference_residuals,
        state_local_radius=state_local_radius_array,
        action_local_radius=action_local_radius_array,
        task_ids=reference_task_ids,
        task_names=task_names,
        language_prototypes=language_prototypes_array,
        proprio_center=proprio_center,
        proprio_scale=proprio_scale,
        baseline_center=baseline_center,
        baseline_scale=baseline_scale,
        residual_scale=residual_scale,
        state_threshold=1e9,
        action_threshold=1e9,
        state_increase_threshold=1e9,
        language_similarity_threshold=args.language_similarity_threshold,
        neighbors=args.neighbors,
        score_neighbors=args.score_neighbors,
    )

    state_scores: list[float] = []
    action_scores: list[float] = []
    candidate_action_scores: list[float] = []
    calibration_tasks: list[str] = []
    state_scores_by_episode: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    device = torch.device(args.device)
    with torch.inference_mode():
        for index in calibration_indices:
            transition = replay.transitions[index]
            baseline = arrays["baseline_actions"][index]
            observed_residual = arrays["executed_actions"][index] - baseline
            observed_decision = support.evaluate(
                observation_feature=arrays["observation_feature"][index],
                proprio=arrays["proprio"][index],
                baseline_actions=baseline,
                candidate_residual_actions=observed_residual,
                language_feature=arrays["language_feature"][index],
            )
            context = np.concatenate(
                [arrays["observation_feature"][index], arrays["proprio"][index]]
            ).astype(np.float32)
            candidate = actor(
                torch.from_numpy(context).unsqueeze(0).to(device),
                torch.from_numpy(baseline).unsqueeze(0).to(device),
                language_feature=torch.from_numpy(
                    arrays["language_feature"][index]
                ).unsqueeze(0).to(device),
            )[0].cpu().numpy()
            candidate_decision = support.evaluate(
                observation_feature=arrays["observation_feature"][index],
                proprio=arrays["proprio"][index],
                baseline_actions=baseline,
                candidate_residual_actions=candidate - baseline,
                language_feature=arrays["language_feature"][index],
            )
            state_scores.append(observed_decision.state_score)
            action_scores.append(observed_decision.action_score)
            candidate_action_scores.append(candidate_decision.action_score)
            calibration_tasks.append(_support_task_name(transition))
            state_scores_by_episode[transition.episode_id].append(
                (
                    transition.transition_index,
                    observed_decision.state_score,
                    _support_task_name(transition),
                )
            )

    state_scores_array = np.asarray(state_scores, dtype=np.float32)
    action_scores_array = np.asarray(action_scores, dtype=np.float32)
    candidate_action_scores_array = np.asarray(candidate_action_scores, dtype=np.float32)
    state_threshold = _task_balanced_quantile(
        state_scores_array, calibration_tasks, args.quantile
    )
    action_threshold = _task_balanced_quantile(
        action_scores_array, calibration_tasks, args.quantile
    )
    increases: list[float] = []
    increase_tasks: list[str] = []
    for values in state_scores_by_episode.values():
        values.sort()
        for previous, current in zip(values, values[1:]):
            increases.append(max(0.0, current[1] - previous[1]))
            increase_tasks.append(current[2])
    state_increase_threshold = (
        state_threshold
        if not increases
        else _task_balanced_quantile(
            np.asarray(increases, dtype=np.float32), increase_tasks, args.quantile
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "arrays.npz",
        observation_features=arrays["observation_feature"][reference_indices],
        proprio=arrays["proprio"][reference_indices],
        baseline_actions=arrays["baseline_actions"][reference_indices],
        residual_actions=reference_residuals,
        state_local_radius=state_local_radius_array,
        action_local_radius=action_local_radius_array,
        task_ids=reference_task_ids,
        language_prototypes=language_prototypes_array,
        proprio_center=proprio_center,
        proprio_scale=proprio_scale,
        baseline_center=baseline_center,
        baseline_scale=baseline_scale,
        residual_scale=residual_scale,
    )
    calibration_summary = {
        "format": SUPPORT_INDEX_FORMAT,
        "task_names": list(task_names),
        "neighbors": args.neighbors,
        "score_neighbors": args.score_neighbors,
        "quantile": args.quantile,
        "calibration_fraction": args.calibration_fraction,
        "state_threshold": state_threshold,
        "action_threshold": action_threshold,
        "state_increase_threshold": state_increase_threshold,
        "language_similarity_threshold": args.language_similarity_threshold,
        "num_reference_episodes": len(reference_episodes),
        "num_calibration_episodes": len(calibration_episodes),
        "num_reference_transitions": int(reference_indices.size),
        "num_calibration_transitions": int(calibration_indices.size),
        "reference_episodes_by_task": {
            task: sum(episode in reference_episodes for episode in episodes)
            for task, episodes in episodes_by_task.items()
        },
        "calibration_episodes_by_task": {
            task: sum(episode in calibration_episodes for episode in episodes)
            for task, episodes in episodes_by_task.items()
        },
        "calibration_state_accept_rate": float(
            np.mean(state_scores_array <= state_threshold)
        ),
        "calibration_observed_action_accept_rate": float(
            np.mean(action_scores_array <= action_threshold)
        ),
        "calibration_actor_action_accept_rate": float(
            np.mean(candidate_action_scores_array <= action_threshold)
        ),
        "calibration_state_score_quantiles": {
            str(q): float(np.quantile(state_scores_array, q))
            for q in (0.0, 0.5, 0.9, 0.95, 1.0)
        },
        "calibration_observed_action_score_quantiles": {
            str(q): float(np.quantile(action_scores_array, q))
            for q in (0.0, 0.5, 0.9, 0.95, 1.0)
        },
        "calibration_actor_action_score_quantiles": {
            str(q): float(np.quantile(candidate_action_scores_array, q))
            for q in (0.0, 0.5, 0.9, 0.95, 1.0)
        },
        "replay_dir": str(args.replay_dir.resolve()),
        "replay_manifest_sha256": _sha256(args.replay_dir / "manifest.json"),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_format": payload["format"],
        "split_seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(calibration_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(calibration_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
