"""Episode-calibrated data-support checks for online residual actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SUPPORT_INDEX_FORMAT = "fastwam_residual_support_v1"


def _finite_array(value: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite non-empty {ndim}D array")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class SupportGateDecision:
    task_name: str | None
    language_similarity: float
    state_score: float
    action_score: float
    state_threshold: float
    action_threshold: float
    state_in_support: bool
    action_in_support: bool

    @property
    def in_support(self) -> bool:
        return self.state_in_support and self.action_in_support


class ResidualSupportIndex:
    """Nearest-neighbour support index built only from successful episodes.

    State distance combines frozen visual features, globally robust-normalized
    proprioception, and globally robust-normalized FastWAM baseline chunks.
    Candidate residual support is measured against residuals actually executed
    in the nearest supported states. Thresholds are stored in the artifact and
    must be calibrated on episode-disjoint data.
    """

    def __init__(
        self,
        *,
        observation_features: np.ndarray,
        proprio: np.ndarray,
        baseline_actions: np.ndarray,
        residual_actions: np.ndarray,
        state_local_radius: np.ndarray,
        action_local_radius: np.ndarray,
        task_ids: np.ndarray,
        task_names: tuple[str, ...],
        language_prototypes: np.ndarray,
        proprio_center: np.ndarray,
        proprio_scale: np.ndarray,
        baseline_center: np.ndarray,
        baseline_scale: np.ndarray,
        residual_scale: np.ndarray,
        state_threshold: float,
        action_threshold: float,
        state_increase_threshold: float,
        language_similarity_threshold: float,
        neighbors: int,
        score_neighbors: int = 3,
        artifact_path: str | Path | None = None,
    ):
        observation_features = _finite_array(
            observation_features, name="observation_features", ndim=2
        )
        proprio = _finite_array(proprio, name="proprio", ndim=2)
        baseline_actions = _finite_array(
            baseline_actions, name="baseline_actions", ndim=3
        )
        residual_actions = _finite_array(
            residual_actions, name="residual_actions", ndim=3
        )
        task_ids = np.asarray(task_ids, dtype=np.int64).reshape(-1)
        state_local_radius = np.asarray(state_local_radius, dtype=np.float32).reshape(-1)
        action_local_radius = np.asarray(action_local_radius, dtype=np.float32).reshape(-1)
        language_prototypes = _finite_array(
            language_prototypes, name="language_prototypes", ndim=2
        )
        count = observation_features.shape[0]
        if not (
            proprio.shape[0]
            == baseline_actions.shape[0]
            == residual_actions.shape[0]
            == task_ids.shape[0]
            == state_local_radius.shape[0]
            == action_local_radius.shape[0]
            == count
        ):
            raise ValueError("support reference arrays have inconsistent row counts")
        if baseline_actions.shape != residual_actions.shape:
            raise ValueError("baseline and residual action shapes must match")
        if len(task_names) != language_prototypes.shape[0]:
            raise ValueError("task names and language prototypes must match")
        if np.any(task_ids < 0) or np.any(task_ids >= len(task_names)):
            raise ValueError("task_ids contain an invalid task index")
        if neighbors <= 0:
            raise ValueError("neighbors must be positive")
        if score_neighbors <= 0 or score_neighbors > neighbors:
            raise ValueError("score_neighbors must be in [1, neighbors]")
        if (
            np.any(~np.isfinite(state_local_radius))
            or np.any(state_local_radius <= 0.0)
            or np.any(~np.isfinite(action_local_radius))
            or np.any(action_local_radius <= 0.0)
        ):
            raise ValueError("local support radii must be finite and positive")
        thresholds = {
            "state_threshold": state_threshold,
            "action_threshold": action_threshold,
            "state_increase_threshold": state_increase_threshold,
        }
        if any(not np.isfinite(value) or value < 0.0 for value in thresholds.values()):
            raise ValueError(f"support thresholds must be finite and non-negative: {thresholds}")
        if not -1.0 <= language_similarity_threshold <= 1.0:
            raise ValueError("language similarity threshold must be in [-1, 1]")

        self.observation_features = observation_features
        self.proprio = proprio
        self.baseline_actions = baseline_actions
        self.residual_actions = residual_actions
        self.state_local_radius = state_local_radius
        self.action_local_radius = action_local_radius
        self.task_ids = task_ids
        self.task_names = tuple(task_names)
        self.language_prototypes = language_prototypes
        self.proprio_center = np.asarray(proprio_center, dtype=np.float32)
        self.proprio_scale = np.asarray(proprio_scale, dtype=np.float32)
        self.baseline_center = np.asarray(baseline_center, dtype=np.float32)
        self.baseline_scale = np.asarray(baseline_scale, dtype=np.float32)
        self.residual_scale = np.asarray(residual_scale, dtype=np.float32).reshape(-1)
        if self.proprio_center.shape != proprio.shape[1:] or self.proprio_scale.shape != proprio.shape[1:]:
            raise ValueError("proprio normalization shape mismatch")
        if self.baseline_center.shape != baseline_actions.shape[1:] or self.baseline_scale.shape != baseline_actions.shape[1:]:
            raise ValueError("baseline normalization shape mismatch")
        if self.residual_scale.shape != (baseline_actions.shape[2],):
            raise ValueError("residual_scale action dimension mismatch")
        if np.any(self.proprio_scale <= 0.0) or np.any(self.baseline_scale <= 0.0):
            raise ValueError("normalization scales must be positive")
        if not np.any(self.residual_scale > 0.0):
            raise ValueError("at least one residual action dimension must be enabled")
        prototype_norms = np.linalg.norm(self.language_prototypes, axis=1, keepdims=True)
        if np.any(prototype_norms <= 0.0):
            raise ValueError("language prototypes must have non-zero norm")
        self.language_prototypes = self.language_prototypes / prototype_norms
        self.state_threshold = float(state_threshold)
        self.action_threshold = float(action_threshold)
        self.state_increase_threshold = float(state_increase_threshold)
        self.language_similarity_threshold = float(language_similarity_threshold)
        self.neighbors = int(neighbors)
        self.score_neighbors = int(score_neighbors)
        self.artifact_path = None if artifact_path is None else str(Path(artifact_path).resolve())

        self._proprio_z = (self.proprio - self.proprio_center) / self.proprio_scale
        self._baseline_z = (
            self.baseline_actions - self.baseline_center
        ) / self.baseline_scale
        enabled = self.residual_scale > 0.0
        self._enabled_action_dimensions = enabled
        self._residual_z = (
            self.residual_actions[..., enabled] / self.residual_scale[enabled]
        )

    @property
    def action_horizon(self) -> int:
        return int(self.baseline_actions.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.baseline_actions.shape[2])

    def _resolve_task(self, language_feature: np.ndarray) -> tuple[int | None, float]:
        language = np.asarray(language_feature, dtype=np.float32).reshape(-1)
        if language.shape != (self.language_prototypes.shape[1],) or not np.all(
            np.isfinite(language)
        ):
            raise ValueError(
                "language_feature shape does not match the support index prototypes"
            )
        norm = float(np.linalg.norm(language))
        if norm <= 0.0:
            raise ValueError("language_feature has zero norm")
        similarities = self.language_prototypes @ (language / norm)
        task_id = int(np.argmax(similarities))
        similarity = float(similarities[task_id])
        if similarity < self.language_similarity_threshold:
            return None, similarity
        return task_id, similarity

    def state_distances(
        self,
        *,
        observation_feature: np.ndarray,
        proprio: np.ndarray,
        baseline_actions: np.ndarray,
        task_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        observation = np.asarray(observation_feature, dtype=np.float32).reshape(-1)
        state = np.asarray(proprio, dtype=np.float32).reshape(-1)
        baseline = np.asarray(baseline_actions, dtype=np.float32)
        if observation.shape != self.observation_features.shape[1:]:
            raise ValueError("observation feature shape does not match support index")
        if state.shape != self.proprio.shape[1:]:
            raise ValueError("proprio shape does not match support index")
        if baseline.shape != self.baseline_actions.shape[1:]:
            raise ValueError("baseline action shape does not match support index")
        candidates = np.flatnonzero(self.task_ids == task_id)
        if candidates.size == 0:
            return candidates, np.empty(0, dtype=np.float32)
        observation_norm = float(np.linalg.norm(observation))
        if observation_norm <= 0.0:
            raise ValueError("observation feature has zero norm")
        reference = self.observation_features[candidates]
        reference_norms = np.linalg.norm(reference, axis=1)
        visual = np.linalg.norm(
            reference / reference_norms[:, None] - observation / observation_norm,
            axis=1,
        )
        proprio_z = (state - self.proprio_center) / self.proprio_scale
        proprio_distance = np.sqrt(
            np.mean(np.square(self._proprio_z[candidates] - proprio_z), axis=1)
        )
        baseline_z = (baseline - self.baseline_center) / self.baseline_scale
        baseline_distance = np.sqrt(
            np.mean(
                np.square(self._baseline_z[candidates] - baseline_z), axis=(1, 2)
            )
        )
        combined = np.sqrt(
            (np.square(visual) + np.square(proprio_distance) + np.square(baseline_distance))
            / 3.0
        )
        return candidates, combined.astype(np.float32, copy=False)

    def evaluate(
        self,
        *,
        observation_feature: np.ndarray,
        proprio: np.ndarray,
        baseline_actions: np.ndarray,
        candidate_residual_actions: np.ndarray,
        language_feature: np.ndarray,
    ) -> SupportGateDecision:
        task_id, similarity = self._resolve_task(language_feature)
        if task_id is None:
            return SupportGateDecision(
                task_name=None,
                language_similarity=similarity,
                state_score=float("inf"),
                action_score=float("inf"),
                state_threshold=self.state_threshold,
                action_threshold=self.action_threshold,
                state_in_support=False,
                action_in_support=False,
            )
        candidates, distances = self.state_distances(
            observation_feature=observation_feature,
            proprio=proprio,
            baseline_actions=baseline_actions,
            task_id=task_id,
        )
        if candidates.size == 0:
            state_score = action_score = float("inf")
        else:
            normalized_state = distances / self.state_local_radius[candidates]
            order = np.argsort(normalized_state)[: min(self.neighbors, candidates.size)]
            neighbors = candidates[order]
            state_score = float(
                np.median(
                    normalized_state[order[: min(self.score_neighbors, order.size)]]
                )
            )
            candidate = np.asarray(candidate_residual_actions, dtype=np.float32)
            expected = self.residual_actions.shape[1:]
            if candidate.shape != expected or not np.all(np.isfinite(candidate)):
                raise ValueError(
                    f"candidate residual must be finite with shape {expected}, got {candidate.shape}"
                )
            candidate_z = (
                candidate[..., self._enabled_action_dimensions]
                / self.residual_scale[self._enabled_action_dimensions]
            )
            residual_distances = np.sqrt(
                np.mean(
                    np.square(self._residual_z[neighbors] - candidate_z),
                    axis=(1, 2),
                )
            )
            normalized_action = (
                residual_distances / self.action_local_radius[neighbors]
            )
            action_score = float(
                np.median(
                    np.sort(normalized_action)[
                        : min(self.score_neighbors, normalized_action.size)
                    ]
                )
            )
        return SupportGateDecision(
            task_name=self.task_names[task_id],
            language_similarity=similarity,
            state_score=state_score,
            action_score=action_score,
            state_threshold=self.state_threshold,
            action_threshold=self.action_threshold,
            state_in_support=state_score <= self.state_threshold,
            action_in_support=action_score <= self.action_threshold,
        )

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "ResidualSupportIndex":
        path = Path(artifact_dir).expanduser().resolve()
        metadata_path = path / "metadata.json"
        arrays_path = path / "arrays.npz"
        if not metadata_path.is_file() or not arrays_path.is_file():
            raise FileNotFoundError(
                f"support index requires metadata.json and arrays.npz in {path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != SUPPORT_INDEX_FORMAT:
            raise ValueError(f"unsupported support index format: {metadata.get('format')!r}")
        with np.load(arrays_path, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        return cls(
            observation_features=arrays["observation_features"],
            proprio=arrays["proprio"],
            baseline_actions=arrays["baseline_actions"],
            residual_actions=arrays["residual_actions"],
            state_local_radius=arrays["state_local_radius"],
            action_local_radius=arrays["action_local_radius"],
            task_ids=arrays["task_ids"],
            task_names=tuple(metadata["task_names"]),
            language_prototypes=arrays["language_prototypes"],
            proprio_center=arrays["proprio_center"],
            proprio_scale=arrays["proprio_scale"],
            baseline_center=arrays["baseline_center"],
            baseline_scale=arrays["baseline_scale"],
            residual_scale=arrays["residual_scale"],
            state_threshold=float(metadata["state_threshold"]),
            action_threshold=float(metadata["action_threshold"]),
            state_increase_threshold=float(metadata["state_increase_threshold"]),
            language_similarity_threshold=float(
                metadata["language_similarity_threshold"]
            ),
            neighbors=int(metadata["neighbors"]),
            score_neighbors=int(metadata.get("score_neighbors", 3)),
            artifact_path=path,
        )
