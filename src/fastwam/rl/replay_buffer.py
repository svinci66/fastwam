"""Versioned, compact replay storage for action-chunk LIBERO transitions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .rewards import (
    CompositeRewardConfig,
    EpisodeShapingBudget,
    IMAGINATION_REWARD_TYPES,
    RewardBreakdown,
    compute_composite_reward,
)


REPLAY_SCHEMA_VERSION = 3
_SUPPORTED_REPLAY_SCHEMA_VERSIONS = {1, 2, REPLAY_SCHEMA_VERSION}
_ARRAY_FILE = "arrays.npz"
_METADATA_FILE = "transitions.jsonl"
_MANIFEST_FILE = "manifest.json"


def _finite_float32(value: np.ndarray, name: str, *, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayTransition:
    episode_id: str
    transition_index: int
    task_suite: str
    task_id: int
    task_description: str
    env_seed: int
    goal_seed: int
    action_seed: int
    policy_version: str
    predictor_version: str
    reward_encoder_version: str
    behavior_mode: str
    action_noise_std: float
    target_k: int
    effective_k: int
    goal_frame_index: int
    goal_tau: float
    terminated: bool
    truncated: bool
    success: bool
    alignment_valid: bool
    observation_feature: np.ndarray
    next_observation_feature: np.ndarray
    goal_feature: np.ndarray
    proprio: np.ndarray
    next_proprio: np.ndarray
    baseline_actions: np.ndarray
    executed_actions: np.ndarray
    environment_rewards: np.ndarray
    reward: RewardBreakdown
    imagination_reward_type: str = "progress_v1"
    language_feature: np.ndarray | None = None
    language_encoder_version: str | None = None

    def validate(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if self.transition_index < 0:
            raise ValueError("transition_index must be non-negative")
        if not self.task_suite.strip() or not self.task_description.strip():
            raise ValueError("task_suite and task_description must not be empty")
        if self.task_id < 0:
            raise ValueError("task_id must be non-negative")
        versions = {
            "policy_version": self.policy_version,
            "predictor_version": self.predictor_version,
            "reward_encoder_version": self.reward_encoder_version,
        }
        if any(not value.strip() for value in versions.values()):
            raise ValueError(f"all component versions must be recorded, got {versions}")
        if self.imagination_reward_type not in IMAGINATION_REWARD_TYPES:
            raise ValueError(
                f"unsupported imagination_reward_type: {self.imagination_reward_type!r}"
            )
        if self.behavior_mode not in {"policy", "noise", "zero", "residual"}:
            raise ValueError(f"unsupported behavior_mode: {self.behavior_mode!r}")
        if not np.isfinite(self.action_noise_std) or self.action_noise_std < 0.0:
            raise ValueError("action_noise_std must be finite and non-negative")
        if self.behavior_mode != "noise" and self.action_noise_std != 0.0:
            raise ValueError("action_noise_std must be zero unless behavior_mode is 'noise'")
        if self.target_k <= 0:
            raise ValueError("target_k must be positive")
        if self.effective_k <= 0 or self.effective_k > self.target_k:
            raise ValueError(
                f"effective_k must be in [1, target_k={self.target_k}], got {self.effective_k}"
            )
        if self.goal_frame_index < 0:
            raise ValueError("goal_frame_index must be non-negative")
        if not np.isfinite(self.goal_tau) or self.goal_tau < 0.0:
            raise ValueError("goal_tau must be finite and non-negative")
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated must not both be true")
        if self.success and not self.terminated:
            raise ValueError("a successful LIBERO transition must be marked terminated")
        if self.alignment_valid and self.effective_k != self.target_k:
            raise ValueError(
                "alignment_valid requires effective_k == target_k; partial chunks need a matched goal"
            )
        if self.reward.alignment_valid != self.alignment_valid:
            raise ValueError("reward and transition alignment_valid flags disagree")

        if (self.language_feature is None) != (self.language_encoder_version is None):
            raise ValueError(
                "language_feature and language_encoder_version must be recorded together"
            )
        if self.language_feature is not None:
            _finite_float32(self.language_feature, "language_feature", ndim=1)
            if not str(self.language_encoder_version).strip():
                raise ValueError("language_encoder_version must not be empty")

        observation = _finite_float32(self.observation_feature, "observation_feature", ndim=1)
        next_observation = _finite_float32(
            self.next_observation_feature, "next_observation_feature", ndim=1
        )
        goal = _finite_float32(self.goal_feature, "goal_feature", ndim=1)
        if observation.shape != next_observation.shape:
            raise ValueError("observation and next_observation feature shapes must match")
        if goal.shape != observation.shape:
            raise ValueError(
                "goal_feature and observation_feature must share the frozen encoder feature shape"
            )

        proprio = _finite_float32(self.proprio, "proprio", ndim=1)
        next_proprio = _finite_float32(self.next_proprio, "next_proprio", ndim=1)
        if proprio.shape != next_proprio.shape:
            raise ValueError("proprio and next_proprio shapes must match")

        baseline = _finite_float32(self.baseline_actions, "baseline_actions", ndim=2)
        executed = _finite_float32(self.executed_actions, "executed_actions", ndim=2)
        if baseline.shape != executed.shape:
            raise ValueError("baseline_actions and executed_actions shapes must match")
        if baseline.shape[0] != self.target_k:
            raise ValueError(
                f"action horizon {baseline.shape[0]} must equal target_k={self.target_k}"
            )
        env_rewards = _finite_float32(self.environment_rewards, "environment_rewards", ndim=1)
        if env_rewards.shape != (self.target_k,):
            raise ValueError(
                f"environment_rewards must have shape {(self.target_k,)}, got {env_rewards.shape}"
            )
        reward_values = asdict(self.reward)
        for key, value in reward_values.items():
            if key == "alignment_valid":
                continue
            if not np.isfinite(float(value)):
                raise ValueError(f"reward.{key} must be finite, got {value}")

    def metadata_dict(self) -> dict[str, Any]:
        self.validate()
        array_fields = {
            "observation_feature",
            "next_observation_feature",
            "goal_feature",
            "proprio",
            "next_proprio",
            "baseline_actions",
            "executed_actions",
            "environment_rewards",
            "reward",
            "language_feature",
        }
        result = {
            key: value
            for key, value in self.__dict__.items()
            if key not in array_fields
        }
        result["reward"] = asdict(self.reward)
        return result


class ReplayBuffer:
    """In-memory MVP buffer with checksummed portable persistence.

    It is intentionally shard-sized rather than an unbounded production replay.
    Collectors should periodically save a new directory and release the large model.
    """

    def __init__(self, transitions: Iterable[ReplayTransition] | None = None):
        self.transitions: list[ReplayTransition] = []
        self._identities: set[tuple[str, int]] = set()
        for transition in transitions or ():
            self.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)

    def append(self, transition: ReplayTransition) -> None:
        transition.validate()
        identity = (transition.episode_id, transition.transition_index)
        if identity in self._identities:
            raise ValueError(f"duplicate replay transition identity: {identity}")
        if self.transitions:
            first = self.transitions[0]
            shape_pairs = {
                "observation_feature": (
                    first.observation_feature.shape,
                    transition.observation_feature.shape,
                ),
                "goal_feature": (first.goal_feature.shape, transition.goal_feature.shape),
                "proprio": (first.proprio.shape, transition.proprio.shape),
                "baseline_actions": (
                    first.baseline_actions.shape,
                    transition.baseline_actions.shape,
                ),
            }
            mismatches = {name: shapes for name, shapes in shape_pairs.items() if shapes[0] != shapes[1]}
            if mismatches:
                raise ValueError(f"transition shapes differ from replay schema: {mismatches}")
            if transition.reward_encoder_version != first.reward_encoder_version:
                raise ValueError("a replay shard cannot mix reward encoder versions")
            if (transition.language_feature is None) != (first.language_feature is None):
                raise ValueError("a replay shard cannot mix transitions with and without language")
            if transition.language_feature is not None:
                if transition.language_feature.shape != first.language_feature.shape:
                    raise ValueError("language feature shapes differ within replay shard")
                if transition.language_encoder_version != first.language_encoder_version:
                    raise ValueError("a replay shard cannot mix language encoder versions")
            if transition.imagination_reward_type != first.imagination_reward_type:
                raise ValueError("a replay shard cannot mix imagination reward types")
            if transition.target_k != first.target_k:
                raise ValueError("a replay shard cannot mix target_k values")
        self.transitions.append(transition)
        self._identities.add(identity)

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.transitions:
            raise ValueError("cannot materialize an empty replay")
        arrays = {
            "observation_feature": np.stack(
                [item.observation_feature for item in self.transitions]
            ).astype(np.float32),
            "next_observation_feature": np.stack(
                [item.next_observation_feature for item in self.transitions]
            ).astype(np.float32),
            "goal_feature": np.stack([item.goal_feature for item in self.transitions]).astype(
                np.float32
            ),
            "proprio": np.stack([item.proprio for item in self.transitions]).astype(np.float32),
            "next_proprio": np.stack([item.next_proprio for item in self.transitions]).astype(
                np.float32
            ),
            "baseline_actions": np.stack(
                [item.baseline_actions for item in self.transitions]
            ).astype(np.float32),
            "executed_actions": np.stack(
                [item.executed_actions for item in self.transitions]
            ).astype(np.float32),
            "environment_rewards": np.stack(
                [item.environment_rewards for item in self.transitions]
            ).astype(np.float32),
            "effective_k": np.asarray(
                [item.effective_k for item in self.transitions], dtype=np.int64
            ),
            "terminated": np.asarray(
                [item.terminated for item in self.transitions], dtype=np.bool_
            ),
            "truncated": np.asarray(
                [item.truncated for item in self.transitions], dtype=np.bool_
            ),
            "total_reward": np.asarray(
                [item.reward.total for item in self.transitions], dtype=np.float32
            ),
        }
        if self.transitions[0].language_feature is not None:
            arrays["language_feature"] = np.stack(
                [item.language_feature for item in self.transitions]
            ).astype(np.float32)
        return arrays

    def monte_carlo_returns(
        self,
        gamma: float,
        *,
        timeout_bootstrap_values: Mapping[str, float] | None = None,
        transition_rewards: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return transition-aligned discounted returns without timeout ambiguity.

        Truncated episodes require an explicit bootstrap value.  This prevents a
        timeout from being silently treated as a true terminal state.
        """

        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        returns = np.zeros(len(self.transitions), dtype=np.float32)
        if transition_rewards is None:
            reward_values = np.asarray(
                [transition.reward.total for transition in self.transitions], dtype=np.float32
            )
        else:
            reward_values = np.asarray(transition_rewards, dtype=np.float32)
            if reward_values.shape != (len(self.transitions),):
                raise ValueError(
                    f"transition_rewards must have shape {(len(self.transitions),)}, "
                    f"got {reward_values.shape}"
                )
            if np.any(~np.isfinite(reward_values)):
                raise ValueError("transition_rewards must contain only finite values")
        episode_indices: dict[str, list[int]] = {}
        for index, transition in enumerate(self.transitions):
            episode_indices.setdefault(transition.episode_id, []).append(index)

        for episode_id, indices in episode_indices.items():
            indices.sort(key=lambda index: self.transitions[index].transition_index)
            ordered_steps = [self.transitions[index].transition_index for index in indices]
            if ordered_steps != list(range(len(indices))):
                raise ValueError(
                    f"episode {episode_id!r} transition indices must be contiguous from zero, "
                    f"got {ordered_steps}"
                )
            last = self.transitions[indices[-1]]
            if not (last.terminated or last.truncated):
                raise ValueError(f"episode {episode_id!r} does not end with terminated or truncated")
            if last.truncated:
                if timeout_bootstrap_values is None or episode_id not in timeout_bootstrap_values:
                    raise ValueError(
                        f"truncated episode {episode_id!r} requires an explicit bootstrap value"
                    )
                running = float(timeout_bootstrap_values[episode_id])
            else:
                running = 0.0
            if not np.isfinite(running):
                raise ValueError(f"bootstrap for episode {episode_id!r} must be finite")
            for index in reversed(indices):
                transition = self.transitions[index]
                running = float(reward_values[index]) + (gamma ** transition.effective_k) * running
                returns[index] = running
        return returns

    def relabel_rewards(
        self,
        config: CompositeRewardConfig,
        *,
        imitation_dimension_scales: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[RewardBreakdown]]:
        """Recompute one reward ablation from immutable raw replay signals.

        A/B/C/D comparisons can therefore share exactly the same transitions.  The
        episode shaping budget is reset per episode and consumed in transition order.
        """

        config.validate()
        reward_types = {transition.imagination_reward_type for transition in self.transitions}
        if reward_types != {config.imagination_reward_type}:
            raise ValueError(
                "reward config cannot reinterpret replay imagination values: "
                f"replay={sorted(reward_types)} config={config.imagination_reward_type!r}"
            )
        order = sorted(
            range(len(self.transitions)),
            key=lambda index: (
                self.transitions[index].episode_id,
                self.transitions[index].transition_index,
            ),
        )
        budgets: dict[str, EpisodeShapingBudget] = {}
        breakdowns: list[RewardBreakdown | None] = [None] * len(self.transitions)
        for index in order:
            transition = self.transitions[index]
            budget = budgets.setdefault(
                transition.episode_id, EpisodeShapingBudget.from_config(config)
            )
            breakdowns[index] = compute_composite_reward(
                environment_rewards=transition.environment_rewards,
                success=transition.success,
                baseline_actions=transition.baseline_actions,
                executed_actions=transition.executed_actions,
                effective_k=transition.effective_k,
                imagination_progress=transition.reward.imagination_raw,
                alignment_valid=transition.alignment_valid,
                config=config,
                shaping_budget=budget,
                imitation_dimension_scales=imitation_dimension_scales,
            )
        typed_breakdowns = [item for item in breakdowns if item is not None]
        if len(typed_breakdowns) != len(self.transitions):
            raise RuntimeError("internal reward relabeling error")
        totals = np.asarray([item.total for item in typed_breakdowns], dtype=np.float32)
        return totals, typed_breakdowns

    def save(
        self,
        directory: str | Path,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> Path:
        if not self.transitions:
            raise ValueError("cannot save an empty replay")
        target = Path(directory).expanduser().resolve()
        if target.exists():
            raise FileExistsError(f"replay directory already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
        try:
            arrays_path = temporary / _ARRAY_FILE
            with arrays_path.open("wb") as stream:
                np.savez_compressed(stream, **self.arrays())
            metadata_path = temporary / _METADATA_FILE
            with metadata_path.open("w", encoding="utf-8") as stream:
                for transition in self.transitions:
                    stream.write(json.dumps(transition.metadata_dict(), sort_keys=True) + "\n")
            arrays = self.arrays()
            manifest = {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "num_transitions": len(self.transitions),
                "target_k": self.transitions[0].target_k,
                "reward_encoder_version": self.transitions[0].reward_encoder_version,
                "language_encoder_version": self.transitions[0].language_encoder_version,
                "imagination_reward_type": self.transitions[0].imagination_reward_type,
                "provenance": dict(provenance or {}),
                "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
                "files": {
                    _ARRAY_FILE: {"sha256": _sha256(arrays_path)},
                    _METADATA_FILE: {"sha256": _sha256(metadata_path)},
                },
            }
            manifest_path = temporary / _MANIFEST_FILE
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return target

    @classmethod
    def load(cls, directory: str | Path, *, verify_checksums: bool = True) -> "ReplayBuffer":
        root = Path(directory).expanduser().resolve()
        manifest = json.loads((root / _MANIFEST_FILE).read_text())
        schema_version = int(manifest.get("schema_version", -1))
        if schema_version not in _SUPPORTED_REPLAY_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported replay schema {manifest.get('schema_version')}; "
                f"expected one of {sorted(_SUPPORTED_REPLAY_SCHEMA_VERSIONS)}"
            )
        if verify_checksums:
            for name, record in manifest["files"].items():
                actual = _sha256(root / name)
                if actual != record["sha256"]:
                    raise ValueError(f"checksum mismatch for {name}: {actual} != {record['sha256']}")
        with np.load(root / _ARRAY_FILE, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        metadata = [
            json.loads(line)
            for line in (root / _METADATA_FILE).read_text().splitlines()
            if line.strip()
        ]
        if len(metadata) != int(manifest["num_transitions"]):
            raise ValueError("metadata transition count does not match manifest")
        if any(arrays[key].shape[0] != len(metadata) for key in arrays):
            raise ValueError("one or more replay arrays have an invalid leading dimension")

        transitions = []
        for index, record in enumerate(metadata):
            reward = RewardBreakdown(**record.pop("reward"))
            if schema_version == 1:
                record.setdefault("imagination_reward_type", "progress_v1")
            if schema_version < 3:
                record.setdefault("language_encoder_version", None)
            transition = ReplayTransition(
                **record,
                observation_feature=arrays["observation_feature"][index],
                next_observation_feature=arrays["next_observation_feature"][index],
                goal_feature=arrays["goal_feature"][index],
                proprio=arrays["proprio"][index],
                next_proprio=arrays["next_proprio"][index],
                baseline_actions=arrays["baseline_actions"][index],
                executed_actions=arrays["executed_actions"][index],
                environment_rewards=arrays["environment_rewards"][index],
                language_feature=(
                    arrays["language_feature"][index]
                    if "language_feature" in arrays
                    else None
                ),
                reward=reward,
            )
            transitions.append(transition)
        return cls(transitions)
