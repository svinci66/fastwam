"""Non-invasive per-action physics audit for RoboTwin evaluations.

The audit wraps an environment instance's ``take_action`` method and records
the task actor immediately before and after every policy action. It does not
change the action, simulator state, physics parameters, or success predicate.
Each JSONL record is flushed immediately so evidence survives an interrupted
or crashed rollout.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _json_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    return [float(item) for item in array]


def _safe_call(obj: Any, method_name: str) -> Any:
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None


def _unwrap_dynamic_component(actor: Any) -> Any:
    entity = getattr(actor, "actor", actor)
    components = _safe_call(entity, "get_components") or []
    for component in components:
        if hasattr(component, "get_linear_velocity"):
            return component
    return None


def _actor_name(actor: Any) -> str | None:
    name = _safe_call(actor, "get_name")
    if name is not None:
        return str(name)
    entity = getattr(actor, "actor", actor)
    name = _safe_call(entity, "get_name")
    if name is not None:
        return str(name)
    name = getattr(entity, "name", None)
    return None if name is None else str(name)


def _contact_snapshot(task_env: Any, actor_name: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "partners": [],
        "gripper_contact": False,
        "contact_point_count": 0,
        "max_impulse": 0.0,
        "total_impulse": 0.0,
    }
    if actor_name is None:
        result["error"] = "actor_name_unavailable"
        return result
    scene = getattr(task_env, "scene", None)
    contacts = _safe_call(scene, "get_contacts") if scene is not None else None
    if contacts is None:
        result["error"] = "contacts_unavailable"
        return result

    gripper_names = set(getattr(getattr(task_env, "robot", None), "gripper_name", []))
    partners: set[str] = set()
    impulses: list[float] = []
    for contact in contacts:
        try:
            names = [str(body.entity.name) for body in contact.bodies]
        except Exception:
            continue
        if actor_name not in names:
            continue
        partner = names[1] if names[0] == actor_name else names[0]
        partners.add(partner)
        result["gripper_contact"] = bool(
            result["gripper_contact"] or partner in gripper_names
        )
        for point in getattr(contact, "points", []):
            impulse = _json_vector(getattr(point, "impulse", None))
            if impulse is not None:
                impulses.append(float(np.linalg.norm(np.asarray(impulse))))

    result["partners"] = sorted(partners)
    result["contact_point_count"] = len(impulses)
    if impulses:
        result["max_impulse"] = max(impulses)
        result["total_impulse"] = sum(impulses)
    return result


class EpisodePhysicsAudit:
    """Record one task actor around each call to ``take_action``."""

    def __init__(
        self,
        *,
        output_path: str | Path,
        task_name: str,
        episode_id: int,
        seed: int,
        actor_attr: str = "can",
        policy_metadata: dict[str, Any] | None = None,
        displacement_threshold_m: float = 0.35,
        linear_speed_threshold_mps: float = 2.0,
        angular_speed_threshold_rps: float = 40.0,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.task_name = str(task_name)
        self.episode_id = int(episode_id)
        self.seed = int(seed)
        self.actor_attr = str(actor_attr)
        self.policy_metadata = dict(policy_metadata or {})
        self.displacement_threshold_m = float(displacement_threshold_m)
        self.linear_speed_threshold_mps = float(linear_speed_threshold_mps)
        self.angular_speed_threshold_rps = float(angular_speed_threshold_rps)

        self._task_env: Any = None
        self._actor: Any = None
        self._actor_name: str | None = None
        self._initial_position: np.ndarray | None = None
        self._original_take_action: Callable[..., Any] | None = None
        self._stream: Any = None
        self._action_index = 0
        self._max_displacement = 0.0
        self._max_linear_speed = 0.0
        self._max_angular_speed = 0.0
        self._first_gripper_contact_action: int | None = None
        self._first_anomaly_action: int | None = None
        self._anomaly_counts: dict[str, int] = {}

    def _write(self, payload: dict[str, Any]) -> None:
        if self._stream is None:
            raise RuntimeError("Physics audit is not installed")
        payload = {
            "schema_version": 1,
            "task_name": self.task_name,
            "episode_id": self.episode_id,
            "seed": self.seed,
            **payload,
        }
        self._stream.write(json.dumps(payload, sort_keys=True) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def _snapshot(self) -> dict[str, Any]:
        pose = _safe_call(self._actor, "get_pose")
        position = _json_vector(getattr(pose, "p", None))
        quaternion = _json_vector(getattr(pose, "q", None))
        component = _unwrap_dynamic_component(self._actor)
        linear_velocity = _json_vector(
            _safe_call(component, "get_linear_velocity")
            if component is not None
            else None
        )
        angular_velocity = _json_vector(
            _safe_call(component, "get_angular_velocity")
            if component is not None
            else None
        )
        mass_value = (
            _safe_call(component, "get_mass") if component is not None else None
        )

        finite_values = []
        for value in (position, quaternion, linear_velocity, angular_velocity):
            if value is not None:
                finite_values.extend(value)
        finite = bool(finite_values) and all(math.isfinite(value) for value in finite_values)
        displacement = None
        if position is not None and self._initial_position is not None:
            displacement = float(
                np.linalg.norm(np.asarray(position) - self._initial_position)
            )
        linear_speed = (
            None
            if linear_velocity is None
            else float(np.linalg.norm(np.asarray(linear_velocity)))
        )
        angular_speed = (
            None
            if angular_velocity is None
            else float(np.linalg.norm(np.asarray(angular_velocity)))
        )
        return {
            "position": position,
            "quaternion_wxyz": quaternion,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "linear_speed_mps": linear_speed,
            "angular_speed_rps": angular_speed,
            "mass_kg": None if mass_value is None else float(mass_value),
            "displacement_from_initial_m": displacement,
            "finite": finite,
            "contacts": _contact_snapshot(self._task_env, self._actor_name),
        }

    def _flags(
        self, pre: dict[str, Any], post: dict[str, Any]
    ) -> tuple[list[str], float | None]:
        flags: list[str] = []
        if not post["finite"]:
            flags.append("non_finite_state")

        action_displacement = None
        if pre["position"] is not None and post["position"] is not None:
            action_displacement = float(
                np.linalg.norm(
                    np.asarray(post["position"]) - np.asarray(pre["position"])
                )
            )
            if action_displacement > self.displacement_threshold_m:
                flags.append("large_single_action_displacement")

        linear_speed = post["linear_speed_mps"]
        if linear_speed is not None and linear_speed > self.linear_speed_threshold_mps:
            flags.append("high_linear_speed")
        angular_speed = post["angular_speed_rps"]
        if angular_speed is not None and angular_speed > self.angular_speed_threshold_rps:
            flags.append("high_angular_speed")

        position = post["position"]
        if position is not None and len(position) >= 3:
            x, y, z = position[:3]
            if abs(x) > 2.0 or abs(y) > 2.0 or z < 0.0 or z > 3.0:
                flags.append("outside_conservative_workspace")

        return flags, action_displacement

    def install(self, task_env: Any) -> None:
        if self._original_take_action is not None:
            raise RuntimeError("Physics audit is already installed")
        self._task_env = task_env
        self._actor = getattr(task_env, self.actor_attr, None)
        if self._actor is None:
            raise AttributeError(
                f"Task {self.task_name!r} has no actor attribute {self.actor_attr!r}"
            )
        self._actor_name = _actor_name(self._actor)
        initial_pose = _safe_call(self._actor, "get_pose")
        initial_position = _json_vector(getattr(initial_pose, "p", None))
        if initial_position is None or len(initial_position) != 3:
            raise RuntimeError("Audited actor initial position is unavailable")
        self._initial_position = np.asarray(initial_position, dtype=np.float64)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.output_path.open("w", encoding="utf-8")
        self._original_take_action = task_env.take_action

        def audited_take_action(action: Any, *args: Any, **kwargs: Any) -> Any:
            if self._original_take_action is None:
                raise RuntimeError("Physics audit was detached during an action")
            action_index = self._action_index
            pre_count = int(getattr(task_env, "take_action_cnt", -1))
            pre = self._snapshot()
            action_array = _json_vector(action)
            result = self._original_take_action(action, *args, **kwargs)
            post = self._snapshot()
            flags, action_displacement = self._flags(pre, post)

            if post["contacts"]["gripper_contact"] and self._first_gripper_contact_action is None:
                self._first_gripper_contact_action = action_index
            if flags and self._first_anomaly_action is None:
                self._first_anomaly_action = action_index
            for flag in flags:
                self._anomaly_counts[flag] = self._anomaly_counts.get(flag, 0) + 1
            displacement = post["displacement_from_initial_m"]
            if displacement is not None:
                self._max_displacement = max(self._max_displacement, displacement)
            if post["linear_speed_mps"] is not None:
                self._max_linear_speed = max(
                    self._max_linear_speed, post["linear_speed_mps"]
                )
            if post["angular_speed_rps"] is not None:
                self._max_angular_speed = max(
                    self._max_angular_speed, post["angular_speed_rps"]
                )

            self._write(
                {
                    "event": "action",
                    "action_index": action_index,
                    "take_action_cnt_before": pre_count,
                    "take_action_cnt_after": int(
                        getattr(task_env, "take_action_cnt", -1)
                    ),
                    "action": action_array,
                    "pre": pre,
                    "post": post,
                    "single_action_displacement_m": action_displacement,
                    "anomaly_flags": flags,
                }
            )
            self._action_index += 1
            return result

        task_env.take_action = audited_take_action
        self._write(
            {
                "event": "start",
                "actor_attr": self.actor_attr,
                "actor_name": self._actor_name,
                "policy": self.policy_metadata,
                "thresholds": {
                    "single_action_displacement_m": self.displacement_threshold_m,
                    "linear_speed_mps": self.linear_speed_threshold_mps,
                    "angular_speed_rps": self.angular_speed_threshold_rps,
                    "workspace_abs_xy_m": 2.0,
                    "workspace_z_m": [0.0, 3.0],
                },
                "initial": self._snapshot(),
            }
        )

    def finish(self, *, success: bool) -> dict[str, Any]:
        summary = {
            "event": "finish",
            "success": bool(success),
            "num_actions": self._action_index,
            "first_gripper_contact_action": self._first_gripper_contact_action,
            "first_anomaly_action": self._first_anomaly_action,
            "max_displacement_from_initial_m": self._max_displacement,
            "max_linear_speed_mps": self._max_linear_speed,
            "max_angular_speed_rps": self._max_angular_speed,
            "anomaly_counts": dict(sorted(self._anomaly_counts.items())),
            "final": self._snapshot(),
        }
        self._write(summary)
        if self._task_env is not None and self._original_take_action is not None:
            self._task_env.take_action = self._original_take_action
        self._original_take_action = None
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        return summary
