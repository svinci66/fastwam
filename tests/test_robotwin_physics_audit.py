import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.physics_audit import EpisodePhysicsAudit


class _Pose:
    def __init__(self):
        self.p = np.array([0.1, 0.2, 0.75], dtype=np.float64)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


class _Component:
    def __init__(self):
        self.linear_velocity = np.zeros(3, dtype=np.float64)
        self.angular_velocity = np.zeros(3, dtype=np.float64)

    def get_linear_velocity(self):
        return self.linear_velocity

    def get_angular_velocity(self):
        return self.angular_velocity

    def get_mass(self):
        return 0.01


class _Entity:
    def __init__(self, component):
        self.name = "071_can"
        self.component = component

    def get_components(self):
        return [self.component]

    def get_name(self):
        return self.name


class _Actor:
    def __init__(self):
        self.pose = _Pose()
        self.component = _Component()
        self.actor = _Entity(self.component)

    def get_pose(self):
        return self.pose

    def get_name(self):
        return self.actor.name


class _Body:
    def __init__(self, name):
        self.entity = type("Entity", (), {"name": name})()


class _Point:
    impulse = np.array([0.03, 0.04, 0.0], dtype=np.float64)


class _Contact:
    bodies = [_Body("071_can"), _Body("left_gripper")]
    points = [_Point()]


class _Scene:
    def __init__(self):
        self.contacts = []

    def get_contacts(self):
        return self.contacts


class _Robot:
    gripper_name = ["left_gripper", "right_gripper"]


class _TaskEnv:
    def __init__(self):
        self.can = _Actor()
        self.scene = _Scene()
        self.robot = _Robot()
        self.take_action_cnt = 0

    def take_action(self, action, action_type="qpos"):
        del action, action_type
        self.take_action_cnt += 1
        self.can.pose.p = self.can.pose.p + np.array([0.5, 0.0, 0.0])
        self.can.component.linear_velocity = np.array([3.0, 0.0, 0.0])
        self.scene.contacts = [_Contact()]
        return "executed"


def test_episode_physics_audit_records_and_restores_action(tmp_path):
    output_path = tmp_path / "episode0_seed7.jsonl"
    task_env = _TaskEnv()
    audit = EpisodePhysicsAudit(
        output_path=output_path,
        task_name="place_can_basket",
        episode_id=0,
        seed=7,
    )

    audit.install(task_env)
    assert task_env.take_action(np.zeros(14), action_type="qpos") == "executed"
    summary = audit.finish(success=False)
    task_env.take_action(np.zeros(14))

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["start", "action", "finish"]
    action_record = records[1]
    assert action_record["take_action_cnt_before"] == 0
    assert action_record["take_action_cnt_after"] == 1
    assert action_record["post"]["mass_kg"] == 0.01
    assert action_record["post"]["contacts"]["gripper_contact"] is True
    assert action_record["post"]["contacts"]["max_impulse"] == 0.05
    assert "large_single_action_displacement" in action_record["anomaly_flags"]
    assert "high_linear_speed" in action_record["anomaly_flags"]
    assert summary["first_gripper_contact_action"] == 0
    assert summary["first_anomaly_action"] == 0
    assert summary["num_actions"] == 1
    assert task_env.take_action_cnt == 2
