"""Summarize paired place_can_basket rollout and physics-audit evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_VARIANTS = ("baseline", "no_imagination", "imagination")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("event") != "start":
        raise ValueError(f"Physics audit has no start record: {path}")
    if records[-1].get("event") != "finish":
        raise ValueError(f"Physics audit is incomplete: {path}")
    return records


def _first_action_with_partner(
    actions: list[dict[str, Any]], partner: str
) -> int | None:
    for record in actions:
        if partner in record["post"]["contacts"]["partners"]:
            return int(record["action_index"])
    return None


def _episode_summary(path: Path) -> dict[str, Any]:
    records = _read_jsonl(path)
    start = records[0]
    finish = records[-1]
    actions = [record for record in records if record.get("event") == "action"]
    if int(finish["num_actions"]) != len(actions):
        raise ValueError(
            f"Action-count mismatch in {path}: finish={finish['num_actions']} "
            f"records={len(actions)}"
        )

    gripper_actions = [
        int(record["action_index"])
        for record in actions
        if bool(record["post"]["contacts"]["gripper_contact"])
    ]
    initial_z = float(start["initial"]["position"][2])
    max_z = max(float(record["post"]["position"][2]) for record in actions)
    ground_action = _first_action_with_partner(actions, "ground")
    basket_action = _first_action_with_partner(actions, "110_basket")
    first_gripper = gripper_actions[0] if gripper_actions else None
    last_gripper = gripper_actions[-1] if gripper_actions else None

    return {
        "audit_path": str(path.resolve()),
        "episode_id": int(start["episode_id"]),
        "seed": int(start["seed"]),
        "success": bool(finish["success"]),
        "num_actions": len(actions),
        "initial_position": start["initial"]["position"],
        "initial_quaternion_wxyz": start["initial"]["quaternion_wxyz"],
        "final_position": finish["final"]["position"],
        "first_gripper_contact_action": first_gripper,
        "last_gripper_contact_action": last_gripper,
        "release_action": None if last_gripper is None else last_gripper + 1,
        "first_basket_contact_action": basket_action,
        "first_ground_contact_action": ground_action,
        "ever_ground_contact": ground_action is not None,
        "pickup_lift_evidence": bool(gripper_actions and max_z > initial_z + 0.1),
        "max_actor_z_m": max_z,
        "max_displacement_from_initial_m": float(
            finish["max_displacement_from_initial_m"]
        ),
        "max_linear_speed_mps": float(finish["max_linear_speed_mps"]),
        "max_angular_speed_rps": float(finish["max_angular_speed_rps"]),
        "first_anomaly_action": finish["first_anomaly_action"],
        "anomaly_counts": finish["anomaly_counts"],
    }


def _online_records_by_variant(
    online_summary: dict[str, Any], task_name: str
) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for row in online_summary.get("rows", []):
        if row.get("task") != task_name:
            continue
        variant = str(row["variant"])
        records: dict[int, dict[str, Any]] = {}
        for record in row.get("episode_records", []):
            seed = int(record["seed"])
            if seed in records:
                raise ValueError(f"Duplicate online seed {seed} for {variant}")
            records[seed] = record
        result[variant] = records
    return result


def analyze(
    *,
    audit_root: Path,
    online_summary_path: Path,
    manifest_path: Path,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    task_name: str = "place_can_basket",
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    seeds = [int(seed) for seed in manifest[task_name]["seeds"]]
    instructions = [str(value) for value in manifest[task_name]["instructions"]]
    if len(seeds) != len(instructions):
        raise ValueError("Manifest seed and instruction counts differ")

    online_summary = _read_json(online_summary_path)
    online_by_variant = _online_records_by_variant(online_summary, task_name)
    initial_audit = online_summary.get("initial_state_audit", {}).get(task_name, {})
    protocol_audit = online_summary.get("protocol_pairing_audit", {}).get(
        task_name, {}
    )
    if not bool(initial_audit.get("exact_match")):
        raise ValueError("Online summary does not prove exact initial-state pairing")
    if not bool(protocol_audit.get("exact_seed_and_instruction_match")):
        raise ValueError("Online summary does not prove exact seed/instruction pairing")

    episodes_by_variant: dict[str, dict[int, dict[str, Any]]] = {}
    for variant in variants:
        variant_dir = audit_root / variant / task_name
        episode_paths = sorted(variant_dir.glob("episode*_seed*.jsonl"))
        if len(episode_paths) != len(seeds):
            raise ValueError(
                f"Expected {len(seeds)} audits for {variant}, found "
                f"{len(episode_paths)} in {variant_dir}"
            )
        episodes: dict[int, dict[str, Any]] = {}
        for path in episode_paths:
            episode = _episode_summary(path)
            seed = int(episode["seed"])
            if seed in episodes:
                raise ValueError(f"Duplicate physics-audit seed {seed} for {variant}")
            episodes[seed] = episode
        if list(sorted(episodes)) != list(sorted(seeds)):
            raise ValueError(f"Physics-audit seeds do not match manifest for {variant}")
        online_records = online_by_variant.get(variant, {})
        if list(sorted(online_records)) != list(sorted(seeds)):
            raise ValueError(f"Online seeds do not match manifest for {variant}")
        for seed, instruction in zip(seeds, instructions):
            online_record = online_records[seed]
            if str(online_record["instruction"]) != instruction:
                raise ValueError(f"Instruction mismatch for {variant} seed {seed}")
            if bool(online_record["success"]) != bool(episodes[seed]["success"]):
                raise ValueError(f"Success mismatch for {variant} seed {seed}")
        episodes_by_variant[variant] = episodes

    paired_rows = []
    initial_pose_match = True
    for seed, instruction in zip(seeds, instructions):
        variant_rows = {
            variant: episodes_by_variant[variant][seed] for variant in variants
        }
        initial_positions = {
            tuple(row["initial_position"]) for row in variant_rows.values()
        }
        initial_quaternions = {
            tuple(row["initial_quaternion_wxyz"]) for row in variant_rows.values()
        }
        seed_pose_match = len(initial_positions) == 1 and len(initial_quaternions) == 1
        initial_pose_match = initial_pose_match and seed_pose_match
        no_imagination = variant_rows["no_imagination"]["success"]
        imagination = variant_rows["imagination"]["success"]
        if imagination and not no_imagination:
            flip = "imagination_win"
        elif no_imagination and not imagination:
            flip = "imagination_loss"
        elif imagination:
            flip = "both_success"
        else:
            flip = "both_failure"
        paired_rows.append(
            {
                "seed": seed,
                "instruction": instruction,
                "initial_pose_exact_match": seed_pose_match,
                "residual_flip": flip,
                "variants": variant_rows,
            }
        )

    aggregate = {}
    for variant in variants:
        rows = list(episodes_by_variant[variant].values())
        aggregate[variant] = {
            "episodes": len(rows),
            "successes": sum(bool(row["success"]) for row in rows),
            "ground_contact_episodes": sum(
                bool(row["ever_ground_contact"]) for row in rows
            ),
            "pickup_lift_evidence_episodes": sum(
                bool(row["pickup_lift_evidence"]) for row in rows
            ),
            "basket_contact_episodes": sum(
                row["first_basket_contact_action"] is not None for row in rows
            ),
            "anomalous_episodes": sum(bool(row["anomaly_counts"]) for row in rows),
        }

    flip_counts = {
        label: sum(row["residual_flip"] == label for row in paired_rows)
        for label in (
            "imagination_win",
            "imagination_loss",
            "both_success",
            "both_failure",
        )
    }
    return {
        "schema_version": "place_can_basket_physics_audit_summary_v1",
        "task_name": task_name,
        "manifest_path": str(manifest_path.resolve()),
        "online_summary_path": str(online_summary_path.resolve()),
        "audit_root": str(audit_root.resolve()),
        "seeds": seeds,
        "variants": list(variants),
        "pairing_audit": {
            "online_initial_observation_exact_match": True,
            "online_seed_and_instruction_exact_match": True,
            "actor_initial_pose_exact_match": initial_pose_match,
        },
        "aggregate": aggregate,
        "flip_counts": flip_counts,
        "paired_rows": paired_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--online-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    result = analyze(
        audit_root=args.audit_root.expanduser().resolve(),
        online_summary_path=args.online_summary.expanduser().resolve(),
        manifest_path=args.manifest.expanduser().resolve(),
    )
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"Physics-audit summary: {output_path}")


if __name__ == "__main__":
    main()
