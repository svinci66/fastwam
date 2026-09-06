#!/usr/bin/env python3
"""Audit a same-data no-imagination/imagination spatial-adapter training pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


SPATIAL_FORMAT = "fastwam_residual_spatial_adapter_awr_v1"


def audit_pair(
    control: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, Any]:
    control_summary = control.get("summary", {})
    treatment_summary = treatment.get("summary", {})
    control_reward = control.get("reward_config", {})
    treatment_reward = treatment.get("reward_config", {})

    reward_keys = sorted(set(control_reward) | set(treatment_reward))
    non_weight_reward_match = all(
        control_reward.get(key) == treatment_reward.get(key)
        for key in reward_keys
        if key != "imagination_weight"
    )
    checks = {
        "checkpoint_format": (
            control.get("format") == SPATIAL_FORMAT
            and treatment.get("format") == SPATIAL_FORMAT
        ),
        "same_replay": (
            control.get("replay_manifest_sha256")
            == treatment.get("replay_manifest_sha256")
        ),
        "same_base_checkpoint": (
            control_summary.get("base_checkpoint_sha256")
            == treatment_summary.get("base_checkpoint_sha256")
        ),
        "same_frozen_base_actor": (
            control_summary.get("base_actor_sha256")
            == treatment_summary.get("base_actor_sha256")
        ),
        "same_initialization": (
            control_summary.get("initialization_sha256")
            == treatment_summary.get("initialization_sha256")
        ),
        "same_awr_config": control.get("awr_config") == treatment.get("awr_config"),
        "same_adapter_config": (
            control.get("adapter_config") == treatment.get("adapter_config")
        ),
        "same_non_imagination_reward_config": non_weight_reward_match,
        "control_imagination_weight_zero": (
            float(control_reward.get("imagination_weight", float("nan"))) == 0.0
        ),
        "treatment_imagination_weight_positive": (
            float(treatment_reward.get("imagination_weight", 0.0)) > 0.0
        ),
        "same_sampler_audit": (
            control_summary.get("sampler_audit")
            == treatment_summary.get("sampler_audit")
        ),
        "control_zero_equivalence": bool(
            control_summary.get("zero_equivalence_audit", {}).get("exact")
        ),
        "treatment_zero_equivalence": bool(
            treatment_summary.get("zero_equivalence_audit", {}).get("exact")
        ),
        "control_trust_region": bool(
            control_summary.get("trained_adapter_audit", {}).get("passed")
        ),
        "treatment_trust_region": bool(
            treatment_summary.get("trained_adapter_audit", {}).get("passed")
        ),
        "control_gripper_unchanged": (
            float(
                control_summary.get("trained_adapter_audit", {}).get(
                    "gripper_adapter_max_abs", float("nan")
                )
            )
            == 0.0
        ),
        "treatment_gripper_unchanged": (
            float(
                treatment_summary.get("trained_adapter_audit", {}).get(
                    "gripper_adapter_max_abs", float("nan")
                )
            )
            == 0.0
        ),
    }
    return {
        "schema_version": "robotwin_imagination_adapter_training_pair_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "control_imagination_weight": control_reward.get("imagination_weight"),
        "treatment_imagination_weight": treatment_reward.get("imagination_weight"),
        "num_transitions": {
            "control": control_summary.get("num_transitions"),
            "treatment": treatment_summary.get("num_transitions"),
        },
        "sampler_audit": control_summary.get("sampler_audit"),
        "trained_adapter_audits": {
            "control": control_summary.get("trained_adapter_audit"),
            "treatment": treatment_summary.get("trained_adapter_audit"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--treatment-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    control = torch.load(
        args.control_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    treatment = torch.load(
        args.treatment_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    report = audit_pair(control, treatment)
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
