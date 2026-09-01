"""Fail closed unless two trained AWR checkpoints form an exact reward ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.audit_awr_training_pair import (
    ALLOWED_CONFIG_DIFFERENCES,
    differing_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--treatment-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_run(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    return config, checkpoint


def main() -> None:
    args = parse_args()
    control_config, control = load_run(args.control_dir.resolve())
    treatment_config, treatment = load_run(args.treatment_dir.resolve())
    differences = differing_paths(control_config, treatment_config)
    weights = {
        "control": float(control["reward_config"]["imagination_weight"]),
        "treatment": float(treatment["reward_config"]["imagination_weight"]),
    }
    checks = {
        "only_registered_config_difference": differences == ALLOWED_CONFIG_DIFFERENCES,
        "expected_reward_weights": weights == {"control": 0.0, "treatment": 1.0},
        "same_initialization": (
            control["summary"]["initialization_sha256"]
            == treatment["summary"]["initialization_sha256"]
        ),
        "same_replay": (
            control["replay_manifest_sha256"] == treatment["replay_manifest_sha256"]
        ),
        "same_actor_structure": control["actor_config"] == treatment["actor_config"],
        "same_awr_configuration": control["awr_config"] == treatment["awr_config"],
        "same_training_seed": (
            int(control["summary"]["training_seed"])
            == int(treatment["summary"]["training_seed"])
        ),
    }
    payload = {
        "schema_version": "robotwin_awr_checkpoint_ablation_audit_v1",
        "control_dir": str(args.control_dir.resolve()),
        "treatment_dir": str(args.treatment_dir.resolve()),
        "config_differences": [".".join(path) for path in sorted(differences)],
        "reward_weights": weights,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit("formal AWR checkpoints are not an exact reward ablation")


if __name__ == "__main__":
    main()
