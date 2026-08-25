"""Audit that RoboTwin AWR treatment/control runs form a fair reward ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED_CONFIG_DIFFERENCES = {
    ("experiment_name",),
    ("reward", "imagination_weight"),
}


def differing_paths(left, right, prefix=()):
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in set(left) | set(right):
            paths.update(differing_paths(left.get(key), right.get(key), (*prefix, key)))
        return paths
    return set() if left == right else {prefix}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows = []
    all_exact = True
    for seed in seeds:
        roots = {
            variant: args.output_root / f"seed{seed}" / variant
            for variant in ("no_imagination", "with_imagination")
        }
        configs = {
            variant: json.loads((root / "run_config.json").read_text())
            for variant, root in roots.items()
        }
        checkpoints = {
            variant: __import__("torch").load(
                root / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            for variant, root in roots.items()
        }
        differences = differing_paths(configs["no_imagination"], configs["with_imagination"])
        hashes = {
            variant: checkpoint["summary"]["initialization_sha256"]
            for variant, checkpoint in checkpoints.items()
        }
        replay_hashes = {
            variant: checkpoint["replay_manifest_sha256"]
            for variant, checkpoint in checkpoints.items()
        }
        exact = (
            differences == ALLOWED_CONFIG_DIFFERENCES
            and len({json.dumps(value, sort_keys=True) for value in hashes.values()}) == 1
            and len(set(replay_hashes.values())) == 1
        )
        all_exact = all_exact and exact
        rows.append(
            {
                "seed": seed,
                "exact": exact,
                "config_differences": [".".join(path) for path in sorted(differences)],
                "initialization_sha256": hashes,
                "replay_manifest_sha256": replay_hashes,
            }
        )
    payload = {"all_exact": all_exact, "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all_exact:
        raise SystemExit("RoboTwin AWR treatment/control audit failed")


if __name__ == "__main__":
    main()
