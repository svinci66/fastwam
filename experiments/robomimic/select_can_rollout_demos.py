#!/usr/bin/env python3
"""List RoboMimic rollout demos in stable numeric order."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def select_demos(
    dataset: Path,
    *,
    split: str = "all",
    require_success: bool = False,
) -> list[str]:
    with h5py.File(dataset.expanduser().resolve(), "r") as source:
        allowed = None
        if split != "all":
            allowed = set(source["mask"][split].asstr()[...])
        demos = []
        for name, group in source["data"].items():
            if allowed is not None and name not in allowed:
                continue
            if require_success and not bool(group.attrs.get("success", 0)):
                continue
            demos.append(name)
    return sorted(demos, key=lambda name: int(name.rsplit("_", 1)[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "train", "valid"), default="all")
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    for demo in select_demos(
        args.dataset,
        split=args.split,
        require_success=args.require_success,
    ):
        print(demo)


if __name__ == "__main__":
    main()
