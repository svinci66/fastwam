"""Exit successfully only for a terminal RoboTwin transition directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    args = parser.parse_args()
    metadata_paths = sorted(args.episode_dir.rglob("metadata.json"))
    if not metadata_paths:
        raise SystemExit(1)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    complete = any(bool(record.get("episode_success", False)) for record in records) or any(
        bool(record.get("truncated", False)) for record in records
    )
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
