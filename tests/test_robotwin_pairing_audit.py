import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.audit_paired_initial_states import audit_collection


def _episode(root: Path, mode: str, image: bytes) -> None:
    target = root / "task_a" / "imagination_transitions" / "task_a" / mode / "episode_0000" / "replan_0000"
    target.mkdir(parents=True)
    (target / "current.png").write_bytes(image)
    (target / "metadata.json").write_text(
        json.dumps({"episode_success": False}), encoding="utf-8"
    )


def test_audit_collection_requires_terminal_and_exact_bytes(tmp_path):
    _episode(tmp_path, "policy", b"same")
    _episode(tmp_path, "mild", b"same")
    _episode(tmp_path, "strong", b"different")

    report = audit_collection(tmp_path, modes=("mild", "strong", "missing"))

    assert report["expected_pair_count"] == 3
    assert report["exact_pair_count"] == 1
    assert report["invalid_pair_count"] == 2
    assert report["all_exact"] is False
