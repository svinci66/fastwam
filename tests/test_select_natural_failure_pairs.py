import json

import pytest

from experiments.robotwin.select_natural_failure_pairs import select_pairs


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_selects_first_valid_natural_failures_without_reward(tmp_path):
    status = tmp_path / "status.jsonl"
    audit = tmp_path / "strict.json"
    rows = []
    strict_rows = []
    for episode, (success, valid) in enumerate(
        [(False, True), (True, True), (False, False), (False, True)]
    ):
        expert = tmp_path / f"expert{episode}.hdf5"
        video = tmp_path / f"failure{episode}.mp4"
        review = tmp_path / f"review{episode}"
        expert.write_bytes(b"expert")
        video.write_bytes(b"video")
        review.mkdir()
        rows.append(
            {
                "task": "place_can_basket",
                "episode_index": episode,
                "environment_seed": 700 + episode,
                "instruction": f"instruction {episode}",
                "success": success,
                "decision": "fastwam_success" if success else "natural_failure",
                "expert_hdf5": str(expert),
                "fastwam_video": str(video),
                "review_dir": str(review),
                "imagination_reward": 1000.0 - episode,
            }
        )
        strict_rows.append(
            {
                "task": "place_can_basket",
                "episode_index": episode,
                "environment_seed": 700 + episode,
                "valid": valid,
            }
        )
    _write_jsonl(status, rows)
    audit.write_text(json.dumps({"pairs": strict_rows}))

    selected, report = select_pairs(
        status_path=status,
        strict_audit_path=audit,
        task="place_can_basket",
        count=2,
    )

    assert [row["episode_index"] for row in selected] == [0, 3]
    assert report["selection_uses_reward"] is False
    assert report["selected_pair_count"] == 2


def test_rejects_insufficient_valid_failures(tmp_path):
    status = tmp_path / "status.jsonl"
    audit = tmp_path / "strict.json"
    _write_jsonl(status, [])
    audit.write_text(json.dumps({"pairs": []}))

    with pytest.raises(ValueError, match="Need 1"):
        select_pairs(
            status_path=status,
            strict_audit_path=audit,
            task="place_can_basket",
            count=1,
        )
