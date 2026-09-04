import json
from pathlib import Path

import pytest

from experiments.robotwin.merge_expert_feasible_manifests import merge_manifests


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, task: str, seeds: list[int]) -> tuple[Path, dict]:
    payload = {
        "_meta": {
            "schema_version": "robotwin_expert_feasible_heldout_manifest_v1",
            "expert_feasibility_records": [
                {"episode_index": index, "seed": seed}
                for index, seed in enumerate(seeds)
            ],
        },
        task: {
            "seeds": seeds,
            "instructions": [f"instruction-{seed}" for seed in seeds],
        },
    }
    path.write_text(json.dumps(payload))
    return path, payload


def test_merge_moves_attestations_to_each_task(tmp_path):
    inputs = [
        _write(tmp_path / "a.json", "a", [1, 2]),
        _write(tmp_path / "b.json", "b", [3, 4]),
    ]
    result = merge_manifests(inputs, ["a", "b"])
    assert result["_meta"]["episodes_per_task"] == 2
    assert result["_meta"]["selection_uses_policy_outcome"] is False
    assert [row["seed"] for row in result["b"]["expert_feasibility_records"]] == [3, 4]


def test_merge_rejects_mismatched_counts(tmp_path):
    with pytest.raises(ValueError, match="same number"):
        merge_manifests(
            [
                _write(tmp_path / "a.json", "a", [1]),
                _write(tmp_path / "b.json", "b", [2, 3]),
            ],
            ["a", "b"],
        )


def test_balanced_candidate_pool_is_fresh_equal_and_disjoint():
    path = (
        PROJECT_ROOT
        / "experiments/robotwin/manifests/robotwin_multitask3_balanced_heldout10_candidate_pool_20260904.json"
    )
    payload = json.loads(path.read_text())
    task_seeds = [
        payload[task]["seeds"]
        for task in ("open_microwave", "hanging_mug", "place_can_basket")
    ]
    assert all(len(seeds) == 40 and len(set(seeds)) == 40 for seeds in task_seeds)
    assert len(set().union(*map(set, task_seeds))) == 120
    assert min(set().union(*map(set, task_seeds))) >= 4_800_200
