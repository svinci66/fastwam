import hashlib
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.build_prevalidated_online_seed_manifest import build_manifest


def write_task_run(root: Path, task: str, seeds: list[int]) -> None:
    (root / f".{task}_{len(seeds)}ep_complete").touch()
    lines = []
    successes = 0
    for episode, seed in enumerate(seeds):
        outcome = episode % 2 == 1
        successes += int(outcome)
        lines.extend(
            [
                f"FASTWAM_ACCEPTED_ENV_SEED episode_id={episode} seed={seed}",
                f"FASTWAM_INITIAL_OBSERVATION episode_id={episode} "
                f"sha256={hashlib.sha256(f'{task}-{seed}'.encode()).hexdigest()}",
                f"FASTWAM_EVAL_INSTRUCTION episode_id={episode} seed={seed} "
                f"instruction='instruction {task} {seed}'",
                f"Success rate: {successes}/{episode + 1} => "
                f"{100.0 * successes / (episode + 1):.1f}%",
            ]
        )
    (root / f"eval_{task}_20260901.log").write_text("\n".join(lines) + "\n")
    OmegaConf.save(
        {
            "EVALUATION": {
                "action_mode": "policy",
                "expert_check": True,
                "paper_aligned": True,
                "strict_paired": True,
            }
        },
        root / f"eval_config_{task}.yaml",
    )
    (root / f"eval_protocol_{task}.json").write_text(
        json.dumps(
            {
                "task": task,
                "episodes": len(seeds),
                "paper_aligned": True,
                "strict_paired": True,
            }
        )
    )


def test_build_multitask_prevalidated_manifest_uses_task_specific_attestations(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    source_path = tmp_path / "source.json"
    task_seeds = {"task_a": [11, 12], "task_b": [21, 22]}
    source_path.write_text(
        json.dumps({task: {"seeds": seeds} for task, seeds in task_seeds.items()})
    )
    for task, seeds in task_seeds.items():
        write_task_run(baseline, task, seeds)

    payload = build_manifest(
        baseline_run_dir=baseline,
        source_manifest_path=source_path,
        tasks=list(task_seeds),
        episodes=2,
    )

    assert payload["_meta"]["selection_uses_policy_outcome"] is False
    for task, seeds in task_seeds.items():
        assert payload[task]["seeds"] == seeds
        assert [
            row["seed"] for row in payload[task]["expert_feasibility_records"]
        ] == seeds
        assert len(payload[task]["instructions"]) == 2
