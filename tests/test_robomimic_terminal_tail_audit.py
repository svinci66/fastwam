from pathlib import Path
import sys

import h5py
import json
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.audit_can_terminal_intervention_tails import (
    summarize_terminal_tails,
)
from experiments.robomimic.prepare_can_terminal_safety_dataset import prepare


def test_terminal_tail_summary_separates_success_loss_from_dense_reward():
    rows = [
        {
            "accepted": True,
            "base_tail_success": 1,
            "residual_tail_success": 0,
            "tail_reward_delta": 0.5,
            "branch_state_linf": 0.0,
        },
        {
            "accepted": True,
            "base_tail_success": 1,
            "residual_tail_success": 1,
            "tail_reward_delta": -0.1,
            "branch_state_linf": 0.0,
        },
        {"accepted": False},
    ]

    summary = summarize_terminal_tails(rows)

    assert summary["terminal_success_losses"] == 1
    assert summary["terminal_success_preserved"] == 1
    assert summary["tail_reward_delta_mean"] == 0.2


def test_prepare_terminal_safety_dataset_adds_intervention_history(tmp_path):
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as output:
        mask = output.create_group("mask")
        mask.create_dataset("train", data=np.asarray(["demo_0"], dtype=h5py.string_dtype()))
        mask.create_dataset("valid", data=np.asarray(["demo_1"], dtype=h5py.string_dtype()))
    row = {
        "start_step": 3,
        "progress": 0.25,
        "accepted": True,
        "accepted_interventions_before": 2,
        "cumulative_residual_norm_before": 0.4,
        "state": [1.0, 2.0],
        "base_action_chunk": [[0.0]],
        "proposal_action_chunk": [[0.1]],
        "base_tail_success": 1,
        "residual_tail_success": 0,
        "tail_reward_delta": 0.5,
    }
    audits = []
    for demo, split in (("demo_0", "train"), ("demo_1", "valid")):
        path = tmp_path / f"{demo}.json"
        path.write_text(
            json.dumps(
                {
                    "complete": True,
                    "demo": demo,
                    "episode_steps": 12,
                    "observation_metadata": {"vision_feature_dim": 1, "proprio_dim": 1},
                    "rows": [row],
                }
            )
        )
        audits.append(path)

    report = prepare(audits, source, tmp_path / "terminal.npz")

    assert report["train_pairs"] == 1
    assert report["valid_pairs"] == 1
    assert report["terminal_success_losses"] == 2
    with np.load(tmp_path / "terminal.npz", allow_pickle=False) as dataset:
        np.testing.assert_allclose(dataset["state"][0], [1.0, 2.0, 0.25, 2.0, 0.4])
