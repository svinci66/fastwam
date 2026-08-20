from pathlib import Path
import sys

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.prepare_can_symmetric_training_datasets import prepare_datasets


def test_symmetric_collection_builds_q_pairs_and_one_actor_target_per_state(tmp_path):
    collection = tmp_path / "branches.hdf5"
    with h5py.File(collection, "w") as output:
        output.attrs.update(
            complete=True,
            format="fastwam.robomimic_symmetric_branches.v1",
            states_committed=2,
            score_margin=0.01,
        )
        states = output.create_group("states")
        states.create_dataset("branch_state", data=np.asarray([[1, 2], [3, 4]], np.float32))
        base = np.zeros((2, 1, 2), dtype=np.float32)
        candidates = np.asarray(
            [[[[0.1, 0]], [[-0.1, 0]], [[0, 0.1]], [[0, -0.1]]]] * 2,
            dtype=np.float32,
        ).reshape(2, 4, 1, 2)
        states.create_dataset("base_action_chunk", data=base)
        states.create_dataset("candidate_action_chunks", data=candidates)
        states.create_dataset("candidate_residual_chunks", data=candidates)
        scores = np.asarray([[0.2, -0.2, 0.005, -0.005], [-0.2, -0.1, 0.0, -0.005]])
        states.create_dataset("base_score", data=np.zeros(2))
        states.create_dataset("candidate_scores", data=scores)
        states.create_dataset("delta_scores", data=scores)
        states.create_dataset("source_split", data=np.asarray(["train", "valid"], dtype="S5"))
        states.create_dataset("source_demo", data=np.asarray(["demo_0", "demo_1"], dtype="S8"))
        states.create_dataset("source_step", data=np.asarray([1, 2], dtype=np.int32))
        states.create_dataset("restore_linf", data=np.zeros(2))
        states.create_dataset("branch_initial_state_linf", data=np.zeros(2))

    q_path = tmp_path / "q.npz"
    actor_path = tmp_path / "actor.npz"
    report = prepare_datasets(
        collection,
        q_path,
        actor_path,
        residual_scale=0.1,
        actor_target_mode="best_candidate",
    )

    # State 0 contributes two base comparisons and one decisive symmetric pair.
    # State 1 contributes two base comparisons and one decisive symmetric pair.
    assert report["q_pairs"] == 6
    with np.load(q_path, allow_pickle=False) as q:
        assert set(q["comparison_type"].tolist()) == {"candidate_vs_base", "symmetric_pair"}
        assert set(q["label"].tolist()) == {-1, 1}
    with np.load(actor_path, allow_pickle=False) as actor:
        assert len(actor["state"]) == 2
        np.testing.assert_allclose(actor["target_residual_chunk"][0], [[0.1, 0.0]])
        np.testing.assert_allclose(actor["target_residual_chunk"][1], [[0.0, 0.0]])
        np.testing.assert_array_equal(actor["has_improvement"], [1, 0])


def test_symmetric_gradient_uses_all_paired_score_differences(tmp_path):
    collection = tmp_path / "gradient_branches.hdf5"
    with h5py.File(collection, "w") as output:
        output.attrs.update(
            complete=True,
            format="fastwam.robomimic_symmetric_branches.v1",
            states_committed=1,
            score_margin=0.001,
        )
        states = output.create_group("states")
        states.create_dataset("branch_state", data=np.asarray([[1, 2]], np.float32))
        base = np.zeros((1, 1, 3), dtype=np.float32)
        residual = np.asarray(
            [[[[0.1, 0, 0]], [[-0.1, 0, 0]], [[0, 0.1, 0]], [[0, -0.1, 0]]]],
            dtype=np.float32,
        )
        scores = np.asarray([[0.4, -0.4, 0.2, -0.2]], dtype=np.float32)
        states.create_dataset("base_action_chunk", data=base)
        states.create_dataset("candidate_action_chunks", data=residual)
        states.create_dataset("candidate_residual_chunks", data=residual)
        states.create_dataset("candidate_scores", data=scores)
        states.create_dataset("delta_scores", data=scores)
        states.create_dataset("source_split", data=np.asarray(["train"], dtype="S5"))
        states.create_dataset("source_demo", data=np.asarray(["demo_0"], dtype="S8"))
        states.create_dataset("source_step", data=np.asarray([1], dtype=np.int32))
        states.create_dataset("restore_linf", data=np.zeros(1))
        states.create_dataset("branch_initial_state_linf", data=np.zeros(1))

    q_path = tmp_path / "q.npz"
    actor_path = tmp_path / "actor.npz"
    report = prepare_datasets(
        collection,
        q_path,
        actor_path,
        residual_scale=0.1,
        actor_target_mode="symmetric_gradient",
    )

    assert report["actor_target_mode"] == "symmetric_gradient"
    with np.load(actor_path, allow_pickle=False) as actor:
        # The x score difference is twice y, so normalization preserves a 2:1 ratio.
        np.testing.assert_allclose(
            actor["target_residual_chunk"][0], [[0.1, 0.05, 0.0]], atol=1e-7
        )
