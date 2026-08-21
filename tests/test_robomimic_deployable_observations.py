from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.collect_can_deployable_observations import group_pending_states
from experiments.robomimic.prepare_can_deployable_training_datasets import prepare


def test_pending_states_are_grouped_by_demo_and_sorted_by_step():
    grouped = group_pending_states(
        np.asarray(["demo_1", "demo_0", "demo_1", "demo_0"]),
        np.asarray([5, 8, 2, 1]),
        np.asarray([0, 1, 0, 0]),
    )

    assert grouped == {"demo_0": [(1, 3)], "demo_1": [(2, 2), (5, 0)]}


def test_deployable_features_replace_privileged_state_by_source_key(tmp_path):
    common = {
        "source_demo": np.asarray(["demo_0", "demo_1"]),
        "source_step": np.asarray([1, 2]),
        "source_split": np.asarray(["train", "valid"]),
        "state": np.ones((2, 71), dtype=np.float32),
    }
    q_source = tmp_path / "q.npz"
    actor_source = tmp_path / "actor.npz"
    np.savez_compressed(q_source, **common, label=np.asarray([1, -1]))
    np.savez_compressed(actor_source, **common, has_improvement=np.asarray([1, 0]))
    observations = tmp_path / "observations.npz"
    np.savez_compressed(
        observations,
        source_demo=np.asarray(["demo_1", "demo_0"]),
        source_step=np.asarray([2, 1]),
        source_split=np.asarray(["valid", "train"]),
        vision_feature=np.asarray([[20, 21], [10, 11]], dtype=np.float32),
        proprio=np.asarray([[22], [12]], dtype=np.float32),
        encoder_path=np.asarray("/encoder"),
        camera_name=np.asarray("wrist"),
        proprio_keys=np.asarray("[]"),
    )
    q_output = tmp_path / "q_out.npz"
    actor_output = tmp_path / "actor_out.npz"

    report = prepare(q_source, actor_source, observations, q_output, actor_output)

    assert report["combined_state_dim"] == 3
    with np.load(actor_output, allow_pickle=False) as actor:
        np.testing.assert_array_equal(actor["state"], [[10, 11, 12], [20, 21, 22]])
        assert str(actor["observation_mode"]) == "siglip_wrist_proprio"


def test_vision_pca_is_fit_only_on_training_observations(tmp_path):
    common = {
        "source_demo": np.asarray(["demo_0", "demo_1", "demo_2"]),
        "source_step": np.asarray([1, 2, 3]),
        "source_split": np.asarray(["train", "train", "valid"]),
        "state": np.zeros((3, 2), dtype=np.float32),
    }
    q_source = tmp_path / "q.npz"
    actor_source = tmp_path / "actor.npz"
    np.savez_compressed(q_source, **common)
    np.savez_compressed(actor_source, **common)
    observations = tmp_path / "observations.npz"
    np.savez_compressed(
        observations,
        source_demo=common["source_demo"],
        source_step=common["source_step"],
        source_split=common["source_split"],
        vision_feature=np.asarray([[0, 0], [2, 0], [100, 100]], dtype=np.float32),
        proprio=np.zeros((3, 1), dtype=np.float32),
        encoder_path=np.asarray("/encoder"),
        camera_name=np.asarray("wrist"),
        proprio_keys=np.asarray("[]"),
    )
    projection = tmp_path / "pca.npz"

    report = prepare(
        q_source,
        actor_source,
        observations,
        tmp_path / "q_out.npz",
        tmp_path / "actor_out.npz",
        vision_pca_dim=1,
        projection_output_path=projection,
    )

    assert report["vision_feature_dim"] == 1
    with np.load(projection, allow_pickle=False) as pca:
        np.testing.assert_allclose(pca["mean"], [1.0, 0.0])
        assert str(pca["fitted_split"]) == "train"


def test_frozen_vision_pca_can_be_reused_without_refitting(tmp_path):
    common = {
        "source_demo": np.asarray(["demo_2"]),
        "source_step": np.asarray([3]),
        "source_split": np.asarray(["valid"]),
        "state": np.zeros((1, 2), dtype=np.float32),
    }
    q_source = tmp_path / "q.npz"
    actor_source = tmp_path / "actor.npz"
    np.savez_compressed(q_source, **common)
    np.savez_compressed(actor_source, **common)
    observations = tmp_path / "observations.npz"
    np.savez_compressed(
        observations,
        source_demo=common["source_demo"],
        source_step=common["source_step"],
        source_split=common["source_split"],
        vision_feature=np.asarray([[5, 7]], dtype=np.float32),
        proprio=np.asarray([[11]], dtype=np.float32),
        encoder_path=np.asarray("/encoder"),
        camera_name=np.asarray("wrist"),
        proprio_keys=np.asarray("[]"),
    )
    projection = tmp_path / "frozen_pca.npz"
    np.savez_compressed(
        projection,
        mean=np.asarray([1, 2], dtype=np.float32),
        components=np.asarray([[0, 1]], dtype=np.float32),
        input_dim=np.asarray(2, dtype=np.int32),
        output_dim=np.asarray(1, dtype=np.int32),
        fitted_split=np.asarray("train"),
    )

    report = prepare(
        q_source,
        actor_source,
        observations,
        tmp_path / "q_out.npz",
        tmp_path / "actor_out.npz",
        projection_input_path=projection,
    )

    assert report["vision_projection_path"] == str(projection)
    with np.load(tmp_path / "actor_out.npz", allow_pickle=False) as actor:
        np.testing.assert_array_equal(actor["state"], [[5, 11]])
