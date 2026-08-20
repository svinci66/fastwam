from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.collect_can_symmetric_branches import make_symmetric_directions


def test_symmetric_directions_are_paired_bounded_and_preserve_gripper():
    directions = make_symmetric_directions(
        seed=7,
        direction_pairs=4,
        intervention_steps=3,
        action_dim=7,
        delta=0.1,
    )

    assert directions.shape == (8, 3, 7)
    for index in range(0, len(directions), 2):
        np.testing.assert_allclose(directions[index], -directions[index + 1])
    np.testing.assert_array_equal(directions[:, :, -1], 0.0)
    assert np.max(np.abs(directions)) <= 0.1 + 1e-7
    np.testing.assert_allclose(
        np.max(np.abs(directions[0::2, :, :-1]), axis=(1, 2)),
        0.1,
    )


def test_symmetric_direction_generation_is_deterministic():
    kwargs = dict(
        seed=11,
        direction_pairs=2,
        intervention_steps=3,
        action_dim=7,
        delta=0.05,
    )
    np.testing.assert_array_equal(
        make_symmetric_directions(**kwargs),
        make_symmetric_directions(**kwargs),
    )
