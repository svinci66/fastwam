import numpy as np
import pytest

from experiments.libero.imagination_reward_utils import (
    build_shared_direction_action_branches,
)


def test_shared_noise_direction_scales_and_preserves_gripper():
    base = np.linspace(-0.4, 0.4, 32 * 7, dtype=np.float32).reshape(32, 7)
    branches, epsilon = build_shared_direction_action_branches(
        base,
        noise_stds=(0.075, 0.15, 0.30),
        rng=np.random.default_rng(2042),
    )

    np.testing.assert_array_equal(branches["policy"], base)
    for std in (0.075, 0.15, 0.30):
        branch = branches[f"noise_{std:.3f}"]
        np.testing.assert_allclose(
            branch[:, :-1] - base[:, :-1],
            std * epsilon,
            rtol=1e-6,
            atol=2e-8,
        )
        np.testing.assert_array_equal(branch[:, -1], base[:, -1])
    np.testing.assert_array_equal(branches["zero"][:, :-1], 0.0)
    np.testing.assert_array_equal(branches["zero"][:, -1], -1.0)


def test_shared_noise_direction_clips_only_non_gripper_actions():
    base = np.full((2, 7), 0.99, dtype=np.float32)
    branches, _ = build_shared_direction_action_branches(
        base,
        noise_stds=(0.30,),
        rng=np.random.default_rng(0),
    )
    assert np.all(branches["noise_0.300"][:, :-1] <= 1.0)
    assert np.all(branches["noise_0.300"][:, :-1] >= -1.0)
    np.testing.assert_array_equal(branches["noise_0.300"][:, -1], base[:, -1])


@pytest.mark.parametrize("noise_stds", [(), (0.0,), (0.15, 0.075)])
def test_invalid_noise_stds_fail(noise_stds):
    with pytest.raises(ValueError):
        build_shared_direction_action_branches(
            np.zeros((8, 7), dtype=np.float32),
            noise_stds=noise_stds,
        )
