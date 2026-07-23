import numpy as np
import pytest

from fastwam.rl.audit import array_sha256, derive_episode_seed, resolve_trial_indices


def test_array_sha256_is_stable_for_equivalent_contiguous_values():
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    non_contiguous = base.T.copy().T

    assert array_sha256(base) == array_sha256(non_contiguous)


def test_array_sha256_binds_dtype_and_shape():
    values = np.arange(4, dtype=np.float32)

    assert array_sha256(values) != array_sha256(values.astype(np.float64))
    assert array_sha256(values) != array_sha256(values.reshape(2, 2))


def test_resolve_trial_indices_defaults_to_num_trials():
    assert resolve_trial_indices(
        num_trials=3, trial_indices=None, available_states=10
    ) == [0, 1, 2]


def test_episode_seed_is_order_independent_and_separates_streams():
    first = derive_episode_seed(base_seed=42, task_id=3, trial_index=11, stream=1)
    repeated = derive_episode_seed(base_seed=42, task_id=3, trial_index=11, stream=1)
    different_task = derive_episode_seed(
        base_seed=42, task_id=4, trial_index=11, stream=1
    )
    different_stream = derive_episode_seed(
        base_seed=42, task_id=3, trial_index=11, stream=2
    )
    assert first == repeated
    assert len({first, different_task, different_stream}) == 3
    assert 0 <= first < 2**32


def test_resolve_trial_indices_preserves_valid_explicit_order():
    assert resolve_trial_indices(
        num_trials=50, trial_indices=[8, 5, 9], available_states=10
    ) == [8, 5, 9]


@pytest.mark.parametrize("indices", [[], [1, 1], [-1], [10]])
def test_resolve_trial_indices_rejects_invalid_explicit_indices(indices):
    with pytest.raises(ValueError):
        resolve_trial_indices(
            num_trials=3, trial_indices=indices, available_states=10
        )
