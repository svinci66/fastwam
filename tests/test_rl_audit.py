import numpy as np
import pytest

from fastwam.rl.audit import array_sha256, resolve_trial_indices


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
