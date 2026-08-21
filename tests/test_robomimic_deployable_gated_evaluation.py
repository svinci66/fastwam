from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.evaluate_can_deployable_gated_branches import (
    conservative_ensemble_advantage,
    source_key_lookup,
)


def test_conservative_ensemble_advantage_penalizes_critic_disagreement():
    advantages = np.asarray([[3.0, 2.0], [1.0, 2.0]], dtype=np.float32)

    conservative = conservative_ensemble_advantage(advantages, uncertainty_weight=1.0)

    np.testing.assert_allclose(conservative, [1.0, 2.0])


def test_source_key_lookup_rejects_duplicate_states():
    with pytest.raises(ValueError, match="Duplicate source key"):
        source_key_lookup(np.asarray(["demo_0", "demo_0"]), np.asarray([4, 4]))
