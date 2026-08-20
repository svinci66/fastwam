from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robomimic.train_can_q_guided_residual_actor import conservative_advantage


def test_conservative_advantage_penalizes_ensemble_disagreement():
    advantages = torch.tensor([[1.0, 2.0], [1.0, 0.0]])

    result = conservative_advantage(advantages, uncertainty_weight=1.0)

    torch.testing.assert_close(result, torch.tensor([1.0, 0.0]))


def test_single_q_conservative_advantage_is_the_q_advantage():
    advantages = torch.tensor([[0.2, -0.1]])

    result = conservative_advantage(advantages, uncertainty_weight=10.0)

    torch.testing.assert_close(result, advantages[0])
