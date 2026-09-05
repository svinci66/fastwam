import torch

from fastwam.rl.adapter_trainer import (
    PairBehaviorBalancedBatchSampler,
    masked_temporal_smoothness,
    sampler_epoch_audit,
)


def _labels():
    tasks = []
    pairs = []
    behaviors = []
    for task in range(3):
        for pair in range(4):
            for behavior, chunks in (("expert", 2), ("policy", 6)):
                for _ in range(chunks):
                    tasks.append(task)
                    pairs.append(f"task{task}-pair{pair}")
                    behaviors.append(behavior)
    return tasks, pairs, behaviors


def test_pair_behavior_sampler_balances_tasks_pairs_and_long_failures():
    tasks, pairs, behaviors = _labels()
    sampler = PairBehaviorBalancedBatchSampler(
        tasks,
        pairs,
        behaviors,
        17,
        generator=torch.Generator().manual_seed(4),
    )
    audit = sampler_epoch_audit(
        sampler,
        task_ids=tasks,
        pair_ids=pairs,
        behaviors=behaviors,
    )
    assert max(audit["task_counts"].values()) - min(audit["task_counts"].values()) <= 1
    assert abs(audit["behavior_counts"]["expert"] - audit["behavior_counts"]["policy"]) <= 1
    assert audit["pair_count_max"] - audit["pair_count_min"] <= 2
    assert audit["unique_pairs"] == 12


def test_pair_behavior_sampler_is_deterministic():
    labels = _labels()
    outputs = []
    for _ in range(2):
        sampler = PairBehaviorBalancedBatchSampler(
            *labels,
            13,
            generator=torch.Generator().manual_seed(9),
        )
        outputs.append(list(sampler))
    assert outputs[0] == outputs[1]


def test_temporal_smoothness_masks_padded_suffix():
    residual = torch.zeros(2, 4, 1)
    residual[0, :, 0] = torch.tensor([0.0, 1.0, 99.0, 99.0])
    residual[1, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    result = masked_temporal_smoothness(residual, torch.tensor([2, 4]))
    assert result.tolist() == [1.0, 1.0]
