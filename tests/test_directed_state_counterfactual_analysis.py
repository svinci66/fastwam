from experiments.libero.analyze_directed_state_counterfactuals import (
    directed_primary_gates,
    summarize_directed_metric,
)


def _rows(num_anchors=10):
    rows = []
    for anchor in range(num_anchors):
        for branch, progress, reward in (
            ("policy", 0.01, 0.4),
            ("toward_bowl", 0.03, 0.8),
            ("away_from_bowl", -0.03, -0.4),
            ("zero", 0.0, 0.0),
        ):
            rows.append(
                {
                    "anchor_index": anchor,
                    "branch": branch,
                    "reward": {"candidate": reward},
                    "geometry": {"eef_target_distance_progress": progress},
                }
            )
    return rows


def test_directed_summary_uses_paired_direction_and_geometry_order():
    summary = summarize_directed_metric(
        _rows(),
        "candidate",
        bootstrap_samples=100,
        bootstrap_seed=2042,
    )
    assert summary["toward_beats_away_count"] == 10
    assert summary["toward_minus_away_bootstrap"]["ci95_lower"] == 1.2
    assert summary["reward_geometry_spearman_positive_count"] == 10
    assert summary["zero_to_policy_abs_reward_ratio"] == 0.0


def test_directed_primary_gates_require_formal_anchor_count_and_all_signals():
    metric = summarize_directed_metric(
        _rows(),
        "candidate",
        bootstrap_samples=100,
        bootstrap_seed=2042,
    )
    geometry = {
        "toward_progress_positive_count": 10,
        "away_progress_negative_count": 10,
        "toward_progress_exceeds_away_count": 10,
    }
    gates = directed_primary_gates(metric, geometry, num_anchors=10)
    assert all(gates.values())
    smoke_gates = directed_primary_gates(metric, geometry, num_anchors=1)
    assert not smoke_gates["formal_anchor_count_is_10"]
