from copy import deepcopy

from experiments.robotwin.audit_imagination_adapter_training_pair import audit_pair


def _checkpoint(weight: float) -> dict:
    return {
        "format": "fastwam_residual_spatial_adapter_awr_v1",
        "replay_manifest_sha256": "replay",
        "awr_config": {"seed": 44},
        "adapter_config": {"type": "spatial"},
        "reward_config": {
            "success_weight": 1.0,
            "imitation_weight": 0.1,
            "imagination_weight": weight,
        },
        "summary": {
            "num_transitions": 100,
            "base_checkpoint_sha256": "base-checkpoint",
            "base_actor_sha256": "base-actor",
            "initialization_sha256": {"adapter": "a", "critic": "c"},
            "sampler_audit": {"samples": 100},
            "zero_equivalence_audit": {"exact": True},
            "trained_adapter_audit": {
                "passed": True,
                "gripper_adapter_max_abs": 0.0,
            },
        },
    }


def test_same_data_adapter_pair_passes_when_only_imagination_weight_differs() -> None:
    report = audit_pair(_checkpoint(0.0), _checkpoint(0.25))
    assert report["passed"]
    assert all(report["checks"].values())


def test_adapter_pair_rejects_different_replay() -> None:
    treatment = deepcopy(_checkpoint(0.25))
    treatment["replay_manifest_sha256"] = "other"
    report = audit_pair(_checkpoint(0.0), treatment)
    assert not report["passed"]
    assert not report["checks"]["same_replay"]
