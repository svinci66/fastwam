from pathlib import Path
import sys

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.audit_awr_training_pair import (
    ALLOWED_CONFIG_DIFFERENCES,
    differing_paths,
)


def load_config(name: str):
    return OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "configs" / "rl" / name), resolve=True
    )


def test_robotwin_awr_configs_are_a_strict_imagination_ablation():
    control = load_config("robotwin_residual_awr_no_imagination.yaml")
    treatment = load_config("robotwin_residual_awr_with_imagination.yaml")
    assert differing_paths(control, treatment) == ALLOWED_CONFIG_DIFFERENCES
    assert control["reward"]["imagination_weight"] == 0.0
    assert treatment["reward"]["imagination_weight"] == 1.0
    assert treatment["reward"]["max_imagination_to_success_ratio"] == 0.1


def test_robotwin_awr_has_no_q_or_online_gate_configuration():
    config = load_config("robotwin_residual_awr_with_imagination.yaml")
    assert "iql" not in config
    assert "q_critic" not in config
    assert "gate" not in config
    scales = config["model"]["residual_scale"]
    assert len(scales) == 14
    assert scales[6] == scales[13] == 0.0
    assert max(scales) == 0.05
    assert config["model"]["zero_init_output"] is True


def test_wan_head_awr_smoke_configs_are_a_strict_imagination_ablation():
    control = load_config("robotwin_residual_awr_wan_head_no_imagination_smoke.yaml")
    treatment = load_config("robotwin_residual_awr_wan_head_with_imagination_smoke.yaml")
    assert differing_paths(control, treatment) == ALLOWED_CONFIG_DIFFERENCES
    assert control["reward"]["imagination_weight"] == 0.0
    assert treatment["reward"]["imagination_weight"] == 1.0
    assert treatment["reward"]["imagination_reward_type"] == (
        "wan_vae_head_trajectory_global_norm_v1"
    )
    assert treatment["awr"]["epochs"] == 3


def test_wan_head_weight_candidates_change_only_the_registered_fields():
    control = load_config("robotwin_residual_awr_wan_head_no_imagination_smoke.yaml")
    for name, expected_weight in (
        ("robotwin_residual_awr_wan_head_weight025_smoke.yaml", 0.25),
        ("robotwin_residual_awr_wan_head_weight010_smoke.yaml", 0.1),
    ):
        candidate = load_config(name)
        assert differing_paths(control, candidate) == ALLOWED_CONFIG_DIFFERENCES
        assert candidate["reward"]["imagination_weight"] == expected_weight


def test_multitask3_smoke_and_formal_configs_share_the_deployable_model():
    smoke = load_config("robotwin_residual_awr_wan_head_multitask3_smoke.yaml")
    formal = load_config("robotwin_residual_awr_wan_head_multitask3_formal.yaml")
    differences = differing_paths(smoke, formal)
    assert differences == {("experiment_name",), ("awr", "epochs")}
    assert smoke["awr"]["epochs"] == 3
    assert formal["awr"]["epochs"] == 20
    assert formal["awr"]["balance_tasks"] is True
    assert formal["model"]["zero_init_output"] is True
    assert max(formal["model"]["residual_scale"]) == 0.1
    assert formal["model"]["residual_scale"][6] == 0.0
    assert formal["model"]["residual_scale"][13] == 0.0
    assert formal["reward"]["imagination_reward_type"] == (
        "wan_vae_head_trajectory_global_norm_v1"
    )


def test_multitask3_formal_configs_are_a_strict_imagination_ablation():
    control = load_config(
        "robotwin_residual_awr_wan_head_multitask3_no_imagination_formal.yaml"
    )
    treatment = load_config("robotwin_residual_awr_wan_head_multitask3_formal.yaml")
    assert differing_paths(control, treatment) == ALLOWED_CONFIG_DIFFERENCES
    assert control["reward"]["imagination_weight"] == 0.0
    assert treatment["reward"]["imagination_weight"] == 1.0
    assert control["awr"]["epochs"] == treatment["awr"]["epochs"] == 20
    assert control["model"] == treatment["model"]


def test_multitask3_weight025_epoch_configs_only_change_training_horizon():
    epoch3 = load_config(
        "robotwin_residual_awr_wan_head_multitask3_weight025_epochs3.yaml"
    )
    epoch20 = load_config(
        "robotwin_residual_awr_wan_head_multitask3_weight025_epochs20.yaml"
    )
    assert differing_paths(epoch3, epoch20) == {
        ("experiment_name",),
        ("awr", "epochs"),
    }
    assert epoch3["awr"]["epochs"] == 3
    assert epoch20["awr"]["epochs"] == 20
    assert epoch3["reward"]["imagination_weight"] == 0.25


def test_multitask3_epoch5_configs_are_a_strict_weight025_ablation():
    control = load_config(
        "robotwin_residual_awr_wan_head_multitask3_no_imagination_epochs5.yaml"
    )
    treatment = load_config(
        "robotwin_residual_awr_wan_head_multitask3_weight025_epochs5.yaml"
    )
    assert differing_paths(control, treatment) == ALLOWED_CONFIG_DIFFERENCES
    assert control["reward"]["imagination_weight"] == 0.0
    assert treatment["reward"]["imagination_weight"] == 0.25
    assert control["awr"]["epochs"] == treatment["awr"]["epochs"] == 5


def test_video_expert_paired_rank_configs_are_a_strict_imagination_ablation():
    control = load_config(
        "robotwin_residual_awr_video_expert_multitask3_paired_rank_no_imagination.yaml"
    )
    treatment = load_config(
        "robotwin_residual_awr_video_expert_multitask3_paired_rank_imagination025.yaml"
    )
    assert differing_paths(control, treatment) == ALLOWED_CONFIG_DIFFERENCES
    assert control["reward"]["imagination_weight"] == 0.0
    assert treatment["reward"]["imagination_weight"] == 0.25
    assert treatment["reward"]["imagination_reward_type"] == (
        "wan_vae_head_trajectory_paired_rank_discount_norm_v1"
    )
    assert treatment["awr"]["epochs"] == 3
