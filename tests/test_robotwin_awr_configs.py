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
