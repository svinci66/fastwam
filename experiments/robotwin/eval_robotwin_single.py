"""
RobotWin single-task evaluation entrypoint (Hydra).

Features:
- Read `configs/sim_robotwin.yaml`.
- Check or create the symlink:
  `RoboTwin/policy/fastwam -> experiments/robotwin/fastwam`.
- Forward config overrides to the official RoboTwin entrypoint
  `script/eval_policy.py` and save logs.

Common arguments:
- `ckpt`: path to the FastWAM checkpoint (required).
- `EVALUATION.task_name`: task name to evaluate (required).
- `gpu_id`: sets `CUDA_VISIBLE_DEVICES`.

Examples:
1) Minimal run
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock

2) Run with more evaluation overrides
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock \
     EVALUATION.task_config=demo_randomized \
     EVALUATION.replan_steps=24 \
     EVALUATION.num_inference_steps=10 \
     gpu_id=0
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "fastwam_policy"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")

    policy_target = policy_root / POLICY_NAME
    source_resolved = policy_source_dir.resolve()

    if not policy_target.exists() and not policy_target.is_symlink():
        policy_target.symlink_to(source_resolved, target_is_directory=True)
        return policy_target

    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            raise RuntimeError(
                f"Policy symlink conflict: {policy_target} -> {target_resolved}, "
                f"expected -> {source_resolved}"
            )
        return policy_target

    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Please handle it manually to avoid overriding existing policy files."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


def _is_none_like(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def _validate_protocol(cfg: DictConfig) -> None:
    """Fail closed when a run is labelled paper-aligned but is not."""

    if not bool(cfg.EVALUATION.paper_aligned):
        if bool(cfg.EVALUATION.strict_paired):
            raise ValueError("EVALUATION.strict_paired requires paper_aligned=true")
        return

    expected = {
        "num_inference_steps": 10,
        "replan_steps": 24,
        "instruction_type": "unseen",
        "text_cfg_scale": 1.0,
    }
    actual = {
        "num_inference_steps": int(cfg.EVALUATION.num_inference_steps),
        "replan_steps": int(cfg.EVALUATION.replan_steps),
        "instruction_type": str(cfg.EVALUATION.instruction_type),
        "text_cfg_scale": float(cfg.EVALUATION.text_cfg_scale),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if str(cfg.EVALUATION.task_config) not in {"demo_clean", "demo_randomized"}:
        mismatches["task_config"] = {
            "expected": "demo_clean or demo_randomized",
            "actual": str(cfg.EVALUATION.task_config),
        }
    if not _is_none_like(cfg.EVALUATION.fixed_instruction):
        mismatches["fixed_instruction"] = {
            "expected": None,
            "actual": str(cfg.EVALUATION.fixed_instruction),
        }
    if not bool(cfg.EVALUATION.expert_check):
        mismatches["expert_check"] = {"expected": True, "actual": False}
    for key in ("action_noise_std", "action_hold_probability"):
        if float(cfg.EVALUATION[key]) != 0.0:
            mismatches[key] = {"expected": 0.0, "actual": float(cfg.EVALUATION[key])}
    if int(cfg.EVALUATION.gripper_close_delay_steps) != 0:
        mismatches["gripper_close_delay_steps"] = {
            "expected": 0,
            "actual": int(cfg.EVALUATION.gripper_close_delay_steps),
        }

    if bool(cfg.EVALUATION.strict_paired):
        if _is_none_like(cfg.EVALUATION.environment_seed_manifest_path):
            mismatches["environment_seed_manifest_path"] = {
                "expected": "a versioned manifest",
                "actual": None,
            }
        if not bool(cfg.EVALUATION.deterministic_instruction_by_seed):
            mismatches["deterministic_instruction_by_seed"] = {
                "expected": True,
                "actual": False,
            }
    if mismatches:
        raise ValueError(
            "Paper-aligned RoboTwin protocol mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")
    _validate_protocol(cfg)

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    policy_source_dir = (PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")

    _ensure_policy_symlink(robotwin_root=robotwin_root, policy_source_dir=policy_source_dir)

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / ckpt_tag
        / run_ts
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_output_dir / (
        f"eval_{str(cfg.EVALUATION.task_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    robotwin_eval_base = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / ckpt_tag
        / run_ts
        / str(cfg.EVALUATION.task_name)
    )

    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()
    sim_task = HydraConfig.get().runtime.choices.get("task")

    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", cfg.EVALUATION.policy_name)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "eval_num_episodes", cfg.EVALUATION.eval_num_episodes)
    _append_override(overrides, "eval_video_log", cfg.EVALUATION.eval_video_log)

    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    _append_override(overrides, "eval_output_dir", str(robotwin_eval_base))
    _append_override(overrides, "mixed_precision", cfg.mixed_precision)
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "action_horizon", cfg.EVALUATION.action_horizon)
    _append_override(overrides, "replan_steps", cfg.EVALUATION.replan_steps)
    _append_override(overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps)
    _append_override(overrides, "sigma_shift", cfg.EVALUATION.sigma_shift)
    _append_override(overrides, "text_cfg_scale", cfg.EVALUATION.text_cfg_scale)
    _append_override(overrides, "negative_prompt", cfg.EVALUATION.negative_prompt)
    _append_override(overrides, "rand_device", cfg.EVALUATION.rand_device)
    _append_override(overrides, "tiled", cfg.EVALUATION.tiled)
    _append_override(
        overrides,
        "capture_decode_tiled",
        cfg.EVALUATION.capture_decode_tiled,
    )
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
    _append_override(
        overrides,
        "skip_get_obs_within_replan",
        cfg.EVALUATION.skip_get_obs_within_replan,
    )
    _append_override(overrides, "action_mode", cfg.EVALUATION.action_mode)
    _append_override(
        overrides, "residual_checkpoint", cfg.EVALUATION.residual_checkpoint
    )
    _append_override(
        overrides, "residual_encoder_path", cfg.EVALUATION.residual_encoder_path
    )
    _append_override(
        overrides, "residual_encoder_version", cfg.EVALUATION.residual_encoder_version
    )
    _append_override(
        overrides, "residual_encoder_dtype", cfg.EVALUATION.residual_encoder_dtype
    )
    _append_override(overrides, "residual_device", cfg.EVALUATION.residual_device)
    _append_override(
        overrides,
        "residual_q_gate_enabled",
        cfg.EVALUATION.residual_q_gate_enabled,
    )
    _append_override(
        overrides,
        "residual_paired_advantage_gate_enabled",
        cfg.EVALUATION.residual_paired_advantage_gate_enabled,
    )
    _append_override(
        overrides,
        "residual_paired_advantage_threshold",
        cfg.EVALUATION.residual_paired_advantage_threshold,
        skip_none=False,
    )
    _append_override(
        overrides,
        "residual_paired_advantage_max_disagreement",
        cfg.EVALUATION.residual_paired_advantage_max_disagreement,
        skip_none=False,
    )
    _append_override(
        overrides, "residual_q_gate_margin", cfg.EVALUATION.residual_q_gate_margin
    )
    _append_override(
        overrides,
        "residual_q_gate_max_disagreement",
        cfg.EVALUATION.residual_q_gate_max_disagreement,
    )
    _append_override(
        overrides,
        "residual_q_gate_risk_scale",
        cfg.EVALUATION.residual_q_gate_risk_scale,
    )
    _append_override(
        overrides,
        "residual_q_gate_risk_decay",
        cfg.EVALUATION.residual_q_gate_risk_decay,
    )
    _append_override(
        overrides,
        "residual_soft_scale_enabled",
        cfg.EVALUATION.residual_soft_scale_enabled,
    )
    _append_override(
        overrides,
        "residual_soft_scale_q_full_advantage",
        cfg.EVALUATION.residual_soft_scale_q_full_advantage,
    )
    _append_override(
        overrides,
        "residual_soft_scale_support_full_margin",
        cfg.EVALUATION.residual_soft_scale_support_full_margin,
    )
    _append_override(
        overrides,
        "residual_q_gate_critic_source",
        cfg.EVALUATION.residual_q_gate_critic_source,
    )
    _append_override(
        overrides,
        "residual_support_index_path",
        cfg.EVALUATION.residual_support_index_path,
    )
    _append_override(
        overrides,
        "residual_support_circuit_breaker_enabled",
        cfg.EVALUATION.residual_support_circuit_breaker_enabled,
    )
    _append_override(
        overrides,
        "residual_shadow_mode",
        cfg.EVALUATION.residual_shadow_mode,
    )
    _append_override(
        overrides,
        "residual_intervention_replans",
        cfg.EVALUATION.residual_intervention_replans,
    )
    _append_override(
        overrides,
        "residual_actor_override_checkpoint",
        cfg.EVALUATION.residual_actor_override_checkpoint,
        skip_none=False,
    )
    _append_override(
        overrides,
        "residual_actor_override_replans",
        cfg.EVALUATION.residual_actor_override_replans,
        skip_none=False,
    )
    _append_override(
        overrides,
        "residual_max_interventions_per_episode",
        cfg.EVALUATION.residual_max_interventions_per_episode,
        skip_none=False,
    )
    _append_override(
        overrides,
        "residual_outcome_confirmation_enabled",
        cfg.EVALUATION.residual_outcome_confirmation_enabled,
    )
    _append_override(
        overrides,
        "residual_outcome_confirmation_min_progress",
        cfg.EVALUATION.residual_outcome_confirmation_min_progress,
    )
    _append_override(
        overrides,
        "residual_outcome_confirmation_reanchor_replans",
        cfg.EVALUATION.residual_outcome_confirmation_reanchor_replans,
    )
    _append_override(
        overrides,
        "residual_language_instruction",
        cfg.EVALUATION.residual_language_instruction,
        skip_none=False,
    )
    _append_override(overrides, "action_noise_std", cfg.EVALUATION.action_noise_std)
    _append_override(overrides, "action_noise_seed", cfg.EVALUATION.action_noise_seed)
    _append_override(
        overrides,
        "action_noise_replans",
        cfg.EVALUATION.action_noise_replans,
    )
    _append_override(
        overrides,
        "action_hold_probability",
        cfg.EVALUATION.action_hold_probability,
    )
    _append_override(
        overrides,
        "action_hold_replans",
        cfg.EVALUATION.action_hold_replans,
    )
    _append_override(
        overrides,
        "gripper_close_delay_steps",
        cfg.EVALUATION.gripper_close_delay_steps,
    )
    _append_override(
        overrides,
        "action_corruption_seed",
        cfg.EVALUATION.action_corruption_seed,
    )
    _append_override(overrides, "trial_offset", cfg.EVALUATION.trial_offset)
    _append_override(
        overrides, "environment_start_seed", cfg.EVALUATION.environment_start_seed
    )
    _append_override(
        overrides, "environment_episode_offset", cfg.EVALUATION.environment_episode_offset
    )
    _append_override(overrides, "expert_check", cfg.EVALUATION.expert_check)
    _append_override(overrides, "fixed_instruction", cfg.EVALUATION.fixed_instruction)
    _append_override(
        overrides,
        "environment_seed_manifest_path",
        cfg.EVALUATION.environment_seed_manifest_path,
    )
    _append_override(
        overrides,
        "deterministic_instruction_by_seed",
        cfg.EVALUATION.deterministic_instruction_by_seed,
    )
    _append_override(
        overrides,
        "save_imagination_transitions",
        cfg.EVALUATION.save_imagination_transitions,
    )
    _append_override(
        overrides,
        "imagination_transition_dir",
        str(robotwin_eval_base / "imagination_transitions"),
    )
    _append_override(
        overrides, "deterministic_algorithms", cfg.EVALUATION.deterministic_algorithms
    )
    _append_override(
        overrides, "deterministic_warn_only", cfg.EVALUATION.deterministic_warn_only
    )

    cmd = [
        sys.executable,
        "-u",
        str((PROJECT_ROOT / "experiments" / "robotwin" / "eval_policy_compat.py").resolve()),
        "--upstream-script",
        str((robotwin_root / "script" / "eval_policy.py").resolve()),
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    protocol = {
        "paper_aligned": bool(cfg.EVALUATION.paper_aligned),
        "strict_paired": bool(cfg.EVALUATION.strict_paired),
        "task": str(cfg.EVALUATION.task_name),
        "task_config": str(cfg.EVALUATION.task_config),
        "episodes": int(cfg.EVALUATION.eval_num_episodes),
        "eval_video_log": bool(cfg.EVALUATION.eval_video_log),
        "num_inference_steps": int(cfg.EVALUATION.num_inference_steps),
        "replan_steps": int(cfg.EVALUATION.replan_steps),
        "text_cfg_scale": float(cfg.EVALUATION.text_cfg_scale),
        "instruction_type": str(cfg.EVALUATION.instruction_type),
        "deterministic_instruction_by_seed": bool(
            cfg.EVALUATION.deterministic_instruction_by_seed
        ),
        "environment_seed_manifest_path": (
            None
            if _is_none_like(cfg.EVALUATION.environment_seed_manifest_path)
            else str(cfg.EVALUATION.environment_seed_manifest_path)
        ),
        "action_mode": str(cfg.EVALUATION.action_mode),
        "action_noise_std": float(cfg.EVALUATION.action_noise_std),
        "action_noise_seed": int(cfg.EVALUATION.action_noise_seed),
        "action_noise_replans": str(cfg.EVALUATION.action_noise_replans),
        "action_hold_probability": float(
            cfg.EVALUATION.action_hold_probability
        ),
        "action_hold_replans": str(cfg.EVALUATION.action_hold_replans),
        "action_corruption_seed": int(cfg.EVALUATION.action_corruption_seed),
        "residual_paired_advantage_gate_enabled": bool(
            cfg.EVALUATION.residual_paired_advantage_gate_enabled
        ),
        "residual_paired_advantage_threshold": (
            None
            if _is_none_like(cfg.EVALUATION.residual_paired_advantage_threshold)
            else float(cfg.EVALUATION.residual_paired_advantage_threshold)
        ),
        "residual_paired_advantage_max_disagreement": (
            None
            if _is_none_like(
                cfg.EVALUATION.residual_paired_advantage_max_disagreement
            )
            else float(
                cfg.EVALUATION.residual_paired_advantage_max_disagreement
            )
        ),
        "residual_q_gate_risk_scale": float(
            cfg.EVALUATION.residual_q_gate_risk_scale
        ),
        "residual_soft_scale_enabled": bool(
            cfg.EVALUATION.residual_soft_scale_enabled
        ),
        "residual_outcome_confirmation_enabled": bool(
            cfg.EVALUATION.residual_outcome_confirmation_enabled
        ),
        "residual_outcome_confirmation_min_progress": float(
            cfg.EVALUATION.residual_outcome_confirmation_min_progress
        ),
        "residual_outcome_confirmation_reanchor_replans": int(
            cfg.EVALUATION.residual_outcome_confirmation_reanchor_replans
        ),
        "save_imagination_transitions": bool(
            cfg.EVALUATION.save_imagination_transitions
        ),
    }
    print("FASTWAM_EVAL_PROTOCOL " + json.dumps(protocol, sort_keys=True), flush=True)
    OmegaConf.save(
        config=cfg,
        f=str(run_output_dir / f"eval_config_{str(cfg.EVALUATION.task_name)}.yaml"),
    )
    (run_output_dir / f"eval_protocol_{str(cfg.EVALUATION.task_name)}.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(robotwin_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")

    print(f"Evaluation finished successfully. Log saved to: {log_file}")


if __name__ == "__main__":
    main()
