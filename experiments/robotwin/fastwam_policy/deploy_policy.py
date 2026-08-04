import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from experiments.robotwin.imagination_reward_utils import (
    ROBOTWIN_CAMERA_NAMES,
    apply_action_chunk_hold,
    apply_first_gripper_close_delay,
    apply_normalized_action_noise,
    array_sha256,
    save_aligned_transition,
    split_robotwin_camera_views,
    update_episode_success,
)
from fastwam.rl.online_policy import (
    ROBOTWIN_RESIDUAL_FEATURE_FUSION,
    OnlineResidualPolicy,
)
from fastwam.rl.language_routing import resolve_residual_language_instruction

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _imagined_goal_progress(
    current_feature: np.ndarray,
    actual_feature: np.ndarray,
    goal_feature: np.ndarray,
) -> float:
    """Cosine progress toward FastWAM's imagined goal in residual feature space."""

    normalized = []
    for name, value in (
        ("current", current_feature),
        ("actual", actual_feature),
        ("goal", goal_feature),
    ):
        feature = np.asarray(value, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(feature))
        if feature.size == 0 or not np.all(np.isfinite(feature)) or norm <= 0.0:
            raise ValueError(f"{name} outcome feature must be finite and non-zero")
        normalized.append(feature / norm)
    current, actual, goal = normalized
    return float(np.dot(actual, goal) - np.dot(current, goal))


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        capture_decode_tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        action_video_freq_ratio: int,
        action_mode: str,
        residual_checkpoint: Optional[str],
        residual_encoder_path: Optional[str],
        residual_encoder_version: Optional[str],
        residual_encoder_dtype: torch.dtype,
        residual_device: str,
        residual_q_gate_enabled: bool,
        residual_q_gate_margin: float,
        residual_q_gate_max_disagreement: float,
        residual_q_gate_risk_scale: float,
        residual_q_gate_risk_decay: float,
        residual_soft_scale_enabled: bool,
        residual_soft_scale_q_full_advantage: float,
        residual_soft_scale_support_full_margin: float,
        residual_q_gate_critic_source: str,
        residual_support_index_path: Optional[str],
        residual_support_circuit_breaker_enabled: bool,
        residual_shadow_mode: bool,
        residual_intervention_replans: Optional[set[int]],
        residual_max_interventions_per_episode: Optional[int],
        residual_outcome_confirmation_enabled: bool,
        residual_outcome_confirmation_min_progress: float,
        residual_outcome_confirmation_reanchor_replans: int,
        residual_language_instruction: Optional[str],
        action_noise_std: float,
        action_noise_seed: int,
        action_hold_probability: float,
        gripper_close_delay_steps: int,
        action_corruption_seed: int,
        trial_offset: int,
        fixed_instruction: Optional[str],
        save_imagination_transitions: bool,
        imagination_transition_dir: Optional[Path],
        task_name: str,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.capture_decode_tiled = bool(capture_decode_tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.action_mode = str(action_mode).strip().lower()
        if self.action_mode not in {
            "policy",
            "noise",
            "hold",
            "gripper_delay",
            "residual",
        }:
            raise ValueError(
                f"Unsupported action_mode={self.action_mode!r}; expected one of "
                "['policy', 'noise', 'hold', 'gripper_delay', 'residual']."
            )
        self.residual_policy: Optional[OnlineResidualPolicy] = None
        self.residual_intervention_replans = residual_intervention_replans
        self.residual_language_instruction = (
            None
            if _is_none_like(residual_language_instruction)
            else str(residual_language_instruction).strip()
        )
        if self.action_mode == "residual":
            if _is_none_like(residual_checkpoint):
                raise ValueError("action_mode='residual' requires residual_checkpoint")
            if _is_none_like(residual_encoder_path):
                raise ValueError("action_mode='residual' requires residual_encoder_path")
            if _is_none_like(residual_encoder_version):
                raise ValueError("action_mode='residual' requires residual_encoder_version")
            self.residual_policy = OnlineResidualPolicy.from_checkpoint(
                checkpoint_path=str(residual_checkpoint),
                encoder_path=str(residual_encoder_path),
                device=residual_device,
                encoder_dtype=residual_encoder_dtype,
                encoder_version=str(residual_encoder_version),
                language_encoder_version="fastwam_umt5_masked_mean_v1",
                camera_names=ROBOTWIN_CAMERA_NAMES,
                feature_fusion=ROBOTWIN_RESIDUAL_FEATURE_FUSION,
                q_gate_enabled=residual_q_gate_enabled,
                q_gate_margin=residual_q_gate_margin,
                q_gate_max_disagreement=residual_q_gate_max_disagreement,
                q_gate_risk_scale=residual_q_gate_risk_scale,
                q_gate_risk_decay=residual_q_gate_risk_decay,
                soft_scale_enabled=residual_soft_scale_enabled,
                soft_scale_q_full_advantage=(
                    residual_soft_scale_q_full_advantage
                ),
                soft_scale_support_full_margin=(
                    residual_soft_scale_support_full_margin
                ),
                q_gate_critic_source=residual_q_gate_critic_source,
                support_index_path=residual_support_index_path,
                support_circuit_breaker_enabled=(
                    residual_support_circuit_breaker_enabled
                ),
                shadow_mode=residual_shadow_mode,
                max_interventions_per_episode=(
                    residual_max_interventions_per_episode
                ),
                outcome_confirmation_enabled=(
                    residual_outcome_confirmation_enabled
                ),
                outcome_confirmation_min_progress=(
                    residual_outcome_confirmation_min_progress
                ),
                outcome_confirmation_reanchor_replans=(
                    residual_outcome_confirmation_reanchor_replans
                ),
            )
            if self.residual_policy.action_dim != 14:
                raise ValueError(
                    "RoboTwin residual checkpoint must use 14 action dimensions, "
                    f"got {self.residual_policy.action_dim}."
                )
            if self.residual_policy.action_horizon > self.action_horizon:
                raise ValueError(
                    "FastWAM action_horizon must cover the residual horizon; "
                    f"got baseline={self.action_horizon}, "
                    f"residual={self.residual_policy.action_horizon}."
                )
        self.action_noise_std = float(action_noise_std)
        if not np.isfinite(self.action_noise_std) or self.action_noise_std < 0.0:
            raise ValueError(
                f"action_noise_std must be finite and non-negative, got {self.action_noise_std}"
            )
        if self.action_mode == "policy" and self.action_noise_std != 0.0:
            raise ValueError("action_mode='policy' requires action_noise_std=0")
        self.action_noise_seed = int(action_noise_seed)
        self.action_hold_probability = float(action_hold_probability)
        if not 0.0 <= self.action_hold_probability <= 1.0:
            raise ValueError(
                "action_hold_probability must be in [0,1], got "
                f"{self.action_hold_probability}"
            )
        self.gripper_close_delay_steps = int(gripper_close_delay_steps)
        if self.gripper_close_delay_steps < 0:
            raise ValueError(
                "gripper_close_delay_steps must be non-negative, got "
                f"{self.gripper_close_delay_steps}"
            )
        self.action_corruption_seed = int(action_corruption_seed)
        if self.action_mode == "hold" and self.action_hold_probability <= 0.0:
            raise ValueError("action_mode='hold' requires action_hold_probability > 0")
        if self.action_mode == "gripper_delay" and self.gripper_close_delay_steps <= 0:
            raise ValueError(
                "action_mode='gripper_delay' requires gripper_close_delay_steps > 0"
            )
        self.trial_offset = int(trial_offset)
        if self.trial_offset < 0:
            raise ValueError(f"trial_offset must be non-negative, got {self.trial_offset}")
        self.fixed_instruction = (
            None if _is_none_like(fixed_instruction) else str(fixed_instruction)
        )
        self.save_imagination_transitions = bool(save_imagination_transitions)
        self.residual_outcome_confirmation_enabled = bool(
            residual_outcome_confirmation_enabled
        )
        self.imagination_transition_dir = (
            None if imagination_transition_dir is None else Path(imagination_transition_dir)
        )
        self.task_name = str(task_name)
        if self.save_imagination_transitions and self.imagination_transition_dir is None:
            raise ValueError(
                "save_imagination_transitions=true requires imagination_transition_dir"
            )
        if (
            self.save_imagination_transitions
            or self.residual_outcome_confirmation_enabled
        ):
            if self.replan_steps % self.action_video_freq_ratio != 0:
                raise ValueError(
                    "Aligned imagination feedback requires replan_steps to be a multiple of "
                    f"action_video_freq_ratio; got {self.replan_steps} and "
                    f"{self.action_video_freq_ratio}."
                )

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = self.trial_offset - 1
        self.step_count = 0
        self.replan_count = 0
        self._pending_transition: Optional[dict[str, Any]] = None
        self._episode_metadata_paths: list[Path] = []
        self._episode_success = False
        self._episode_initial_hash: Optional[str] = None
        self._gripper_delay_triggered = np.zeros(2, dtype=bool)
        self._gripper_delay_remaining = np.zeros(2, dtype=np.int64)
        self._language_feature_cache: dict[str, np.ndarray] = {}
        self._timing_rollout = {"infer_s": 0.0, "residual_s": 0.0, "sim_s": 0.0}
        self._residual_rollout_rms: list[float] = []
        self._residual_rollout_max_abs: list[float] = []
        self._residual_gate_decisions: list[bool] = []
        self._residual_gate_approvals: list[bool] = []

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | "
            "replan=%d | mode=%s | noise_std=%.3f | hold_prob=%.3f | "
            "gripper_delay=%d | capture=%s | outcome_confirmation=%s",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self.action_mode,
            self.action_noise_std,
            self.action_hold_probability,
            self.gripper_close_delay_steps,
            self.save_imagination_transitions,
            self.residual_outcome_confirmation_enabled,
        )

    def _behavior_tag(self) -> str:
        if self.action_mode == "policy":
            return "policy"
        if self.action_mode == "noise":
            return f"noise_{self.action_noise_std:.3f}"
        if self.action_mode == "hold":
            return f"hold_{self.action_hold_probability:.3f}"
        if self.action_mode == "gripper_delay":
            return f"gripper_delay_{self.gripper_close_delay_steps:03d}"
        return "residual"

    def _encode_language_feature(self, instruction: str) -> np.ndarray:
        cached = self._language_feature_cache.get(instruction)
        if cached is not None:
            return cached
        prompt = DEFAULT_PROMPT.format(task=instruction)
        pooled = self.model.encode_prompt_pooled([prompt])
        feature = pooled[0].detach().float().cpu().numpy().astype(
            np.float32, copy=False
        )
        self._language_feature_cache[instruction] = feature
        return feature

    def _residual_instruction(self, policy_instruction: str) -> str:
        return resolve_residual_language_instruction(
            policy_instruction,
            self.residual_language_instruction,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image(self, observation: Dict[str, Any]) -> np.ndarray:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        return np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        image = self._build_robotwin_image(observation)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(
        self, observation: Dict[str, Any], instruction: str
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Optional[list[Image.Image]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Any,
    ]:
        image_tensor = self._build_robotwin_image_tensor(observation)
        current_image = self._build_robotwin_image(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        action_parameters = inspect.signature(self.model.infer_action).parameters
        if "num_video_frames" in action_parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            predicted_frames = None
            action_kwargs = {
                key: value for key, value in infer_kwargs.items() if key in action_parameters
            }
            pred = self.model.infer_action(**action_kwargs)

        normalized_baseline = pred["action"].detach().float().cpu().numpy()
        if normalized_baseline.ndim == 3:
            normalized_baseline = normalized_baseline[0]
        if normalized_baseline.ndim != 2:
            raise ValueError(
                f"Expected normalized action [T,D], got {normalized_baseline.shape}"
            )
        normalized_executed = normalized_baseline.copy()
        residual_output = None
        epsilon = np.zeros_like(normalized_baseline, dtype=np.float32)
        if self.action_mode == "noise":
            noise_seed = (
                self.action_noise_seed
                + self.episode_count * 100_003
                + self.replan_count * 1_009
            )
            normalized_executed, epsilon = apply_normalized_action_noise(
                normalized_baseline,
                noise_std=self.action_noise_std,
                rng=np.random.default_rng(noise_seed),
            )

        baseline_actions = self._denormalize_action(
            torch.from_numpy(normalized_baseline)
        )[0]
        executed_actions = self._denormalize_action(
            torch.from_numpy(normalized_executed)
        )[0]
        if self.action_mode == "residual":
            if self.residual_policy is None:
                raise RuntimeError("residual action mode was initialized without a policy")
            residual_t0 = time.perf_counter() if self.timing_enabled else 0.0
            residual_instruction = self._residual_instruction(instruction)
            residual_output = self.residual_policy.correct_action_chunk(
                camera_images=split_robotwin_camera_views(current_image),
                proprio=state_vector,
                baseline_actions=baseline_actions,
                language_feature=self._encode_language_feature(residual_instruction),
                intervention_allowed=(
                    self.residual_intervention_replans is None
                    or self.replan_count in self.residual_intervention_replans
                ),
            )
            executed_actions = residual_output.corrected_actions
            if self.timing_enabled:
                self._timing_rollout["residual_s"] += time.perf_counter() - residual_t0
            self._residual_rollout_rms.append(residual_output.residual_rms)
            self._residual_rollout_max_abs.append(residual_output.residual_max_abs)
            self._residual_gate_decisions.append(residual_output.gate_applied)
            self._residual_gate_approvals.append(residual_output.gate_approved)
            gripper_residual_max = float(
                np.max(np.abs(residual_output.residual_actions[..., [6, 13]]))
            )
            q_gate_fields = ""
            if residual_output.q_advantages is not None:
                q_gate_fields = (
                    f" gate_applied={int(residual_output.gate_applied)}"
                    f" q_advantage_min={residual_output.q_advantage_min:.6f}"
                    " q_advantage_disagreement="
                    f"{residual_output.q_advantage_disagreement:.6f}"
                )
            support_fields = (
                f" gate_approved={int(residual_output.gate_approved)}"
                f" shadow_mode={int(residual_output.shadow_mode)}"
                f" circuit_breaker_active={int(residual_output.circuit_breaker_active)}"
                " circuit_breaker_triggered="
                f"{int(residual_output.circuit_breaker_triggered)}"
            )
            if residual_output.support_decision is not None:
                decision = residual_output.support_decision
                support_fields += (
                    f" support_state_score={decision.state_score:.6f}"
                    f" support_state_threshold={decision.state_threshold:.6f}"
                    f" support_action_score={decision.action_score:.6f}"
                    f" support_action_threshold={decision.action_threshold:.6f}"
                    f" support_in_distribution={int(decision.in_support)}"
                    f" support_language_similarity={decision.language_similarity:.6f}"
                )
            support_fields += (
                " residual_language_canonicalized="
                f"{int(residual_instruction != instruction)}"
                " intervention_allowed="
                f"{int(residual_output.intervention_allowed)}"
                " intervention_count="
                f"{residual_output.intervention_count}"
                " intervention_budget_remaining="
                f"{residual_output.intervention_budget_remaining}"
                " intervention_budget_exhausted="
                f"{int(residual_output.intervention_budget_exhausted)}"
                " q_gate_effective_margin="
                f"{residual_output.q_gate_effective_margin}"
                " candidate_residual_rms="
                f"{residual_output.candidate_residual_rms:.6f}"
                " residual_risk_before="
                f"{residual_output.residual_risk_before:.6f}"
                " residual_risk_after="
                f"{residual_output.residual_risk_after:.6f}"
                " residual_scale_factor="
                f"{residual_output.residual_scale_factor:.6f}"
                " q_scale_confidence="
                f"{residual_output.q_scale_confidence:.6f}"
                " support_scale_confidence="
                f"{residual_output.support_scale_confidence:.6f}"
                " outcome_confirmation_pending="
                f"{int(residual_output.outcome_confirmation_pending)}"
                " last_outcome_progress="
                f"{residual_output.last_outcome_progress}"
                " last_outcome_confirmed="
                f"{residual_output.last_outcome_confirmed}"
                " outcome_reanchor_remaining="
                f"{residual_output.outcome_reanchor_remaining}"
                " outcome_blocked="
                f"{int(residual_output.outcome_blocked)}"
            )
            print(
                "[fastwam-residual] "
                f"replan={self.replan_count} "
                f"rms={residual_output.residual_rms:.6f} "
                f"max_abs={residual_output.residual_max_abs:.6f} "
                f"gripper_max_abs={gripper_residual_max:.6f}"
                f"{q_gate_fields}"
                f"{support_fields}",
                flush=True,
            )
        corruption_mask = np.zeros_like(executed_actions, dtype=np.float32)
        if self.action_mode == "hold":
            corruption_seed = (
                self.action_corruption_seed
                + self.episode_count * 100_003
                + self.replan_count * 1_009
            )
            hold_chunk = bool(
                np.random.default_rng(corruption_seed).random()
                < self.action_hold_probability
            )
            executed_actions, corruption_mask = apply_action_chunk_hold(
                executed_actions,
                current_action=state_vector,
                hold_chunk=hold_chunk,
            )
        elif self.action_mode == "gripper_delay":
            (
                executed_actions,
                corruption_mask,
                self._gripper_delay_triggered,
                self._gripper_delay_remaining,
            ) = apply_first_gripper_close_delay(
                executed_actions,
                current_action=state_vector,
                delay_steps=self.gripper_close_delay_steps,
                already_triggered=self._gripper_delay_triggered,
                remaining_steps=self._gripper_delay_remaining,
            )
        needs_outcome_goal = bool(
            self.residual_outcome_confirmation_enabled
            and residual_output is not None
            and residual_output.gate_applied
        )
        if self.save_imagination_transitions or needs_outcome_goal:
            joint_kwargs = dict(infer_kwargs)
            joint_kwargs["num_video_frames"] = int(self._num_video_frames)
            if "decode_tiled" in inspect.signature(self.model.infer_joint).parameters:
                joint_kwargs["tiled"] = False
                joint_kwargs["decode_tiled"] = self.capture_decode_tiled
            if "test_action_with_infer_action" in inspect.signature(
                self.model.infer_joint
            ).parameters:
                joint_kwargs["test_action_with_infer_action"] = False
            with torch.no_grad():
                joint_pred = self.model.infer_joint(**joint_kwargs)
            keep_frames = 1 + self.replan_steps // self.action_video_freq_ratio
            predicted_frames = list(joint_pred["video"][:keep_frames])
            if len(predicted_frames) != keep_frames:
                raise ValueError(
                    "Predicted video is too short for aligned feedback: "
                    f"expected {keep_frames}, got {len(predicted_frames)}"
                )
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
        return (
            baseline_actions,
            executed_actions,
            predicted_frames,
            current_image,
            epsilon,
            corruption_mask,
            residual_output,
        )

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        (
            baseline_actions,
            executed_actions,
            predicted_frames,
            current_image,
            epsilon,
            corruption_mask,
            residual_output,
        ) = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, executed_actions.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(executed_actions[i], dtype=np.float32))

        state_vector = np.asarray(
            observation["joint_action"]["vector"], dtype=np.float32
        )
        if self._episode_initial_hash is None:
            self._episode_initial_hash = array_sha256(
                np.concatenate([current_image.reshape(-1), state_vector.reshape(-1)])
            )
            print(
                "FASTWAM_INITIAL_OBSERVATION "
                f"episode_id={self.episode_count} "
                f"sha256={self._episode_initial_hash}",
                flush=True,
            )

        capture_transition = self.save_imagination_transitions or bool(
            self.residual_outcome_confirmation_enabled
            and residual_output is not None
            and residual_output.gate_applied
        )
        if capture_transition:
            if predicted_frames is None:
                raise RuntimeError("feedback capture lacks an imagined video")
            residual_instruction = (
                self._residual_instruction(instruction)
                if self.action_mode == "residual"
                else instruction
            )
            language_feature = self._encode_language_feature(residual_instruction)
            self._pending_transition = {
                "replan_idx": self.replan_count,
                "instruction": instruction,
                "residual_language_instruction": residual_instruction,
                "current_image": current_image,
                "predicted_goal": predicted_frames[-1],
                "start_proprio": state_vector.copy(),
                "baseline_actions": baseline_actions[:n_exec].copy(),
                "planned_actions": executed_actions[:n_exec].copy(),
                "normalized_noise_direction": epsilon[:n_exec].copy(),
                "action_corruption_mask": corruption_mask[:n_exec].copy(),
                "executed_actions": [],
                "initial_observation_sha256": self._episode_initial_hash,
                "residual_gate_applied": bool(
                    residual_output is not None and residual_output.gate_applied
                ),
            }
            if self.residual_outcome_confirmation_enabled:
                if self.residual_policy is None or residual_output is None:
                    raise RuntimeError(
                        "outcome confirmation requires an initialized residual policy"
                    )
                self._pending_transition["current_residual_feature"] = (
                    residual_output.observation_feature.copy()
                )
                self._pending_transition["goal_residual_feature"] = (
                    self.residual_policy.encode_observation(
                        split_robotwin_camera_views(predicted_frames[-1])
                    )
                )
        self.replan_count += 1

    def _save_pending_transition(self, task_env, actual_observation: Dict[str, Any]) -> None:
        transition = self._pending_transition
        if transition is None:
            return
        executed_actions = np.asarray(transition["executed_actions"], dtype=np.float32)
        effective_k = int(executed_actions.shape[0])
        target_k = int(self.replan_steps)
        success = bool(task_env.eval_success)
        truncated = bool(task_env.take_action_cnt >= task_env.step_lim and not success)
        if self.residual_outcome_confirmation_enabled:
            if self.residual_policy is None:
                raise RuntimeError("outcome confirmation lacks a residual policy")
            actual_feature = self.residual_policy.encode_observation(
                split_robotwin_camera_views(
                    self._build_robotwin_image(actual_observation)
                )
            )
            outcome_progress = _imagined_goal_progress(
                transition["current_residual_feature"],
                actual_feature,
                transition["goal_residual_feature"],
            )
            self.residual_policy.record_intervention_outcome(outcome_progress)
            print(
                "[fastwam-residual-outcome] "
                f"replan={int(transition['replan_idx'])} "
                f"gate_applied={int(transition['residual_gate_applied'])} "
                f"imagination_progress={outcome_progress:.6f}",
                flush=True,
            )
        if not self.save_imagination_transitions:
            self._pending_transition = None
            return
        mode_tag = self._behavior_tag()
        record_dir = (
            self.imagination_transition_dir
            / self.task_name
            / mode_tag
            / f"episode_{self.episode_count:04d}"
            / f"replan_{int(transition['replan_idx']):04d}"
        )
        metadata = {
            "schema_version": "robotwin_imagination_transition_v1",
            "task_suite": "robotwin2.0",
            "task_name": self.task_name,
            "task_description": transition["instruction"],
            "trial_idx": self.episode_count,
            "replan_idx": int(transition["replan_idx"]),
            "action_mode": self.action_mode,
            "behavior_tag": mode_tag,
            "action_noise_std": (
                self.action_noise_std if self.action_mode == "noise" else 0.0
            ),
            "action_noise_seed": self.action_noise_seed,
            "action_hold_probability": (
                self.action_hold_probability if self.action_mode == "hold" else 0.0
            ),
            "gripper_close_delay_steps": (
                self.gripper_close_delay_steps
                if self.action_mode == "gripper_delay"
                else 0
            ),
            "action_corruption_seed": self.action_corruption_seed,
            "initial_observation_sha256": transition["initial_observation_sha256"],
            "environment_seed": (
                int(os.environ["FASTWAM_ENVIRONMENT_SEED"])
                if os.environ.get("FASTWAM_ENVIRONMENT_SEED") is not None
                else None
            ),
            "target_step": target_k,
            "effective_k": effective_k,
            "goal_frame_index": target_k // self.action_video_freq_ratio,
            "goal_tau": target_k,
            "terminated": success,
            "truncated": truncated,
            "transition_success": success,
            "episode_success": success,
            "alignment_valid": effective_k == target_k,
            "camera_layout": "head_256x320_over_left_right_128x160_v1",
            "policy_version": "fastwam_infer_action",
            "predictor_version": "fastwam_infer_joint",
            "language_encoder_version": "fastwam_umt5_masked_mean_v1",
            "language_prompt_template": DEFAULT_PROMPT,
            "residual_language_instruction": transition[
                "residual_language_instruction"
            ],
        }
        metadata_path = save_aligned_transition(
            record_dir,
            current_frame=transition["current_image"],
            predicted_goal_frame=transition["predicted_goal"],
            actual_frame=self._build_robotwin_image(actual_observation),
            metadata=metadata,
            rollout_arrays={
                "proprio": transition["start_proprio"],
                "next_proprio": np.asarray(
                    actual_observation["joint_action"]["vector"], dtype=np.float32
                ),
                "baseline_actions": transition["baseline_actions"][:effective_k],
                "planned_actions": transition["planned_actions"][:effective_k],
                "executed_actions": executed_actions,
                "normalized_noise_direction": transition["normalized_noise_direction"][:effective_k],
                "action_corruption_mask": transition["action_corruption_mask"][:effective_k],
                "environment_rewards": np.zeros(effective_k, dtype=np.float32),
                "language_feature": self._language_feature_cache[
                    transition["residual_language_instruction"]
                ],
            },
        )
        self._episode_metadata_paths.append(metadata_path)
        self._pending_transition = None
        if success and not self._episode_success:
            self._episode_success = True
            update_episode_success(self._episode_metadata_paths, True)

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            instruction = self.fixed_instruction or task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1
        if self._pending_transition is not None:
            self._pending_transition["executed_actions"].append(action.copy())
        if self._pending_transition is not None and (
            not self.pending_actions
            or bool(task_env.eval_success)
            or task_env.take_action_cnt >= task_env.step_lim
        ):
            actual_observation = task_env.get_obs()
            self._save_pending_transition(task_env, actual_observation)

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["residual_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "residual_s": float(self._timing_rollout["residual_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
            "residual_rms_mean": (
                0.0
                if not self._residual_rollout_rms
                else float(np.mean(self._residual_rollout_rms))
            ),
            "residual_max_abs": (
                0.0
                if not self._residual_rollout_max_abs
                else float(np.max(self._residual_rollout_max_abs))
            ),
            "residual_gate_apply_rate": (
                0.0
                if not self._residual_gate_decisions
                else float(np.mean(self._residual_gate_decisions))
            ),
            "residual_gate_approval_rate": (
                0.0
                if not self._residual_gate_approvals
                else float(np.mean(self._residual_gate_approvals))
            ),
        }

    def reset(self) -> None:
        if self._episode_metadata_paths:
            update_episode_success(self._episode_metadata_paths, self._episode_success)
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.replan_count = 0
        self._pending_transition = None
        self._episode_metadata_paths = []
        self._episode_success = False
        self._episode_initial_hash = None
        self._gripper_delay_triggered[:] = False
        self._gripper_delay_remaining[:] = 0
        self._residual_rollout_rms.clear()
        self._residual_rollout_max_abs.clear()
        self._residual_gate_decisions.clear()
        self._residual_gate_approvals.clear()
        if self.residual_policy is not None:
            self.residual_policy.reset()
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    deterministic_algorithms = _parse_bool(
        usr_args.get(
            "deterministic_algorithms",
            cfg.EVALUATION.get("deterministic_algorithms", True),
        )
    )
    deterministic_warn_only = _parse_bool(
        usr_args.get(
            "deterministic_warn_only",
            cfg.EVALUATION.get("deterministic_warn_only", False),
        )
    )
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    capture_decode_tiled = _parse_bool(
        usr_args.get(
            "capture_decode_tiled",
            cfg.EVALUATION.get("capture_decode_tiled", False),
        )
    )
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    action_mode = str(usr_args.get("action_mode", cfg.EVALUATION.get("action_mode", "policy")))
    residual_checkpoint_value = usr_args.get(
        "residual_checkpoint", cfg.EVALUATION.get("residual_checkpoint")
    )
    residual_checkpoint = (
        None
        if _is_none_like(residual_checkpoint_value)
        else str(residual_checkpoint_value)
    )
    residual_encoder_path_value = usr_args.get(
        "residual_encoder_path", cfg.EVALUATION.get("residual_encoder_path")
    )
    residual_encoder_path = (
        None
        if _is_none_like(residual_encoder_path_value)
        else str(residual_encoder_path_value)
    )
    residual_encoder_version_value = usr_args.get(
        "residual_encoder_version", cfg.EVALUATION.get("residual_encoder_version")
    )
    residual_encoder_version = (
        None
        if _is_none_like(residual_encoder_version_value)
        else str(residual_encoder_version_value)
    )
    residual_encoder_precision = str(
        usr_args.get(
            "residual_encoder_dtype",
            cfg.EVALUATION.get("residual_encoder_dtype", "bf16"),
        )
    )
    residual_encoder_dtype = _mixed_precision_to_model_dtype(
        residual_encoder_precision
    )
    residual_device = str(
        usr_args.get(
            "residual_device",
            cfg.EVALUATION.get("residual_device", "same"),
        )
    ).strip()
    if not residual_device or residual_device.lower() == "same":
        residual_device = device
    residual_q_gate_enabled = _parse_bool(
        usr_args.get(
            "residual_q_gate_enabled",
            cfg.EVALUATION.get("residual_q_gate_enabled", False),
        )
    )
    residual_q_gate_margin = float(
        usr_args.get(
            "residual_q_gate_margin",
            cfg.EVALUATION.get("residual_q_gate_margin", 0.0),
        )
    )
    residual_q_gate_max_disagreement = float(
        usr_args.get(
            "residual_q_gate_max_disagreement",
            cfg.EVALUATION.get("residual_q_gate_max_disagreement", 0.05),
        )
    )
    residual_q_gate_risk_scale = float(
        usr_args.get(
            "residual_q_gate_risk_scale",
            cfg.EVALUATION.get("residual_q_gate_risk_scale", 0.0),
        )
    )
    residual_q_gate_risk_decay = float(
        usr_args.get(
            "residual_q_gate_risk_decay",
            cfg.EVALUATION.get("residual_q_gate_risk_decay", 1.0),
        )
    )
    residual_soft_scale_enabled = _parse_bool(
        usr_args.get(
            "residual_soft_scale_enabled",
            cfg.EVALUATION.get("residual_soft_scale_enabled", False),
        )
    )
    residual_soft_scale_q_full_advantage = float(
        usr_args.get(
            "residual_soft_scale_q_full_advantage",
            cfg.EVALUATION.get("residual_soft_scale_q_full_advantage", 0.005),
        )
    )
    residual_soft_scale_support_full_margin = float(
        usr_args.get(
            "residual_soft_scale_support_full_margin",
            cfg.EVALUATION.get("residual_soft_scale_support_full_margin", 0.25),
        )
    )
    residual_q_gate_critic_source = str(
        usr_args.get(
            "residual_q_gate_critic_source",
            cfg.EVALUATION.get("residual_q_gate_critic_source", "target"),
        )
    )
    residual_support_index_value = usr_args.get(
        "residual_support_index_path",
        cfg.EVALUATION.get("residual_support_index_path"),
    )
    residual_support_index_path = (
        None
        if _is_none_like(residual_support_index_value)
        else str(residual_support_index_value)
    )
    residual_support_circuit_breaker_enabled = _parse_bool(
        usr_args.get(
            "residual_support_circuit_breaker_enabled",
            cfg.EVALUATION.get(
                "residual_support_circuit_breaker_enabled", True
            ),
        )
    )
    residual_shadow_mode = _parse_bool(
        usr_args.get(
            "residual_shadow_mode",
            cfg.EVALUATION.get("residual_shadow_mode", False),
        )
    )
    residual_intervention_value = usr_args.get(
        "residual_intervention_replans",
        cfg.EVALUATION.get("residual_intervention_replans", "all"),
    )
    if _is_none_like(residual_intervention_value) or str(
        residual_intervention_value
    ).strip().lower() == "all":
        residual_intervention_replans = None
    else:
        residual_intervention_replans = {
            int(item.strip())
            for item in str(residual_intervention_value).split(",")
            if item.strip()
        }
        if not residual_intervention_replans or min(
            residual_intervention_replans
        ) < 0:
            raise ValueError(
                "residual_intervention_replans must be 'all' or non-negative CSV indices"
            )
    residual_max_interventions_value = usr_args.get(
        "residual_max_interventions_per_episode",
        cfg.EVALUATION.get("residual_max_interventions_per_episode"),
    )
    residual_max_interventions_per_episode = (
        None
        if _is_none_like(residual_max_interventions_value)
        else int(residual_max_interventions_value)
    )
    if (
        residual_max_interventions_per_episode is not None
        and residual_max_interventions_per_episode <= 0
    ):
        raise ValueError(
            "residual_max_interventions_per_episode must be positive or null"
        )
    residual_outcome_confirmation_enabled = _parse_bool(
        usr_args.get(
            "residual_outcome_confirmation_enabled",
            cfg.EVALUATION.get("residual_outcome_confirmation_enabled", False),
        )
    )
    residual_outcome_confirmation_min_progress = float(
        usr_args.get(
            "residual_outcome_confirmation_min_progress",
            cfg.EVALUATION.get(
                "residual_outcome_confirmation_min_progress", 0.0
            ),
        )
    )
    residual_outcome_confirmation_reanchor_replans = int(
        usr_args.get(
            "residual_outcome_confirmation_reanchor_replans",
            cfg.EVALUATION.get(
                "residual_outcome_confirmation_reanchor_replans", 1
            ),
        )
    )
    residual_language_instruction_value = usr_args.get(
        "residual_language_instruction",
        cfg.EVALUATION.get("residual_language_instruction"),
    )
    residual_language_instruction = (
        None
        if _is_none_like(residual_language_instruction_value)
        else str(residual_language_instruction_value)
    )
    action_noise_std = float(
        usr_args.get("action_noise_std", cfg.EVALUATION.get("action_noise_std", 0.0))
    )
    action_noise_seed = int(
        usr_args.get("action_noise_seed", cfg.EVALUATION.get("action_noise_seed", 0))
    )
    action_hold_probability = float(
        usr_args.get(
            "action_hold_probability",
            cfg.EVALUATION.get("action_hold_probability", 0.0),
        )
    )
    gripper_close_delay_steps = int(
        usr_args.get(
            "gripper_close_delay_steps",
            cfg.EVALUATION.get("gripper_close_delay_steps", 0),
        )
    )
    action_corruption_seed = int(
        usr_args.get(
            "action_corruption_seed",
            cfg.EVALUATION.get("action_corruption_seed", 0),
        )
    )
    trial_offset = int(usr_args.get("trial_offset", cfg.EVALUATION.get("trial_offset", 0)))
    fixed_instruction_value = usr_args.get(
        "fixed_instruction", cfg.EVALUATION.get("fixed_instruction")
    )
    fixed_instruction = (
        None if _is_none_like(fixed_instruction_value) else str(fixed_instruction_value)
    )
    save_imagination_transitions = _parse_bool(
        usr_args.get(
            "save_imagination_transitions",
            cfg.EVALUATION.get("save_imagination_transitions", False),
        )
    )
    transition_dir_value = usr_args.get("imagination_transition_dir")
    imagination_transition_dir = (
        None
        if _is_none_like(transition_dir_value)
        else Path(str(transition_dir_value)).expanduser().resolve()
    )
    task_name = str(usr_args.get("task_name") or "unknown_task")

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        capture_decode_tiled=capture_decode_tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        action_video_freq_ratio=int(cfg.data.train.action_video_freq_ratio),
        action_mode=action_mode,
        residual_checkpoint=residual_checkpoint,
        residual_encoder_path=residual_encoder_path,
        residual_encoder_version=residual_encoder_version,
        residual_encoder_dtype=residual_encoder_dtype,
        residual_device=residual_device,
        residual_q_gate_enabled=residual_q_gate_enabled,
        residual_q_gate_margin=residual_q_gate_margin,
        residual_q_gate_max_disagreement=residual_q_gate_max_disagreement,
        residual_q_gate_risk_scale=residual_q_gate_risk_scale,
        residual_q_gate_risk_decay=residual_q_gate_risk_decay,
        residual_soft_scale_enabled=residual_soft_scale_enabled,
        residual_soft_scale_q_full_advantage=(
            residual_soft_scale_q_full_advantage
        ),
        residual_soft_scale_support_full_margin=(
            residual_soft_scale_support_full_margin
        ),
        residual_q_gate_critic_source=residual_q_gate_critic_source,
        residual_support_index_path=residual_support_index_path,
        residual_support_circuit_breaker_enabled=(
            residual_support_circuit_breaker_enabled
        ),
        residual_shadow_mode=residual_shadow_mode,
        residual_intervention_replans=residual_intervention_replans,
        residual_max_interventions_per_episode=(
            residual_max_interventions_per_episode
        ),
        residual_outcome_confirmation_enabled=(
            residual_outcome_confirmation_enabled
        ),
        residual_outcome_confirmation_min_progress=(
            residual_outcome_confirmation_min_progress
        ),
        residual_outcome_confirmation_reanchor_replans=(
            residual_outcome_confirmation_reanchor_replans
        ),
        residual_language_instruction=residual_language_instruction,
        action_noise_std=action_noise_std,
        action_noise_seed=action_noise_seed,
        action_hold_probability=action_hold_probability,
        gripper_close_delay_steps=gripper_close_delay_steps,
        action_corruption_seed=action_corruption_seed,
        trial_offset=trial_offset,
        fixed_instruction=fixed_instruction,
        save_imagination_transitions=save_imagination_transitions,
        imagination_transition_dir=imagination_transition_dir,
        task_name=task_name,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
