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
    apply_normalized_action_noise,
    array_sha256,
    save_aligned_transition,
    update_episode_success,
)

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
        timing_enabled: bool,
        num_video_frames: int,
        action_video_freq_ratio: int,
        action_mode: str,
        action_noise_std: float,
        action_noise_seed: int,
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
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.action_mode = str(action_mode).strip().lower()
        if self.action_mode not in {"policy", "noise"}:
            raise ValueError(
                f"Unsupported action_mode={self.action_mode!r}; expected 'policy' or 'noise'."
            )
        self.action_noise_std = float(action_noise_std)
        if not np.isfinite(self.action_noise_std) or self.action_noise_std < 0.0:
            raise ValueError(
                f"action_noise_std must be finite and non-negative, got {self.action_noise_std}"
            )
        if self.action_mode == "policy" and self.action_noise_std != 0.0:
            raise ValueError("action_mode='policy' requires action_noise_std=0")
        self.action_noise_seed = int(action_noise_seed)
        self.trial_offset = int(trial_offset)
        if self.trial_offset < 0:
            raise ValueError(f"trial_offset must be non-negative, got {self.trial_offset}")
        self.fixed_instruction = (
            None if _is_none_like(fixed_instruction) else str(fixed_instruction)
        )
        self.save_imagination_transitions = bool(save_imagination_transitions)
        self.imagination_transition_dir = (
            None if imagination_transition_dir is None else Path(imagination_transition_dir)
        )
        self.task_name = str(task_name)
        if self.save_imagination_transitions:
            if self.imagination_transition_dir is None:
                raise ValueError(
                    "save_imagination_transitions=true requires imagination_transition_dir"
                )
            if self.replan_steps % self.action_video_freq_ratio != 0:
                raise ValueError(
                    "Aligned capture requires replan_steps to be a multiple of "
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
        self._language_feature_cache: dict[str, np.ndarray] = {}
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | "
            "replan=%d | mode=%s | noise_std=%.3f | capture=%s",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self.action_mode,
            self.action_noise_std,
            self.save_imagination_transitions,
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
    ) -> tuple[np.ndarray, np.ndarray, Optional[list[Image.Image]], np.ndarray, np.ndarray]:
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
            if self.save_imagination_transitions:
                joint_kwargs = dict(infer_kwargs)
                joint_kwargs["num_video_frames"] = int(self._num_video_frames)
                if "test_action_with_infer_action" in inspect.signature(
                    self.model.infer_joint
                ).parameters:
                    joint_kwargs["test_action_with_infer_action"] = False
                joint_pred = self.model.infer_joint(**joint_kwargs)
                keep_frames = 1 + self.replan_steps // self.action_video_freq_ratio
                predicted_frames = list(joint_pred["video"][:keep_frames])
                if len(predicted_frames) != keep_frames:
                    raise ValueError(
                        "Predicted video is too short for aligned capture: "
                        f"expected {keep_frames}, got {len(predicted_frames)}"
                    )
            action_kwargs = {
                key: value for key, value in infer_kwargs.items() if key in action_parameters
            }
            pred = self.model.infer_action(**action_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        normalized_baseline = pred["action"].detach().float().cpu().numpy()
        if normalized_baseline.ndim == 3:
            normalized_baseline = normalized_baseline[0]
        if normalized_baseline.ndim != 2:
            raise ValueError(
                f"Expected normalized action [T,D], got {normalized_baseline.shape}"
            )
        normalized_executed = normalized_baseline.copy()
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
        return (
            baseline_actions,
            executed_actions,
            predicted_frames,
            current_image,
            epsilon,
        )

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        (
            baseline_actions,
            executed_actions,
            predicted_frames,
            current_image,
            epsilon,
        ) = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, executed_actions.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(executed_actions[i], dtype=np.float32))

        if self.save_imagination_transitions:
            if predicted_frames is None:
                raise RuntimeError("capture enabled but infer_joint produced no video")
            state_vector = np.asarray(
                observation["joint_action"]["vector"], dtype=np.float32
            )
            if self._episode_initial_hash is None:
                self._episode_initial_hash = array_sha256(
                    np.concatenate([current_image.reshape(-1), state_vector.reshape(-1)])
                )
            language_feature = self._language_feature_cache.get(instruction)
            if language_feature is None:
                prompt = DEFAULT_PROMPT.format(task=instruction)
                pooled = self.model.encode_prompt_pooled([prompt])
                language_feature = (
                    pooled[0].detach().float().cpu().numpy().astype(np.float32, copy=False)
                )
                self._language_feature_cache[instruction] = language_feature
            self._pending_transition = {
                "replan_idx": self.replan_count,
                "instruction": instruction,
                "current_image": current_image,
                "predicted_goal": predicted_frames[-1],
                "start_proprio": state_vector.copy(),
                "baseline_actions": baseline_actions[:n_exec].copy(),
                "planned_actions": executed_actions[:n_exec].copy(),
                "normalized_noise_direction": epsilon[:n_exec].copy(),
                "executed_actions": [],
                "initial_observation_sha256": self._episode_initial_hash,
            }
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
        mode_tag = (
            "policy"
            if self.action_mode == "policy"
            else f"noise_{self.action_noise_std:.3f}"
        )
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
            "action_noise_std": (
                self.action_noise_std if self.action_mode == "noise" else 0.0
            ),
            "action_noise_seed": self.action_noise_seed,
            "initial_observation_sha256": transition["initial_observation_sha256"],
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
                "environment_rewards": np.zeros(effective_k, dtype=np.float32),
                "language_feature": self._language_feature_cache[transition["instruction"]],
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
        if self.save_imagination_transitions and (
            not self.pending_actions
            or bool(task_env.eval_success)
            or task_env.take_action_cnt >= task_env.step_lim
        ):
            actual_observation = task_env.get_obs()
            self._save_pending_transition(task_env, actual_observation)

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
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
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    action_mode = str(usr_args.get("action_mode", cfg.EVALUATION.get("action_mode", "policy")))
    action_noise_std = float(
        usr_args.get("action_noise_std", cfg.EVALUATION.get("action_noise_std", 0.0))
    )
    action_noise_seed = int(
        usr_args.get("action_noise_seed", cfg.EVALUATION.get("action_noise_seed", 0))
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
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        action_video_freq_ratio=int(cfg.data.train.action_video_freq_ratio),
        action_mode=action_mode,
        action_noise_std=action_noise_std,
        action_noise_seed=action_noise_seed,
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
