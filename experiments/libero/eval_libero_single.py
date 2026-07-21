import json
import inspect
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from experiments.libero.imagination_reward_utils import (
    ACTION_MODES,
    apply_action_mode,
    save_aligned_transition,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.rl.online_policy import OnlineResidualPolicy
from fastwam.rl.audit import array_sha256, resolve_trial_indices
from libero.libero import benchmark
from action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _configure_deterministic_algorithms(cfg: DictConfig) -> dict[str, Any]:
    enabled = bool(cfg.EVALUATION.get("deterministic_algorithms", False))
    warn_only = bool(cfg.EVALUATION.get("deterministic_warn_only", True))
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            # Transformers 4.49 selects SDPA for SigLIP automatically.  The
            # fused Flash and memory-efficient CUDA implementations can return
            # different low-order bits across fresh processes even when
            # torch.use_deterministic_algorithms(True) is enabled.  Those bits
            # are enough to change the residual actor output and eventually the
            # simulated trajectory, so strict audits use the math backend only.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    return {
        "enabled": enabled,
        "warn_only": warn_only,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cuda_flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "cuda_mem_efficient_sdp_enabled": bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "cuda_math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _validate_imagination_reward_cfg(cfg: DictConfig) -> None:
    action_mode = str(cfg.EVALUATION.get("action_mode", "policy")).strip().lower()
    if action_mode not in ACTION_MODES:
        raise ValueError(
            f"EVALUATION.action_mode must be one of {sorted(ACTION_MODES)}, got {action_mode!r}."
        )
    noise_std = float(cfg.EVALUATION.get("action_noise_std", 0.15))
    if noise_std < 0:
        raise ValueError(f"EVALUATION.action_noise_std must be non-negative, got {noise_std}.")
    residual_checkpoint = cfg.EVALUATION.get("residual_checkpoint")
    residual_encoder_path = cfg.EVALUATION.get("residual_encoder_path")
    residual_encoder_version = cfg.EVALUATION.get("residual_encoder_version")
    if action_mode == "residual":
        if (
            residual_checkpoint is None
            or residual_encoder_path is None
            or residual_encoder_version is None
        ):
            raise ValueError(
                "EVALUATION.action_mode=residual requires residual_checkpoint, "
                "residual_encoder_path, and residual_encoder_version."
            )
    elif (
        residual_checkpoint is not None
        or residual_encoder_path is not None
        or residual_encoder_version is not None
    ):
        raise ValueError(
            "residual checkpoint/encoder settings may only be set when "
            "EVALUATION.action_mode=residual."
        )

    if not bool(cfg.EVALUATION.get("save_imagination_transitions", False)):
        return
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError(
            "EVALUATION.save_imagination_transitions=true requires "
            "EVALUATION.visualize_future_video=true."
        )
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    if replan_steps <= 0 or replan_steps % action_video_freq_ratio != 0:
        raise ValueError(
            "Initial imagination-reward capture requires replan_steps to be a positive "
            "multiple of data.train.action_video_freq_ratio; got "
            f"replan_steps={replan_steps}, ratio={action_video_freq_ratio}."
        )
    if bool(cfg.EVALUATION.get("imagination_use_direct_action", False)) and not bool(
        cfg.EVALUATION.get("visualize_future_video", False)
    ):
        raise ValueError(
            "EVALUATION.imagination_use_direct_action=true requires "
            "EVALUATION.visualize_future_video=true."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    save_rollout_video_enabled = bool(cfg.EVALUATION.get("save_rollout_video", True))
    save_prediction_videos = bool(cfg.EVALUATION.get("save_prediction_videos", True))
    predicted_future_frames = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    with torch.no_grad():
        if visualize_future_video:
            joint_kwargs = dict(infer_kwargs)
            if "test_action_with_infer_action" in inspect.signature(model.infer_joint).parameters:
                joint_kwargs["test_action_with_infer_action"] = False
            pred = model.infer_joint(**joint_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
            if bool(cfg.EVALUATION.get("imagination_use_direct_action", False)):
                action_parameters = inspect.signature(model.infer_action).parameters
                action_kwargs = {key: value for key, value in infer_kwargs.items() if key in action_parameters}
                pred["action"] = model.infer_action(**action_kwargs)["action"]
        else:
            pred = model.infer_action(**infer_kwargs)
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs, predicted_future_frames


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    residual_policy: Optional[OnlineResidualPolicy] = None,
    language_feature: Optional[np.ndarray] = None,
) -> tuple[
    bool,
    list,
    list[dict[str, Any]],
    Optional[float],
    int,
    Optional[float],
    Optional[dict[str, Any]],
]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    episode_policy_steps = 0
    episode_residual_values: list[float] = []
    action_replan_index = -1
    record_action_hashes = bool(cfg.EVALUATION.get("record_action_hashes", False))
    action_audit = (
        {
            "initial_state_sha256": array_sha256(initial_state),
            "replans": [],
        }
        if record_action_hashes
        else None
    )
    action_mode = str(cfg.EVALUATION.get("action_mode", "policy")).strip().lower()
    action_noise_std = float(cfg.EVALUATION.get("action_noise_std", 0.15))
    base_seed = 0 if cfg.get("seed") is None else int(cfg.seed)
    action_rng = np.random.default_rng(base_seed + episode_idx * 100_003)
    if (action_mode == "residual") != (residual_policy is not None):
        raise ValueError(
            "OnlineResidualPolicy must be provided if and only if action_mode='residual'."
        )

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_replan_index += 1
            action_chunk, imgs, predicted_future_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            baseline_action_chunk = np.asarray(action_chunk, dtype=np.float32).copy()
            residual_output = None
            if action_mode == "residual":
                residual_output = residual_policy.correct_action_chunk(
                    camera_images={
                        "agent": imgs["image"],
                        "wrist": imgs["wrist_image"],
                    },
                    proprio=_extract_sim_state(obs),
                    baseline_actions=baseline_action_chunk,
                    language_feature=language_feature,
                )
                action_chunk = residual_output.corrected_actions
                episode_residual_values.append(float(residual_output.residual_rms))
            else:
                action_chunk = apply_action_mode(
                    action_chunk,
                    mode=action_mode,
                    noise_std=action_noise_std,
                    rng=action_rng,
                )
            if action_audit is not None:
                audit_record = {
                    "replan_index": action_replan_index,
                    "agent_image_sha256": array_sha256(imgs["image"]),
                    "wrist_image_sha256": array_sha256(imgs["wrist_image"]),
                    "proprio_sha256": array_sha256(_extract_sim_state(obs)),
                    "baseline_actions_sha256": array_sha256(baseline_action_chunk),
                    "corrected_actions_sha256": array_sha256(action_chunk),
                    "executed_prefix_sha256": array_sha256(action_chunk[:replan_steps]),
                    "residual_actions_sha256": None,
                }
                if residual_output is not None:
                    audit_record["residual_actions_sha256"] = array_sha256(
                        residual_output.residual_actions
                    )
                    audit_record["observation_feature_sha256"] = array_sha256(
                        residual_output.observation_feature
                    )
                    if language_feature is not None:
                        audit_record["language_feature_sha256"] = array_sha256(
                            language_feature
                        )
                action_audit["replans"].append(audit_record)
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                    "action_mode": action_mode,
                    "action_noise_std": (
                        action_noise_std if action_mode == "noise" else 0.0
                    ),
                    "action_source": (
                        "residual_actor"
                        if action_mode == "residual"
                        else "direct_infer_action"
                        if bool(cfg.EVALUATION.get("imagination_use_direct_action", False))
                        else "infer_joint"
                    ),
                    "target_step": replan_steps,
                    "start_proprio": _extract_sim_state(obs).copy(),
                    "baseline_actions": baseline_action_chunk[:replan_steps].copy(),
                    "planned_executed_actions": np.asarray(
                        action_chunk[:replan_steps], dtype=np.float32
                    ).copy(),
                    "executed_actions": [],
                    "sim_rewards": [],
                }
                if language_feature is not None:
                    current_predicted_future_clip["language_feature"] = language_feature.copy()
                if residual_output is not None:
                    current_predicted_future_clip.update(
                        residual_checkpoint=residual_policy.checkpoint_path,
                        residual_encoder_path=residual_policy.encoder_path,
                        residual_encoder_version=residual_policy.encoder_version,
                        residual_rms=residual_output.residual_rms,
                        residual_max_abs=residual_output.residual_max_abs,
                    )
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        executed_action = np.asarray(pending_actions.pop(0), dtype=np.float32)
        obs, sim_reward, done, _ = env.step(executed_action)
        episode_policy_steps += 1
        if current_predicted_future_clip is not None:
            current_predicted_future_clip["executed_actions"].append(executed_action.copy())
            current_predicted_future_clip["sim_rewards"].append(float(sim_reward))
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            will_truncate = bool(
                not done and t + 1 >= max_steps + num_steps_wait
            )
            if done or len(pending_actions) == 0 or will_truncate:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                current_predicted_future_clip["effective_k"] = current_replan_step
                current_predicted_future_clip["terminated"] = bool(done)
                current_predicted_future_clip["truncated"] = will_truncate
                current_predicted_future_clip["transition_success"] = bool(done)
                current_predicted_future_clip["next_proprio"] = _extract_sim_state(obs).copy()
                current_predicted_future_clip["goal_frame_index"] = expected_frame_count - 1
                current_predicted_future_clip["alignment_valid"] = bool(
                    current_replan_step == replan_steps
                    and expected_frame_count == len(capture_steps) + 1
                )
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    episode_residual_rms = (
        float(np.mean(episode_residual_values)) if episode_residual_values else None
    )
    if action_audit is not None:
        action_audit.update(
            final_sim_state_sha256=array_sha256(env.get_sim_state()),
            policy_steps=episode_policy_steps,
            success=bool(done),
        )
    return (
        bool(done),
        replay_images,
        predicted_future_video_clips,
        episode_mean_psnr,
        episode_policy_steps,
        episode_residual_rms,
        action_audit,
    )


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    imagination_transition_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    residual_policy: Optional[OnlineResidualPolicy] = None,
    trial_indices: Optional[list[int]] = None,
) -> dict:
    seed = 0 if cfg.get("seed") is None else int(cfg.seed)
    deterministic_env = bool(cfg.EVALUATION.get("deterministic_env", False))
    if deterministic_env:
        random.seed(seed)
        np.random.seed(seed)
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
    if deterministic_env:
        env.env.hard_reset = False
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None
    save_imagination_transitions = bool(cfg.EVALUATION.get("save_imagination_transitions", False))
    needs_language_feature = save_imagination_transitions or (
        residual_policy is not None and residual_policy.requires_language_conditioning
    )
    language_feature = None
    language_encoder_version = None
    if needs_language_feature:
        if not hasattr(model, "encode_prompt_pooled"):
            raise ValueError("FastWAM model does not expose encode_prompt_pooled")
        language_prompt = DEFAULT_PROMPT.format(task=task_description)
        pooled = model.encode_prompt_pooled([language_prompt])
        language_feature = pooled[0].detach().cpu().numpy().astype(np.float32, copy=False)
        language_encoder_version = str(
            cfg.EVALUATION.get("language_encoder_version", "")
        ).strip()
        if not language_encoder_version:
            raise ValueError(
                "EVALUATION.language_encoder_version is required when collecting or using "
                "language-conditioned residual data"
            )
        results["language_encoder_version"] = language_encoder_version
        results["language_feature_sha256"] = array_sha256(language_feature)
    results["action_mode"] = str(cfg.EVALUATION.get("action_mode", "policy")).strip().lower()
    results["imagination_transition_count"] = 0
    results["valid_imagination_transition_count"] = 0
    results["episode_policy_steps"] = []
    if residual_policy is not None:
        results["residual_checkpoint"] = residual_policy.checkpoint_path
        results["residual_encoder_path"] = residual_policy.encoder_path
        results["residual_encoder_version"] = residual_policy.encoder_version
        results["episode_residual_rms"] = []
        results["residual_rms_mean"] = None
    record_action_hashes = bool(cfg.EVALUATION.get("record_action_hashes", False))
    if record_action_hashes:
        results["episode_action_audit"] = []

    if trial_indices is None:
        trial_indices = list(range(int(cfg.EVALUATION.num_trials)))
    results["trial_indices"] = list(trial_indices)

    for trial_idx in trial_indices:
        (
            success,
            replay_images,
            predicted_future_video_clips,
            episode_mean_psnr,
            episode_policy_steps,
            episode_residual_rms,
            action_audit,
        ) = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            residual_policy=residual_policy,
            language_feature=language_feature,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)
        results["episode_policy_steps"].append(int(episode_policy_steps))
        if residual_policy is not None:
            results["episode_residual_rms"].append(episode_residual_rms)
        if record_action_hashes:
            if action_audit is None:
                raise RuntimeError("Action audit was requested but not returned")
            action_audit["trial_index"] = int(trial_idx)
            results["episode_action_audit"].append(action_audit)

        if save_rollout_video_enabled:
            save_rollout_video(
                video_dir,
                replay_images,
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                success=success,
                task_description=task_description,
            )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    if save_prediction_videos:
                        save_prediction_video(
                            predicted_video_dir,
                            clip["gt_frames"],
                            clip["pred_frames"],
                            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                            clip["replan_idx"],
                            success=success,
                            task_description=task_description,
                        )
                    if save_imagination_transitions:
                        replan_idx = int(clip["replan_idx"])
                        alignment_valid = bool(clip.get("alignment_valid", False))
                        metadata = {
                            "task_suite": str(cfg.EVALUATION.task_suite_name),
                            "task_id": int(cfg.EVALUATION.task_id),
                            "task_description": task_description,
                            "trial_idx": trial_idx,
                            "replan_idx": replan_idx,
                            "success": bool(success),
                            "episode_success": bool(success),
                            "transition_success": bool(clip.get("transition_success", False)),
                            "terminated": bool(clip.get("terminated", False)),
                            "truncated": bool(clip.get("truncated", False)),
                            "action_mode": str(clip.get("action_mode", "policy")),
                            "action_noise_std": float(clip.get("action_noise_std", 0.0)),
                            "action_source": str(clip.get("action_source", "infer_joint")),
                            "target_step": int(clip.get("target_step", 0)),
                            "effective_k": int(clip.get("effective_k", 0)),
                            "episode_policy_steps": episode_policy_steps,
                            "alignment_valid": alignment_valid,
                            "policy_seed": None if cfg.get("seed") is None else int(cfg.seed),
                            "env_seed": None if cfg.get("seed") is None else int(cfg.seed),
                            "goal_seed": None if cfg.get("seed") is None else int(cfg.seed),
                            "action_seed": (
                                trial_idx * 100_003
                                if cfg.get("seed") is None
                                else int(cfg.seed) + trial_idx * 100_003
                            ),
                            "goal_frame_index": int(clip.get("goal_frame_index", -1)),
                            "goal_tau": float(clip.get("effective_k", 0)),
                            "policy_version": str(cfg.ckpt),
                            "predictor_version": str(cfg.ckpt),
                            "reward_encoder_version": "raw_images_not_encoded",
                            "language_encoder_version": language_encoder_version,
                            "language_pooling": "umt5_masked_mean_v1",
                            "language_prompt_template": DEFAULT_PROMPT,
                        }
                        if clip.get("residual_checkpoint") is not None:
                            metadata.update(
                                residual_checkpoint=str(clip["residual_checkpoint"]),
                                residual_encoder_path=str(clip["residual_encoder_path"]),
                                residual_encoder_version=str(
                                    clip["residual_encoder_version"]
                                ),
                                residual_rms=float(clip["residual_rms"]),
                                residual_max_abs=float(clip["residual_max_abs"]),
                            )
                        target_k = int(clip.get("target_step", 0))
                        effective_k = int(clip.get("effective_k", 0))
                        baseline_actions = np.asarray(clip["baseline_actions"], dtype=np.float32)
                        executed_actions = np.zeros_like(baseline_actions)
                        recorded_executed = np.asarray(clip["executed_actions"], dtype=np.float32)
                        executed_actions[:effective_k] = recorded_executed[:effective_k]
                        environment_rewards = np.zeros(target_k, dtype=np.float32)
                        recorded_rewards = np.asarray(clip["sim_rewards"], dtype=np.float32)
                        environment_rewards[:effective_k] = recorded_rewards[:effective_k]
                        save_aligned_transition(
                            imagination_transition_dir
                            / f"task{int(cfg.EVALUATION.task_id):02d}"
                            / f"trial{trial_idx:04d}"
                            / f"replan{replan_idx:04d}",
                            current_frame=clip["gt_frames"][0],
                            predicted_goal_frame=clip["pred_frames"][-1],
                            actual_frame=clip["gt_frames"][-1],
                            metadata=metadata,
                            rollout_arrays={
                                "proprio": np.asarray(clip["start_proprio"], dtype=np.float32),
                                "next_proprio": np.asarray(
                                    clip["next_proprio"], dtype=np.float32
                                ),
                                "baseline_actions": baseline_actions,
                                "planned_executed_actions": np.asarray(
                                    clip["planned_executed_actions"], dtype=np.float32
                                ),
                                "executed_actions": executed_actions,
                                "environment_rewards": environment_rewards,
                                "language_feature": np.asarray(
                                    clip["language_feature"], dtype=np.float32
                                ),
                            },
                        )
                        results["imagination_transition_count"] += 1
                        results["valid_imagination_transition_count"] += int(alignment_valid)
                if save_prediction_videos:
                    save_prediction_video(
                        predicted_video_dir,
                        all_gt_frames,
                        all_pred_frames,
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        "all",
                        success=success,
                        task_description=task_description,
                    )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    if residual_policy is not None:
        valid_residual_rms = [
            value for value in results["episode_residual_rms"] if value is not None
        ]
        if valid_residual_rms:
            results["residual_rms_mean"] = float(np.mean(valid_residual_rms))
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    determinism = _configure_deterministic_algorithms(cfg)
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)
    _validate_imagination_reward_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    action_mode = str(cfg.EVALUATION.get("action_mode", "policy")).strip().lower()
    residual_policy = None
    if action_mode == "residual":
        residual_device = str(cfg.EVALUATION.get("residual_device", model_device))
        residual_dtype = _mixed_precision_to_model_dtype(
            str(cfg.EVALUATION.get("residual_encoder_dtype", "bf16"))
        )
        residual_policy = OnlineResidualPolicy.from_checkpoint(
            checkpoint_path=str(cfg.EVALUATION.residual_checkpoint),
            encoder_path=str(cfg.EVALUATION.residual_encoder_path),
            device=residual_device,
            encoder_dtype=residual_dtype,
            encoder_version=str(cfg.EVALUATION.residual_encoder_version),
            language_encoder_version=str(
                cfg.EVALUATION.get("language_encoder_version", "")
            ),
            camera_image_size=int(
                cfg.EVALUATION.get("residual_camera_image_size", 224)
            ),
            allow_legacy_provenance=bool(
                cfg.EVALUATION.get("residual_allow_legacy_provenance", False)
            ),
        )
        logging.info(
            "Loaded online residual actor from %s with encoder %s on %s.",
            residual_policy.checkpoint_path,
            residual_policy.encoder_path,
            residual_device,
        )

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")
    if residual_policy is not None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
        if residual_policy.action_horizon != replan_steps:
            raise ValueError(
                "Residual actor action_horizon must equal EVALUATION.replan_steps: "
                f"actor={residual_policy.action_horizon} replan_steps={replan_steps}."
            )
        if residual_policy.action_dim != 7:
            raise ValueError(
                f"LIBERO residual actor action_dim must be 7, got {residual_policy.action_dim}."
            )

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)
    imagination_transition_dir = (
        local_log_dir / cfg.EVALUATION.task_suite_name / "imagination_transitions"
    )
    if bool(cfg.EVALUATION.get("save_imagination_transitions", False)):
        imagination_transition_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)
    explicit_trial_indices = cfg.EVALUATION.get("trial_indices", None)
    trial_indices = resolve_trial_indices(
        num_trials=int(cfg.EVALUATION.num_trials),
        trial_indices=explicit_trial_indices,
        available_states=len(initial_states),
    )
    if explicit_trial_indices is None:
        while len(initial_states) < int(cfg.EVALUATION.num_trials):
            initial_states.extend(
                initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))]
            )

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": len(trial_indices),
        "trial_indices": trial_indices,
        "deterministic_env": bool(cfg.EVALUATION.get("deterministic_env", False)),
        "deterministic_algorithms": determinism,
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        imagination_transition_dir=imagination_transition_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
        residual_policy=residual_policy,
        trial_indices=trial_indices,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{results['total_episodes']} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
