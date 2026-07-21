"""Online adapter for applying a trained residual actor to FastWAM actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from .models import ResidualActor, ResidualActorConfig


LIBERO_RESIDUAL_CAMERA_NAMES = ("agent", "wrist")
RESIDUAL_CHECKPOINT_FORMAT = "fastwam_residual_awr_v2"
_SUPPORTED_RESIDUAL_CHECKPOINT_FORMATS = {
    "fastwam_residual_awr_v1",
    RESIDUAL_CHECKPOINT_FORMAT,
}
RESIDUAL_FEATURE_FUSION = "per_camera_l2_then_agent_wrist_concat_l2_v1"


def _tuple_actor_config(payload: Mapping[str, Any]) -> ResidualActorConfig:
    config = dict(payload)
    for key in ("hidden_dims", "residual_scale", "action_low", "action_high"):
        if key in config:
            config[key] = tuple(config[key])
    return ResidualActorConfig(**config)


def load_residual_actor_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[ResidualActor, dict[str, Any]]:
    """Load and strictly validate a residual-AWR actor checkpoint."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Residual checkpoint must be a mapping, got {type(payload)}")
    if payload.get("format") not in _SUPPORTED_RESIDUAL_CHECKPOINT_FORMATS:
        raise ValueError(
            f"Unsupported residual checkpoint format {payload.get('format')!r}; "
            f"expected one of {sorted(_SUPPORTED_RESIDUAL_CHECKPOINT_FORMATS)}."
        )
    if not isinstance(payload.get("actor"), dict) or not isinstance(
        payload.get("actor_config"), dict
    ):
        raise ValueError("Residual checkpoint must contain actor and actor_config mappings.")
    awr_config = payload.get("awr_config")
    if not isinstance(awr_config, dict):
        raise ValueError("Residual checkpoint must contain an awr_config mapping.")
    if bool(awr_config.get("use_goal_conditioning", False)):
        raise ValueError(
            "Online residual evaluation currently requires use_goal_conditioning=false."
        )

    actor = ResidualActor(_tuple_actor_config(payload["actor_config"]))
    actor.load_state_dict(payload["actor"], strict=True)
    actor.to(device=device, dtype=torch.float32).eval()
    return actor, payload


def combine_normalized_camera_features(
    camera_features: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Match replay construction: agent then wrist, followed by L2 normalization."""

    if set(camera_features) != set(LIBERO_RESIDUAL_CAMERA_NAMES):
        raise ValueError(
            "Expected residual camera features "
            f"{LIBERO_RESIDUAL_CAMERA_NAMES}, got {sorted(camera_features)}."
        )
    features = []
    feature_dims = set()
    for camera in LIBERO_RESIDUAL_CAMERA_NAMES:
        feature = np.asarray(camera_features[camera], dtype=np.float32).reshape(-1)
        if feature.size == 0 or not np.all(np.isfinite(feature)):
            raise ValueError(f"Camera feature {camera!r} must be finite and non-empty.")
        norm = float(np.linalg.norm(feature))
        if norm <= 0.0:
            raise ValueError(f"Camera feature {camera!r} has zero norm.")
        features.append(feature / norm)
        feature_dims.add(int(feature.size))
    if len(feature_dims) != 1:
        raise ValueError(f"Camera feature dimensions differ: {sorted(feature_dims)}")
    combined = np.concatenate(features).astype(np.float32, copy=False)
    return combined / float(np.linalg.norm(combined))


def _rgb_pil(image: Any, *, camera: str, image_size: int) -> Image.Image:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Camera {camera!r} must be RGB [H,W,3], got {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Camera {camera!r} must use an integer image dtype, got {array.dtype}.")
    array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    return Image.fromarray(array, mode="RGB").resize(
        (image_size, image_size), resample=Image.BILINEAR
    )


@dataclass(frozen=True)
class ResidualPolicyOutput:
    corrected_actions: np.ndarray
    residual_actions: np.ndarray
    observation_feature: np.ndarray

    @property
    def residual_rms(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.residual_actions))))

    @property
    def residual_max_abs(self) -> float:
        return float(np.max(np.abs(self.residual_actions)))


class OnlineResidualPolicy:
    """Frozen SigLIP observation encoder plus a trained residual actor."""

    def __init__(
        self,
        *,
        actor: ResidualActor,
        image_processor: Any,
        vision_encoder: torch.nn.Module,
        device: torch.device | str,
        encoder_dtype: torch.dtype,
        checkpoint_path: str | Path,
        encoder_path: str | Path,
        encoder_version: str,
        language_encoder_version: str | None = None,
        camera_image_size: int = 224,
    ):
        if camera_image_size <= 0:
            raise ValueError("camera_image_size must be positive")
        self.actor = actor
        self.image_processor = image_processor
        self.vision_encoder = vision_encoder
        self.device = torch.device(device)
        self.encoder_dtype = encoder_dtype
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.encoder_path = str(Path(encoder_path).expanduser().resolve())
        self.encoder_version = str(encoder_version)
        self.language_encoder_version = (
            None
            if language_encoder_version is None
            else str(language_encoder_version).strip()
        )
        self.camera_image_size = int(camera_image_size)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str | Path,
        encoder_path: str | Path,
        device: torch.device | str,
        encoder_dtype: torch.dtype,
        encoder_version: str,
        language_encoder_version: str | None = None,
        camera_image_size: int = 224,
        allow_legacy_provenance: bool = False,
    ) -> "OnlineResidualPolicy":
        from transformers import SiglipImageProcessor, SiglipVisionModel

        device = torch.device(device)
        if device.type == "cpu" and encoder_dtype != torch.float32:
            encoder_dtype = torch.float32
        encoder_path = Path(encoder_path).expanduser().resolve()
        if not encoder_path.is_dir():
            raise FileNotFoundError(encoder_path)
        actor, payload = load_residual_actor_checkpoint(checkpoint_path, device=device)
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            raise ValueError("Residual checkpoint must contain a summary mapping.")
        encoder_version = str(encoder_version).strip()
        if not encoder_version:
            raise ValueError("encoder_version must be a non-empty immutable identifier.")
        provenance = payload.get("replay_provenance")
        if not provenance:
            if not allow_legacy_provenance:
                raise ValueError(
                    "Residual checkpoint lacks replay encoder provenance; pass an explicitly "
                    "audited legacy override only for pipeline smoke evaluation."
                )
        elif not isinstance(provenance, dict):
            raise ValueError("Residual checkpoint replay_provenance must be a mapping.")
        else:
            expected = {
                "reward_encoder_version": encoder_version,
                "camera_names": list(LIBERO_RESIDUAL_CAMERA_NAMES),
                "camera_image_size": int(camera_image_size),
                "feature_fusion": RESIDUAL_FEATURE_FUSION,
            }
            mismatches = {
                key: {"checkpoint": provenance.get(key), "requested": value}
                for key, value in expected.items()
                if provenance.get(key) != value
            }
            if mismatches:
                raise ValueError(f"Residual encoder provenance mismatch: {mismatches}")

        image_processor = SiglipImageProcessor.from_pretrained(
            encoder_path, local_files_only=True
        )
        vision_encoder = SiglipVisionModel.from_pretrained(
            encoder_path,
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=encoder_dtype,
        ).to(device).eval()
        hidden_size = int(vision_encoder.config.hidden_size)
        expected_feature_dim = len(LIBERO_RESIDUAL_CAMERA_NAMES) * hidden_size
        if int(summary.get("feature_dim", -1)) != expected_feature_dim:
            raise ValueError(
                "Residual checkpoint feature_dim does not match the SigLIP encoder: "
                f"checkpoint={summary.get('feature_dim')} encoder={expected_feature_dim}."
            )
        expected_context_dim = expected_feature_dim + int(summary.get("proprio_dim", -1))
        if actor.config.context_dim != expected_context_dim:
            raise ValueError(
                "Residual actor context_dim does not match feature_dim + proprio_dim: "
                f"actor={actor.config.context_dim} expected={expected_context_dim}."
            )
        if actor.config.language_feature_dim > 0:
            if int(summary.get("language_feature_dim", -1)) != actor.config.language_feature_dim:
                raise ValueError(
                    "Residual language feature dimension is inconsistent between actor and summary."
                )
            checkpoint_language_version = (
                ""
                if not isinstance(provenance, dict)
                else str(provenance.get("language_encoder_version", "")).strip()
            )
            requested_language_version = str(language_encoder_version or "").strip()
            if not checkpoint_language_version:
                raise ValueError(
                    "Language-conditioned residual checkpoint lacks language encoder provenance."
                )
            if not requested_language_version:
                raise ValueError(
                    "language_encoder_version is required for a language-conditioned checkpoint."
                )
            if checkpoint_language_version != requested_language_version:
                raise ValueError(
                    "Residual language encoder provenance mismatch: "
                    f"checkpoint={checkpoint_language_version!r} "
                    f"requested={requested_language_version!r}."
                )
        return cls(
            actor=actor,
            image_processor=image_processor,
            vision_encoder=vision_encoder,
            device=device,
            encoder_dtype=encoder_dtype,
            checkpoint_path=checkpoint_path,
            encoder_path=encoder_path,
            encoder_version=encoder_version,
            language_encoder_version=language_encoder_version,
            camera_image_size=camera_image_size,
        )

    @property
    def action_horizon(self) -> int:
        return int(self.actor.config.action_horizon)

    @property
    def action_dim(self) -> int:
        return int(self.actor.config.action_dim)

    @property
    def requires_language_conditioning(self) -> bool:
        return self.actor.config.language_feature_dim > 0

    def encode_observation(self, camera_images: Mapping[str, Any]) -> np.ndarray:
        if set(camera_images) != set(LIBERO_RESIDUAL_CAMERA_NAMES):
            raise ValueError(
                f"Expected camera images {LIBERO_RESIDUAL_CAMERA_NAMES}, "
                f"got {sorted(camera_images)}."
            )
        images = [
            _rgb_pil(
                camera_images[camera],
                camera=camera,
                image_size=self.camera_image_size,
            )
            for camera in LIBERO_RESIDUAL_CAMERA_NAMES
        ]
        inputs = self.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=self.device, dtype=self.encoder_dtype
        )
        with torch.inference_mode():
            pooled = self.vision_encoder(pixel_values=pixel_values).pooler_output
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy()
        return combine_normalized_camera_features(
            {
                camera: feature
                for camera, feature in zip(LIBERO_RESIDUAL_CAMERA_NAMES, pooled)
            }
        )

    def correct_from_feature(
        self,
        *,
        observation_feature: np.ndarray,
        proprio: np.ndarray,
        baseline_actions: np.ndarray,
        language_feature: np.ndarray | None = None,
    ) -> ResidualPolicyOutput:
        feature = np.asarray(observation_feature, dtype=np.float32).reshape(-1)
        state = np.asarray(proprio, dtype=np.float32).reshape(-1)
        baseline = np.asarray(baseline_actions, dtype=np.float32)
        if not np.all(np.isfinite(feature)) or not np.all(np.isfinite(state)):
            raise ValueError("Residual context contains non-finite values.")
        expected = (self.action_horizon, self.action_dim)
        if baseline.ndim != 2 or baseline.shape[0] < self.action_horizon:
            raise ValueError(
                "baseline_actions must contain at least the residual horizon; "
                f"got {baseline.shape}, required first dimensions >= {expected}."
            )
        if baseline.shape[1] != self.action_dim:
            raise ValueError(
                f"baseline action_dim must be {self.action_dim}, got {baseline.shape[1]}."
            )
        context = np.concatenate([feature, state]).astype(np.float32, copy=False)
        if context.shape != (self.actor.config.context_dim,):
            raise ValueError(
                f"Residual context must have shape {(self.actor.config.context_dim,)}, "
                f"got {context.shape}."
            )

        actor_context = torch.from_numpy(context).unsqueeze(0).to(self.device)
        actor_baseline = (
            torch.from_numpy(baseline[: self.action_horizon].copy())
            .unsqueeze(0)
            .to(self.device)
        )
        actor_language = None
        if self.requires_language_conditioning:
            if language_feature is None:
                raise ValueError("language_feature is required by this residual checkpoint")
            language = np.asarray(language_feature, dtype=np.float32).reshape(-1)
            expected_language = (self.actor.config.language_feature_dim,)
            if language.shape != expected_language or not np.all(np.isfinite(language)):
                raise ValueError(
                    f"language_feature must be finite with shape {expected_language}, "
                    f"got {language.shape}"
                )
            actor_language = torch.from_numpy(language).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            corrected_prefix = self.actor(
                actor_context,
                actor_baseline,
                language_feature=actor_language,
            )[0].cpu().numpy()
        corrected = baseline.copy()
        corrected[: self.action_horizon] = corrected_prefix
        residual = corrected_prefix - baseline[: self.action_horizon]
        if not np.all(np.isfinite(corrected)):
            raise ValueError("Residual actor produced non-finite actions.")
        return ResidualPolicyOutput(
            corrected_actions=corrected,
            residual_actions=residual,
            observation_feature=feature,
        )

    def correct_action_chunk(
        self,
        *,
        camera_images: Mapping[str, Any],
        proprio: np.ndarray,
        baseline_actions: np.ndarray,
        language_feature: np.ndarray | None = None,
    ) -> ResidualPolicyOutput:
        return self.correct_from_feature(
            observation_feature=self.encode_observation(camera_images),
            proprio=proprio,
            baseline_actions=baseline_actions,
            language_feature=language_feature,
        )
