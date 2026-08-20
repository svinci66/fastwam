#!/usr/bin/env python3
"""Encode rendered RoboMimic wrist observations with a frozen local SigLIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image


def encode(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import SiglipImageProcessor, SiglipVisionModel

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = SiglipImageProcessor.from_pretrained(args.encoder_path, local_files_only=True)
    model = SiglipVisionModel.from_pretrained(
        args.encoder_path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()
    with h5py.File(args.observations, "r") as source:
        if not bool(source.attrs.get("complete", False)):
            raise ValueError("Deployable observation collection is incomplete")
        states = source["states"]
        count = len(states["completed"])
        features = []
        for start in range(0, count, args.batch_size):
            images = [
                Image.fromarray(image).convert("RGB")
                for image in np.asarray(states["wrist_rgb"][start : start + args.batch_size])
            ]
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(
                device=device, dtype=dtype
            )
            with torch.no_grad():
                batch = model(pixel_values=pixel_values).pooler_output.float().cpu().numpy()
            features.append(batch)
            print(json.dumps({"encoded": min(start + args.batch_size, count), "total": count}), flush=True)
        feature = np.concatenate(features).astype(np.float32)
        arrays = {
            "vision_feature": feature,
            "proprio": np.asarray(states["proprio"], dtype=np.float32),
            "source_demo": states["source_demo"].asstr()[...].astype("U32"),
            "source_split": states["source_split"].asstr()[...].astype("U5"),
            "source_step": np.asarray(states["source_step"], dtype=np.int32),
            "observation_mode": np.asarray("siglip_wrist_proprio"),
            "encoder_path": np.asarray(str(args.encoder_path.resolve())),
            "camera_name": np.asarray(str(source.attrs["camera_name"])),
            "proprio_keys": np.asarray(str(source.attrs["proprio_keys"])),
        }
    if not np.all(np.isfinite(feature)):
        raise RuntimeError("SigLIP produced non-finite features")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {
        "observations": str(args.observations.resolve()),
        "output": str(args.output.resolve()),
        "encoder_path": str(args.encoder_path.resolve()),
        "device": str(device),
        "states": len(feature),
        "vision_feature_shape": list(feature.shape),
        "proprio_shape": list(arrays["proprio"].shape),
        "feature_norm_mean": float(np.mean(np.linalg.norm(feature, axis=1))),
        "all_finite": True,
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    encode(args)


if __name__ == "__main__":
    main()
