#!/usr/bin/env bash

# Activate the persistent FastWAM server runtime in the current shell.
#
# Usage:
#   source scripts/activate_fastwam_runtime.sh
#
# The defaults match the persistent AIMP layout used by this project. Every
# path can be overridden before sourcing the script. The script intentionally
# lives in the repository and only exports paths to large assets; model/data
# files under /data/share are never copied into Git.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Error: this script must be sourced, not executed." >&2
    echo "Use: source $0" >&2
    exit 2
fi

export FASTWAM_ROOT="${FASTWAM_ROOT:-/data/share/1919650160032350208/sj/fastwam}"
export FASTWAM_REPO="${FASTWAM_REPO:-$FASTWAM_ROOT/FastWAM-git}"
export FASTWAM_ENV="${FASTWAM_ENV:-$FASTWAM_ROOT/envs/fastwam}"

export CUDA_HOME="${CUDA_HOME:-$FASTWAM_ROOT/toolchains/cuda-12.2}"
export FASTWAM_GL_ROOT="${FASTWAM_GL_ROOT:-$FASTWAM_ROOT/toolchains/glvnd-ubuntu2204}"
export OSMESA_ROOT="${OSMESA_ROOT:-$FASTWAM_ROOT/toolchains/osmesa-ubuntu2204}"

export LIBERO_ROOT="${LIBERO_ROOT:-$FASTWAM_ROOT/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$FASTWAM_ROOT/config/libero}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$FASTWAM_ROOT/checkpoints}"

export HF_HOME="${HF_HOME:-/data/share/1919650160032350208/sj/hf_cache_shared}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export SIGLIP_REVISION="${SIGLIP_REVISION:-7067f6db2baa594bab7c6d965fe488c7ac62f1c8}"
export SIGLIP_PATH="${SIGLIP_PATH:-$HF_HUB_CACHE/models--google--siglip-so400m-patch14-384/snapshots/$SIGLIP_REVISION}"

export FASTWAM_RUN_DIR="${FASTWAM_RUN_DIR:-$FASTWAM_REPO/runs/libero_uncond_2cam224_1e-4/smoke_h20_8gpu_2steps}"
export FASTWAM_CKPT="${FASTWAM_CKPT:-$FASTWAM_RUN_DIR/checkpoints/weights/step_248000.pt}"
export DATASET_STATS="${DATASET_STATS:-$FASTWAM_RUN_DIR/dataset_stats.json}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$FASTWAM_ROOT/cache/triton-autotune}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$FASTWAM_ROOT/cache/torch-extensions}"
export TORCH_HOME="${TORCH_HOME:-$FASTWAM_ROOT/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$FASTWAM_ROOT/cache/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$FASTWAM_ROOT/cache/matplotlib}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
# Official LIBERO init-state files contain NumPy objects and predate PyTorch
# 2.6's weights_only=True default. They come from the pinned trusted LIBERO
# checkout, so retain the legacy loader behavior without patching that source.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

mkdir -p \
    "$TRITON_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" \
    "$TORCH_HOME" \
    "$XDG_CACHE_HOME" \
    "$MPLCONFIGDIR"

if [[ ! -x "$FASTWAM_ENV/bin/python" ]]; then
    echo "Error: persistent FastWAM Python not found: $FASTWAM_ENV/bin/python" >&2
    return 1
fi
if [[ ! -f "$FASTWAM_ENV/bin/activate" ]]; then
    echo "Error: persistent FastWAM activation script not found: $FASTWAM_ENV/bin/activate" >&2
    return 1
fi

# shellcheck disable=SC1091
source "$FASTWAM_ENV/bin/activate"

_fastwam_prepend_unique() {
    local variable_name="$1"
    local directory="$2"
    local -n variable_reference="$variable_name"

    [[ -d "$directory" ]] || return 0
    case ":${variable_reference-}:" in
        *":$directory:"*) ;;
        *)
            variable_reference="$directory${variable_reference:+:$variable_reference}"
            export "$variable_name"
            ;;
    esac
}

# The environment Python wins over system Python; persistent nvcc remains
# available for DeepSpeed/Torch extension builds.
_fastwam_prepend_unique PATH "$CUDA_HOME/bin"
_fastwam_prepend_unique PATH "$FASTWAM_ENV/bin"

# GLVND supplies libGL/libEGL/GLib. OSMesa supplies a software renderer for
# compute-only containers that do not mount NVIDIA graphics driver libraries.
_fastwam_prepend_unique LD_LIBRARY_PATH "$FASTWAM_GL_ROOT/usr/lib/x86_64-linux-gnu"
_fastwam_prepend_unique LD_LIBRARY_PATH "$OSMESA_ROOT/lib/x86_64-linux-gnu"
_fastwam_prepend_unique LD_LIBRARY_PATH "$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"

# Keep imports independent of stale editable-install metadata after conda-pack.
_fastwam_prepend_unique PYTHONPATH "$LIBERO_ROOT"
_fastwam_prepend_unique PYTHONPATH "$FASTWAM_REPO"
_fastwam_prepend_unique PYTHONPATH "$FASTWAM_REPO/src"

unset -f _fastwam_prepend_unique

_fastwam_nvidia_egl_available() {
    [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]] || return 1
    # Avoid grep -q here: with pipefail enabled by the caller, its early exit
    # can propagate ldconfig's SIGPIPE as a false negative.
    ldconfig -p 2>/dev/null | grep 'libEGL_nvidia' >/dev/null
}

# FASTWAM_RENDER_REQUEST accepts auto, egl, or osmesa. The default is safe for
# both compute-only notebook containers and graphics-enabled evaluation jobs.
export FASTWAM_RENDER_REQUEST="${FASTWAM_RENDER_REQUEST:-auto}"
case "$FASTWAM_RENDER_REQUEST" in
    auto)
        if _fastwam_nvidia_egl_available; then
            export FASTWAM_RENDER_BACKEND=nvidia-egl
        else
            export FASTWAM_RENDER_BACKEND=osmesa
        fi
        ;;
    egl|nvidia-egl)
        if ! _fastwam_nvidia_egl_available; then
            echo "Error: NVIDIA EGL was requested but the container has no NVIDIA EGL vendor runtime." >&2
            echo "Use FASTWAM_RENDER_REQUEST=osmesa or recreate the container with NVIDIA graphics capability." >&2
            unset -f _fastwam_nvidia_egl_available
            return 1
        fi
        export FASTWAM_RENDER_BACKEND=nvidia-egl
        ;;
    osmesa)
        export FASTWAM_RENDER_BACKEND=osmesa
        ;;
    *)
        echo "Error: FASTWAM_RENDER_REQUEST must be auto, egl, or osmesa; got: $FASTWAM_RENDER_REQUEST" >&2
        unset -f _fastwam_nvidia_egl_available
        return 1
        ;;
esac
unset -f _fastwam_nvidia_egl_available

if [[ "$FASTWAM_RENDER_BACKEND" == "nvidia-egl" ]]; then
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
else
    export MUJOCO_GL=osmesa
    export PYOPENGL_PLATFORM=osmesa
    unset __EGL_VENDOR_LIBRARY_FILENAMES 2>/dev/null || true
fi

fastwam_runtime_check() {
    local missing=0
    local required_paths=(
        "$FASTWAM_ENV/bin/python"
        "$CUDA_HOME/bin/nvcc"
        "$LIBERO_ROOT/libero/libero/bddl_files"
        "$LIBERO_ROOT/libero/libero/init_files"
        "$LIBERO_ROOT/libero/libero/assets"
        "$LIBERO_CONFIG_PATH/config.yaml"
        "$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json"
        "$DIFFSYNTH_MODEL_BASE_PATH/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
        "$SIGLIP_PATH/config.json"
        "$SIGLIP_PATH/preprocessor_config.json"
        "$SIGLIP_PATH/model.safetensors"
        "$FASTWAM_CKPT"
        "$DATASET_STATS"
    )

    if [[ "$FASTWAM_RENDER_BACKEND" == "osmesa" ]]; then
        required_paths+=("$OSMESA_ROOT/usr/lib/x86_64-linux-gnu/libOSMesa.so.8")
    else
        required_paths+=("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    fi

    local path
    for path in "${required_paths[@]}"; do
        if [[ ! -e "$path" ]]; then
            echo "MISSING: $path" >&2
            missing=1
        fi
    done
    if ((missing != 0)); then
        echo "FastWAM persistent asset check: FAILED" >&2
        return 1
    fi

    MUJOCO_GL="$MUJOCO_GL" PYOPENGL_PLATFORM="$PYOPENGL_PLATFORM" python - <<'PY'
import os
import sys

import cv2
import fastwam
import mujoco
import robosuite
import torch
from libero.libero import benchmark

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() >= 1, "No CUDA GPU is visible"

ctx = mujoco.GLContext(64, 64)
ctx.make_current()
ctx.free()

print("FastWAM persistent runtime check: OK")
print("python:", sys.executable)
print("torch:", torch.__version__)
print("gpu_count:", torch.cuda.device_count())
print("gpu:", torch.cuda.get_device_name(0))
print("cv2:", cv2.__version__)
print("mujoco:", getattr(mujoco, "__version__", "unknown"))
print("robosuite:", getattr(robosuite, "__version__", "unknown"))
print("renderer:", os.environ["MUJOCO_GL"])
print("LIBERO benchmarks:", sorted(benchmark.get_benchmark_dict().keys()))
PY
}

echo "FastWAM persistent environment activated"
echo "  repository:    $FASTWAM_REPO"
echo "  environment:   $FASTWAM_ENV"
echo "  CUDA_HOME:     $CUDA_HOME"
echo "  renderer:      $FASTWAM_RENDER_BACKEND"
echo "  LIBERO_ROOT:   $LIBERO_ROOT"
echo "  checkpoints:   $DIFFSYNTH_MODEL_BASE_PATH"
echo "  FastWAM ckpt:  $FASTWAM_CKPT"
echo "  SigLIP:        $SIGLIP_PATH"
echo
echo "Run validation with: fastwam_runtime_check"
echo "Enter repository with: cd \"$FASTWAM_REPO\""
