#!/usr/bin/env bash
# Source this file before running FastWAM on the Midea training servers:
#   source /data/share/1919650160032350208/sj/fastwam/FastWAM-git/scripts/activate_fastwam_server.sh

export FASTWAM_ROOT=/data/share/1919650160032350208/sj/fastwam
export FASTWAM_REPO="$FASTWAM_ROOT/FastWAM-git"
export FASTWAM_ENV="$FASTWAM_ROOT/envs/fastwam"

if [[ ! -f "$FASTWAM_ENV/bin/activate" ]]; then
  echo "FastWAM environment is missing: $FASTWAM_ENV" >&2
  return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="$FASTWAM_ROOT/toolchains/cuda-12.2"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "Persistent CUDA compiler is missing: $CUDA_HOME/bin/nvcc" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "$FASTWAM_REPO/pyproject.toml" ]]; then
  echo "FastWAM repository is missing: $FASTWAM_REPO" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$FASTWAM_ENV/bin/activate"

export PYTHONNOUSERSITE=1
export PATH="$CUDA_HOME/bin:$FASTWAM_ENV/bin:$PATH"

# Keep all reusable model, compiler, and application caches on persistent storage.
export DIFFSYNTH_MODEL_BASE_PATH="$FASTWAM_ROOT/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HOME="$FASTWAM_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$FASTWAM_ROOT/cache/huggingface/datasets"
export MODELSCOPE_CACHE="$FASTWAM_ROOT/cache/modelscope"
export XDG_CACHE_HOME="$FASTWAM_ROOT/cache/xdg"
export TORCH_HOME="$FASTWAM_ROOT/cache/torch"
export TRITON_CACHE_DIR="$FASTWAM_ROOT/cache/triton-autotune"
export TORCH_EXTENSIONS_DIR="$FASTWAM_ROOT/cache/torch-extensions"
export WANDB_DIR="$FASTWAM_ROOT/logs/wandb"
export WANDB_CACHE_DIR="$FASTWAM_ROOT/cache/wandb"

mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$MODELSCOPE_CACHE" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR"

echo "FastWAM environment activated"
echo "  repository: $FASTWAM_REPO"
echo "  environment: $FASTWAM_ENV"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  checkpoints: $DIFFSYNTH_MODEL_BASE_PATH"
