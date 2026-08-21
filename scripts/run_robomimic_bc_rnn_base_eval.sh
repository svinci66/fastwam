#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${ROBOMIMIC_WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
ROBOMIMIC_SOURCE="${ROBOMIMIC_SOURCE:-${WORKSPACE_ROOT}/robomimic-upstream}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-${WORKSPACE_ROOT}/robosuite-v1.5.1-src}"
TRAIN_ROOT="${ROBOMIMIC_BC_RNN_TRAIN_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_base/train_seed20260821}"
EPISODES="${ROBOMIMIC_BC_RNN_EVAL_EPISODES:-50}"
SEED="${ROBOMIMIC_BC_RNN_EVAL_SEED:-20260822}"
OUTPUT_ROOT="${ROBOMIMIC_BC_RNN_EVAL_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_base_eval}"

if [[ -z "${ROBOMIMIC_BC_RNN_CHECKPOINT:-}" ]]; then
    CHECKPOINT="$(find "${TRAIN_ROOT}" -path '*/models/model_epoch_275_*.pth' -print -quit)"
else
    CHECKPOINT="${ROBOMIMIC_BC_RNN_CHECKPOINT}"
fi
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
    echo "BC-RNN checkpoint not found" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}" /tmp/fastwam_matplotlib /tmp/fastwam_hf /tmp/fastwam_numba_cache
export PYTHONPATH="${ROBOMIMIC_SOURCE}:${ROBOSUITE_SOURCE}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fastwam_matplotlib}"
export HF_HOME="${HF_HOME:-/tmp/fastwam_hf}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/fastwam_numba_cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/evaluate_can_bc_rnn_base.py" \
    --checkpoint "${CHECKPOINT}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --horizon 400 \
    --output-json "${OUTPUT_ROOT}/epoch275_seed${SEED}_${EPISODES}ep.json" \
    2>&1 | tee "${OUTPUT_ROOT}/epoch275_seed${SEED}_${EPISODES}ep.log"
