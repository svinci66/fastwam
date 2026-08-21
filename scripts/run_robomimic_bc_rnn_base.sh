#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${ROBOMIMIC_WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
ROBOMIMIC_SOURCE="${ROBOMIMIC_SOURCE:-${WORKSPACE_ROOT}/robomimic-upstream}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-${WORKSPACE_ROOT}/robosuite-v1.5.1-src}"
DATASET="${ROBOMIMIC_PAIRED_DATASET:-${WORKSPACE_ROOT}/datasets/robomimic_hf/v1.5/can/paired/low_dim_v15.hdf5}"
MODE="${ROBOMIMIC_BC_RNN_MODE:-smoke}"
SEED="${ROBOMIMIC_BC_RNN_SEED:-20260821}"
OUTPUT_ROOT="${ROBOMIMIC_BC_RNN_OUTPUT_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_base}"
RUN_ROOT="${OUTPUT_ROOT}/${MODE}_seed${SEED}"

mkdir -p "${RUN_ROOT}" /tmp/fastwam_matplotlib /tmp/fastwam_hf /tmp/fastwam_numba_cache
export PYTHONPATH="${ROBOMIMIC_SOURCE}:${ROBOSUITE_SOURCE}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fastwam_matplotlib}"
export HF_HOME="${HF_HOME:-/tmp/fastwam_hf}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/fastwam_numba_cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

EXTRA_ARGS=()
if [[ -n "${ROBOMIMIC_BC_RNN_EPOCHS:-}" ]]; then
    EXTRA_ARGS+=(--epochs "${ROBOMIMIC_BC_RNN_EPOCHS}")
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/train_can_bc_rnn_base.py" \
    --dataset "${DATASET}" \
    --output-dir "${RUN_ROOT}" \
    --config-output "${RUN_ROOT}/config.json" \
    --mode "${MODE}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/train.log"
