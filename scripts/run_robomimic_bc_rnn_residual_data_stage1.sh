#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${ROBOMIMIC_WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
ROBOMIMIC_SOURCE="${ROBOMIMIC_SOURCE:-${WORKSPACE_ROOT}/robomimic-upstream}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-${WORKSPACE_ROOT}/robosuite-v1.5.1-src}"
TRAIN_ROOT="${ROBOMIMIC_BC_RNN_TRAIN_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_base/train_seed20260821}"
SEED="${ROBOMIMIC_BC_RESIDUAL_DATA_SEED:-20260824}"
EPISODES="${ROBOMIMIC_BC_RESIDUAL_EPISODES:-40}"
TRAIN_STATES="${ROBOMIMIC_BC_RESIDUAL_TRAIN_STATES:-300}"
VALID_STATES="${ROBOMIMIC_BC_RESIDUAL_VALID_STATES:-100}"
OUTPUT_ROOT="${ROBOMIMIC_BC_RESIDUAL_DATA_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_residual_data/seed${SEED}}"

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

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/collect_can_bc_rnn_rollouts.py" \
    --checkpoint "${CHECKPOINT}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --horizon 400 \
    --valid-every 5 \
    --branch-warmup 10 \
    --branch-horizon 20 \
    --branch-stride 5 \
    --output "${OUTPUT_ROOT}/rollouts.hdf5" \
    --state-index "${OUTPUT_ROOT}/state_index.npz" \
    --summary-json "${OUTPUT_ROOT}/rollouts.summary.json" \
    2>&1 | tee "${OUTPUT_ROOT}/rollouts.log"

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/collect_can_symmetric_branches.py" \
    --source-dataset "${OUTPUT_ROOT}/rollouts.hdf5" \
    --state-dataset "${OUTPUT_ROOT}/state_index.npz" \
    --output "${OUTPUT_ROOT}/symmetric_branches.hdf5" \
    --summary-json "${OUTPUT_ROOT}/symmetric_branches.summary.json" \
    --train-states "${TRAIN_STATES}" \
    --valid-states "${VALID_STATES}" \
    --seed "${SEED}" \
    --horizon 20 \
    --intervention-steps 3 \
    --direction-pairs 4 \
    --delta 0.1 \
    --success-bonus 10 \
    --score-margin 0.0001 \
    --flush-every 5 \
    2>&1 | tee "${OUTPUT_ROOT}/symmetric_branches.log"
