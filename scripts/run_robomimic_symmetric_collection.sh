#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${ROBOMIMIC_WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"
SOURCE_DATASET="${ROBOMIMIC_PAIRED_DATASET:-${WORKSPACE_ROOT}/datasets/robomimic_hf/v1.5/can/paired/low_dim_v15.hdf5}"
STATE_DATASET="${ROBOMIMIC_RESIDUAL_STATE_DATASET:-${PROJECT_ROOT}/evaluate_results/robomimic_residual_actor_phase1/can_residual_actor.npz}"
ROBOMIMIC_SOURCE="${ROBOMIMIC_SOURCE:-${WORKSPACE_ROOT}/robomimic-upstream}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-${WORKSPACE_ROOT}/robosuite-v1.5.1-src}"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
OUTPUT_ROOT="${ROBOMIMIC_SYMMETRIC_OUTPUT_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_symmetric_collection}"
SEED="${ROBOMIMIC_SYMMETRIC_SEED:-20260820}"

case "${MODE}" in
    smoke)
        TRAIN_STATES="${ROBOMIMIC_SYMMETRIC_TRAIN_STATES:-40}"
        VALID_STATES="${ROBOMIMIC_SYMMETRIC_VALID_STATES:-10}"
        OUTPUT_PATH="${OUTPUT_ROOT}/can_symmetric_smoke_${TRAIN_STATES}train_${VALID_STATES}valid_seed${SEED}.hdf5"
        QUALITY_FLAG="--require-smoke-quality"
        ;;
    long)
        TRAIN_STATES="${ROBOMIMIC_SYMMETRIC_TRAIN_STATES:-400}"
        VALID_STATES="${ROBOMIMIC_SYMMETRIC_VALID_STATES:-100}"
        OUTPUT_PATH="${OUTPUT_ROOT}/can_symmetric_long_${TRAIN_STATES}train_${VALID_STATES}valid_seed${SEED}.hdf5"
        QUALITY_FLAG=""
        ;;
    expanded)
        TRAIN_STATES="${ROBOMIMIC_SYMMETRIC_TRAIN_STATES:-1200}"
        VALID_STATES="${ROBOMIMIC_SYMMETRIC_VALID_STATES:-300}"
        OUTPUT_PATH="${OUTPUT_ROOT}/can_symmetric_expanded_${TRAIN_STATES}train_${VALID_STATES}valid_seed${SEED}.hdf5"
        QUALITY_FLAG=""
        ;;
    *)
        echo "usage: $0 [smoke|long|expanded]" >&2
        exit 2
        ;;
esac

mkdir -p "${OUTPUT_ROOT}" /tmp/fastwam-matplotlib
export PYTHONPATH="${ROBOMIMIC_SOURCE}:${ROBOSUITE_SOURCE}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fastwam-matplotlib}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/collect_can_symmetric_branches.py" \
    --source-dataset "${SOURCE_DATASET}" \
    --state-dataset "${STATE_DATASET}" \
    --output "${OUTPUT_PATH}" \
    --train-states "${TRAIN_STATES}" \
    --valid-states "${VALID_STATES}" \
    --seed "${SEED}" \
    --horizon 20 \
    --intervention-steps 3 \
    --direction-pairs 4 \
    --delta 0.1 \
    --flush-every 5 \
    ${QUALITY_FLAG}
