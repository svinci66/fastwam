#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${ROBOMIMIC_WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"
DATASET="${ROBOMIMIC_PAIRED_DATASET:-${WORKSPACE_ROOT}/datasets/robomimic_hf/v1.5/can/paired/low_dim_v15.hdf5}"
ROBOMIMIC_SOURCE="${ROBOMIMIC_SOURCE:-${WORKSPACE_ROOT}/robomimic-upstream}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-${WORKSPACE_ROOT}/robosuite-v1.5.1-src}"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
OUTPUT_ROOT="${ROBOMIMIC_COUNTERFACTUAL_OUTPUT_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_counterfactual}"
SEED="${ROBOMIMIC_COUNTERFACTUAL_SEED:-20260819}"

case "${MODE}" in
    smoke)
        NUM_SAMPLES="${ROBOMIMIC_COUNTERFACTUAL_SAMPLES:-20}"
        OUTPUT_PATH="${OUTPUT_ROOT}/can_smoke_${NUM_SAMPLES}_seed${SEED}.hdf5"
        QUALITY_FLAG="--require-smoke-quality"
        ;;
    long)
        NUM_SAMPLES="${ROBOMIMIC_COUNTERFACTUAL_SAMPLES:-5000}"
        OUTPUT_PATH="${OUTPUT_ROOT}/can_long_${NUM_SAMPLES}_seed${SEED}.hdf5"
        QUALITY_FLAG=""
        ;;
    *)
        echo "usage: $0 [smoke|long]" >&2
        exit 2
        ;;
esac

mkdir -p "${OUTPUT_ROOT}" /tmp/fastwam-matplotlib
export PYTHONPATH="${ROBOMIMIC_SOURCE}:${ROBOSUITE_SOURCE}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fastwam-matplotlib}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/collect_can_counterfactual_branches.py" \
    --dataset "${DATASET}" \
    --output "${OUTPUT_PATH}" \
    --num-samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --horizon "${ROBOMIMIC_COUNTERFACTUAL_HORIZON:-20}" \
    --intervention-steps "${ROBOMIMIC_COUNTERFACTUAL_INTERVENTION_STEPS:-3}" \
    --noise-sigmas 0.1 0.2 0.35 \
    --source-splits train valid \
    --late-state-fraction 0.5 \
    --flush-every "${ROBOMIMIC_COUNTERFACTUAL_FLUSH_EVERY:-10}" \
    ${QUALITY_FLAG}
