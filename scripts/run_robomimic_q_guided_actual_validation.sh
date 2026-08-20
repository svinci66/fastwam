#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
COLLECTION="${ROBOMIMIC_SYMMETRIC_COLLECTION:-${PROJECT_ROOT}/evaluate_results/robomimic_symmetric_collection/can_symmetric_long_400train_100valid_seed20260820.hdf5}"
OUTPUT_ROOT="${ROBOMIMIC_Q_GUIDED_OUTPUT:-${PROJECT_ROOT}/evaluate_results/robomimic_q_guided_actor_constrained}"
SEEDS_CSV="${ROBOMIMIC_Q_GUIDED_SEEDS:-20260820,20260821,20260822}"
VALID_STATES="${ROBOMIMIC_Q_GUIDED_VALID_STATES:-100}"

mkdir -p /tmp/fastwam_numba_cache
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR=/tmp/fastwam_numba_cache
export MUJOCO_GL=egl

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
for SEED in "${SEEDS[@]}"; do
    ACTOR_DIR="${OUTPUT_ROOT}/actor_seed${SEED}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/evaluate_can_residual_branches.py" \
        --collection "${COLLECTION}" \
        --actor-checkpoint "${ACTOR_DIR}/checkpoint.pt" \
        --output-json "${ACTOR_DIR}/actual_valid${VALID_STATES}.json" \
        --states "${VALID_STATES}" \
        --seed 20260820 \
        > "${ACTOR_DIR}/actual_valid${VALID_STATES}.log" 2>&1
done

if [[ "${VALID_STATES}" == "100" ]]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/summarize_can_q_guided_validation.py" \
        --output-root "${OUTPUT_ROOT}" \
        --output-json "${OUTPUT_ROOT}/actual_valid100_multiseed.json"
fi
