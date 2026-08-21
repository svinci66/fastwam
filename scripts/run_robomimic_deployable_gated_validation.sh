#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${ROBOMIMIC_WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
ROBOMIMIC_SOURCE="${ROBOMIMIC_SOURCE:-${WORKSPACE_ROOT}/robomimic-upstream}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-${WORKSPACE_ROOT}/robosuite-v1.5.1-src}"
COLLECTION="${ROBOMIMIC_DEPLOYABLE_COLLECTION:-${PROJECT_ROOT}/evaluate_results/robomimic_symmetric_collection_expanded/can_symmetric_expanded_1200train_300valid_seed20260820.hdf5}"
SOURCE_ROOT="${ROBOMIMIC_DEPLOYABLE_Q_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_deployable_q_expanded_posttrain}"
ACTOR_ROOT="${ROBOMIMIC_DEPLOYABLE_ACTOR_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_deployable_q_guided_actor}"
ACTOR_SEED="${ROBOMIMIC_DEPLOYABLE_ACTOR_SEED:-20260820}"
VALID_STATES="${ROBOMIMIC_DEPLOYABLE_VALID_STATES:-100}"
OUTPUT_ROOT="${ROBOMIMIC_DEPLOYABLE_VALIDATION_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_deployable_gated_validation}"

mkdir -p "${OUTPUT_ROOT}" /tmp/fastwam_numba_cache
export PYTHONPATH="${ROBOMIMIC_SOURCE}:${ROBOSUITE_SOURCE}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/fastwam_numba_cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

Q_ARGS=()
for Q_SEED in 20260820 20260821 20260822; do
    Q_ARGS+=(--q-checkpoint "${SOURCE_ROOT}/full_state_seed${Q_SEED}/checkpoint.pt")
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/evaluate_can_deployable_gated_branches.py" \
    --collection "${COLLECTION}" \
    --actor-dataset "${SOURCE_ROOT}/can_symmetric_residual_actor.npz" \
    --actor-checkpoint "${ACTOR_ROOT}/actor_seed${ACTOR_SEED}/checkpoint.pt" \
    "${Q_ARGS[@]}" \
    --output-json "${OUTPUT_ROOT}/seed${ACTOR_SEED}_valid${VALID_STATES}.json" \
    --states "${VALID_STATES}" \
    --selection-seed 20260820
