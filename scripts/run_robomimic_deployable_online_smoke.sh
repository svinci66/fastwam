#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${ROBOMIMIC_WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
ROBOMIMIC_SOURCE="${ROBOMIMIC_SOURCE:-${WORKSPACE_ROOT}/robomimic-upstream}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-${WORKSPACE_ROOT}/robosuite-v1.5.1-src}"
DATASET="${ROBOMIMIC_PAIRED_DATASET:-${WORKSPACE_ROOT}/datasets/robomimic_hf/v1.5/can/paired/low_dim_v15.hdf5}"
SOURCE_ROOT="${ROBOMIMIC_DEPLOYABLE_Q_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_deployable_q_expanded_posttrain}"
ACTOR_ROOT="${ROBOMIMIC_DEPLOYABLE_ACTOR_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_deployable_q_guided_actor}"
ACTOR_SEED="${ROBOMIMIC_DEPLOYABLE_ACTOR_SEED:-20260820}"
EPISODES="${ROBOMIMIC_DEPLOYABLE_ONLINE_EPISODES:-5}"
OUTPUT_ROOT="${ROBOMIMIC_DEPLOYABLE_ONLINE_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_deployable_online}"

mkdir -p "${OUTPUT_ROOT}" /tmp/fastwam_numba_cache
export PYTHONPATH="${ROBOMIMIC_SOURCE}:${ROBOSUITE_SOURCE}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/fastwam_numba_cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

Q_ARGS=()
for Q_SEED in 20260820 20260821 20260822; do
    Q_ARGS+=(--q-checkpoint "${SOURCE_ROOT}/full_state_seed${Q_SEED}/checkpoint.pt")
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/evaluate_can_deployable_online_episodes.py" \
    --dataset "${DATASET}" \
    --actor-dataset "${SOURCE_ROOT}/can_symmetric_residual_actor.npz" \
    --actor-checkpoint "${ACTOR_ROOT}/actor_seed${ACTOR_SEED}/checkpoint.pt" \
    "${Q_ARGS[@]}" \
    --episodes "${EPISODES}" \
    --output-json "${OUTPUT_ROOT}/seed${ACTOR_SEED}_valid${EPISODES}.json"
