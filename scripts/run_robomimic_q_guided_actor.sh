#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
SOURCE_ROOT="${ROBOMIMIC_Q_GUIDED_SOURCE_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_symmetric_gradient_posttrain}"
OUTPUT_ROOT="${ROBOMIMIC_Q_GUIDED_OUTPUT:-${PROJECT_ROOT}/evaluate_results/robomimic_q_guided_actor}"
SEEDS_CSV="${ROBOMIMIC_Q_GUIDED_SEEDS:-20260820,20260821,20260822}"
DATASET="${SOURCE_ROOT}/can_symmetric_residual_actor.npz"

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

Q_ARGS=()
for Q_SEED in 20260820 20260821 20260822; do
    Q_ARGS+=(--q-checkpoint "${SOURCE_ROOT}/full_state_seed${Q_SEED}/checkpoint.pt")
done

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
for SEED in "${SEEDS[@]}"; do
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/train_can_q_guided_residual_actor.py" \
        --dataset "${DATASET}" \
        "${Q_ARGS[@]}" \
        --output-dir "${OUTPUT_ROOT}/actor_seed${SEED}" \
        --residual-l2-weight 5.0 \
        --zero-target-weight 10.0 \
        --actor-residual-scale 0.03 \
        --seed "${SEED}"
done
