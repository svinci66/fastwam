#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
DATA_ROOT="${ROBOMIMIC_TERMINAL_SAFETY_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_terminal_safety_seed20260825}"
DATASET="${DATA_ROOT}/terminal_safety_pairwise.npz"
OUTPUT_ROOT="${ROBOMIMIC_TERMINAL_Q_OUTPUT:-${DATA_ROOT}/critic_outcome_weight20}"
SEEDS_CSV="${ROBOMIMIC_TERMINAL_Q_SEEDS:-20260820,20260821,20260822}"
OUTCOME_WEIGHT="${ROBOMIMIC_TERMINAL_OUTCOME_WEIGHT:-20}"

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
for SEED in "${SEEDS[@]}"; do
    COMMON_ARGS=(
        --dataset "${DATASET}"
        --seed "${SEED}"
        --hidden-dims 64 64 32
        --sample-weight-key terminal_outcome_changed
        --sample-weight-multiplier "${OUTCOME_WEIGHT}"
    )
    ACTION_DIR="${OUTPUT_ROOT}/action_only_seed${SEED}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/train_can_pairwise_q.py" \
        "${COMMON_ARGS[@]}" \
        --output-dir "${ACTION_DIR}" \
        --state-mode action_only
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/train_can_pairwise_q.py" \
        "${COMMON_ARGS[@]}" \
        --output-dir "${OUTPUT_ROOT}/full_state_seed${SEED}" \
        --state-mode full \
        --initialize-action-checkpoint "${ACTION_DIR}/checkpoint.pt" \
        --teacher-regularization 1.0
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/summarize_can_pairwise_q.py" \
    --output-root "${OUTPUT_ROOT}" \
    --output-json "${OUTPUT_ROOT}/q_comparison_multiseed.json"
