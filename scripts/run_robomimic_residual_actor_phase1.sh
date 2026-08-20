#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
PAIRWISE_ROOT="${ROBOMIMIC_PAIRWISE_Q_OUTPUT:-${PROJECT_ROOT}/evaluate_results/robomimic_pairwise_q_phase1}"
OUTPUT_ROOT="${ROBOMIMIC_RESIDUAL_ACTOR_OUTPUT:-${PROJECT_ROOT}/evaluate_results/robomimic_residual_actor_phase1}"
SEEDS_CSV="${ROBOMIMIC_RESIDUAL_ACTOR_SEEDS:-20260820,20260821,20260822}"
DATASET="${OUTPUT_ROOT}/can_residual_actor.npz"

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/prepare_can_residual_actor_dataset.py" \
    --pairwise-dataset "${PAIRWISE_ROOT}/can_pairwise_q_clean.npz" \
    --output "${DATASET}" \
    --report-json "${OUTPUT_ROOT}/preparation_report.json" \
    --residual-scale 0.1

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
for SEED in "${SEEDS[@]}"; do
    ACTOR_DIR="${OUTPUT_ROOT}/actor_seed${SEED}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/train_can_residual_actor.py" \
        --dataset "${DATASET}" \
        --output-dir "${ACTOR_DIR}" \
        --seed "${SEED}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/audit_can_residual_q_support.py" \
        --dataset "${DATASET}" \
        --actor-checkpoint "${ACTOR_DIR}/checkpoint.pt" \
        --q-checkpoint "${PAIRWISE_ROOT}/full_state_seed${SEED}/checkpoint.pt" \
        --output-support "${ACTOR_DIR}/support_index_k5_q95.npz" \
        --output-json "${ACTOR_DIR}/gate_audit.json" \
        --k 5 \
        --support-quantile 0.95 \
        --q-threshold-quantile 0.95
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/summarize_can_residual_actor.py" \
    --output-root "${OUTPUT_ROOT}" \
    --output-json "${OUTPUT_ROOT}/summary_multiseed.json"
