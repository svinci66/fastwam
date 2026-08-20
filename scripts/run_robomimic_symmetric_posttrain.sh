#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
COLLECTION="${ROBOMIMIC_SYMMETRIC_COLLECTION:-${PROJECT_ROOT}/evaluate_results/robomimic_symmetric_collection/can_symmetric_long_400train_100valid_seed20260820.hdf5}"
ACTOR_TARGET_MODE="${ROBOMIMIC_ACTOR_TARGET_MODE:-symmetric_gradient}"
OUTPUT_ROOT="${ROBOMIMIC_SYMMETRIC_POSTTRAIN_OUTPUT:-${PROJECT_ROOT}/evaluate_results/robomimic_symmetric_gradient_posttrain}"
SEEDS_CSV="${ROBOMIMIC_SYMMETRIC_POSTTRAIN_SEEDS:-20260820,20260821,20260822}"
Q_DATASET="${OUTPUT_ROOT}/can_symmetric_pairwise_q.npz"
ACTOR_DATASET="${OUTPUT_ROOT}/can_symmetric_residual_actor.npz"

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/prepare_can_symmetric_training_datasets.py" \
    --collection "${COLLECTION}" \
    --q-output "${Q_DATASET}" \
    --actor-output "${ACTOR_DATASET}" \
    --report-json "${OUTPUT_ROOT}/preparation_report.json" \
    --residual-scale 0.1 \
    --actor-target-mode "${ACTOR_TARGET_MODE}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
for SEED in "${SEEDS[@]}"; do
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/train_can_pairwise_q.py" \
        --dataset "${Q_DATASET}" \
        --output-dir "${OUTPUT_ROOT}/full_state_seed${SEED}" \
        --state-mode full \
        --seed "${SEED}"

    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/train_can_pairwise_q.py" \
        --dataset "${Q_DATASET}" \
        --output-dir "${OUTPUT_ROOT}/action_only_seed${SEED}" \
        --state-mode action_only \
        --seed "${SEED}"

    ACTOR_DIR="${OUTPUT_ROOT}/actor_seed${SEED}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/train_can_residual_actor.py" \
        --dataset "${ACTOR_DATASET}" \
        --output-dir "${ACTOR_DIR}" \
        --seed "${SEED}"

    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/audit_can_residual_q_support.py" \
        --dataset "${ACTOR_DATASET}" \
        --actor-checkpoint "${ACTOR_DIR}/checkpoint.pt" \
        --q-checkpoint "${OUTPUT_ROOT}/full_state_seed${SEED}/checkpoint.pt" \
        --output-support "${ACTOR_DIR}/support_index_k5_q95.npz" \
        --output-json "${ACTOR_DIR}/gate_audit.json" \
        --k 5 \
        --support-quantile 0.95 \
        --q-threshold-quantile 0.95 \
        --seed "${SEED}"
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/summarize_can_residual_actor.py" \
    --output-root "${OUTPUT_ROOT}" \
    --output-json "${OUTPUT_ROOT}/actor_summary_multiseed.json"

# Keep the strict Q research gate last: a failed ablation gate must not prevent
# the actor and Q/OOD summaries above from being written.
"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/summarize_can_pairwise_q.py" \
    --output-root "${OUTPUT_ROOT}" \
    --output-json "${OUTPUT_ROOT}/q_comparison_multiseed.json"
