#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
COLLECTION="${ROBOMIMIC_COUNTERFACTUAL_COLLECTION:-${PROJECT_ROOT}/evaluate_results/robomimic_counterfactual/can_long_prefix_5000_seed20260819.hdf5}"
OUTPUT_ROOT="${ROBOMIMIC_PAIRWISE_Q_OUTPUT:-${PROJECT_ROOT}/evaluate_results/robomimic_pairwise_q_phase1}"
CLEAN_DATA="${OUTPUT_ROOT}/can_pairwise_q_clean.npz"
SEEDS_CSV="${ROBOMIMIC_PAIRWISE_Q_SEEDS:-20260820,20260821,20260822}"

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/prepare_can_pairwise_q_dataset.py" \
    --collection "${COLLECTION}" \
    --output "${CLEAN_DATA}" \
    --report-json "${OUTPUT_ROOT}/cleaning_report.json"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
for SEED in "${SEEDS[@]}"; do
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/train_can_pairwise_q.py" \
        --dataset "${CLEAN_DATA}" \
        --output-dir "${OUTPUT_ROOT}/full_state_seed${SEED}" \
        --state-mode full \
        --seed "${SEED}"

    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/train_can_pairwise_q.py" \
        --dataset "${CLEAN_DATA}" \
        --output-dir "${OUTPUT_ROOT}/action_only_seed${SEED}" \
        --state-mode action_only \
        --seed "${SEED}"
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/summarize_can_pairwise_q.py" \
    --output-root "${OUTPUT_ROOT}" \
    --output-json "${OUTPUT_ROOT}/comparison_multiseed.json"
